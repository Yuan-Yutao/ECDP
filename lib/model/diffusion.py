import math

import torch
import torch.utils.checkpoint
from torch import nn
from torchdiffeq import odeint_adjoint as odeint


def make_rand_like(x):
    return torch.randn_like(x)


class Diffusion(nn.Module):
    def __init__(self, network):
        super().__init__()

        t_max = 300
        self.b_min = 0.1
        self.b_max = 20
        ts = torch.arange(0, t_max + 1) / t_max
        alpha_bars_ext = self.t_to_aa(ts)
        alphas = alpha_bars_ext[1:] / alpha_bars_ext[:-1]
        betas = 1 - alphas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars_ext[1:])

        self.network = network

        self.t_count = 0

        self.gen_steps = 0
        self.gen_verbose = False

        self.ddim = False
        self.use_ode = False

    def t_to_aa(self, ts):
        return torch.exp(-ts * (ts * (self.b_max - self.b_min) / 2 + self.b_min))

    def normalize(self, x, *, cond):
        cond, diff_mean, diff_scale = cond
        if self.training:
            ts = (
                torch.rand(x.shape[0], device=x.device) * (len(self.alpha_bars) - 0) + 0
            )
            sampled_aa = self.t_to_aa(ts / len(self.alpha_bars))
            self.t_count = 0
        else:
            ts = (
                torch.arange(self.t_count, self.t_count + x.shape[0], device=x.device)
                * 83
                % len(self.alpha_bars)
            ) + 1
            sampled_aa = self.alpha_bars[ts - 1]
            self.t_count += x.shape[0]

        eps = make_rand_like(x)
        xt = (
            x * sampled_aa.sqrt()[:, None, None, None]
            + eps * (1 - sampled_aa).sqrt()[:, None, None, None]
        )
        network_pred_eps_x0 = self.network(xt, ts, cond)
        network_pred_eps, network_pred_x0 = network_pred_eps_x0.split(
            network_pred_eps_x0.shape[1] // 2, dim=1
        )
        network_pred = (
            network_pred_eps * sampled_aa.sqrt()[:, None, None, None]
            - network_pred_x0 * (1 - sampled_aa).sqrt()[:, None, None, None]
        )

        loss1 = (network_pred_eps - eps).square().sum(dim=[1, 2, 3])
        loss2 = (network_pred_x0 - x).square().sum(dim=[1, 2, 3])
        loss = loss1 + loss2

        return loss

    def generate(self, x, *, cond):
        def next_step(x, cond, t, the_eps):
            network_pred_eps_x0 = self.network(
                x, torch.tensor([t + 1], device=x.device).expand(x.shape[0]), cond
            )
            network_pred_eps, network_pred_x0 = network_pred_eps_x0.split(
                network_pred_eps_x0.shape[1] // 2, dim=1
            )
            network_pred = (
                network_pred_eps * self.alpha_bars[t].sqrt()
                - network_pred_x0 * (1 - self.alpha_bars[t]).sqrt()
            )
            eps_pred = (
                x * (1 - self.alpha_bars[t]).sqrt()
                + network_pred * self.alpha_bars[t].sqrt()
            )
            if self.ddim:
                eps_coeff = (1 - self.alphas[t]) / (
                    (1 - self.alpha_bars[t]).sqrt()
                    + (self.alphas[t] - self.alpha_bars[t]).sqrt()
                )
                x = (x - eps_coeff * eps_pred) / self.alphas[t].sqrt()
            else:
                if t == end_step:
                    x = (
                        x - (1 - self.alpha_bars[t]).sqrt() * eps_pred
                    ) / self.alpha_bars[t].sqrt()
                else:
                    eps_coeff = (1 - self.alphas[t]) / (1 - self.alpha_bars[t]).sqrt()
                    x = (x - eps_coeff * eps_pred) / self.alphas[t].sqrt()
                    x = x + the_eps * (self.get_inject_noise_var(t) ** 0.5)
            return x

        cond, diff_mean, diff_scale = cond
        if self.use_ode:
            x = odeint(
                lambda t, net_in: self.ode_func(net_in, t, cond),
                x,
                torch.tensor([len(self.alpha_bars), 0.1], device=x.device),
                atol=1e-3,
                rtol=1e-3,
                options={"jump_t": torch.tensor([0.1], device=x.device)},
                adjoint_params=self.parameters(),
            )[1]
        else:
            end_step = 0 if self.ddim else 5
            for t in range(self.gen_steps - 1, end_step - 1, -1):
                if t == self.gen_steps - 1:
                    x = next_step(x, cond, t, make_rand_like(x))
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        next_step, x, cond, t, make_rand_like(x)
                    )
                if self.gen_verbose:
                    print(t, x.mean().item(), x.std().item())
        return x

    def ode_func(self, net_in, t, cond):
        if self.gen_verbose:
            print("eval at", t)
        sampled_aa = self.t_to_aa(t / len(self.alpha_bars))
        sampled_beta = self.b_min + (self.b_max - self.b_min) * (
            t / len(self.alpha_bars)
        )
        network_pred_eps_x0 = self.network(net_in, t[None], cond)
        network_pred_eps, network_pred_x0 = network_pred_eps_x0.split(
            network_pred_eps_x0.shape[1] // 2, dim=1
        )
        network_pred = (
            network_pred_eps * sampled_aa.sqrt()
            - network_pred_x0 * (1 - sampled_aa).sqrt()
        )
        eps_pred = net_in * (1 - sampled_aa).sqrt() + network_pred * sampled_aa.sqrt()
        value = (
            -0.5 * sampled_beta * net_in
            + 0.5 * sampled_beta * eps_pred / (1 - sampled_aa).sqrt()
        )
        return value / len(self.alpha_bars)

    def get_inject_noise_var(self, t):
        if t == 0:
            return 0
        beta_tilde = (
            (1 - self.alpha_bars[t] / self.alphas[t])
            / (1 - self.alpha_bars[t])
            * self.betas[t]
        )
        return math.exp(math.log(beta_tilde) * 0.0 + math.log(self.betas[t]) * 1.0)

    def set_generate_steps(self, steps):
        self.gen_steps = steps

    def set_generate_verbose(self, verbose):
        self.gen_verbose = verbose
