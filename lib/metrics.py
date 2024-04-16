import lpips
import piq
import torch
import torch.nn.functional as F
from torch import nn


def psnr(x, y):
    if x.isnan().any() or y.isnan().any():
        return torch.tensor(float("nan"), device=x.device)
    x = x.clamp(0, 1)
    y = y.clamp(0, 1)
    return piq.psnr(x, y, data_range=1.0, reduction="none")


def ssim(x, y):
    # piq.ssim() with downsample=False is equivalent to skimage's ssim
    # with gaussian_weights=True and use_sample_covariance=False, i.e.
    # the same calculation as the original paper.
    if x.isnan().any() or y.isnan().any():
        return torch.tensor(float("nan"), device=x.device)
    x = x.clamp(0, 1)
    y = y.clamp(0, 1)
    return piq.ssim(x, y, data_range=1.0, downsample=False, reduction="none")


class LPIPS(nn.Module):
    def __init__(self):
        super().__init__()

        # We need alex net, so can't use piq.LPIPS
        self.lpips = lpips.LPIPS(net="alex", verbose=False)

    def forward(self, x, y):
        x = x.clamp(0, 1)
        y = y.clamp(0, 1)
        return self.lpips(x, y, normalize=True).view(x.shape[0])


def gan_loss(x, target):
    return F.binary_cross_entropy_with_logits(
        x, torch.full_like(x, target), reduction="none"
    )
