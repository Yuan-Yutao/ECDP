import math

import torch
from torch import nn

from .utils import ImageSizeMixin


class Prior(ImageSizeMixin, nn.Module):
    def __init__(self, shape_gen):
        super().__init__()

        self.shape_gen = shape_gen

    def forward(self, x):
        raise NotImplementedError

    def sample(self, batch_size, *, device=None):
        raise NotImplementedError

    @property
    def latent_size(self):
        return self.shape_gen(self.image_size_x, self.image_size_y)


class LogisticPrior(Prior):
    def forward(self, x):
        return -2 * torch.logaddexp(x / 2, -x / 2).sum(dim=1)

    def sample(self, batch_size, *, device=None):
        x = torch.rand((batch_size, self.latent_size), device=device)
        x = torch.logit(x, eps=1e-6)
        return x


class GaussianPrior(Prior):
    def forward(self, x):
        c = math.log(2 * math.pi)
        return (-(x**2) - c).sum(dim=1) / 2

    def sample(self, batch_size, *, device=None):
        x = torch.randn((batch_size, self.latent_size), device=device)
        return x
