from torch import nn

from .utils import ImageSizeManagerMixin
from .ecdp import ECDP
from .prior import GaussianPrior, LogisticPrior
from .rrdbnet import RRDBNet


def _build_model_from_options(options):
    if options.model.type == "ecdp":
        return ECDP(options)
    elif options.model.type == "rrdbnet":
        return RRDBNet(options)
    else:
        raise ValueError("unknown model type")


def _build_prior_from_options(options):
    input_channels = options.model.input_channels

    def shape_gen(x, y):
        return input_channels * x * y

    if options.model.prior == "logistic":
        return LogisticPrior(shape_gen)
    elif options.model.prior == "gaussian":
        return GaussianPrior(shape_gen)
    else:
        raise ValueError("unknown prior type")


class ConditionalDensityModel(ImageSizeManagerMixin, nn.Module):
    def __init__(self, options):
        super().__init__()

        self.model = _build_model_from_options(options)
        self.prior = _build_prior_from_options(options)

    def forward(self, *args, mode, **kwargs):
        if mode == "loss":
            return self._calculate_loss(*args, **kwargs)
        elif mode == "generate":
            return self._generate_sample(*args, **kwargs)
        elif mode == "random-generate":
            return self._generate_random_sample(*args, **kwargs)
        elif mode == "sample-latent":
            return self.prior.sample(*args, **kwargs)
        else:
            raise ValueError("invalid forward mode")

    def _calculate_loss(self, x, cond):
        x = self.model(x, cond=cond, mode="loss")
        return x

    def _generate_sample(self, z, cond):
        x = self.model(z, cond=cond, mode="generate")
        return x

    def _generate_random_sample(self, cond, t):
        z = self.prior.sample(cond.shape[0], device=cond.device) * t
        return self._generate_sample(z, cond)


def make_model(options):
    return ConditionalDensityModel(options)
