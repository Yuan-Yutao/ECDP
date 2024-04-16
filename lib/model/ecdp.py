import torch
import torchvision.transforms.functional as FV
from torch import nn

from .diffusion import Diffusion
from .utils import ImageSizeMixin
from .rrdbnet import RRDBNet
from .unet import UNet


class ECDP(ImageSizeMixin, nn.Module):
    def __init__(self, options):
        super().__init__()

        self.in_channels = options.model.input_channels
        self.diffusion = Diffusion(
            UNet(
                self.in_channels,
                128,
                options.model.rrdb_channels,
                out_channels=2 * self.in_channels,
            )
        )

        # TODO: load pretrained rrdb
        # you should place the parameters of pretrained RRDBNet to the
        # files here
        rrdbnet = RRDBNet(options, rrdb_channels=options.model.rrdb_network_features)
        if options.train.dataset == "df2k":
            t = torch.load(
                "pretrained-rrdbnet-df2k.pt", map_location=torch.device("cpu")
            )
        elif options.train.dataset == "celeba":
            t = torch.load(
                "pretrained-rrdbnet-celeba.pt", map_location=torch.device("cpu")
            )
        elif options.train.dataset == "imagenet":
            t = torch.load(
                "pretrained-rrdbnet-imagenet.pt", map_location=torch.device("cpu")
            )
        elif options.train.dataset == "ffhq":
            t = torch.load(
                "pretrained-rrdbnet-ffhq.pt", map_location=torch.device("cpu")
            )
        else:
            raise ValueError("unknown dataset")
        rrdbnet.load_state_dict(t)
        self.lr_feats = rrdbnet.rrdb

    def forward(self, *args, mode, **kwargs):
        if mode == "loss":
            return self._calculate_loss(*args, **kwargs)
        elif mode == "generate":
            return self._generate_sample(*args, **kwargs)
        else:
            raise ValueError("invalid forward mode")

    def _calculate_loss(self, x, *, cond):
        lr_feats = self.lr_feats(cond)
        cond_scaled = FV.resize(
            cond, (x.shape[2], x.shape[3]), interpolation=FV.InterpolationMode.BICUBIC
        )
        scale = 5
        x = x - cond_scaled
        x = x * scale
        return self.diffusion.normalize(x, cond=(lr_feats, cond_scaled, scale))

    def _generate_sample(self, x, *, cond):
        x = x.view(x.shape[0], self.in_channels, self.image_size_x, self.image_size_y)
        if self.diffusion.ddim:
            import torch.utils.checkpoint

            lr_feats = torch.utils.checkpoint.checkpoint(self.lr_feats, cond)
        else:
            lr_feats = self.lr_feats(cond)
        cond_scaled = FV.resize(
            cond, (x.shape[2], x.shape[3]), interpolation=FV.InterpolationMode.BICUBIC
        )
        scale = 5
        x = self.diffusion.generate(x, cond=(lr_feats, cond_scaled, scale))
        x = x / scale
        x = x + cond_scaled
        return x

    def set_generate_steps(self, steps):
        def visit(module):
            if isinstance(module, Diffusion):
                module.set_generate_steps(steps)

        self.apply(visit)

    def set_generate_verbose(self, verbose):
        def visit(module):
            if isinstance(module, Diffusion):
                module.set_generate_verbose(verbose)

        self.apply(visit)
