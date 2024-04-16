import math

from torch import nn

from .rrdb import RRDBEncoder


class RRDBNet(nn.Module):
    def __init__(self, options, rrdb_channels=None):
        super().__init__()

        self.in_channels = options.model.input_channels
        self.rrdb = RRDBEncoder(self.in_channels, options)

        if rrdb_channels is None:
            rrdb_channels = options.model.rrdb_channels
        self.final = nn.Sequential()
        for i in range(round(math.log2(options.model.sr_factor))):
            self.final.append(nn.Upsample(scale_factor=2, mode="nearest"))
            self.final.append(nn.Conv2d(rrdb_channels, rrdb_channels, 3, padding=1))
            self.final.append(nn.LeakyReLU(0.2, inplace=True))
        self.final.append(nn.Conv2d(rrdb_channels, rrdb_channels, 3, padding=1))
        self.final.append(nn.LeakyReLU(0.2, inplace=True))
        self.final.append(nn.Conv2d(rrdb_channels, self.in_channels, 3, padding=1))

    def forward(self, *args, mode, **kwargs):
        if mode == "loss":
            return self._calculate_loss(*args, **kwargs)
        elif mode == "generate":
            return self._generate_sample(*args, **kwargs)
        else:
            raise ValueError("invalid forward mode")

    def _calculate_loss(self, x, *, cond):
        gen = self.final(self.rrdb(cond))
        loss = (x - gen).abs().sum(dim=[1, 2, 3])
        return loss

    def _generate_sample(self, x, *, cond):
        return self.final(self.rrdb(cond))
