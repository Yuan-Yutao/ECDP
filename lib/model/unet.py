import math

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as FV
from torch import nn

# ACT = functools.partial(F.leaky_relu, inplace=True)
ACT = F.silu


class LambdaModule(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class MyConv2d(nn.Module):
    def __init__(self, *args, init_scale=1, **kwargs):
        super().__init__()

        self.network = nn.Conv2d(*args, **kwargs)

        with torch.no_grad():
            self.network.weight.mul_(init_scale)
            if self.network.bias is not None:
                self.network.bias.zero_()

    def forward(self, x):
        return self.network(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, extra_channels):
        super().__init__()

        self.time_in = nn.Linear(time_channels, out_channels)
        self.extra_in = MyConv2d(extra_channels, out_channels, 1)
        self.norm1 = nn.GroupNorm(min(in_channels // 4, 32), in_channels)
        self.conv1 = MyConv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(out_channels // 4, 32), out_channels)
        self.conv2 = MyConv2d(out_channels, out_channels, 3, padding=1, init_scale=1e-3)
        # self.dropout = nn.Dropout(p=0.1)
        self.dropout = nn.Identity()

        if in_channels != out_channels:
            self.skip = MyConv2d(in_channels, out_channels, 1, init_scale=0.1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, t_embed, extra_feats):
        t_embed = self.time_in(t_embed)
        if extra_feats is not None:
            extra_feats = FV.resize(
                extra_feats,
                (x.shape[2], x.shape[3]),
                interpolation=FV.InterpolationMode.BILINEAR,
            )
            extra_feats = self.extra_in(extra_feats)
        y = self.norm1(x)
        y = ACT(y)
        y = self.conv1(y)
        y = y + t_embed[:, :, None, None]
        if extra_feats is not None:
            y = y + extra_feats
        del t_embed, extra_feats
        y = self.norm2(y)
        y = ACT(y)
        y = self.dropout(y)
        y = self.conv2(y)
        x = self.skip(x)
        return x + y


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.downconv = MyConv2d(in_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x, t_embed, extra_feats):
        x = self.downconv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upconv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            MyConv2d(in_channels, out_channels, 3, padding=1),
        )

    def forward(self, x, t_embed, extra_feats):
        x = self.upconv(x)
        return x


def concat(x, prev_x):
    assert 0 <= x.shape[2] - prev_x.shape[2] <= 1
    assert 0 <= x.shape[3] - prev_x.shape[3] <= 1
    x = x[:, :, : prev_x.shape[2], : prev_x.shape[3]]
    x = torch.cat([x, prev_x], dim=1)
    return x


def make_embed(t, channels):
    assert channels % 2 == 0
    half_dim = channels // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = (torch.arange(half_dim, device=t.device) * -emb).exp()
    emb = t[:, None] * emb[None, :]
    emb = torch.cat([emb.sin(), emb.cos()], dim=1)
    return emb


class UNet(nn.Module):
    def __init__(self, num_channels, time_channels, extra_channels, out_channels=None):
        super().__init__()

        if out_channels is None:
            out_channels = num_channels
        self.time_channels = time_channels
        self.time_in = nn.Sequential(
            nn.Linear(time_channels, time_channels * 4),
            LambdaModule(ACT),
            nn.Linear(time_channels * 4, time_channels * 4),
            LambdaModule(ACT),
        )
        time_channels = time_channels * 4

        self.in_conv = MyConv2d(num_channels, 64, 3, padding=1)
        prev_channel = 64

        channels = [64, 128, 256, 256]
        num_blocks = [2, 2, 2, 2]

        self.downs = nn.ModuleList()
        feat_channels = [prev_channel]
        for i in range(len(channels)):
            for j in range(num_blocks[i]):
                self.downs.append(
                    ResnetBlock(
                        prev_channel, channels[i], time_channels, extra_channels
                    )
                )
                feat_channels.append(channels[i])
                prev_channel = channels[i]
            if i < len(channels) - 1:
                self.downs.append(Down(prev_channel, prev_channel))
                feat_channels.append(prev_channel)

        self.mid = nn.ModuleList(
            [
                ResnetBlock(prev_channel, prev_channel, time_channels, extra_channels),
                ResnetBlock(prev_channel, prev_channel, time_channels, extra_channels),
            ]
        )

        self.ups = nn.ModuleList()
        self.up_need_res = []
        for i in range(len(channels) - 1, -1, -1):
            for j in range(num_blocks[i] + 1):
                res_channel = feat_channels.pop()
                self.ups.append(
                    ResnetBlock(
                        prev_channel + res_channel,
                        channels[i],
                        time_channels,
                        extra_channels,
                    )
                )
                self.up_need_res.append(True)
                prev_channel = channels[i]
            if i > 0:
                self.ups.append(Up(prev_channel, prev_channel))
                self.up_need_res.append(False)

        self.out_conv = nn.Sequential(
            nn.GroupNorm(16, prev_channel),
            LambdaModule(ACT),
            MyConv2d(prev_channel, out_channels, 3, padding=1),
        )

    def forward(self, x, t, extra_feats):
        t_embed = make_embed(t, self.time_channels)
        t_embed = self.time_in(t_embed)

        x = self.in_conv(x)
        feats = [x]

        for layer in self.downs:
            x = layer(x, t_embed, extra_feats)
            feats.append(x)

        for layer in self.mid:
            x = layer(x, t_embed, extra_feats)

        for layer, need_res in zip(self.ups, self.up_need_res):
            if need_res:
                x = concat(x, feats.pop())
            x = layer(x, t_embed, extra_feats)

        assert not feats

        return self.out_conv(x)
