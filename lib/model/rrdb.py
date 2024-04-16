import torch
import torch.nn.functional as F
from torch import nn


class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, num_features):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, num_features, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels + num_features, num_features, 3, padding=1)
        self.conv3 = nn.Conv2d(
            in_channels + 2 * num_features, num_features, 3, padding=1
        )
        self.conv4 = nn.Conv2d(
            in_channels + 3 * num_features, num_features, 3, padding=1
        )
        self.conv5 = nn.Conv2d(
            in_channels + 4 * num_features, in_channels, 3, padding=1
        )

        with torch.no_grad():
            self.conv1.weight.mul_(0.1)
            self.conv1.bias.mul_(0.1)
            self.conv2.weight.mul_(0.1)
            self.conv2.bias.mul_(0.1)
            self.conv3.weight.mul_(0.1)
            self.conv3.bias.mul_(0.1)
            self.conv4.weight.mul_(0.1)
            self.conv4.bias.mul_(0.1)
            self.conv5.weight.mul_(0.1)
            self.conv5.bias.mul_(0.1)

    def forward(self, x):
        x1 = F.leaky_relu(self.conv1(x), 0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(torch.cat([x, x1], dim=1)), 0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(torch.cat([x, x1, x2], dim=1)), 0.2, inplace=True)
        x4 = F.leaky_relu(
            self.conv4(torch.cat([x, x1, x2, x3], dim=1)), 0.2, inplace=True
        )
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x + x5 * 0.2


class RRDBBlock(nn.Module):
    def __init__(self, in_channels, num_features):
        super().__init__()

        self.block1 = ResidualDenseBlock(in_channels, num_features)
        self.block2 = ResidualDenseBlock(in_channels, num_features)
        self.block3 = ResidualDenseBlock(in_channels, num_features)

    def forward(self, x):
        y = self.block1(x)
        y = self.block2(y)
        y = self.block3(y)
        return x + y * 0.2


class RRDBEncoder(nn.Module):
    def __init__(self, in_channels, options):
        super().__init__()

        self.conv_first = nn.Conv2d(
            in_channels,
            options.model.rrdb_network_features,
            3,
            padding=1,
        )

        self.blocks = nn.ModuleList()
        for _ in range(options.model.rrdb_num_blocks):
            self.blocks.append(
                RRDBBlock(
                    options.model.rrdb_network_features,
                    options.model.rrdb_intermediate_features,
                )
            )

        self.block_selection = options.model.rrdb_block_selection

    def forward(self, x):
        x = self.conv_first(x)
        results = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.block_selection:
                results.append(x)
        return torch.cat(results, dim=1)
