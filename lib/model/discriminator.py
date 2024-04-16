import torch
from torch import nn


class VGGDiscriminator(nn.Module):
    def __init__(self, options):
        super().__init__()

        in_channels = options.model.input_channels
        num_features = options.model.discriminator_channels
        image_size = options.train.train_image_size

        assert image_size % 32 == 0

        # [64, 160, 160]
        self.conv0_0 = nn.Conv2d(in_channels, num_features, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(num_features, num_features, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(num_features, affine=True)
        # [64, 80, 80]
        self.conv1_0 = nn.Conv2d(num_features, num_features * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(num_features * 2, affine=True)
        self.conv1_1 = nn.Conv2d(
            num_features * 2, num_features * 2, 4, 2, 1, bias=False
        )
        self.bn1_1 = nn.BatchNorm2d(num_features * 2, affine=True)
        # [128, 40, 40]
        self.conv2_0 = nn.Conv2d(
            num_features * 2, num_features * 4, 3, 1, 1, bias=False
        )
        self.bn2_0 = nn.BatchNorm2d(num_features * 4, affine=True)
        self.conv2_1 = nn.Conv2d(
            num_features * 4, num_features * 4, 4, 2, 1, bias=False
        )
        self.bn2_1 = nn.BatchNorm2d(num_features * 4, affine=True)
        # [256, 20, 20]
        self.conv3_0 = nn.Conv2d(
            num_features * 4, num_features * 8, 3, 1, 1, bias=False
        )
        self.bn3_0 = nn.BatchNorm2d(num_features * 8, affine=True)
        self.conv3_1 = nn.Conv2d(
            num_features * 8, num_features * 8, 4, 2, 1, bias=False
        )
        self.bn3_1 = nn.BatchNorm2d(num_features * 8, affine=True)
        # [512, 10, 10]
        self.conv4_0 = nn.Conv2d(
            num_features * 8, num_features * 8, 3, 1, 1, bias=False
        )
        self.bn4_0 = nn.BatchNorm2d(num_features * 8, affine=True)
        self.conv4_1 = nn.Conv2d(
            num_features * 8, num_features * 8, 4, 2, 1, bias=False
        )
        self.bn4_1 = nn.BatchNorm2d(num_features * 8, affine=True)
        # [512, 5, 5]

        self.linear1 = nn.Linear(num_features * 8 * (image_size // 32) ** 2, 100)
        self.linear2 = nn.Linear(100, 1)

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv0_0(x)
        x = self.act(x)
        x = self.conv0_1(x)
        x = self.bn0_1(x)
        x = self.act(x)

        x = self.conv1_0(x)
        x = self.bn1_0(x)
        x = self.act(x)
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.act(x)

        x = self.conv2_0(x)
        x = self.bn2_0(x)
        x = self.act(x)
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.act(x)

        x = self.conv3_0(x)
        x = self.bn3_0(x)
        x = self.act(x)
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.act(x)

        x = self.conv4_0(x)
        x = self.bn4_0(x)
        x = self.act(x)
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.act(x)

        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x
