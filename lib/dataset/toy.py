import numpy as np
import torch
import torchvision.transforms.functional as FV
from torch.utils.data import Dataset

from .utils import image_to_hr_lr_tensor


class ToyDataset(Dataset):
    def __init__(self, num):
        color_f1 = FV.resize(
            torch.rand(num, 3, 1, 1),
            (16, 16),
            interpolation=FV.InterpolationMode.BILINEAR,
        )
        color_f2 = FV.resize(
            torch.rand(num, 3, 4, 4),
            (16, 16),
            interpolation=FV.InterpolationMode.BILINEAR,
        )
        color_f4 = FV.resize(
            torch.rand(num, 3, 8, 8),
            (16, 16),
            interpolation=FV.InterpolationMode.BILINEAR,
        )
        color_f8 = FV.resize(
            torch.rand(num, 3, 32, 32),
            (16, 16),
            interpolation=FV.InterpolationMode.BILINEAR,
        )
        images = (color_f1 * 1 + color_f2 * 1 + color_f4 * 1 + color_f8 * 1) / 4
        images = (images.clamp(0, 1) * 255).round()
        images = images.numpy().astype(np.uint8)
        self.images = images

        self.downscale_factor = 4

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return image_to_hr_lr_tensor(
            self.images[idx].transpose(1, 2, 0), self.downscale_factor
        )
