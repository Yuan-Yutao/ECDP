import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10

from .utils import image_to_hr_lr_tensor


class CIFAR10Dataset(Dataset):
    def __init__(self, data_dir, downscale_factor, train: bool):
        root = data_dir / "cifar10"
        self.downscale_factor = downscale_factor
        self.cifar10 = CIFAR10(root=root, train=train)

    def __len__(self):
        return len(self.cifar10)

    def __getitem__(self, idx):
        image, _ = self.cifar10[idx]
        image = np.asarray(image)
        return image_to_hr_lr_tensor(image, self.downscale_factor)
