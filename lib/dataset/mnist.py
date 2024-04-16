import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import MNIST

from .utils import image_to_hr_lr_tensor


class MNISTDataset(Dataset):
    def __init__(self, data_dir, downscale_factor, train: bool):
        root = data_dir / "mnist"
        self.downscale_factor = downscale_factor
        self.mnist = MNIST(root=root, train=train)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        image, _ = self.mnist[idx]
        image = np.asarray(image)[:, :, None]
        return image_to_hr_lr_tensor(image, self.downscale_factor)
