import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .utils import image_to_tensor


class DIV2kTestDataset(Dataset):
    def __init__(self, data_dir):
        self.data_range = range(801, 901)
        self.hr_dir = data_dir / "div2k" / "DIV2K_valid_HR"
        self.lr_dir = data_dir / "div2k" / "DIV2K_valid_LR_bicubic" / "X4"

    def __len__(self):
        return len(self.data_range)

    def __getitem__(self, idx):
        num = self.data_range[idx]
        hr_path = self.hr_dir / f"{num:04d}.png"
        lr_path = self.lr_dir / f"{num:04d}x4.png"
        with Image.open(hr_path) as image_file:
            image_hr = np.asarray(image_file)
        with Image.open(lr_path) as image_file:
            image_lr = np.asarray(image_file)

        image_hr = image_to_tensor(image_hr)
        image_lr = image_to_tensor(image_lr)
        return {"image": image_hr, "image_lr": image_lr}
