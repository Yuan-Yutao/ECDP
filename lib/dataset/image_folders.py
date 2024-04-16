import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .imresize import imresize
from .utils import image_to_hr_lr_tensor

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


class ImageFolders(Dataset):
    def __init__(
        self,
        paths,
        downscale_factor,
        *,
        random_crop_size=None,
        deterministic=False,
        repeat=1,
        pre_resize=None,
        pre_crop=False,
    ):
        self.files = []
        for path in paths:
            paths = [
                file
                for file in path.iterdir()
                if file.is_file() and str(file).lower().endswith(IMAGE_EXTENSIONS)
            ]
            paths.sort()
            self.files.extend(paths)

        self.downscale_factor = downscale_factor
        self.random_crop_size = random_crop_size
        self.deterministic = deterministic
        self.repeat = repeat
        self.pre_resize = pre_resize
        self.pre_crop = pre_crop

        if self.deterministic:
            g = torch.Generator()
            g.manual_seed(123456789)
            self.crop_indices_frac = torch.rand(
                size=(len(self.files) * self.repeat, 2),
                dtype=torch.float64,
                generator=g,
            )
        else:
            self.crop_indices_frac = None

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        path = self.files[idx % len(self.files)]
        with Image.open(path) as image_file:
            image = image_file.convert('RGB')
            image = np.asarray(image)
        if self.pre_crop:
            target_size = min(image.shape[0], image.shape[1])
            if self.deterministic:
                start_x = int(round((image.shape[0] - target_size) / 2.0))
                start_y = int(round((image.shape[1] - target_size) / 2.0))
            else:
                start_x = np.random.randint(0, image.shape[0] - target_size + 1)
                start_y = np.random.randint(0, image.shape[1] - target_size + 1)
            image = image[start_x : start_x + target_size, start_y : start_y + target_size, :]
        if self.pre_resize is not None:
            image = imresize(image, output_shape=self.pre_resize)

        if self.random_crop_size is not None:
            x_idx_limit = image.shape[0] - self.random_crop_size + 1
            y_idx_limit = image.shape[1] - self.random_crop_size + 1
            if not self.deterministic:
                x_idx = torch.randint(0, x_idx_limit, size=()).item()
                y_idx = torch.randint(0, y_idx_limit, size=()).item()
            else:
                x_idx = (
                    (self.crop_indices_frac[idx, 0] * x_idx_limit)
                    .floor()
                    .int()
                    .clamp(0, x_idx_limit - 1)
                    .item()
                )
                y_idx = (
                    (self.crop_indices_frac[idx, 1] * y_idx_limit)
                    .floor()
                    .int()
                    .clamp(0, y_idx_limit - 1)
                    .item()
                )

            image = image[
                x_idx : x_idx + self.random_crop_size,
                y_idx : y_idx + self.random_crop_size,
                :,
            ]

        return image_to_hr_lr_tensor(image, self.downscale_factor)
