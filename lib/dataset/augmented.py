import torch
import torchvision.transforms.functional as FV
from torch.utils.data import Dataset

from .imresize import imresize


class AugmentedDataset(Dataset):
    def __init__(
        self,
        dataset,
        *,
        random_flip=True,
        random_rotate=True,
        dequantize=True,
        aggressive=False
    ):
        self.dataset = dataset
        self.random_flip = random_flip
        self.random_rotate = random_rotate
        self.dequantize = dequantize
        self.aggressive = aggressive

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        image = data["image"]
        image_lr = data["image_lr"]

        if self.random_flip and torch.rand(()) < 0.5:
            image = FV.hflip(image)
            image_lr = FV.hflip(image_lr)

        if self.random_rotate:
            k = torch.randint(0, 4, size=()).item()
            image = torch.rot90(image, k, dims=[-2, -1])
            image_lr = torch.rot90(image_lr, k, dims=[-2, -1])

        # if not self.aggressive and self.dequantize:
        #     image += (torch.rand_like(image) - 0.5) / 255.0

        if self.aggressive:
            image_orig = image
            image_lr_orig = image_lr
            noise_std = torch.rand(()) * 0.05
            noise = torch.randn_like(image) * noise_std
            image = image + noise
            image_lr = imresize(
                image.permute(1, 2, 0).numpy(),
                output_shape=(image.shape[1] // 4, image.shape[2] // 4),
            )
            image_lr = torch.tensor(image_lr, dtype=torch.get_default_dtype()).permute(
                2, 0, 1
            )
            return {
                "image": image,
                "image_lr": image_lr,
                "noise_std": noise_std,
                "noise": noise,
                "image_orig": image_orig,
                "image_lr_orig": image_lr_orig,
            }

        return {"image": image, "image_lr": image_lr}
