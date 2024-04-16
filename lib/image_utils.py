import numpy as np
import torch.nn.functional as F
from PIL import Image


def quantize(x):
    return (x.clamp(0, 1) * 255).round() / 255


def pad_to_multiple(image, k):
    size_x = image.shape[2]
    size_y = image.shape[3]
    new_x = (size_x + k - 1) // k * k
    new_y = (size_y + k - 1) // k * k
    image = F.pad(image, (0, new_y - size_y, 0, new_x - size_x), mode="replicate")
    return image


def save_image(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = (image.clamp(0, 1) * 255).round()
    image = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    Image.fromarray(image).save(path)
