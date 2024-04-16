import torch

from .imresize import imresize


def image_to_tensor(image):
    image = image.transpose(2, 0, 1).copy()
    image = torch.from_numpy(image)
    image = image.to(torch.get_default_dtype()) / 255.0
    return image


def image_to_hr_lr_tensor(image, downscale_factor):
    image_lr = imresize(
        image,
        output_shape=(
            image.shape[0] // downscale_factor,
            image.shape[1] // downscale_factor,
        ),
    )

    image = image_to_tensor(image)
    image_lr = image_to_tensor(image_lr)
    return {"image": image, "image_lr": image_lr}
