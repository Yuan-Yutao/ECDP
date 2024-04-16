from .image_folders import ImageFolders


class FFHQDataset(ImageFolders):
    def __init__(
        self,
        data_dir,
        downscale_factor,
        split,
        *,
        random_crop_size=None,
        deterministic=False,
        repeat=1
    ):
        if split == "train":
            paths = [
                data_dir / "ffhq" / "train",
            ]
        elif split == "val":
            paths = [
                data_dir / "ffhq" / "val",
            ]
        elif split == "test":
            paths = [
                data_dir / "ffhq" / "test",
            ]

        super().__init__(
            paths,
            downscale_factor,
            random_crop_size=random_crop_size,
            deterministic=deterministic,
            repeat=repeat,
            pre_resize=(256, 256),
        )
