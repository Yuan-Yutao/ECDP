from .image_folders import ImageFolders


class DF2kDataset(ImageFolders):
    def __init__(
        self,
        data_dir,
        downscale_factor,
        train: bool,
        *,
        random_crop_size=None,
        deterministic=False,
        repeat=1
    ):
        if train:
            paths = [
                data_dir / "div2k" / "DIV2K_train_HR",
                data_dir / "flickr2k" / "Flickr2K_HR",
            ]
        else:
            paths = [
                data_dir / "div2k" / "DIV2K_valid_HR",
            ]

        super().__init__(
            paths,
            downscale_factor,
            random_crop_size=random_crop_size,
            deterministic=deterministic,
            repeat=repeat,
        )
