import warnings

import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from .augmented import AugmentedDataset
from .celeba import CelebADataset
from .cifar10 import CIFAR10Dataset
from .df2k import DF2kDataset
from .div2k_test import DIV2kTestDataset
from .ffhq import FFHQDataset
from .imagenet import ImagenetDataset
from .mnist import MNISTDataset
from .toy import ToyDataset


def get_train_dataloader(options):
    if options.train.dataset == "mnist":
        dataset = MNISTDataset(options.train.data_dir, 4, True)
    elif options.train.dataset == "cifar10":
        dataset = CIFAR10Dataset(options.train.data_dir, 4, True)
    elif options.train.dataset == "df2k":
        dataset = DF2kDataset(
            options.train.data_dir,
            4,
            True,
            random_crop_size=options.train.train_image_size,
        )
    elif options.train.dataset == "celeba":
        dataset = CelebADataset(options.train.data_dir, 8, "train")
    elif options.train.dataset == "ffhq":
        dataset = FFHQDataset(options.train.data_dir, 8, "train")
    elif options.train.dataset == "imagenet":
        dataset = ImagenetDataset(options.train.data_dir, 4, "train")
    elif options.train.dataset == "toy":
        dataset = ToyDataset(10000)
    else:
        raise ValueError("unknown dataset")
    dataset = AugmentedDataset(dataset, aggressive=False)

    if options.train.distributed:
        world_size = dist.get_world_size()
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
        if options.train.batch_size % world_size != 0:
            warnings.warn(
                "batch size is not divisible by world size, batch size will be inaccurate"
            )
        return DataLoader(
            dataset,
            batch_size=options.train.batch_size // world_size,
            sampler=sampler,
            drop_last=True,
            num_workers=8,
            pin_memory=True,
        )
    else:
        return DataLoader(
            dataset,
            batch_size=options.train.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        )


def get_val_dataloader(options):
    if options.train.dataset == "mnist":
        dataset = MNISTDataset(options.train.data_dir, 4, False)
    elif options.train.dataset == "cifar10":
        dataset = CIFAR10Dataset(options.train.data_dir, 4, False)
    elif options.train.dataset == "df2k":
        dataset = DF2kDataset(
            options.train.data_dir,
            4,
            False,
            random_crop_size=options.train.train_image_size,
            deterministic=True,
            repeat=2,
        )
    elif options.train.dataset == "celeba":
        dataset = CelebADataset(options.train.data_dir, 8, "val")
    elif options.train.dataset == "ffhq":
        dataset = FFHQDataset(options.train.data_dir, 8, "val")
    elif options.train.dataset == "imagenet":
        dataset = ImagenetDataset(options.train.data_dir, 4, "val", deterministic=True)
    elif options.train.dataset == "toy":
        dataset = ToyDataset(1000)
    else:
        raise ValueError("unknown dataset")
    dataset = AugmentedDataset(dataset, random_flip=False, random_rotate=False)

    if options.train.distributed:
        world_size = dist.get_world_size()
        if len(dataset) % world_size != 0:
            warnings.warn(
                "validation dataset size is not divisible by world size, validation results will be inaccurate"
            )
        sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
        if options.train.batch_size % world_size != 0:
            warnings.warn(
                "batch size is not divisible by world size, batch size will be inaccurate"
            )
        return DataLoader(
            dataset,
            batch_size=options.train.batch_size // world_size,
            sampler=sampler,
            num_workers=8,
            pin_memory=True,
        )
    else:
        return DataLoader(
            dataset,
            batch_size=options.train.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        )


def get_test_dataset(options):
    if options.train.dataset == "df2k":
        dataset = DIV2kTestDataset(options.train.data_dir)
    elif options.train.dataset == "celeba":
        dataset = CelebADataset(options.train.data_dir, 8, "test")
    elif options.train.dataset == "ffhq":
        dataset = FFHQDataset(options.train.data_dir, 8, "test")
    elif options.train.dataset == "imagenet":
        dataset = ImagenetDataset(options.train.data_dir, 4, "val", deterministic=True)
    else:
        raise ValueError("unknown dataset")
    return dataset
