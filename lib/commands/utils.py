import os
import random
import shutil
from contextlib import contextmanager

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ..options import read_options
from ..trainer import Trainer


@contextmanager
def setup_trainer(options):
    torch.backends.cudnn.benchmark = True
    set_random_seed(options)
    trainer = Trainer(options)
    try:
        yield trainer
    finally:
        trainer.flush_writer()


def distributed_train_work(rank, world_size, options):
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size, init_method="tcp://127.0.0.1:29500/"
    )
    torch.cuda.set_device(rank)
    if rank != 0:
        devnull = os.open("/dev/null", os.O_WRONLY)
        os.dup2(devnull, 2)
    with setup_trainer(options) as trainer:
        trainer.train()
    dist.destroy_process_group()


def start_train(options):
    if options.train.distributed:
        assert options.train.use_cuda
        world_size = torch.cuda.device_count()
        assert world_size > 0
        mp.spawn(
            distributed_train_work,
            args=(world_size, options),
            nprocs=world_size,
            join=True,
        )
    else:
        with setup_trainer(options) as trainer:
            trainer.train()


def set_random_seed(options):
    seed = options.train.random_seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_deterministic(True)


def copy_code(src_dir, dst_dir):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src_file in src_dir.glob("**/*.py"):
        if not src_file.is_file():
            continue
        dst_file = dst_dir / src_file.relative_to(src_dir)
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src_file, dst_file)


def load_saved_options(root_dir, name):
    result_dir = root_dir / "results" / name
    assert result_dir.is_dir()

    config_path = result_dir / "config.yaml"
    return read_options(config_path)
