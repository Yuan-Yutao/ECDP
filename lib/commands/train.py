import datetime

import click

from ..options import MyPath, read_options, write_options
from .utils import copy_code, start_train


def make_name(name):
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{time}-{name}"


@click.command()
@click.argument("name")
@click.option("--config", required=True)
def train(name, config):
    root_dir = MyPath.root_dir
    name = make_name(name)

    options = read_options(config)
    options.train.data_dir = root_dir / "data"
    options.train.result_dir = root_dir / "results" / name

    copy_code(root_dir / "lib", options.train.result_dir / "lib")
    write_options(options, options.train.result_dir / "config.yaml")

    start_train(options)
