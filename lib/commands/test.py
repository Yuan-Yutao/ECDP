import click

from ..options import MyPath
from .utils import load_saved_options, setup_trainer


@click.command()
@click.argument("name")
@click.option("--save-images", is_flag=True)
def test(name, save_images):
    options = load_saved_options(MyPath.root_dir, name)

    options.train.tensorboard = False
    options.train.distributed = False

    with setup_trainer(options) as trainer:
        trainer.test_one(save_images=save_images)
