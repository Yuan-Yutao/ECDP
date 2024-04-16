import click

from ..options import MyPath
from .utils import load_saved_options, start_train


@click.command()
@click.argument("name")
def resume(name):
    options = load_saved_options(MyPath.root_dir, name)

    start_train(options)
