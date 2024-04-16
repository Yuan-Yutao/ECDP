#!/usr/bin/env python3

import click

from lib.commands import ALL_COMMANDS


def main():
    cli = click.Group(commands=ALL_COMMANDS)
    cli.main()


if __name__ == "__main__":
    main()
