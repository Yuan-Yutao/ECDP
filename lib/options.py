from pathlib import Path

import yaml


class MyPath(yaml.YAMLObject):
    yaml_loader = yaml.SafeLoader
    yaml_dumper = yaml.SafeDumper

    yaml_tag = "!path"

    root_dir = (Path.cwd() / __file__).parent.parent

    def __init__(self, path):
        self.path = path

    @classmethod
    def from_yaml(cls, loader, node):
        value = loader.construct_scalar(node)
        return cls(value)

    @classmethod
    def to_yaml(cls, dumper, data):
        return dumper.represent_scalar("!path", data.path)

    def to_python_path(self):
        return self.root_dir / self.path

    @classmethod
    def from_python_path(cls, path):
        return cls(str(path.relative_to(cls.root_dir)))


class Namespace:
    def __init__(self, items):
        self._items = dict(items)

    def __getattr__(self, name):
        try:
            return self._items[name]
        except KeyError as e:
            raise AttributeError from e

    def __setattr__(self, name, value):
        if not name.startswith("_"):
            self._items[name] = value
        else:
            super().__setattr__(name, value)


def dict_to_object(x):
    if isinstance(x, dict):
        return Namespace((k, dict_to_object(v)) for k, v in x.items())
    elif isinstance(x, list):
        return [dict_to_object(v) for v in x]
    elif isinstance(x, MyPath):
        return x.to_python_path()
    else:
        return x


def object_to_dict(x):
    if isinstance(x, Namespace):
        return {k: object_to_dict(v) for k, v in x._items.items()}
    elif isinstance(x, (list, tuple)):
        return [object_to_dict(v) for v in x]
    elif isinstance(x, Path):
        return MyPath.from_python_path(x)
    else:
        return x


def read_options(path):
    with open(path, "r", encoding="utf-8") as f:
        options = yaml.safe_load(f)
    options = dict_to_object(options)
    return options


def write_options(options, path):
    options = object_to_dict(options)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(options, f)
