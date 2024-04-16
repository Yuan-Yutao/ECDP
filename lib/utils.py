import torch


class LazyValues:
    def __init__(self, **kwargs):
        self._producers = kwargs
        self._cached = {}

    def __getattr__(self, attr):
        try:
            return self._cached[attr]
        except KeyError:
            p = self._producers.pop(attr)
            val = p()
            self._cached[attr] = val
            return val

    def get_or_none(self, key):
        return self._cached.get(key)

    def set_value(self, key, value):
        self._cached[key] = value
        self._producers.pop(key, None)


def sample_call(sample, func):
    if torch.is_tensor(sample):
        return func(sample)
    if isinstance(sample, list):
        return [sample_call(x, func) for x in sample]
    if isinstance(sample, tuple):
        return tuple(sample_call(x, func) for x in sample)
    if isinstance(sample, dict):
        return {key: sample_call(value, func) for key, value in sample.items()}
    return sample


def sample_to_device(sample, device):
    return sample_call(sample, lambda x: x.to(device))


def sample_to_cpu(sample):
    return sample_call(sample, lambda x: x.detach().cpu())


def sample_detach(sample):
    return sample_call(sample, lambda x: x.detach())
