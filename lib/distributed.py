from torch.nn.parallel import DistributedDataParallel


class DDP(DistributedDataParallel):
    def load_state_dict(self, *args, **kwargs):
        self.module.load_state_dict(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)
