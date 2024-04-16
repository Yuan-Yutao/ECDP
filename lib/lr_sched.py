class LRScheduler:
    def __init__(self, optimizer, factor, milestones, warmup_steps):
        self.optimizer = optimizer
        self.factor = factor
        self.milestones = milestones
        self.warmup_steps = warmup_steps

        self.num_steps = 0
        self.num_epochs = 0
        self.milestone_idx = 0

        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.update_lr()

    def state_dict(self):
        return {k: v for k, v in self.__dict__.items() if k != "optimizer"}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def step(self):
        self.num_steps += 1
        self.update_lr()

    def step_epoch(self):
        self.num_epochs += 1
        self.update_lr()

    def update_lr(self):
        while (
            self.milestone_idx < len(self.milestones)
            and self.milestones[self.milestone_idx] <= self.num_epochs
        ):
            self.milestone_idx += 1
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            warmup_factor = (
                min(self.num_steps + 1, self.warmup_steps) / self.warmup_steps
            )
            decay_factor = self.factor**self.milestone_idx
            lr = base_lr * warmup_factor * decay_factor
            group["lr"] = lr
