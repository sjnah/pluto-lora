from torch.optim.lr_scheduler import _LRScheduler


class WarmupConstantLR(_LRScheduler):
    """Linear step warmup followed by a constant per-group learning rate."""

    def __init__(
        self,
        optimizer,
        *,
        warmup_steps: int,
        total_steps: int,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        if self.warmup_steps <= 0:
            raise ValueError("warmup_steps must be positive")
        if self.total_steps < self.warmup_steps:
            raise ValueError("total_steps must be greater than or equal to warmup_steps")
        super().__init__(optimizer, last_epoch, verbose)

    def _get_scheduled_lr(self, base_lr: float, current_step: int) -> float:
        if current_step <= self.warmup_steps:
            return base_lr * current_step / self.warmup_steps
        return base_lr

    def get_lr(self):
        current_step = self.last_epoch + 1
        return [
            self._get_scheduled_lr(base_lr, current_step)
            for base_lr in self.base_lrs
        ]

