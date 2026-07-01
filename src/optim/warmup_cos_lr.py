import math

from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosLR(_LRScheduler):
    def __init__(
        self, optimizer, min_lr, lr, warmup_epochs=None, warmup_steps=None, epochs=None, total_steps=None, last_epoch=-1, verbose=False
    ) -> None:
        self.min_lr = min_lr
        self.lr = lr
        
        # Support both warmup_epochs and warmup_steps
        # PyTorch Lightning scheduler is step-based, so last_epoch is actually step count
        if warmup_steps is not None:
            self.warmup_steps = warmup_steps
            self.warmup_epochs = None  # Use steps-based warmup
            if total_steps is None:
                if epochs is None:
                    raise ValueError("Either total_steps or epochs must be provided when using warmup_steps")
                # Will be set later when we know steps_per_epoch
                self.total_steps = None
                self.epochs = epochs
            else:
                self.total_steps = total_steps
                self.epochs = None
        else:
            if warmup_epochs is None:
                raise ValueError("Either warmup_epochs or warmup_steps must be provided")
            self.warmup_epochs = warmup_epochs
            self.warmup_steps = None  # Use epoch-based warmup
            if epochs is None:
                raise ValueError("epochs must be provided when using warmup_epochs")
            self.epochs = epochs
            self.total_steps = total_steps  # May be None for epoch-based
        
        super(WarmupCosLR, self).__init__(optimizer, last_epoch, verbose)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_init_lr(self):
        if self.warmup_steps is not None:
            lr = self.lr / self.warmup_steps
        else:
            lr = self.lr / self.warmup_epochs
        return lr

    def get_lr(self):
        # PyTorch Lightning scheduler: last_epoch is step count (0-indexed)
        current_step = self.last_epoch + 1  # Convert to 1-indexed
        
        if self.warmup_steps is not None:
            # Step-based warmup
            if current_step <= self.warmup_steps:
                lr = self.lr * current_step / self.warmup_steps
            else:
                # Need total_steps for cosine annealing
                if self.total_steps is None:
                    # Fallback: use a large number (will be corrected when total_steps is known)
                    # This is a temporary workaround
                    remaining_steps = max(1, self.epochs * 1000 - self.warmup_steps)  # Estimate
                else:
                    remaining_steps = self.total_steps - self.warmup_steps
                
                current_step_in_cosine = current_step - self.warmup_steps
                lr = self.min_lr + 0.5 * (self.lr - self.min_lr) * (
                    1
                    + math.cos(
                        math.pi
                        * current_step_in_cosine
                        / remaining_steps
                    )
                )
        else:
            # Epoch-based warmup (original behavior)
            # Note: In PyTorch Lightning, last_epoch is step count, not epoch count
            # This requires steps_per_epoch to convert, but for backward compatibility
            # we'll assume last_epoch represents epochs (old behavior)
            if current_step <= self.warmup_epochs:
                lr = self.lr * current_step / self.warmup_epochs
            else:
                lr = self.min_lr + 0.5 * (self.lr - self.min_lr) * (
                    1
                    + math.cos(
                        math.pi
                        * (current_step - self.warmup_epochs)
                        / (self.epochs - self.warmup_epochs)
                    )
                )
        
        if "lr_scale" in self.optimizer.param_groups[0]:
            return [lr * group["lr_scale"] for group in self.optimizer.param_groups]

        return [lr for _ in self.optimizer.param_groups]
