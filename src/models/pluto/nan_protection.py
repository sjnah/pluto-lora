"""
NaN Protection Callback for PyTorch Lightning
Detects and prevents NaN from corrupting model weights
"""
import torch
from pytorch_lightning.callbacks import Callback
import logging

logger = logging.getLogger(__name__)


class NaNProtectionCallback(Callback):
    """
    Callback that monitors gradients and parameters for NaN values.
    If NaN is detected, it skips the optimizer step to prevent weight corruption.
    """
    
    def __init__(self, check_frequency: int = 1):
        super().__init__()
        self.check_frequency = check_frequency
        self.nan_steps = 0
        self.total_steps = 0
    
    def on_after_backward(self, trainer, pl_module):
        """Check gradients after backward pass but before optimizer step."""
        self.total_steps += 1
        
        if self.total_steps % self.check_frequency != 0:
            return
        
        # Check all trainable parameters for NaN/inf gradients
        has_nan_grad = False
        has_inf_grad = False
        has_nan_param = False
        nan_param_names = []
        
        for name, param in pl_module.named_parameters():
            if param.requires_grad:
                # Check parameter value
                if torch.isnan(param).any():
                    has_nan_param = True
                    nan_param_names.append(f"{name} (param NaN)")
                
                # Check gradient
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        has_nan_grad = True
                        nan_param_names.append(f"{name} (grad NaN)")
                    if torch.isinf(param.grad).any():
                        has_inf_grad = True
                        nan_param_names.append(f"{name} (grad Inf)")
        
        if has_nan_grad or has_inf_grad or has_nan_param:
            self.nan_steps += 1
            logger.error(
                f"[NaN PROTECTION] Detected NaN/Inf at step {self.total_steps}!\n"
                f"  NaN in parameters: {has_nan_param}\n"
                f"  NaN in gradients: {has_nan_grad}\n"
                f"  Inf in gradients: {has_inf_grad}\n"
                f"  Affected: {nan_param_names[:5]}{'...' if len(nan_param_names) > 5 else ''}\n"
                f"  Total NaN steps: {self.nan_steps}/{self.total_steps}"
            )
            
            # Zero out all gradients to prevent corruption
            pl_module.zero_grad(set_to_none=True)
            logger.warning("  → Zeroed all gradients to prevent optimizer update")
            
            # Set flag to skip optimizer step
            pl_module._skip_optimizer_step = True
    
    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        """Check and skip optimizer step if NaN/inf was detected."""
        # Check if we should skip optimizer step (set by on_after_backward)
        if hasattr(pl_module, '_skip_optimizer_step') and pl_module._skip_optimizer_step:
            logger.warning(f"[NaN PROTECTION] Skipping optimizer step at step {self.total_steps}")
            pl_module._skip_optimizer_step = False
            # Return False to skip optimizer step
            return False
        
        # Additional check: verify no NaN/inf in gradients right before optimizer step
        for name, param in pl_module.named_parameters():
            if param.requires_grad and param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    logger.error(f"[NaN PROTECTION] NaN/Inf detected in {name} right before optimizer step! Skipping.")
                    pl_module.zero_grad(set_to_none=True)
                    return False

