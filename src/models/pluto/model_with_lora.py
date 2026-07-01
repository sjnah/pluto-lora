"""
PLUTO model wrapper with LoRA support and ultra-minimal head control (3 layer biases).

This module provides:
- PLUTOModelWithLoRA: Wrapper that applies LoRA and controls trainable parameters
- Optimizer group configuration for separate LoRA and head learning rates
- Ultra-minimal mode: 3 layer biases (mlp.0.bias, mlp.1 LN gamma/beta, mlp.3.bias) are trainable (weight frozen)
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper

from .lora_layers import (
    get_lora_state_dict,
    inject_lora_into_encoder,
    mark_only_lora_and_ultra_head_as_trainable,
    merge_lora_weights,
)

logger = logging.getLogger(__name__)


class PLUTOModelWithLoRA(nn.Module):
    """
    PLUTO model wrapper with LoRA adapters and ultra-minimal head fine-tuning.
    
    This wrapper:
    1. Applies LoRA ONLY to encoder_blocks attention layers
    2. Controls trainable parameters (LoRA + ultra-minimal head: 3 layer biases)
    3. Provides optimizer groups for separate learning rates (head params: weight_decay=0.0)
    """
    
    def __init__(
        self,
        base_model: TorchModuleWrapper,
        lora_enabled: bool = True,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        ultra_minimal: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize PLUTO model with LoRA.
        
        Args:
            base_model: Base PLUTO model (PlanningModel wrapped in TorchModuleWrapper)
            lora_enabled: Whether to enable LoRA
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha scaling factor
            lora_dropout: LoRA dropout probability
            ultra_minimal: If True, only 3 layer biases (mlp.0.bias, mlp.1 LN gamma/beta, mlp.3.bias) of heads are trainable (weight frozen)
            verbose: Whether to print configuration info
        """
        super().__init__()
        
        self.base_model = base_model
        self.lora_enabled = lora_enabled
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.ultra_minimal = ultra_minimal
        
        # Extract the actual PlanningModel from TorchModuleWrapper
        # TorchModuleWrapper wraps the model in self._model
        if hasattr(base_model, '_model'):
            self.model = base_model._model
        elif hasattr(base_model, 'model'):
            self.model = base_model.model
        else:
            # Assume base_model is the PlanningModel itself
            self.model = base_model
        
        # Apply LoRA if enabled
        if self.lora_enabled:
            logger.info("="*80)
            logger.info("Applying LoRA to encoder_blocks ONLY...")
            logger.info("="*80)
            
            lora_count = inject_lora_into_encoder(
                self.model,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout,
                verbose=verbose,
            )
            
            if lora_count == 0:
                logger.warning("⚠ No LoRA adapters were injected!")
            else:
                logger.info(f"✓ Injected {lora_count} LoRA adapters into encoder_blocks")
        
        # Mark trainable parameters
        mark_only_lora_and_ultra_head_as_trainable(
            self.model,
            ultra_minimal=self.ultra_minimal,
            verbose=verbose,
        )
        
        logger.info("✓ Model with LoRA configured successfully")
    
    def forward(self, *args, **kwargs):
        """Forward pass through base model."""
        return self.base_model(*args, **kwargs)
    
    def get_optimizer_groups(
        self,
        lora_lr: float = 1e-5,
        head_lr: float = 1e-5,
        weight_decay: float = 0.01,
    ) -> List[Dict]:
        """
        Get optimizer parameter groups with separate learning rates.
        
        Args:
            lora_lr: Learning rate for LoRA parameters
            head_lr: Learning rate for head parameters
            weight_decay: Weight decay
            
        Returns:
            List of parameter group dictionaries
        """
        lora_params = []
        head_params = []
        
        # Separate parameters by type
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Check if LoRA parameter
            is_lora = "lora_A" in name or "lora_B" in name
            
            # Check if head parameter
            is_head = any(head in name for head in ['loc_head', 'yaw_head', 'vel_head', 'pi_head'])
            
            # Determine weight decay
            # For LoRA: apply weight decay based on parameter type (bias/norm -> no decay)
            # For head: always use weight_decay=0.0 (head bias/LN parameters)
            apply_decay = "bias" not in name and "norm" not in name.lower()
            
            if is_lora:
                if apply_decay:
                    lora_params.append({"params": [param], "lr": lora_lr, "weight_decay": weight_decay})
                else:
                    lora_params.append({"params": [param], "lr": lora_lr, "weight_decay": 0.0})
            elif is_head:
                # Head parameters (bias + LN gamma/beta): always weight_decay=0.0
                head_params.append({"params": [param], "lr": head_lr, "weight_decay": 0.0})
        
        # Combine groups
        optim_groups = []
        if lora_params:
            # Group LoRA params by weight decay
            lora_decay = [g for g in lora_params if g.get("weight_decay", 0) > 0]
            lora_no_decay = [g for g in lora_params if g.get("weight_decay", 0) == 0]
            
            if lora_decay:
                optim_groups.append({
                    "params": [p for g in lora_decay for p in g["params"]],
                    "lr": lora_lr,
                    "weight_decay": weight_decay,
                    "name": "lora_decay",
                })
            if lora_no_decay:
                optim_groups.append({
                    "params": [p for g in lora_no_decay for p in g["params"]],
                    "lr": lora_lr,
                    "weight_decay": 0.0,
                    "name": "lora_no_decay",
                })
        
        if head_params:
            # Head parameters: all use weight_decay=0.0 (bias + LN gamma/beta)
            optim_groups.append({
                "params": [p for g in head_params for p in g["params"]],
                "lr": head_lr,
                "weight_decay": 0.0,
                "name": "head_bias_ln",
            })
        
        logger.info(f"\n{'='*80}")
        logger.info("Optimizer Parameter Groups:")
        logger.info(f"{'='*80}")
        
        # Build param name mapping for logging
        param_to_name = {id(param): name for name, param in self.model.named_parameters()}
        
        for group in optim_groups:
            n_params = sum(p.numel() for p in group["params"])
            logger.info(f"  {group['name']}: {n_params:,} params, lr={group['lr']:.2e}, wd={group['weight_decay']}")
            
            # Log first few parameter names for verification
            param_names = []
            for param in group["params"]:
                param_id = id(param)
                if param_id in param_to_name:
                    param_names.append(param_to_name[param_id])
            if param_names:
                logger.info(f"    Sample params (first 5): {param_names[:5]}")
                if len(param_names) > 5:
                    logger.info(f"    ... and {len(param_names) - 5} more")
        logger.info(f"{'='*80}\n")
        
        return optim_groups
    
    def save_lora_only(self, filepath: str) -> None:
        """
        Save only LoRA parameters to a file.
        
        Args:
            filepath: Path to save LoRA parameters
        """
        if not self.lora_enabled:
            logger.warning("LoRA is not enabled, nothing to save")
            return
        
        lora_state = {
            "lora_state_dict": get_lora_state_dict(self.model),
            "lora_config": {
                "rank": self.lora_rank,
                "alpha": self.lora_alpha,
                "dropout": self.lora_dropout,
            },
        }
        torch.save(lora_state, filepath)
        logger.info(f"✓ Saved LoRA parameters to {filepath}")
    
    def merge_and_save(self, filepath: str) -> None:
        """
        Merge LoRA weights into base model and save.
        
        Args:
            filepath: Path to save merged model
        """
        if not self.lora_enabled:
            logger.warning("LoRA is not enabled, saving base model only")
            torch.save({"state_dict": self.model.state_dict()}, filepath)
            return
        
        logger.info("Merging LoRA weights into base model...")
        merged_model = merge_lora_weights(self.model, verbose=True)
        torch.save({"state_dict": merged_model.state_dict()}, filepath)
        logger.info(f"✓ Saved merged model to {filepath}")

