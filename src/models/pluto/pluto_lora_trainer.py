"""
PLUTO LoRA Trainer with Uniform and Curriculum Fine-tuning.

This module provides:
- Uniform fine-tuning mode (all difficulty levels mixed)
- Curriculum fine-tuning mode (3-stage progressive training)
- NaN/Inf handling for training stability
- Identical total update steps for fair comparison
"""

import logging
from typing import Dict, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import (
    FeaturesType,
    ScenarioListType,
    TargetsType,
)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from .model_with_lora import PLUTOModelWithLoRA
from .pluto_trainer import LightningTrainer

logger = logging.getLogger(__name__)


class PLUTOLoRATrainer(LightningTrainer):
    """
    PLUTO Lightning trainer with LoRA support and curriculum/uniform training modes.
    
    Features:
    - Encoder-only LoRA (encoder_blocks attention layers)
    - Ultra-minimal head fine-tuning (3 layer biases: mlp.0.bias, mlp.1 LN gamma/beta, mlp.3.bias, weight frozen)
    - L2-SP regularization for head parameters (λ=1.5e-3)
    - EMA (Exponential Moving Average, decay=0.999)
    - NaN/Inf training step skip
    - Gradient clipping
    - Separate learning rates for LoRA and head
    """
    
    def __init__(
        self,
        model: TorchModuleWrapper,
        lr: float,
        weight_decay: float,
        epochs: int,
        warmup_steps: int = 300,
        warmup_epochs: int = 0,  # For backward compatibility (we use warmup_steps instead)
        use_collision_loss: bool = True,
        use_contrast_loss: bool = False,
        regulate_yaw: bool = False,
        objective_aggregate_mode: str = "mean",
        # Loss weights
        weight_reg_loss: float = 1.0,
        weight_cls_loss: float = 1.0,
        weight_prediction_loss: float = 1.0,
        weight_collision_loss: float = 1.0,
        weight_contrastive_loss: float = 1.0,
        weight_ref_free_reg_loss: float = 1.0,
        weight_auxiliary: float = 1.0,  # Global multiplier for auxiliary losses
        # LoRA-specific parameters
        lora_enabled: bool = True,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        lora_lr: Optional[float] = None,
        head_lr: Optional[float] = None,
        policy_head_lr: Optional[float] = None,  # Alias for head_lr (backward compatibility)
        ultra_minimal: bool = True,
        trainable_modules: Optional[list] = None,  # For backward compatibility (not used with encoder-only LoRA)
        is_curriculum_stage: bool = False,  # For backward compatibility
        # Training stability
        gradient_clip_val: float = 1.0,
        skip_nan_steps: bool = True,
        remove_invalid_goals: bool = True,
    ) -> None:
        """
        Initialize PLUTO LoRA trainer.
        
        Args:
            model: Base PLUTO model
            lr: Base learning rate
            weight_decay: Weight decay
            epochs: Total training epochs
            warmup_steps: Number of warmup steps
            use_collision_loss: Whether to use collision loss
            use_contrast_loss: Whether to use contrastive loss
            regulate_yaw: Whether to regulate yaw
            objective_aggregate_mode: How to aggregate objectives
            lora_enabled: Whether to enable LoRA
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha scaling factor
            lora_dropout: LoRA dropout probability
            lora_lr: Learning rate for LoRA parameters (if None, uses lr)
            head_lr: Learning rate for head parameters (if None, uses lr)
            ultra_minimal: If True, only 3 layer biases (mlp.0.bias, mlp.1 LN gamma/beta, mlp.3.bias) are trainable (weight frozen)
            gradient_clip_val: Gradient clipping value
            skip_nan_steps: Whether to skip steps with NaN/Inf loss
            remove_invalid_goals: Whether to remove invalid goals (configurable)
        """
        # Initialize parent class
        super().__init__(
            model=model,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            warmup_epochs=warmup_epochs,  # Pass through for backward compatibility
            use_collision_loss=use_collision_loss,
            use_contrast_loss=use_contrast_loss,
            regulate_yaw=regulate_yaw,
            objective_aggregate_mode=objective_aggregate_mode,
            # Loss weights
            weight_reg_loss=weight_reg_loss,
            weight_cls_loss=weight_cls_loss,
            weight_prediction_loss=weight_prediction_loss,
            weight_collision_loss=weight_collision_loss,
            weight_contrastive_loss=weight_contrastive_loss,
            weight_ref_free_reg_loss=weight_ref_free_reg_loss,
            weight_auxiliary=weight_auxiliary,
        )
        
        # LoRA configuration
        self.lora_enabled = lora_enabled
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_lr = lora_lr if lora_lr is not None else lr
        # Support both head_lr and policy_head_lr (backward compatibility)
        self.head_lr = head_lr if head_lr is not None else (policy_head_lr if policy_head_lr is not None else lr)
        self.ultra_minimal = ultra_minimal
        self.warmup_steps = warmup_steps
        self.trainable_modules = trainable_modules  # Store for compatibility (not used)
        self.is_curriculum_stage = is_curriculum_stage  # Store for compatibility
        
        # Training stability
        self.gradient_clip_val = gradient_clip_val
        self.skip_nan_steps = skip_nan_steps
        self.remove_invalid_goals = remove_invalid_goals
        
        # L2-SP regularization (for head bias/LN parameters)
        self.l2sp_lambda = 1.5e-3  # Default: 1.5e-3 (between 1e-3 and 2e-3)
        self.pretrained_head_state = None  # Will be set after model initialization
        
        # EMA (Exponential Moving Average)
        self.ema_decay = 0.999
        self.ema_model_state = None
        self.ema_initialized = False
        self.base_model_state = None  # Store base weights for restoration after validation/test
        
        # Statistics
        self.nan_steps_skipped = 0
        self.total_steps = 0
        
        # Wrap model with LoRA if enabled
        if self.lora_enabled:
            logger.info("="*80)
            logger.info("Wrapping model with LoRA...")
            logger.info("="*80)
            
            self.lora_model = PLUTOModelWithLoRA(
                base_model=model,
                lora_enabled=True,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                ultra_minimal=self.ultra_minimal,
                verbose=True,
            )
            # Replace model with LoRA-wrapped version
            self.model = self.lora_model.base_model
        else:
            self.lora_model = None
            logger.info("LoRA disabled - using ultra-minimal head-only fine-tuning")
    
    def on_fit_start(self) -> None:
        """Initialize pretrained state for L2-SP and EMA."""
        super().on_fit_start()
        
        # Verify LayerNorm affine parameters
        if self.ultra_minimal:
            self.verify_layer_norm_affine()
        
        # Save pretrained state for L2-SP (only head bias/LN parameters)
        if self.ultra_minimal:
            self.pretrained_head_state = {}
            expected_params = ['.mlp.0.bias', '.mlp.1.weight', '.mlp.1.bias', '.mlp.3.bias']
            found_params = {pattern: [] for pattern in expected_params}
            
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    # Only save head parameters (bias + LN gamma/beta)
                    is_head = any(head in name for head in ['loc_head', 'yaw_head', 'vel_head', 'pi_head'])
                    if is_head:
                        for pattern in expected_params:
                            if pattern in name:
                                self.pretrained_head_state[name] = param.data.clone().detach()
                                found_params[pattern].append(name)
                                break
            
            logger.info(f"✓ Saved pretrained state for {len(self.pretrained_head_state)} head parameters (L2-SP)")
            logger.info(f"  L2-SP parameter breakdown:")
            for pattern, names in found_params.items():
                logger.info(f"    {pattern}: {len(names)} params")
                if names:
                    logger.info(f"      Examples: {names[:2]}")
            
            # Verify all expected patterns are found
            missing = [p for p, names in found_params.items() if not names]
            if missing:
                logger.warning(f"  ⚠ Missing L2-SP patterns: {missing}")
        
        # Initialize EMA
        if not self.ema_initialized:
            self.ema_model_state = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.ema_model_state[name] = param.data.clone().detach()
            self.ema_initialized = True
            logger.info(f"✓ Initialized EMA for {len(self.ema_model_state)} trainable parameters")
    
    def training_step(
        self,
        batch: Tuple[FeaturesType, TargetsType, ScenarioListType],
        batch_idx: int,
    ) -> Optional[torch.Tensor]:
        """
        Training step with NaN/Inf handling, L2-SP loss, and EMA update.
        
        Args:
            batch: Training batch
            batch_idx: Batch index
            
        Returns:
            Loss tensor or None to skip backward pass
        """
        self.total_steps += 1
        self.current_batch_idx = batch_idx
        self._nan_debug_batch_context = self._format_nan_debug_batch_context(batch, batch_idx)
        
        # Forward pass
        loss = self._step(batch, prefix="train")
        
        # Add L2-SP regularization loss (for head bias/LN parameters)
        if loss is not None and self.pretrained_head_state is not None and len(self.pretrained_head_state) > 0:
            l2sp_loss = 0.0
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.pretrained_head_state:
                    pretrained_param = self.pretrained_head_state[name]
                    # L2-SP: λ * ||θ - θ₀||²
                    l2sp_loss += self.l2sp_lambda * torch.sum((param - pretrained_param) ** 2)
            
            if l2sp_loss > 0:
                loss = loss + l2sp_loss
                if self.trainer.global_step % 100 == 0:
                    self.log("train/l2sp_loss", l2sp_loss, on_step=True, on_epoch=False)
        
        # Check for NaN/Inf
        if self.skip_nan_steps:
            if loss is not None and (torch.isnan(loss) or torch.isinf(loss)):
                self.nan_steps_skipped += 1
                logger.warning(
                    f"Step {batch_idx}: NaN/Inf loss detected (skipped {self.nan_steps_skipped} total)\n"
                    f"{self._nan_debug_batch_context}"
                )
                # Zero gradients and skip backward pass
                self.optimizers().zero_grad()
                return None
        
        # Update EMA after optimizer step (will be called in on_train_batch_end)
        
        # Log gradient norm if available
        if self.trainer.global_step % 100 == 0:
            total_norm = 0.0
            param_count = 0
            for p in self.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
            if param_count > 0:
                total_norm = total_norm ** (1.0 / 2)
                self.log("train/grad_norm", total_norm, on_step=True, on_epoch=False)
        
        return loss

    def _format_nan_debug_batch_context(
        self,
        batch: Tuple[FeaturesType, TargetsType, ScenarioListType],
        batch_idx: int,
    ) -> str:
        """Return compact scenario identifiers for NaN diagnostics."""
        try:
            _features, _targets, scenarios = batch
        except Exception:
            return f"  batch_idx: {batch_idx}\n  scenarios: <unavailable>"

        rows = []
        for scenario in scenarios:
            token = getattr(scenario, "token", None)
            log_name = getattr(scenario, "log_name", None)
            scenario_type = getattr(scenario, "scenario_type", None)
            scenario_name = getattr(scenario, "scenario_name", None)
            rows.append(
                {
                    "token": token,
                    "log_name": log_name,
                    "scenario_type": scenario_type,
                    "scenario_name": scenario_name,
                }
            )

        return (
            f"  batch_idx: {batch_idx}\n"
            f"  global_step: {getattr(self.trainer, 'global_step', 'unknown')}\n"
            f"  scenarios: {rows}"
        )
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Log NaN step statistics and update EMA."""
        if self.trainer.global_step % 100 == 0:
            nan_rate = self.nan_steps_skipped / max(self.total_steps, 1) * 100
            self.log("train/nan_steps_skipped", self.nan_steps_skipped, on_step=True, on_epoch=False)
            self.log("train/nan_rate_pct", nan_rate, on_step=True, on_epoch=False)
        
        # Update EMA (Exponential Moving Average)
        if self.ema_initialized and self.ema_model_state is not None:
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if param.requires_grad and name in self.ema_model_state:
                        # EMA update: θ_ema = decay * θ_ema + (1 - decay) * θ
                        self.ema_model_state[name].mul_(self.ema_decay).add_(
                            param.data, alpha=1.0 - self.ema_decay
                        )
    
    def on_validation_start(self) -> None:
        """Load EMA weights for validation."""
        super().on_validation_start()
        if self.ema_initialized and self.ema_model_state is not None:
            # Save current base weights
            self.base_model_state = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.ema_model_state:
                    self.base_model_state[name] = param.data.clone().detach()
            
            # Load EMA weights
            logger.info("🔄 Loading EMA weights for validation...")
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.ema_model_state:
                    param.data.copy_(self.ema_model_state[name])
    
    def on_validation_end(self) -> None:
        """Restore base weights after validation."""
        super().on_validation_end()
        if self.base_model_state is not None:
            # Restore base weights for training
            logger.info("🔄 Restoring base weights after validation...")
            for name, param in self.model.named_parameters():
                if name in self.base_model_state:
                    param.data.copy_(self.base_model_state[name])
            self.base_model_state = None
    
    def on_test_start(self) -> None:
        """Load EMA weights for testing."""
        super().on_test_start()
        if self.ema_initialized and self.ema_model_state is not None:
            # Save current base weights
            self.base_model_state = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.ema_model_state:
                    self.base_model_state[name] = param.data.clone().detach()
            
            # Load EMA weights
            logger.info("🔄 Loading EMA weights for testing...")
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.ema_model_state:
                    param.data.copy_(self.ema_model_state[name])
    
    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, _LRScheduler]]]:
        """
        Configure optimizer with separate learning rates for LoRA and head.
        
        Returns:
            Optimizer and scheduler configuration
        """
        if self.lora_enabled and self.lora_model is not None:
            # Use LoRA model's optimizer groups
            optim_groups = self.lora_model.get_optimizer_groups(
                lora_lr=self.lora_lr,
                head_lr=self.head_lr,
                weight_decay=self.weight_decay,
            )
        else:
            # LoRA disabled: collect only trainable parameters (ultra-minimal head)
            all_trainable_params = [
                (n, p) for n, p in self.model.named_parameters() if p.requires_grad
            ]
            
            if not all_trainable_params:
                raise RuntimeError("No trainable parameters found!")
            
            # Organize by weight decay
            decay_params = []
            no_decay_params = []
            
            for param_name, param in all_trainable_params:
                apply_decay = "bias" not in param_name and "norm" not in param_name.lower()
                
                if apply_decay:
                    decay_params.append(param)
                else:
                    no_decay_params.append(param)
            
            optim_groups = []
            if decay_params:
                optim_groups.append({
                    "params": decay_params,
                    "lr": self.head_lr,
                    "weight_decay": self.weight_decay,
                })
            if no_decay_params:
                optim_groups.append({
                    "params": no_decay_params,
                    "lr": self.head_lr,
                    "weight_decay": 0.0,
                })
        
        # Create optimizer
        optimizer = torch.optim.AdamW(optim_groups)
        
        # Create scheduler with warmup
        from src.optim.warmup_cos_lr import WarmupCosLR
        
        # Calculate total steps
        # Assume we can get this from trainer or datamodule
        # For now, use a reasonable estimate
        total_steps = self.trainer.estimated_stepping_batches if hasattr(self.trainer, 'estimated_stepping_batches') else self.epochs * 1000
        
        # WarmupCosLR may use different parameter names, check the actual implementation
        # For now, use epochs-based scheduler
        try:
            scheduler = WarmupCosLR(
                optimizer=optimizer,
                lr=self.lora_lr if self.lora_enabled else self.head_lr,
                min_lr=1e-6,
                warmup_steps=self.warmup_steps,
                epochs=self.epochs,
                total_steps=total_steps,
            )
        except TypeError:
            # Fallback: use epochs if total_steps not supported
            scheduler = WarmupCosLR(
                optimizer=optimizer,
                lr=self.lora_lr if self.lora_enabled else self.head_lr,
                min_lr=1e-6,
                epochs=self.epochs,
                warmup_epochs=max(1, self.warmup_steps // 1000),  # Approximate
            )
        
        # Log actual learning rates for verification
        logger.info(f"\n{'='*80}")
        logger.info("Optimizer Configuration:")
        logger.info(f"{'='*80}")
        for i, group in enumerate(optimizer.param_groups):
            n_params = sum(p.numel() for p in group["params"])
            logger.info(f"  Group {i}: {n_params:,} params, lr={group['lr']:.2e}, wd={group['weight_decay']}")
        logger.info(f"{'='*80}\n")
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
    
    def on_train_epoch_start(self) -> None:
        """Log actual learning rates at epoch start."""
        super().on_train_epoch_start()
        # Log learning rates at epoch start (on_step=False is required for epoch hooks)
        if hasattr(self, 'optimizers'):
            optimizer = self.optimizers()
            if optimizer is not None:
                for i, group in enumerate(optimizer.param_groups):
                    self.log(f"lr/param_group_{i}", group['lr'], on_step=False, on_epoch=True)
    
    def on_train_batch_start(self, batch, batch_idx: int) -> None:
        """Log actual learning rates during training (every 100 steps)."""
        super().on_train_batch_start(batch, batch_idx)
        if self.trainer.global_step % 100 == 0 and hasattr(self, 'optimizers'):
            optimizer = self.optimizers()
            if optimizer is not None:
                for i, group in enumerate(optimizer.param_groups):
                    self.log(f"lr_step/param_group_{i}", group['lr'], on_step=True, on_epoch=False)
    
    def on_before_optimizer_step(self, optimizer, optimizer_idx: int = 0):
        """Apply gradient clipping before optimizer step."""
        # Call parent method first (for NaN/inf handling)
        super().on_before_optimizer_step(optimizer, optimizer_idx)
        
        # Apply gradient clipping
        if self.gradient_clip_val > 0:
            # Calculate gradient norm before clipping
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(),
                max_norm=self.gradient_clip_val,
            )
            
            # Log gradient clipping statistics
            if self.trainer.global_step % 100 == 0:
                self.log("train/grad_norm_before_clip", total_norm.item(), on_step=True, on_epoch=False)
                if total_norm.item() > self.gradient_clip_val:
                    self.log("train/grad_clipped", 1.0, on_step=True, on_epoch=False)
                else:
                    self.log("train/grad_clipped", 0.0, on_step=True, on_epoch=False)
    
    def on_load_checkpoint(self, checkpoint: Dict) -> None:
        """
        Handle checkpoint loading, especially for curriculum stages.
        
        For curriculum stages, we skip loading optimizer state because:
        1. Parameter groups may have changed
        2. We want fresh optimizer state for the new stage
        
        Note: This modifies the checkpoint dict in-place, which Lightning will use.
        """
        if self.is_curriculum_stage:
            logger.info("🔄 Curriculum stage detected: Skipping optimizer state loading")
            # Remove optimizer and lr_scheduler states to force re-initialization
            # Set to empty list instead of deleting to avoid KeyError
            checkpoint["optimizer_states"] = []
            checkpoint["lr_schedulers"] = []
            logger.info("  → Cleared optimizer_states and lr_schedulers from checkpoint")
            # Reset Lightning's loop progress so stage max_epochs is interpreted as
            # stage-local epochs while still loading the previous stage's weights.
            checkpoint.pop("loops", None)
            checkpoint["epoch"] = 0
            checkpoint["global_step"] = 0
            logger.info("  → Reset loop progress, epoch, and global_step for a fresh curriculum stage")
            logger.info("  ✓ Will load model weights but re-initialize optimizer/scheduler")
    
    def save_lora_only(self, filepath: str) -> None:
        """Save LoRA-only checkpoint."""
        if self.lora_model is not None:
            self.lora_model.save_lora_only(filepath)
        else:
            logger.warning("LoRA is not enabled, nothing to save")
    
    def merge_and_save(self, filepath: str, use_ema: bool = False) -> None:
        """
        Merge LoRA weights and save.
        
        Args:
            filepath: Path to save the merged model
            use_ema: If True, use EMA weights instead of current weights
        """
        if use_ema and self.ema_initialized and self.ema_model_state is not None:
            logger.info("Saving model with EMA weights...")
            # Temporarily save current state
            current_state = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.ema_model_state:
                    current_state[name] = param.data.clone()
                    param.data.copy_(self.ema_model_state[name])
            
            # Save with EMA weights
            if self.lora_model is not None:
                self.lora_model.merge_and_save(filepath)
            else:
                torch.save({"state_dict": self.model.state_dict()}, filepath)
            
            # Restore current state
            for name, param in self.model.named_parameters():
                if name in current_state:
                    param.data.copy_(current_state[name])
            
            logger.info(f"✓ Saved EMA model to {filepath}")
        else:
            if self.lora_model is not None:
                self.lora_model.merge_and_save(filepath)
            else:
                logger.warning("LoRA is not enabled, saving base model only")
                torch.save({"state_dict": self.model.state_dict()}, filepath)
    
    def load_ema_weights(self) -> None:
        """Load EMA weights into the model."""
        if not self.ema_initialized or self.ema_model_state is None:
            logger.warning("EMA not initialized, cannot load EMA weights")
            return
        
        logger.info("Loading EMA weights into model...")
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.ema_model_state:
                param.data.copy_(self.ema_model_state[name])
        logger.info("✓ EMA weights loaded")
    
    def verify_layer_norm_affine(self) -> None:
        """Verify that LayerNorm layers have affine=True (have weight/bias)."""
        logger.info(f"\n{'='*80}")
        logger.info("Verifying LayerNorm affine parameters:")
        logger.info(f"{'='*80}")
        
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.LayerNorm):
                has_weight = hasattr(module, 'weight') and module.weight is not None
                has_bias = hasattr(module, 'bias') and module.bias is not None
                is_affine = module.elementwise_affine if hasattr(module, 'elementwise_affine') else True
                
                logger.info(f"  {name}:")
                logger.info(f"    elementwise_affine: {is_affine}")
                logger.info(f"    has weight: {has_weight}")
                logger.info(f"    has bias: {has_bias}")
                
                if 'mlp.1' in name and not is_affine:
                    logger.warning(f"    ⚠ WARNING: {name} does not have affine=True! mlp.1.weight/bias will not exist!")
        
        logger.info(f"{'='*80}\n")


# Alias for backward compatibility with finetune_pluto.py
LoRALightningTrainer = PLUTOLoRATrainer
