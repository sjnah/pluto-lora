"""
Curriculum Fine-tuning Controller.

This module provides:
- CurriculumFTController: Manages 3-stage curriculum training
- Ensures identical total update steps with uniform FT
- Stage progression with configurable sampling weights
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class CurriculumFTController:
    """
    Controller for curriculum fine-tuning with 3 stages.
    
    Stages:
    1. Easy scenarios (curriculum_stage1)
    2. Easy + Medium scenarios (curriculum_stage1 + curriculum_stage2)
    3. Easy + Medium + Hard scenarios (all stages)
    
    Ensures total update steps match uniform FT for fair comparison.
    """
    
    def __init__(
        self,
        stage_configs: Dict[str, DictConfig],
        uniform_total_steps: int,
        batch_size: int,
        reset_optimizer_each_stage: bool = False,
        verbose: bool = True,
    ):
        """
        Initialize curriculum controller.
        
        Args:
            stage_configs: Dictionary of stage configs:
                {
                    "stage1": {"splits": [...], "epochs": 5, "sampling_weights": [1.0]},
                    "stage2": {"splits": [...], "epochs": 5, "sampling_weights": [0.7, 0.3]},
                    "stage3": {"splits": [...], "epochs": 5, "sampling_weights": [0.5, 0.3, 0.2]},
                }
            uniform_total_steps: Total steps in uniform FT (for alignment)
            batch_size: Batch size
            reset_optimizer_each_stage: Whether to reset optimizer state between stages
            verbose: Whether to print progress
        """
        self.stage_configs = stage_configs
        self.uniform_total_steps = uniform_total_steps
        self.batch_size = batch_size
        self.reset_optimizer_each_stage = reset_optimizer_each_stage
        self.verbose = verbose
        
        self.current_stage = 1
        self.stage_history = []
        
        # Calculate steps per stage to match uniform FT
        self._calculate_stage_steps()
    
    def _calculate_stage_steps(self):
        """Calculate steps per stage to match uniform FT total steps."""
        # Get dataset sizes for each stage
        stage_sizes = {}
        for stage_name, config in self.stage_configs.items():
            # Estimate dataset size from splits (placeholder - actual implementation depends on data loading)
            # For now, assume we can get this from config
            splits = config.get("splits", [])
            # This is a placeholder - actual size calculation depends on your data loading
            estimated_size = config.get("estimated_size", 1000)  # Placeholder
            stage_sizes[stage_name] = estimated_size
        
        # Calculate steps per stage
        # Stage 1: only easy
        # Stage 2: easy + medium (with sampling weights)
        # Stage 3: easy + medium + hard (with sampling weights)
        
        # For simplicity, divide total steps equally across stages
        # More sophisticated: weight by dataset size and sampling weights
        steps_per_stage = self.uniform_total_steps // 3
        
        self.stage_steps = {
            "stage1": steps_per_stage,
            "stage2": steps_per_stage,
            "stage3": self.uniform_total_steps - 2 * steps_per_stage,  # Remaining steps
        }
        
        if self.verbose:
            logger.info("="*80)
            logger.info("Curriculum Stage Steps Calculation")
            logger.info("="*80)
            logger.info(f"Uniform FT total steps: {self.uniform_total_steps:,}")
            for stage_name, steps in self.stage_steps.items():
                logger.info(f"  {stage_name}: {steps:,} steps")
            logger.info("="*80)
    
    def get_current_stage_config(self) -> DictConfig:
        """Get configuration for current stage."""
        stage_name = f"stage{self.current_stage}"
        if stage_name not in self.stage_configs:
            raise ValueError(f"Stage {self.current_stage} not found in configs")
        return self.stage_configs[stage_name]
    
    def get_current_stage_splits(self) -> List[str]:
        """Get data splits for current stage."""
        config = self.get_current_stage_config()
        return config.get("splits", [])
    
    def get_current_stage_sampling_weights(self) -> List[float]:
        """Get sampling weights for current stage."""
        config = self.get_current_stage_config()
        return config.get("sampling_weights", [1.0])
    
    def get_current_stage_epochs(self) -> int:
        """Get epochs for current stage."""
        config = self.get_current_stage_config()
        return config.get("epochs", 5)
    
    def get_current_stage_steps(self) -> int:
        """Get target steps for current stage."""
        stage_name = f"stage{self.current_stage}"
        return self.stage_steps.get(stage_name, 1000)
    
    def advance_stage(self) -> bool:
        """
        Advance to next stage.
        
        Returns:
            True if advanced, False if already at final stage
        """
        if self.current_stage >= 3:
            return False
        
        self.current_stage += 1
        
        if self.verbose:
            logger.info("="*80)
            logger.info(f"Advancing to Stage {self.current_stage}/3")
            logger.info("="*80)
            config = self.get_current_stage_config()
            logger.info(f"Splits: {config.get('splits', [])}")
            logger.info(f"Sampling weights: {config.get('sampling_weights', [])}")
            logger.info(f"Target steps: {self.get_current_stage_steps():,}")
            logger.info("="*80)
        
        return True
    
    def is_final_stage(self) -> bool:
        """Check if current stage is the final stage."""
        return self.current_stage >= 3
    
    def should_reset_optimizer(self) -> bool:
        """Check if optimizer should be reset for current stage."""
        return self.reset_optimizer_each_stage and self.current_stage > 1
    
    def get_checkpoint_path(self, base_dir: str) -> str:
        """Get checkpoint path for current stage."""
        stage_name = f"stage{self.current_stage}"
        return f"{base_dir}/checkpoints/{stage_name}_last.ckpt"
    
    def save_stage_info(self, output_dir: str):
        """Save stage information to file."""
        import json
        from pathlib import Path
        
        info = {
            "current_stage": self.current_stage,
            "uniform_total_steps": self.uniform_total_steps,
            "stage_steps": self.stage_steps,
            "stage_configs": {
                k: {
                    "splits": v.get("splits", []),
                    "epochs": v.get("epochs", 5),
                    "sampling_weights": v.get("sampling_weights", [1.0]),
                }
                for k, v in self.stage_configs.items()
            },
        }
        
        output_path = Path(output_dir) / "curriculum_info.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(info, f, indent=2)
        
        logger.info(f"✓ Saved curriculum info to {output_path}")

