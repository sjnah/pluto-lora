"""
Fine-tune PLUTO with LoRA adapters.

This script fine-tunes a pretrained PLUTO model using LoRA (Low-Rank Adaptation)
for parameter-efficient training on nuplan-mini or other datasets.

Example usage:
    python scripts/training/finetune_pluto.py \
        experiment=lora_finetune \
        pretrained_ckpt=checkpoints/pluto_1M_aux_cil.ckpt \
        lora.enabled=true \
        lora.rank=8 \
        lora.alpha=16 \
        epochs=20

    Or with a config file:
    python scripts/training/finetune_pluto.py --config-name training/train_pluto_lora
"""

import logging
import os
from pathlib import Path
from typing import Optional

import hydra
import pytorch_lightning as pl
import torch
from nuplan.planning.script.builders.folder_builder import (
    build_training_experiment_folder,
)
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from nuplan.planning.script.profiler_context_manager import ProfilerContextManager
from nuplan.planning.script.utils import set_default_path
from omegaconf import DictConfig, OmegaConf

from src.custom_training import (
    TrainingEngine,
    build_lightning_datamodule,
    update_config_for_training,
)
from src.custom_training.custom_training_builder import (
    build_custom_trainer,
)
from src.models.pluto.pluto_lora_trainer import LoRALightningTrainer
from src.models.pluto.pluto_model import PlanningModel

logging.getLogger("numba").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# If set, use the env. variable to overwrite the default dataset and experiment paths
set_default_path()

# Hydra config
CONFIG_PATH = "../../config"
CONFIG_NAME = "default_training"


def load_pretrained_pluto(
    checkpoint_path: str,
    device: str = "cuda",
) -> PlanningModel:
    """
    Load a pretrained PLUTO model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on
        
    Returns:
        Loaded PlanningModel
    """
    logger.info(f"Loading pretrained PLUTO from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract state dict
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    
    # Remove "model." prefix if present (from Lightning wrapper)
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            cleaned_state_dict[key[6:]] = value
        else:
            cleaned_state_dict[key] = value
    
    return cleaned_state_dict


def build_lora_training_engine(
    cfg: DictConfig,
    worker,
    pretrained_state_dict: Optional[dict] = None,
    has_checkpoint: bool = False,
) -> TrainingEngine:
    """
    Build training engine with LoRA-enabled model.
    
    Args:
        cfg: Configuration
        worker: Worker pool
        pretrained_state_dict: Pretrained model state dict
        
    Returns:
        TrainingEngine with LoRA model
    """
    logger.info("="*80)
    logger.info("Building LoRA fine-tuning training engine...")
    logger.info("="*80)
    
    # Build trainer
    trainer = build_custom_trainer(cfg)
    
    # Create base PLUTO model
    logger.info("Creating base PLUTO model...")
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    
    # Extract loss config before instantiating model (to avoid passing it to PlanningModel)
    loss_config = None
    if hasattr(cfg.model, 'loss'):
        # Extract loss config as a dict
        loss_config = OmegaConf.to_container(cfg.model.loss, resolve=True)
        # Create a copy of model config without loss for instantiation
        model_cfg_dict = OmegaConf.to_container(cfg.model, resolve=True)
        if isinstance(model_cfg_dict, dict) and 'loss' in model_cfg_dict:
            model_cfg_dict = {k: v for k, v in model_cfg_dict.items() if k != 'loss'}
        model_cfg = OmegaConf.create(model_cfg_dict)
    else:
        model_cfg = cfg.model
    
    torch_module_wrapper = instantiate(model_cfg)
    
    # Load pretrained weights if provided (only if not using Lightning checkpoint)
    # If using Lightning checkpoint, the model will be loaded by Lightning during fit()
    if pretrained_state_dict is not None and not has_checkpoint:
        logger.info("Loading pretrained weights into model...")
        missing_keys, unexpected_keys = torch_module_wrapper.load_state_dict(
            pretrained_state_dict, strict=False
        )
        if missing_keys:
            logger.warning(f"Missing keys: {missing_keys[:5]}...")  # Show first 5
        if unexpected_keys:
            logger.warning(f"Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5
        logger.info("✓ Pretrained weights loaded successfully")
    elif has_checkpoint:
        logger.info("⏭️  Skipping pretrained weight loading - Lightning will load from checkpoint during fit()")
    
    # Build datamodule
    logger.info("Building datamodule...")
    datamodule = build_lightning_datamodule(cfg, worker, torch_module_wrapper)
    
    # Extract LoRA config
    lora_config = cfg.get("lora", {})
    
    # Check if this is a curriculum learning stage (Stage 2 or 3)
    is_curriculum_stage = lora_config.get("is_curriculum_stage", False)
    if is_curriculum_stage:
        logger.info("🔄 Curriculum Learning Stage: Will keep existing LoRA weights from previous stage")
    else:
        logger.info("🆕 Initial Stage: Will initialize new LoRA weights")
    
    # Build LoRA Lightning module
    logger.info("Creating LoRA Lightning module...")
    
    # Extract loss weights from config (support both model.loss and loss paths)
    # Use the loss_config we extracted earlier, or try to get it from cfg
    if loss_config is None:
        if "model" in cfg and hasattr(cfg.model, "loss"):
            loss_config = OmegaConf.to_container(cfg.model.loss, resolve=True)
        elif "loss" in cfg:
            loss_config = OmegaConf.to_container(cfg.loss, resolve=True) if hasattr(cfg.loss, '__dict__') else dict(cfg.loss)
        else:
            loss_config = {}
    
    # Ensure loss_config is a dict
    if loss_config is None:
        loss_config = {}
    elif not isinstance(loss_config, dict):
        loss_config = OmegaConf.to_container(loss_config, resolve=True) if hasattr(loss_config, '__dict__') else {}
    
    model = LoRALightningTrainer(
        model=torch_module_wrapper,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        epochs=cfg.epochs,
        warmup_epochs=cfg.get("warmup_epochs", 0),
        warmup_steps=cfg.get("warmup_steps", None),
        use_collision_loss=lora_config.get("use_collision_loss", True),
        use_contrast_loss=lora_config.get("use_contrast_loss", False),
        regulate_yaw=lora_config.get("regulate_yaw", False),
        objective_aggregate_mode=cfg.objective_aggregate_mode,
        # Loss weights (from config or defaults)
        weight_reg_loss=loss_config.get("weight_reg_loss", 1.0) if loss_config else 1.0,
        weight_cls_loss=loss_config.get("weight_cls_loss", 1.0) if loss_config else 1.0,
        weight_prediction_loss=loss_config.get("weight_prediction_loss", 1.0) if loss_config else 1.0,
        weight_collision_loss=loss_config.get("weight_collision_loss", 1.0) if loss_config else 1.0,
        weight_contrastive_loss=loss_config.get("weight_contrastive_loss", 1.0) if loss_config else 1.0,
        weight_ref_free_reg_loss=loss_config.get("weight_ref_free_reg_loss", 1.0) if loss_config else 1.0,
        weight_auxiliary=loss_config.get("weight_auxiliary", 1.0) if loss_config else 1.0,
        # LoRA-specific parameters
        lora_enabled=lora_config.get("enabled", True),
        lora_rank=lora_config.get("rank", 8),
        lora_alpha=lora_config.get("alpha", 16.0),
        lora_dropout=lora_config.get("dropout", 0.05),
        lora_lr=lora_config.get("lora_lr", cfg.lr),
        policy_head_lr=lora_config.get("policy_head_lr", cfg.lr * 0.5),
        trainable_modules=lora_config.get("trainable_modules", None),
        is_curriculum_stage=is_curriculum_stage,
        ultra_minimal=lora_config.get("ultra_minimal", False),
    )
    
    engine = TrainingEngine(trainer=trainer, datamodule=datamodule, model=model)
    
    logger.info("✓ LoRA training engine built successfully\n")
    return engine


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> Optional[TrainingEngine]:
    """
    Main entrypoint for LoRA fine-tuning.
    
    Args:
        cfg: Hydra configuration
        
    Returns:
        TrainingEngine
    """
    # Set random seed
    pl.seed_everything(cfg.seed, workers=True)
    
    # Configure logger
    build_logger(cfg)
    
    # Override configs based on setup
    update_config_for_training(cfg)
    
    # Create output storage folder
    build_training_experiment_folder(cfg=cfg)
    
    # Build worker
    worker = build_worker(cfg)
    
    # Determine checkpoint loading strategy:
    # - If 'checkpoint' is specified: Lightning will load model + optimizer + lr_scheduler state
    # - If 'pretrained_ckpt' is specified: Only load model weights (optimizer will be initialized)
    # - If both are specified: 'checkpoint' takes precedence (for resuming training with optimizer state)
    pretrained_state_dict = None
    has_checkpoint = "checkpoint" in cfg and cfg.checkpoint is not None
    
    if has_checkpoint:
        logger.info(f"📦 Using Lightning checkpoint: {cfg.checkpoint}")
        logger.info("   This will load model + optimizer + lr_scheduler state (resuming training)")
        # Don't load pretrained weights - Lightning will load everything from checkpoint
        pretrained_state_dict = None
    elif "pretrained_ckpt" in cfg and cfg.pretrained_ckpt is not None:
        logger.info(f"📦 Using pretrained checkpoint: {cfg.pretrained_ckpt}")
        logger.info("   This will load only model weights (optimizer will be initialized)")
        pretrained_state_dict = load_pretrained_pluto(
            cfg.pretrained_ckpt,
            device="cpu",  # Load on CPU first
        )
    else:
        logger.warning("⚠ No checkpoint specified! Training from scratch.")
    
    # Build LoRA training engine
    with ProfilerContextManager(
        cfg.output_dir, cfg.enable_profiling, "build_lora_training_engine"
    ):
        engine = build_lora_training_engine(cfg, worker, pretrained_state_dict, has_checkpoint=has_checkpoint)
    
    # Start training
    if cfg.py_func == "train":
        logger.info("="*80)
        logger.info("Starting LoRA fine-tuning...")
        logger.info("="*80)
        
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "training"):
            engine.trainer.fit(
                model=engine.model,
                datamodule=engine.datamodule,
                ckpt_path=cfg.checkpoint if "checkpoint" in cfg else None,
            )
        
        # Save LoRA-only checkpoint
        lora_save_dir = Path(cfg.output_dir) / "lora_checkpoints"
        lora_save_dir.mkdir(exist_ok=True)
        lora_path = lora_save_dir / "lora_final.pt"
        
        logger.info(f"Saving LoRA-only weights to {lora_path}...")
        engine.model.save_lora_only(str(lora_path))
        
        # Optionally save merged checkpoint
        if cfg.get("lora", {}).get("save_merged", True):
            merged_path = lora_save_dir / "merged_final.ckpt"
            logger.info(f"Saving merged model to {merged_path}...")
            engine.model.merge_and_save(str(merged_path), use_ema=False)
            
            # Also save EMA version if available
            if hasattr(engine.model, 'ema_initialized') and engine.model.ema_initialized:
                ema_merged_path = lora_save_dir / "merged_final_ema.ckpt"
                logger.info(f"Saving EMA merged model to {ema_merged_path}...")
                engine.model.merge_and_save(str(ema_merged_path), use_ema=True)
        
        logger.info("✓ Training complete!")
        
        return engine
    
    elif cfg.py_func == "validate":
        logger.info("Starting validation...")
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "validate"):
            engine.trainer.validate(
                model=engine.model,
                datamodule=engine.datamodule,
                ckpt_path=cfg.checkpoint if "checkpoint" in cfg else None,
            )
        return engine
    
    elif cfg.py_func == "test":
        logger.info("Starting testing...")
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "testing"):
            engine.trainer.test(
                model=engine.model,
                datamodule=engine.datamodule,
            )
        return engine
    
    else:
        raise ValueError(f"Unknown py_func: {cfg.py_func}")


if __name__ == "__main__":
    main()
