"""
Merge LoRA weights into base model.

This script merges LoRA adapters into the base PLUTO model weights,
creating a standard checkpoint without LoRA components.

Usage:
    python scripts/training/merge_lora.py \
        --checkpoint checkpoints/model_with_lora.ckpt \
        --output checkpoints/merged_model.ckpt
"""

import argparse
import logging
import torch
from pathlib import Path

from src.models.pluto.lora_layers import merge_lora_weights
from src.models.pluto.pluto_model import PlanningModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA weights into base model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint with LoRA weights",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save merged checkpoint",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Path to model config file (optional)",
    )
    
    args = parser.parse_args()
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    
    # Extract state dict
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    
    # Remove "model." prefix if present
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            cleaned_state_dict[key[6:]] = value
        else:
            cleaned_state_dict[key] = value
    
    # Create model (you may need to adjust this based on your config system)
    logger.info("Creating model...")
    model = PlanningModel()
    
    # Load weights
    model.load_state_dict(cleaned_state_dict, strict=False)
    
    # Merge LoRA weights
    logger.info("Merging LoRA weights...")
    merged_model = merge_lora_weights(model, verbose=True)
    
    # Save merged checkpoint
    logger.info(f"Saving merged checkpoint to: {args.output}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(
        {
            "state_dict": merged_model.state_dict(),
            "merged_from": args.checkpoint,
        },
        output_path,
    )
    
    logger.info("✓ LoRA weights merged successfully!")
    logger.info(f"✓ Saved merged model to: {args.output}")


if __name__ == "__main__":
    main()
