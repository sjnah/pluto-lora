"""
Compatibility exports for PLUTO LoRA utilities.

The active implementation lives in :mod:`src.models.pluto.lora_layers`.
The older debug-heavy experimental implementation was archived under
``archive/legacy_lora_debug/`` to keep the model package focused while
preserving the historical code.
"""

from .lora_layers import (  # noqa: F401
    LinearWithLoRA,
    LoRALinear,
    MultiheadAttentionInProjLoRA,
    get_lora_state_dict,
    inject_lora_into_encoder,
    mark_only_lora_and_ultra_head_as_trainable,
    merge_lora_weights,
)
