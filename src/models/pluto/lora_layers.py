"""
LoRA (Low-Rank Adaptation) layers for encoder-only fine-tuning.

This module provides:
- LoRALinear: Core LoRA linear layer implementation
- inject_lora_into_encoder(): Inject LoRA ONLY into encoder_blocks attention layers
- mark_only_lora_and_ultra_head_as_trainable(): Freeze all except LoRA + ultra-minimal head bias
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn


def _as_bool_mask(mask):
    if mask is None or mask.dtype == torch.bool:
        return mask
    return mask.bool()


class LoRALinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation) Linear Layer.
    
    Implements: y = W(x) + (alpha / rank) * B(A(x))
    where:
        - W is the frozen pretrained weight
        - A is (rank, in_features) low-rank matrix
        - B is (out_features, rank) low-rank matrix
        - rank is the LoRA rank
        - alpha is the scaling factor
    
    Args:
        in_features: Size of input features
        out_features: Size of output features
        rank: LoRA rank (r)
        alpha: LoRA alpha (scaling factor)
        dropout: Dropout probability for LoRA path
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 0.0
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Optional dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        
        # Initialize LoRA weights
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize LoRA parameters."""
        # A: normal initialization with small std
        std = 1.0 / math.sqrt(self.in_features)
        nn.init.normal_(self.lora_A, mean=0.0, std=std)
        # B: zero initialization
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: y = (alpha / rank) * B(A(x))
        
        Args:
            x: Input tensor of shape (..., in_features)
            
        Returns:
            LoRA output of shape (..., out_features)
        """
        # Apply dropout to input
        x_dropped = self.dropout(x)
        
        # Compute LoRA: scaling * B @ A @ x
        # A @ x: (rank, in_features) @ (..., in_features) -> (..., rank)
        step1 = x_dropped @ self.lora_A.T  # (..., rank)
        
        # B @ step1: (out_features, rank) @ (..., rank) -> (..., out_features)
        step2 = step1 @ self.lora_B.T  # (..., out_features)
        
        # Scale by alpha/rank
        lora_out = self.scaling * step2
        
        return lora_out


class LinearWithLoRA(nn.Module):
    """
    Wrapper that combines a frozen Linear layer with a LoRA adapter.
    
    output = original_linear(x) + lora(x)
    """
    
    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # Store original linear (will be frozen)
        self.original_linear = original_linear
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        
        # Freeze original weights
        for param in self.original_linear.parameters():
            param.requires_grad = False
        
        # Add LoRA adapter
        self.lora = LoRALinear(
            in_features=self.in_features,
            out_features=self.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: original + LoRA."""
        original_out = self.original_linear(x)
        lora_out = self.lora(x)
        return original_out + lora_out


class MultiheadAttentionInProjLoRA(nn.Module):
    """
    LoRA wrapper for MultiheadAttention's in_proj (combined QKV projection).
    
    PyTorch's MultiheadAttention uses in_proj_weight with shape (3*embed_dim, embed_dim)
    to compute Q, K, V in one shot. We add LoRA adapters to this projection.
    """
    
    def __init__(
        self,
        embed_dim: int,
        in_proj_weight: nn.Parameter,
        in_proj_bias: Optional[nn.Parameter],
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Store frozen original weights
        self.register_buffer('in_proj_weight', in_proj_weight.data.clone())
        if in_proj_bias is not None:
            self.register_buffer('in_proj_bias', in_proj_bias.data.clone())
        else:
            self.register_buffer('in_proj_bias', None)
        
        # LoRA for the full in_proj (3*embed_dim output)
        self.lora = LoRALinear(
            in_features=embed_dim,
            out_features=3 * embed_dim,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply in_proj with LoRA.
        
        Args:
            x: Input of shape (..., embed_dim)
            
        Returns:
            Combined QKV of shape (..., 3*embed_dim)
        """
        # Original projection
        original_out = torch.nn.functional.linear(x, self.in_proj_weight, self.in_proj_bias)
        
        # Add LoRA
        lora_out = self.lora(x)
        
        return original_out + lora_out


def _patch_multihead_attention_forward(attn: nn.MultiheadAttention):
    """
    Monkey-patch MultiheadAttention to use our custom in_proj with LoRA.
    """
    # Store the original forward for later restoration
    if not hasattr(attn, '_original_forward'):
        attn._original_forward = attn.forward.__func__.__get__(attn, type(attn))
    
    original_forward = attn._original_forward
    
    def custom_forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
        is_causal=False,
    ):
        has_lora_in_proj = hasattr(self, 'in_proj_lora') and self.in_proj_lora is not None
        key_padding_mask = _as_bool_mask(key_padding_mask)
        
        if has_lora_in_proj:
            # Self-attention: use LoRA-augmented in_proj for Q, K, V
            is_self_attention = query is key and key is value
            
            if is_self_attention:
                qkv = self.in_proj_lora(query)
                embed_dim = self.embed_dim
                qkv = qkv.unflatten(-1, (3, embed_dim)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
                q, k, v = qkv[0], qkv[1], qkv[2]
            else:
                # Cross-attention
                embed_dim = self.embed_dim
                q_qkv = self.in_proj_lora(query)
                q = q_qkv[..., :embed_dim]
                kv_qkv = self.in_proj_lora(key)
                k = kv_qkv[..., embed_dim:2*embed_dim]
                v = kv_qkv[..., 2*embed_dim:]
            
            # Continue with standard multi-head attention computation
            embed_dim = self.embed_dim
            q = q / (embed_dim ** 0.5)
            
            bsz, tgt_len, embed_dim = query.shape
            _, src_len, _ = key.shape if not is_self_attention else (bsz, tgt_len, embed_dim)
            
            q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
            
            attn_weights = torch.matmul(q, k.transpose(-2, -1))
            
            if attn_mask is not None:
                attn_weights = attn_weights + attn_mask
            
            if key_padding_mask is not None:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    float('-inf')
                )
            
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
            attn_weights = torch.nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
            
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, embed_dim)
            
            # Apply output projection (which may have LoRA)
            attn_output = self.out_proj(attn_output)
            
            if need_weights:
                if average_attn_weights:
                    attn_weights = attn_weights.mean(dim=1)
                return attn_output, attn_weights
            else:
                return attn_output, None
        else:
            # No LoRA: fall back to original forward
            return original_forward(
                query, key, value, 
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights
            )
    
    # Replace the forward method
    import types
    attn.forward = types.MethodType(custom_forward, attn)


def inject_lora_into_encoder(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    verbose: bool = True,
) -> int:
    """
    Inject LoRA adapters ONLY into encoder_blocks attention layers.
    
    Targets:
    - encoder_blocks[*].attn.in_proj_weight (QKV combined)
    - encoder_blocks[*].attn.out_proj (output projection)
    
    Does NOT touch:
    - planning_decoder
    - agent_encoder
    - map_encoder
    - static_objects_encoder
    - agent_predictor
    - any other modules
    
    Args:
        model: PLUTO model to inject LoRA into
        rank: LoRA rank
        alpha: LoRA alpha scaling factor
        dropout: Dropout probability for LoRA
        verbose: Whether to print injection info
        
    Returns:
        Number of LoRA adapters injected
    """
    lora_count = 0
    
    if verbose:
        print("="*80)
        print("Injecting LoRA into encoder_blocks attention layers ONLY...")
        print("="*80)
    
    # Check if model has encoder_blocks
    if not hasattr(model, 'encoder_blocks'):
        if verbose:
            print("⚠ Warning: Model does not have 'encoder_blocks' attribute")
        return 0
    
    # Iterate through each TransformerEncoderLayer in encoder_blocks
    for block_idx, block in enumerate(model.encoder_blocks):
        # Each block should have 'attn' (MultiheadAttention)
        if not hasattr(block, 'attn'):
            if verbose:
                print(f"⚠ Warning: encoder_blocks[{block_idx}] does not have 'attn'")
            continue
        
        attn = block.attn
        
        # Inject LoRA into in_proj (QKV combined)
        if hasattr(attn, 'in_proj_weight') and attn.in_proj_weight is not None:
            embed_dim = attn.embed_dim
            in_proj_weight = attn.in_proj_weight
            in_proj_bias = attn.in_proj_bias
            
            # Create LoRA wrapper for in_proj
            in_proj_lora = MultiheadAttentionInProjLoRA(
                embed_dim=embed_dim,
                in_proj_weight=in_proj_weight,
                in_proj_bias=in_proj_bias,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
            
            # Replace the in_proj mechanism
            attn._qkv_same_embed_dim = False
            attn.q_proj_weight = None
            attn.k_proj_weight = None
            attn.v_proj_weight = None
            attn.in_proj_weight = None
            attn.in_proj_bias = None
            attn.in_proj_lora = in_proj_lora
            
            # Monkey-patch the forward method
            _patch_multihead_attention_forward(attn)
            
            lora_count += 1
            if verbose:
                print(f"✓ Block {block_idx}: attn.in_proj (QKV) - {3*embed_dim}×{embed_dim}")
        
        # Inject LoRA into out_proj
        if hasattr(attn, 'out_proj') and isinstance(attn.out_proj, nn.Linear):
            original_out_proj = attn.out_proj
            lora_out_proj = LinearWithLoRA(
                original_out_proj,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
            attn.out_proj = lora_out_proj
            
            lora_count += 1
            if verbose:
                print(f"✓ Block {block_idx}: attn.out_proj - {original_out_proj.out_features}×{original_out_proj.in_features}")
    
    if verbose:
        print("="*80)
        print(f"✓ Total LoRA adapters injected: {lora_count}")
        print("="*80)
    
    return lora_count


def mark_only_lora_and_ultra_head_as_trainable(
    model: nn.Module,
    ultra_minimal: bool = True,
    verbose: bool = True,
) -> None:
    """
    Mark only LoRA parameters and ultra-minimal head (3 layer biases: mlp.0.bias, mlp.1 LN gamma/beta, mlp.3.bias) as trainable.
    
    Behavior:
    - If ultra_minimal=True:
        - Freeze all parameters except:
          - LoRA parameters (lora_A, lora_B)
          - First linear layer bias (mlp.0.bias) of each planning head:
            - planning_decoder.loc_head.mlp.0.bias
            - planning_decoder.yaw_head.mlp.0.bias
            - planning_decoder.vel_head.mlp.0.bias
            - planning_decoder.pi_head.mlp.0.bias
          - LayerNorm parameters (mlp.1.weight/gamma and mlp.1.bias/beta) of each planning head:
            - planning_decoder.loc_head.mlp.1.weight (gamma)
            - planning_decoder.loc_head.mlp.1.bias (beta)
            - planning_decoder.yaw_head.mlp.1.weight (gamma)
            - planning_decoder.yaw_head.mlp.1.bias (beta)
            - planning_decoder.vel_head.mlp.1.weight (gamma)
            - planning_decoder.vel_head.mlp.1.bias (beta)
            - planning_decoder.pi_head.mlp.1.weight (gamma)
            - planning_decoder.pi_head.mlp.1.bias (beta)
          - Last linear layer bias (mlp.3.bias) of each planning head:
            - planning_decoder.loc_head.mlp.3.bias
            - planning_decoder.yaw_head.mlp.3.bias
            - planning_decoder.vel_head.mlp.3.bias
            - planning_decoder.pi_head.mlp.3.bias
          - Note: All weights (mlp.0.weight, mlp.3.weight) are frozen (only biases are trainable)
    - If ultra_minimal=False:
        - LoRA parameters are trainable
        - All planning head parameters are trainable (not just biases)
    
    Args:
        model: The model to configure
        ultra_minimal: If True, only 3 layer biases (mlp.0.bias, mlp.1 LN gamma/beta, mlp.3.bias) are trainable. If False, entire heads are trainable.
        verbose: Whether to print configuration info
    """
    trainable_params = 0
    frozen_params = 0
    lora_params = 0
    head_params = 0
    
    # First, freeze ALL parameters
    for name, param in model.named_parameters():
        param.requires_grad = False
        frozen_params += param.numel()
    
    # Then, enable trainable parameters
    for name, param in model.named_parameters():
        is_trainable = False
        
        # Check if this is a LoRA parameter
        is_lora = "lora_A" in name or "lora_B" in name
        
        if is_lora:
            is_trainable = True
            lora_params += param.numel()
        
        # Check if this is a planning head parameter
        if 'planning_decoder' in name:
            is_head = any(head in name for head in ['loc_head', 'yaw_head', 'vel_head', 'pi_head'])
            if is_head:
                if ultra_minimal:
                    # 3 layer biases are trainable:
                    # 1. mlp.0.bias (first linear layer bias)
                    # 2. mlp.1.weight (LayerNorm gamma) and mlp.1.bias (LayerNorm beta)
                    # 3. mlp.3.bias (last linear layer bias)
                    # All weights (mlp.0.weight, mlp.3.weight) are frozen
                    if ('.mlp.0.bias' in name or 
                        '.mlp.1.weight' in name or 
                        '.mlp.1.bias' in name or 
                        '.mlp.3.bias' in name):
                        is_trainable = True
                        head_params += param.numel()
                else:
                    # Entire head is trainable
                    is_trainable = True
                    head_params += param.numel()
        
        if is_trainable:
            param.requires_grad = True
            trainable_params += param.numel()
            if verbose:
                param_type = "LoRA" if is_lora else "Head"
                print(f"✓ Trainable [{param_type}]: {name} ({param.numel():,} params)")
        else:
            frozen_params += param.numel()
    
    total_params = trainable_params + frozen_params
    trainable_pct = 100.0 * trainable_params / total_params if total_params > 0 else 0.0
    
    if verbose:
        print(f"\n{'='*80}")
        if ultra_minimal:
            print(f"ULTRA-MINIMAL MODE: LoRA + 3 layer biases (mlp.0.bias, mlp.1 LN gamma/beta, mlp.3.bias)")
        else:
            print(f"NORMAL MODE: LoRA + entire planning heads")
        print(f"Parameter Summary:")
        print(f"  🔥 LoRA parameters (A/B):     {lora_params:,}")
        print(f"  🔥 Head parameters (3 biases): {head_params:,}")
        print(f"  ────────────────────────────────────────────")
        print(f"  ✅ Total trainable:           {trainable_params:,} ({trainable_pct:.2f}%)")
        print(f"  ❄️  Frozen parameters:        {frozen_params:,} ({100-trainable_pct:.2f}%)")
        print(f"  📊 Total parameters:          {total_params:,}")
        print(f"{'='*80}\n")


def get_lora_state_dict(model: nn.Module) -> dict:
    """
    Extract only LoRA parameters from a model.
    
    Args:
        model: Model containing LoRA adapters
        
    Returns:
        State dict containing only LoRA parameters
    """
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_state_dict[name] = param.data.clone()
    
    return lora_state_dict


def merge_lora_weights(model: nn.Module, verbose: bool = True) -> nn.Module:
    """
    Merge all LoRA weights into the base model weights.
    
    Args:
        model: The model with LoRA adapters
        verbose: Whether to print merge progress
        
    Returns:
        A new model with LoRA weights merged into base weights
    """
    import copy
    
    # Create a deep copy to avoid modifying the original
    merged_model = copy.deepcopy(model)
    
    def merge_module(module: nn.Module, name: str = ""):
        """Recursively merge LoRA weights in all submodules"""
        for child_name, child in list(module.named_children()):
            full_name = f"{name}.{child_name}" if name else child_name
            
            if verbose:
                # Only print if we're actually merging something
                pass  # Will print when we actually merge
            
            # Handle LinearWithLoRA
            if isinstance(child, LinearWithLoRA):
                # Compute merged weight: W = W_0 + (alpha/r) * B @ A
                lora_weight = child.lora.scaling * (child.lora.lora_B @ child.lora.lora_A)
                merged_weight = child.original_linear.weight.data + lora_weight
                
                # Replace with standard Linear
                merged_linear = nn.Linear(
                    child.in_features,
                    child.out_features,
                    bias=child.original_linear.bias is not None,
                )
                merged_linear.weight.data = merged_weight
                if child.original_linear.bias is not None:
                    merged_linear.bias.data = child.original_linear.bias.data
                
                if verbose:
                    print(f"Merging LoRA weights for: {full_name}")
                setattr(module, child_name, merged_linear)
            
            # Handle MultiheadAttention with in_proj_lora
            elif isinstance(child, nn.MultiheadAttention) and hasattr(child, 'in_proj_lora'):
                if verbose:
                    print(f"Merging LoRA in_proj weights for: {full_name}")
                in_proj_lora = child.in_proj_lora
                
                # Merge LoRA weights
                lora_weight = in_proj_lora.lora.scaling * (in_proj_lora.lora.lora_B @ in_proj_lora.lora.lora_A)
                merged_weight = in_proj_lora.in_proj_weight + lora_weight
                
                # Restore the original MultiheadAttention structure
                child.in_proj_weight = nn.Parameter(merged_weight)
                if in_proj_lora.in_proj_bias is not None:
                    child.in_proj_bias = nn.Parameter(in_proj_lora.in_proj_bias.clone())
                
                child._qkv_same_embed_dim = True
                del child.in_proj_lora
                
                if hasattr(child, '_original_forward'):
                    child.forward = child._original_forward
                    del child._original_forward
                
                if verbose:
                    print(f"  → Merged LoRA parameters into in_proj_weight")
            
            # Recursively process children
            merge_module(child, full_name)
    
    # Start merging from the root
    merge_module(merged_model)
    
    if verbose:
        print("✓ All LoRA weights merged successfully!")
    
    return merged_model
