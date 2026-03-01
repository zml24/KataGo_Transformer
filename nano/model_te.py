"""Transformer model architecture for KataGo nano training (TransformerEngine version).

This module provides the same Model API as model.py but uses NVIDIA TransformerEngine
for fused kernels and optional FP8 training support.

Three integration levels (te_mode):
    - "dpa":  Only DotProductAttention from TE; Q/K/V/out_proj as te.Linear;
              RoPE applied manually (rotate_every_two, same as model.py).
              Weights are directly compatible with model.py.
    - "mha":  te.MultiheadAttention (fused QKV + attention);
              RoPE handled internally by TE (rotate_half).
    - "full": te.TransformerLayer (full fused block including FFN + residual);
              RoPE handled internally by TE (rotate_half).

Usage:
    - Drop-in replacement for model.py's Model class
    - Requires: pip install transformer-engine[pytorch]
    - FP8 requires Hopper (H100/H200) or Ada (RTX 4090) GPU
    - Non-FP8 mode still benefits from TE's fused kernels
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te

from configs import get_num_bin_input_features, get_num_global_input_features
from model import (
    EXTRA_SCORE_DISTR_RADIUS,
    SoftPlusWithGradientFloor,
    cross_entropy,
    precompute_freqs_cos_sin_2d,
    apply_rotary_emb,
    PolicyHead,
    ValueHead,
)


# ---------------------------------------------------------------------------
# Level 1 — DPA: Only DotProductAttention from TE
# ---------------------------------------------------------------------------
class TransformerBlockTE_DPA(nn.Module):
    """Uses te.Linear for Q/K/V/out projections and te.DotProductAttention.

    RoPE is applied manually using rotate_every_two (same as model.py),
    so weights are directly compatible with model.py checkpoints.
    """

    def __init__(self, c_main: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = c_main // num_heads

        # Pre-attention RMSNorm
        self.norm1 = te.RMSNorm(c_main, eps=1e-6)

        # Q/K/V/Out projections (te.Linear for FP8 GEMM support)
        self.q_proj = te.Linear(c_main, c_main, bias=False)
        self.k_proj = te.Linear(c_main, c_main, bias=False)
        self.v_proj = te.Linear(c_main, c_main, bias=False)
        self.out_proj = te.Linear(c_main, c_main, bias=False)

        # TE DotProductAttention (fused flash/memory-efficient attention)
        self.attn = te.DotProductAttention(
            num_heads, self.head_dim,
            attn_mask_type="no_mask",
            qkv_format="bshd",
        )

        # Fused RMSNorm + SwiGLU MLP
        self.ffn = te.LayerNormMLP(
            c_main, ffn_dim,
            eps=1e-6, bias=False,
            normalization="RMSNorm",
            activation="swiglu",
        )

    def forward(self, x, attn_mask, rope_cos, rope_sin):
        """
        x: (N, L, C)
        attn_mask: unused (None)
        rope_cos, rope_sin: (L, 1, 1, head_dim) precomputed
        """
        B, L, C = x.shape
        x_normed = self.norm1(x)

        q = self.q_proj(x_normed).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x_normed).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x_normed).view(B, L, self.num_heads, self.head_dim)

        q, k = apply_rotary_emb(q, k, rope_cos, rope_sin)

        attn_out = self.attn(q, k, v).view(B, L, C)
        x = x + self.out_proj(attn_out)

        # Fused RMSNorm + SwiGLU (residual added manually)
        x = x + self.ffn(x)
        return x


# ---------------------------------------------------------------------------
# Level 2 — MHA: te.MultiheadAttention (fused QKV + attention)
# ---------------------------------------------------------------------------
class TransformerBlockTE_MHA(nn.Module):
    """Uses te.MultiheadAttention with built-in layernorm.

    RoPE is handled internally by TE using rotate_half.
    """

    def __init__(self, c_main: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = c_main // num_heads

        self.attn = te.MultiheadAttention(
            c_main, num_heads,
            attention_dropout=0,
            layernorm_epsilon=1e-6,
            attn_mask_type="no_mask",
            input_layernorm=True,
            normalization="RMSNorm",
            bias=False,
            qkv_format="bshd",
        )

        # Fused RMSNorm + SwiGLU MLP
        self.ffn = te.LayerNormMLP(
            c_main, ffn_dim,
            eps=1e-6, bias=False,
            normalization="RMSNorm",
            activation="swiglu",
        )

    def forward(self, x, attn_mask, rope):
        """
        x: (N, L, C)
        attn_mask: unused (None)
        rope: (L, 1, 1, dim_half) raw RoPE embeddings for TE
        """
        x = x + self.attn(x, rotary_pos_emb=rope)
        x = x + self.ffn(x)
        return x


# ---------------------------------------------------------------------------
# Level 3 — Full: te.TransformerLayer (complete fused block)
# ---------------------------------------------------------------------------
class TransformerBlockTE_Full(nn.Module):
    """Uses te.TransformerLayer for the entire block including residual connections.

    RoPE is handled internally by TE using rotate_half.
    """

    def __init__(self, c_main: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.layer = te.TransformerLayer(
            c_main, ffn_dim, num_heads,
            layernorm_epsilon=1e-6,
            hidden_dropout=0,
            attention_dropout=0,
            self_attn_mask_type="no_mask",
            normalization="RMSNorm",
            bias=False,
            activation="swiglu",
            attn_input_format="bshd",
        )

    def forward(self, x, attn_mask, rope):
        """
        x: (N, L, C)
        attn_mask: unused (None)
        rope: (L, 1, 1, dim_half) raw RoPE embeddings for TE
        """
        return self.layer(x, rotary_pos_emb=rope)


# Block class lookup
_BLOCK_CLASSES = {
    "dpa": TransformerBlockTE_DPA,
    "mha": TransformerBlockTE_MHA,
    "full": TransformerBlockTE_Full,
}


# ---------------------------------------------------------------------------
# Model (same API as model.py)
# ---------------------------------------------------------------------------
class Model(nn.Module):
    def __init__(self, config: dict, pos_len: int, score_mode: str = "mixop",
                 te_mode: str = "mha"):
        super().__init__()
        self.config = config
        self.pos_len = pos_len
        self.c_trunk = config["hidden_size"]
        self.te_mode = te_mode
        num_bin_features = get_num_bin_input_features(config)
        num_global_features = get_num_global_input_features(config)

        num_heads = config["num_heads"]
        ffn_dim = config["ffn_dim"]
        head_dim = self.c_trunk // num_heads

        # Stem (Conv2d and global Linear stay as nn — input dims not FP8-aligned)
        self.conv_spatial = nn.Conv2d(num_bin_features, self.c_trunk, kernel_size=3, padding="same", bias=False)
        self.linear_global = nn.Linear(num_global_features, self.c_trunk, bias=False)

        # Precompute RoPE embeddings
        emb = precompute_freqs_cos_sin_2d(head_dim, pos_len)  # (L, 1, 1, dim_half)
        if te_mode == "dpa":
            # DPA uses rotate_every_two (same as model.py)
            emb_expanded = emb.repeat_interleave(2, dim=-1)  # (L, 1, 1, dim)
            self.register_buffer("rope_cos", emb_expanded.cos(), persistent=False)
            self.register_buffer("rope_sin", emb_expanded.sin(), persistent=False)
        else:
            # MHA/Full: TE handles RoPE internally (rotate_half)
            # TE expects raw emb; it does cat([emb, emb]) internally
            self.register_buffer("rope", emb, persistent=False)

        # Transformer blocks
        BlockClass = _BLOCK_CLASSES[te_mode]
        self.blocks = nn.ModuleList()
        for _ in range(config["num_layers"]):
            self.blocks.append(BlockClass(
                c_main=self.c_trunk,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
            ))

        # Final normalization (TE fused kernel)
        self.norm_final = te.RMSNorm(self.c_trunk, eps=1e-6)

        # Output heads (kept as nn.Linear — small dims not FP8-aligned, not a bottleneck)
        num_scorebeliefs = config["num_scorebeliefs"]
        self.policy_head = PolicyHead(self.c_trunk, pos_len)
        self.value_head = ValueHead(self.c_trunk, num_scorebeliefs, pos_len, score_mode=score_mode)

        # Seki dynamic weight moving average state
        self.moving_unowned_proportion_sum = 0.0
        self.moving_unowned_proportion_weight = 0.0

    def initialize(self, init_std=0.02):
        """Megatron-LM style initialization (adapted for TE parameter naming)."""
        num_blocks = len(self.blocks)
        output_std = init_std / math.sqrt(2.0 * num_blocks)

        for name, p in self.named_parameters():
            if p.dim() < 2:
                # Skip norm params (keep at default: weight=1, bias=0)
                if "rms_norm" not in name and "norm" not in name:
                    nn.init.zeros_(p)
            else:
                # Output projections get smaller init for residual scaling
                if ".out_proj." in name or "fc2_weight" in name:
                    nn.init.normal_(p, mean=0.0, std=output_std)
                else:
                    nn.init.normal_(p, mean=0.0, std=init_std)

    def forward(self, input_spatial, input_global):
        """
        input_spatial: (N, C_bin, H, W)
        input_global:  (N, C_global)
        """
        N = input_spatial.shape[0]
        H = W = self.pos_len
        L = H * W

        mask = input_spatial[:, 0:1, :, :].contiguous()
        mask_sum_hw = torch.sum(mask, dim=(2, 3), keepdim=True)

        # Stem: NCHW -> NLC
        x_spatial = self.conv_spatial(input_spatial)
        x_global = self.linear_global(input_global)
        x = x_spatial + x_global.unsqueeze(-1).unsqueeze(-1)
        x = x.view(N, self.c_trunk, L).permute(0, 2, 1)

        # Trunk
        for block in self.blocks:
            if self.te_mode == "dpa":
                x = block(x, None, self.rope_cos, self.rope_sin)
            else:
                x = block(x, None, self.rope)

        x = self.norm_final(x)

        # Output heads
        out_policy = self.policy_head(x, mask, mask_sum_hw)
        (
            out_value, out_misc, out_moremisc,
            out_ownership, out_scoring, out_futurepos, out_seki,
            out_scorebelief,
        ) = self.value_head(x, mask, mask_sum_hw, input_global[:, -1:])

        return (
            out_policy.float(), out_value.float(), out_misc.float(), out_moremisc.float(),
            out_ownership.float(), out_scoring.float(), out_futurepos.float(), out_seki.float(),
            out_scorebelief.float(),
        )

    def postprocess(self, outputs):
        (
            out_policy, out_value, out_misc, out_moremisc,
            out_ownership, out_scoring, out_futurepos, out_seki,
            out_scorebelief,
        ) = outputs

        td_score_multiplier = 20.0
        scoremean_multiplier = 20.0
        scorestdev_multiplier = 20.0
        lead_multiplier = 20.0
        variance_time_multiplier = 40.0
        shortterm_value_error_multiplier = 0.25
        shortterm_score_error_multiplier = 150.0

        policy_logits = out_policy
        value_logits = out_value
        td_value_logits = torch.stack(
            (out_misc[:, 4:7], out_misc[:, 7:10], out_moremisc[:, 2:5]), dim=1
        )
        pred_td_score = out_moremisc[:, 5:8] * td_score_multiplier
        ownership_pretanh = out_ownership
        pred_scoring = out_scoring
        futurepos_pretanh = out_futurepos
        seki_logits = out_seki
        pred_scoremean = out_misc[:, 0] * scoremean_multiplier
        pred_scorestdev = SoftPlusWithGradientFloor.apply(out_misc[:, 1], 0.05, False) * scorestdev_multiplier
        pred_lead = out_misc[:, 2] * lead_multiplier
        pred_variance_time = SoftPlusWithGradientFloor.apply(out_misc[:, 3], 0.05, False) * variance_time_multiplier

        pred_shortterm_value_error = SoftPlusWithGradientFloor.apply(out_moremisc[:, 0], 0.05, True) * shortterm_value_error_multiplier
        pred_shortterm_score_error = SoftPlusWithGradientFloor.apply(out_moremisc[:, 1], 0.05, True) * shortterm_score_error_multiplier

        scorebelief_logits = out_scorebelief

        return (
            policy_logits, value_logits, td_value_logits, pred_td_score,
            ownership_pretanh, pred_scoring, futurepos_pretanh, seki_logits,
            pred_scoremean, pred_scorestdev, pred_lead, pred_variance_time,
            pred_shortterm_value_error, pred_shortterm_score_error,
            scorebelief_logits,
        )


# ---------------------------------------------------------------------------
# Checkpoint format detection and conversion
# ---------------------------------------------------------------------------
def detect_checkpoint_format(state_dict):
    """Detect checkpoint format: 'model', 'te_dpa', 'te_mha', or 'te_full'.

    Detection logic:
        - 'te_full': has te.TransformerLayer keys like '.layer.self_attention.'
        - 'te_mha':  has te.MultiheadAttention keys like '.attn.layernorm_qkv.'
        - 'te_dpa':  has te.DotProductAttention keys like '.attn.core_attention.' + '.q_proj.'
        - 'model':   has model.py keys like '.ffn_w1.weight' or '.norm2.weight'
    """
    has_transformer_layer = False
    has_mha_qkv = False
    has_dpa_q_proj = False
    has_model_ffn = False

    for key in state_dict:
        if "_extra_state" in key:
            continue
        if ".layer.self_attention." in key:
            has_transformer_layer = True
        if ".attn.layernorm_qkv." in key:
            has_mha_qkv = True
        if ".q_proj." in key and ".attn." not in key:
            has_dpa_q_proj = True
        if ".ffn_w1.weight" in key or ".norm2.weight" in key:
            has_model_ffn = True

    if has_transformer_layer:
        return "te_full"
    if has_mha_qkv:
        return "te_mha"
    if has_dpa_q_proj and not has_model_ffn:
        return "te_dpa"
    return "model"


def convert_checkpoint_model_to_te(state_dict, te_mode="mha"):
    """Convert model.py state_dict to model_te.py format.

    Args:
        state_dict: model.py format state dict
        te_mode: target TE mode ("dpa", "mha", or "full")

    For DPA mode, only FFN keys are remapped (Q/K/V/out_proj names match model.py).
    For MHA mode, attention keys are also remapped to te.MultiheadAttention format.
    For Full mode, all keys are remapped to te.TransformerLayer format.
    """
    if te_mode == "dpa":
        return _convert_model_to_te_dpa(state_dict)
    elif te_mode == "mha":
        return _convert_model_to_te_mha(state_dict)
    elif te_mode == "full":
        return _convert_model_to_te_full(state_dict)
    else:
        raise ValueError(f"Unknown te_mode: {te_mode}")


def _convert_model_to_te_dpa(state_dict):
    """model.py -> TE DPA: only FFN keys change, attention keys stay the same."""
    new_sd = {}
    for key, value in state_dict.items():
        if ".ffn_w1.weight" in key:
            block_prefix = key.rsplit(".ffn_w1.weight", 1)[0]
            wgate_key = block_prefix + ".ffn_wgate.weight"
            new_sd[block_prefix + ".ffn.fc1_weight"] = torch.cat([value, state_dict[wgate_key]], dim=0)
        elif ".ffn_wgate.weight" in key:
            continue
        elif ".ffn_w2.weight" in key:
            new_sd[key.replace(".ffn_w2.weight", ".ffn.fc2_weight")] = value
        elif ".norm2.weight" in key:
            new_sd[key.replace(".norm2.weight", ".ffn.layer_norm_weight")] = value
        elif ".norm2.bias" in key:
            new_sd[key.replace(".norm2.bias", ".ffn.layer_norm_bias")] = value
        elif ".norm1.weight" in key:
            new_sd[key.replace(".norm1.weight", ".norm1.weight")] = value
        elif ".norm1.bias" in key:
            new_sd[key.replace(".norm1.bias", ".norm1.bias")] = value
        else:
            new_sd[key] = value
    return new_sd


def _convert_model_to_te_mha(state_dict):
    """model.py -> TE MHA: remap norm1+Q/K/V and FFN keys."""
    new_sd = {}
    for key, value in state_dict.items():
        if ".ffn_w1.weight" in key:
            block_prefix = key.rsplit(".ffn_w1.weight", 1)[0]
            wgate_key = block_prefix + ".ffn_wgate.weight"
            new_sd[block_prefix + ".ffn.fc1_weight"] = torch.cat([value, state_dict[wgate_key]], dim=0)
        elif ".ffn_wgate.weight" in key:
            continue
        elif ".ffn_w2.weight" in key:
            new_sd[key.replace(".ffn_w2.weight", ".ffn.fc2_weight")] = value
        elif ".norm2.weight" in key:
            new_sd[key.replace(".norm2.weight", ".ffn.layer_norm_weight")] = value
        elif ".norm2.bias" in key:
            new_sd[key.replace(".norm2.bias", ".ffn.layer_norm_bias")] = value
        else:
            new_sd[key] = value
    return new_sd


def _convert_model_to_te_full(state_dict):
    """model.py -> TE Full: remap all keys to te.TransformerLayer format."""
    new_sd = {}
    for key, value in state_dict.items():
        if ".ffn_w1.weight" in key:
            block_prefix = key.rsplit(".ffn_w1.weight", 1)[0]
            wgate_key = block_prefix + ".ffn_wgate.weight"
            new_sd[block_prefix + ".layer.layernorm_mlp.fc1_weight"] = torch.cat([value, state_dict[wgate_key]], dim=0)
        elif ".ffn_wgate.weight" in key:
            continue
        elif ".ffn_w2.weight" in key:
            new_sd[key.replace(".ffn_w2.weight", ".layer.layernorm_mlp.fc2_weight")] = value
        elif ".norm1.weight" in key:
            new_sd[key.replace(".norm1.weight", ".layer.self_attention.layernorm_qkv.layer_norm_weight")] = value
        elif ".norm1.bias" in key:
            new_sd[key.replace(".norm1.bias", ".layer.self_attention.layernorm_qkv.layer_norm_bias")] = value
        elif ".norm2.weight" in key:
            new_sd[key.replace(".norm2.weight", ".layer.layernorm_mlp.layer_norm_weight")] = value
        elif ".norm2.bias" in key:
            new_sd[key.replace(".norm2.bias", ".layer.layernorm_mlp.layer_norm_bias")] = value
        elif ".q_proj.weight" in key:
            new_sd[key.replace(".q_proj.weight", ".layer.self_attention.layernorm_qkv.query_weight")] = value
        elif ".k_proj.weight" in key:
            new_sd[key.replace(".k_proj.weight", ".layer.self_attention.layernorm_qkv.key_weight")] = value
        elif ".v_proj.weight" in key:
            new_sd[key.replace(".v_proj.weight", ".layer.self_attention.layernorm_qkv.value_weight")] = value
        elif ".out_proj.weight" in key:
            new_sd[key.replace(".out_proj.weight", ".layer.self_attention.proj.weight")] = value
        else:
            new_sd[key] = value
    return new_sd


def convert_checkpoint_te_to_model(state_dict):
    """Convert any TE format state_dict back to model.py format.

    Auto-detects the TE sub-format and converts accordingly.
    Filters out TE-specific _extra_state keys.
    """
    fmt = detect_checkpoint_format(state_dict)
    if fmt == "model":
        return state_dict
    elif fmt == "te_dpa":
        return _convert_te_dpa_to_model(state_dict)
    elif fmt == "te_mha":
        return _convert_te_mha_to_model(state_dict)
    elif fmt == "te_full":
        return _convert_te_full_to_model(state_dict)
    else:
        raise ValueError(f"Unknown format: {fmt}")


def _convert_te_dpa_to_model(state_dict):
    """TE DPA -> model.py: reverse FFN key mapping."""
    new_sd = {}
    for key, value in state_dict.items():
        if "_extra_state" in key:
            continue
        if ".ffn.fc1_weight" in key:
            block_prefix = key.rsplit(".ffn.fc1_weight", 1)[0]
            half = value.shape[0] // 2
            new_sd[block_prefix + ".ffn_w1.weight"] = value[:half]
            new_sd[block_prefix + ".ffn_wgate.weight"] = value[half:]
        elif ".ffn.fc2_weight" in key:
            new_sd[key.replace(".ffn.fc2_weight", ".ffn_w2.weight")] = value
        elif ".ffn.layer_norm_weight" in key:
            new_sd[key.replace(".ffn.layer_norm_weight", ".norm2.weight")] = value
        elif ".ffn.layer_norm_bias" in key:
            new_sd[key.replace(".ffn.layer_norm_bias", ".norm2.bias")] = value
        else:
            new_sd[key] = value
    return new_sd


def _convert_te_mha_to_model(state_dict):
    """TE MHA -> model.py: reverse all key mappings."""
    new_sd = {}
    for key, value in state_dict.items():
        if "_extra_state" in key:
            continue
        if ".ffn.fc1_weight" in key:
            block_prefix = key.rsplit(".ffn.fc1_weight", 1)[0]
            half = value.shape[0] // 2
            new_sd[block_prefix + ".ffn_w1.weight"] = value[:half]
            new_sd[block_prefix + ".ffn_wgate.weight"] = value[half:]
        elif ".ffn.fc2_weight" in key:
            new_sd[key.replace(".ffn.fc2_weight", ".ffn_w2.weight")] = value
        elif ".ffn.layer_norm_weight" in key:
            new_sd[key.replace(".ffn.layer_norm_weight", ".norm2.weight")] = value
        elif ".ffn.layer_norm_bias" in key:
            new_sd[key.replace(".ffn.layer_norm_bias", ".norm2.bias")] = value
        else:
            new_sd[key] = value
    return new_sd


def _convert_te_full_to_model(state_dict):
    """TE Full -> model.py: reverse te.TransformerLayer key mappings."""
    new_sd = {}
    for key, value in state_dict.items():
        if "_extra_state" in key:
            continue
        if ".layer.layernorm_mlp.fc1_weight" in key:
            block_prefix = key.rsplit(".layer.layernorm_mlp.fc1_weight", 1)[0]
            half = value.shape[0] // 2
            new_sd[block_prefix + ".ffn_w1.weight"] = value[:half]
            new_sd[block_prefix + ".ffn_wgate.weight"] = value[half:]
        elif ".layer.layernorm_mlp.fc2_weight" in key:
            new_sd[key.replace(".layer.layernorm_mlp.fc2_weight", ".ffn_w2.weight")] = value
        elif ".layer.self_attention.layernorm_qkv.layer_norm_weight" in key:
            new_sd[key.replace(".layer.self_attention.layernorm_qkv.layer_norm_weight", ".norm1.weight")] = value
        elif ".layer.self_attention.layernorm_qkv.layer_norm_bias" in key:
            new_sd[key.replace(".layer.self_attention.layernorm_qkv.layer_norm_bias", ".norm1.bias")] = value
        elif ".layer.layernorm_mlp.layer_norm_weight" in key:
            new_sd[key.replace(".layer.layernorm_mlp.layer_norm_weight", ".norm2.weight")] = value
        elif ".layer.layernorm_mlp.layer_norm_bias" in key:
            new_sd[key.replace(".layer.layernorm_mlp.layer_norm_bias", ".norm2.bias")] = value
        elif ".layer.self_attention.layernorm_qkv.query_weight" in key:
            new_sd[key.replace(".layer.self_attention.layernorm_qkv.query_weight", ".q_proj.weight")] = value
        elif ".layer.self_attention.layernorm_qkv.key_weight" in key:
            new_sd[key.replace(".layer.self_attention.layernorm_qkv.key_weight", ".k_proj.weight")] = value
        elif ".layer.self_attention.layernorm_qkv.value_weight" in key:
            new_sd[key.replace(".layer.self_attention.layernorm_qkv.value_weight", ".v_proj.weight")] = value
        elif ".layer.self_attention.proj.weight" in key:
            new_sd[key.replace(".layer.self_attention.proj.weight", ".out_proj.weight")] = value
        else:
            new_sd[key] = value
    return new_sd
