"""Transformer model architecture for KataGo nano training (TransformerEngine version).

This module provides the same Model API as model.py but uses NVIDIA TransformerEngine
(te.TransformerLayer) for fused kernels and optional FP8 training support.

All weights are directly compatible with model.py via checkpoint conversion utilities.

Usage:
    - Drop-in replacement for model.py's Model class
    - Requires: pip install transformer-engine[pytorch]
    - FP8 requires Hopper (H100/H200) or Ada (RTX 4090) GPU
    - Non-FP8 mode still benefits from TE's fused kernels

Torch compile note:
    - TE kernels currently call several PyCapsule ops that torch._dynamo cannot trace.
    - The TE trunk is isolated from torch.compile via @torch._dynamo.disable to avoid graph-break warnings.
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te

from configs import get_num_bin_input_features, get_num_global_input_features
from model import (
    EXTRA_SCORE_DISTR_RADIUS,
    GABTemplateMLP,
    RMSNormFP32,
    SoftPlusWithGradientFloor,
    apply_rotary_emb,
    build_edge_index_map,
    build_rpb_index_map,
    cross_entropy,
    precompute_freqs_cos_sin_2d,
    PolicyHead,
    ValueHead,
)


# ---------------------------------------------------------------------------
# Transformer block: te.TransformerLayer (complete fused block)
# ---------------------------------------------------------------------------
class TransformerBlockTE(nn.Module):
    """Uses te.TransformerLayer for the entire block including residual connections.

    RoPE is handled internally by TE using rotate_half.
    """

    def __init__(self, c_main: int, num_heads: int, ffn_dim: int,
                 init_method=None, output_layer_init_method=None):
        super().__init__()
        self.layer = te.TransformerLayer(
            c_main, ffn_dim, num_heads,
            layernorm_epsilon=1e-6,
            hidden_dropout=0,
            attention_dropout=0,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            self_attn_mask_type="no_mask",
            normalization="RMSNorm",
            bias=False,
            activation="swiglu",
            attn_input_format="bshd",
        )

    def forward(self, x, rope=None, core_attention_bias=None):
        """
        x: (N, L, C)
        rope: (L, 1, 1, dim_half) raw RoPE embeddings for TE (None when using RPB)
        core_attention_bias: (B, H, L, L) attention bias — RPB, GAB, or both (additive)
        """
        kwargs = {}
        if rope is not None:
            kwargs["rotary_pos_emb"] = rope
        if core_attention_bias is not None:
            kwargs["core_attention_bias_type"] = "post_scale_bias"
            kwargs["core_attention_bias"] = core_attention_bias
        return self.layer(x, **kwargs)


class TransformerBlockTEDecomposed(nn.Module):
    """Export-only TE block with manual RoPE outside TE custom fused kernels."""

    def __init__(self, c_main: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = c_main // num_heads
        self.ln_qkv = te.LayerNormLinear(
            c_main,
            3 * c_main,
            eps=1e-6,
            bias=False,
            normalization="RMSNorm",
        )
        self.attention = te.DotProductAttention(
            num_attention_heads=num_heads,
            kv_channels=self.head_dim,
            attention_dropout=0.0,
            attn_mask_type="no_mask",
            qkv_format="bshd",
        )
        self.proj = te.Linear(c_main, c_main, bias=False)
        self.ln_mlp = te.LayerNormMLP(
            c_main,
            ffn_dim,
            eps=1e-6,
            bias=False,
            normalization="RMSNorm",
            activation="swiglu",
        )

    def forward(self, x, rope_cos=None, rope_sin=None, attn_bias=None):
        batch_size, seq_len, _ = x.shape
        residual = x
        qkv = self.ln_qkv(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        if rope_cos is not None:
            q, k = apply_rotary_emb(q, k, rope_cos, rope_sin)
        if attn_bias is not None:
            x = residual + self.proj(self.attention(
                q, k, v,
                core_attention_bias_type="post_scale_bias",
                core_attention_bias=attn_bias,
            ))
        else:
            x = residual + self.proj(self.attention(q, k, v))
        return x + self.ln_mlp(x)


def _replace_nn_linear_with_te(module):
    """Recursively replace nn.Linear with te.Linear in a module."""
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, te.Linear(
                child.in_features, child.out_features, bias=(child.bias is not None),
            ))
        else:
            _replace_nn_linear_with_te(child)


def _rms_norm_fp32(x, weight, eps=1e-6):
    """Functional RMSNorm in FP32, matching RMSNormFP32 / TE RMSNorm behavior."""
    with torch.amp.autocast(x.device.type, enabled=False):
        x_f = x.float()
        return (x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype)


class GABMixer(nn.Module):
    """Per-layer GAB mixing: projects normalized tokens to per-head template mixing weights.

    Weight-compatible with model.py TransformerBlock.gab_* parameters via
    checkpoint conversion (blocks.X.gab_* <-> gab_mixers.X.gab_*).
    """
    def __init__(self, c_main, num_heads, gab_num_templates, gab_d1=16, gab_d2=16):
        super().__init__()
        self.num_heads = num_heads
        self.gab_num_templates = gab_num_templates
        self.gab_proj1 = nn.Linear(c_main, gab_d1, bias=False)
        self.gab_proj2 = nn.Linear(gab_d1, gab_d2, bias=False)
        self.gab_norm1 = nn.LayerNorm(gab_d2)
        self.gab_proj3 = nn.Linear(gab_d2, num_heads * gab_num_templates, bias=False)
        self.gab_norm2 = nn.LayerNorm(num_heads * gab_num_templates)

    def forward(self, x_norm, gab_templates):
        B = x_norm.shape[0]
        pooled = self.gab_proj1(x_norm).mean(dim=1)       # (B, d1)
        z = F.gelu(self.gab_proj2(pooled))                 # (B, d2)
        z = self.gab_norm1(z)
        z = F.gelu(self.gab_proj3(z))                      # (B, H*T)
        z = self.gab_norm2(z)
        z = z.view(B, self.num_heads, self.gab_num_templates)  # (B, H, T)
        return torch.einsum("bhd,std->bhst", z, gab_templates)


# ---------------------------------------------------------------------------
# Model (same API as model.py)
# ---------------------------------------------------------------------------
class Model(nn.Module):
    def __init__(self, config: dict, pos_len: int, score_mode: str = "mixop",
                 use_fp8: bool = False):
        super().__init__()
        self.config = config
        self.pos_len = pos_len
        self.c_trunk = config["hidden_size"]
        num_bin_features = get_num_bin_input_features(config)
        num_global_features = get_num_global_input_features(config)

        num_heads = config["num_heads"]
        ffn_dim = config["ffn_dim"]
        head_dim = self.c_trunk // num_heads

        # Stem
        self.stem = config.get("stem", "cnn3")
        self.ape = config.get("ape", "none")
        self.rpe = config.get("rpe", "rope")
        self.use_rpb = self.rpe in ("rpb", "rope+rpb")
        self.use_rope = self.rpe in ("rope", "rope+rpb")
        kernel_size = {"cnn1": 1, "cnn3": 3, "cnn5": 5}[self.stem]
        self.conv_spatial = nn.Conv2d(num_bin_features, self.c_trunk,
                                      kernel_size=kernel_size, padding="same", bias=False)
        if self.ape == "d4":
            half = (pos_len - 1) // 2
            num_edge_positions = (half + 1) * (half + 2) // 2
            self.register_buffer("edge_index_map", build_edge_index_map(pos_len), persistent=False)
            self.pos_embed = nn.Embedding(num_edge_positions, self.c_trunk)
        elif self.ape == "per_pos":
            self.pos_embed = nn.Embedding(pos_len * pos_len, self.c_trunk)
        # Non-FP8: use te.Linear for fused kernels; FP8: nn.Linear (dims not FP8-aligned)
        Linear = nn.Linear if use_fp8 else te.Linear
        self.linear_global = Linear(num_global_features, self.c_trunk, bias=False)

        # Stem normalization: per-component RMSNorm before summing
        self.stem_norm = config.get("stem_norm", False)
        if self.stem_norm:
            self.norm_spatial = RMSNormFP32(self.c_trunk, eps=1e-6)
            self.norm_global = RMSNormFP32(self.c_trunk, eps=1e-6)
            if self.ape in ("d4", "per_pos"):
                self.norm_ape = RMSNormFP32(self.c_trunk, eps=1e-6)

        # RPE: RPB parameters
        if self.use_rpb:
            num_rpb_classes = pos_len * (pos_len + 1) // 2  # 190 for 19x19
            self.register_buffer("rpb_index_map", build_rpb_index_map(pos_len), persistent=False)
            self.rpb_tables = nn.ParameterList([
                nn.Parameter(torch.zeros(num_heads, num_rpb_classes))
                for _ in range(config["num_layers"])
            ])

        # RPE: precompute RoPE embeddings (rotate_half)
        if self.use_rope:
            emb = precompute_freqs_cos_sin_2d(head_dim, pos_len)  # (L, 1, 1, dim_half)
            emb_full = torch.cat([emb, emb], dim=-1)              # (L, 1, 1, dim)
            self.register_buffer("rope", emb_full, persistent=False)

        # GAB: shared template MLP + per-layer mixers
        self.use_gab = config.get("use_gab", False)
        if self.use_gab:
            self.gab_template_mlp = GABTemplateMLP(
                num_templates=config.get("gab_num_templates", 16),
                num_fourier_features=config.get("gab_num_fourier_features", 8),
                mlp_hidden=config.get("gab_mlp_hidden", 64),
                pos_len=pos_len,
            )
            self.gab_mixers = nn.ModuleList([
                GABMixer(
                    self.c_trunk, num_heads,
                    config.get("gab_num_templates", 16),
                    config.get("gab_d1", 16),
                    config.get("gab_d2", 16),
                )
                for _ in range(config["num_layers"])
            ])

        # Transformer blocks
        BlockClass = TransformerBlockTE
        self.blocks = nn.ModuleList()
        for _ in range(config["num_layers"]):
            self.blocks.append(BlockClass(
                c_main=self.c_trunk,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
            ))

        # Final normalization (TE fused kernel)
        self.norm_final = te.RMSNorm(self.c_trunk, eps=1e-6)

        # Output heads: non-FP8 uses te.Linear for fused kernels; FP8 keeps nn.Linear (dims not FP8-aligned)
        num_scorebeliefs = config["num_scorebeliefs"]
        self.policy_head = PolicyHead(self.c_trunk, pos_len)
        self.value_head = ValueHead(self.c_trunk, num_scorebeliefs, pos_len, score_mode=score_mode)
        if not use_fp8:
            _replace_nn_linear_with_te(self.policy_head)
            _replace_nn_linear_with_te(self.value_head)

        # Seki dynamic weight moving average state
        self.moving_unowned_proportion_sum = 0.0
        self.moving_unowned_proportion_weight = 0.0

    def _run_trunk_impl(self, x):
        gab_templates = self.gab_template_mlp() if self.use_gab else None

        for i, block in enumerate(self.blocks):
            # GAB bias: use block's internal norm weight to compute x_norm
            attn_bias = None
            if self.use_gab:
                norm_weight = block.layer.self_attention.layernorm_qkv.layer_norm_weight
                x_norm = _rms_norm_fp32(x, norm_weight)
                attn_bias = self.gab_mixers[i](x_norm, gab_templates)

            if self.use_rpb:
                rpb = self.rpb_tables[i][:, self.rpb_index_map].unsqueeze(0).to(x.dtype)
                attn_bias = rpb if attn_bias is None else rpb + attn_bias

            if self.use_rope:
                x = block(x, rope=self.rope, core_attention_bias=attn_bias)
            else:
                x = block(x, core_attention_bias=attn_bias)

        return self.norm_final(x)

    @torch._dynamo.disable
    def _run_trunk_no_compile(self, x):
        return self._run_trunk_impl(x)

    def initialize(self, init_std=0.02, use_fan_in_init=True, stem_init_aligned=False):
        """Weight initialization using TE-native init_method.

        When use_fan_in_init=True: Megatron-LM style, Linear/Conv std = 1/sqrt(fan_in).
        When use_fan_in_init=False: all Linear/Conv layers use fixed init_std.
        In both modes, output layers additionally scale by 1/sqrt(2*num_blocks).
        init_std is always used for non-linear/conv parameters (APE, RPB, etc.).
        When stem_init_aligned=True: override stem weights so output std ≈ init_std.
        """
        num_blocks = len(self.blocks)

        def init_fn(tensor):
            if use_fan_in_init:
                std = 1.0 / math.sqrt(tensor[0].numel())
            else:
                std = init_std
            nn.init.normal_(tensor, mean=0.0, std=std)

        def output_init_fn(tensor):
            if use_fan_in_init:
                std = 1.0 / math.sqrt(tensor[0].numel()) / math.sqrt(2.0 * num_blocks)
            else:
                std = init_std / math.sqrt(2.0 * num_blocks)
            nn.init.normal_(tensor, mean=0.0, std=std)

        # Rebuild blocks with TE-native init methods
        BlockClass = TransformerBlockTE
        num_heads = self.config["num_heads"]
        ffn_dim = self.config["ffn_dim"]
        self.blocks = nn.ModuleList([
            BlockClass(
                c_main=self.c_trunk, num_heads=num_heads, ffn_dim=ffn_dim,
                init_method=init_fn, output_layer_init_method=output_init_fn,
            )
            for _ in range(num_blocks)
        ])

        # Stem
        init_fn(self.conv_spatial.weight)
        init_fn(self.linear_global.weight)

        # Heads (nn.Linear from model.py, all bias=False)
        for m in (self.policy_head, self.value_head):
            for p in m.parameters():
                if p.dim() >= 2:
                    init_fn(p)

        # APE embedding (fixed init_std, not fan_in)
        if self.ape in ("d4", "per_pos"):
            nn.init.normal_(self.pos_embed.weight, mean=0.0, std=init_std)
        # RPB
        if self.use_rpb:
            for table in self.rpb_tables:
                nn.init.zeros_(table)
        # GAB
        if self.use_gab:
            init_fn(self.gab_template_mlp.linear1.weight)
            nn.init.zeros_(self.gab_template_mlp.linear1.bias)
            init_fn(self.gab_template_mlp.linear2.weight)
            nn.init.zeros_(self.gab_template_mlp.linear2.bias)
            num_ff = self.config.get("gab_num_fourier_features", 8)
            with torch.no_grad():
                self.gab_template_mlp.gab_freqs.copy_(
                    torch.exp(torch.linspace(math.log(1.0), math.log(1.0 / 50.0), num_ff))
                )
            for mixer in self.gab_mixers:
                for name, p in mixer.named_parameters():
                    if p.dim() >= 2:
                        init_fn(p)
                    elif "norm" not in name:
                        nn.init.zeros_(p)

        if stem_init_aligned:
            # Override stem weights so output std ≈ init_std (matching APE)
            fan_in_spatial = self.conv_spatial.weight[0].numel()
            nn.init.normal_(self.conv_spatial.weight, mean=0.0,
                            std=init_std / math.sqrt(fan_in_spatial))
            fan_in_global = self.linear_global.weight.shape[1]
            nn.init.normal_(self.linear_global.weight, mean=0.0,
                            std=init_std / math.sqrt(fan_in_global))

    @torch.no_grad()
    def compute_stem_norms(self, input_spatial, input_global):
        """Compute per-component RMS of stem signals for diagnostics."""
        x_spatial = self.conv_spatial(input_spatial)
        x_global = self.linear_global(input_global)
        norms = {
            "spatial_rms": x_spatial.float().pow(2).mean().sqrt().item(),
            "global_rms": x_global.float().pow(2).mean().sqrt().item(),
        }
        if self.ape == "d4":
            ape = self.pos_embed(self.edge_index_map)
            norms["ape_rms"] = ape.float().pow(2).mean().sqrt().item()
        elif self.ape == "per_pos":
            norms["ape_rms"] = self.pos_embed.weight.float().pow(2).mean().sqrt().item()
        return norms

    def forward_trunk_for_onnx_export(self, input_spatial, input_global):
        x = self._forward_stem_impl(input_spatial, input_global)
        return self._forward_blocks_impl(x).float()

    def _forward_stem_impl(self, input_spatial, input_global):
        N = input_spatial.shape[0]
        L = self.pos_len * self.pos_len
        x_global = self.linear_global(input_global)
        x_spatial = self.conv_spatial(input_spatial)
        if self.stem_norm:
            x_spatial_nlc = x_spatial.view(N, self.c_trunk, L).permute(0, 2, 1)
            x_global_nlc = x_global.unsqueeze(1).expand(-1, L, -1)
            x = self.norm_spatial(x_spatial_nlc) + self.norm_global(x_global_nlc)
        else:
            x = x_spatial + x_global.unsqueeze(-1).unsqueeze(-1)
            x = x.view(N, self.c_trunk, L).permute(0, 2, 1)
        if self.ape == "d4":
            ape = self.pos_embed(self.edge_index_map).to(dtype=x.dtype)
            x = x + (self.norm_ape(ape) if self.stem_norm else ape)
        elif self.ape == "per_pos":
            ape = self.pos_embed.weight.to(dtype=x.dtype)
            x = x + (self.norm_ape(ape) if self.stem_norm else ape)
        return x

    def forward_stem_for_onnx_export(self, input_spatial, input_global):
        return self._forward_stem_impl(input_spatial, input_global).float()

    def _forward_blocks_impl(self, x):
        return self._run_trunk_impl(x)

    def forward_blocks_for_onnx_export(self, input_stem):
        return self._forward_blocks_impl(input_stem).float()

    def _forward_impl(self, input_spatial, input_global, for_onnx_export: bool):
        """
        input_spatial: (N, C_bin, H, W)
        input_global:  (N, C_global)
        """
        N = input_spatial.shape[0]
        H = W = self.pos_len
        L = H * W

        # Stem: NCHW -> NLC
        x = self._forward_stem_impl(input_spatial, input_global)

        # ONNX export needs the full TE graph visible to torch.export / torch.onnx.
        if for_onnx_export:
            x = self._forward_blocks_impl(x)
        else:
            # Trunk is isolated from torch.compile for TE compatibility during training/inference.
            x = self._run_trunk_no_compile(x)

        # Output heads
        out_policy = self.policy_head(x)
        (
            out_value, out_misc, out_moremisc,
            out_ownership, out_scoring, out_futurepos, out_seki,
            out_scorebelief,
        ) = self.value_head(x, input_global[:, -1:])

        return (
            out_policy.float(), out_value.float(), out_misc.float(), out_moremisc.float(),
            out_ownership.float(), out_scoring.float(), out_futurepos.float(), out_seki.float(),
            out_scorebelief.float(),
        )

    def forward(self, input_spatial, input_global):
        return self._forward_impl(input_spatial, input_global, for_onnx_export=False)

    def forward_for_onnx_export(self, input_spatial, input_global):
        return self._forward_impl(input_spatial, input_global, for_onnx_export=True)

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


class ModelDecomposedExport(nn.Module):
    """Export-only TE model that keeps TE modules but applies RoPE via plain PyTorch ops."""

    def __init__(self, config: dict, pos_len: int, score_mode: str = "mixop",
                 use_fp8: bool = False):
        super().__init__()
        self.config = config
        self.pos_len = pos_len
        self.c_trunk = config["hidden_size"]
        num_bin_features = get_num_bin_input_features(config)
        num_global_features = get_num_global_input_features(config)

        num_heads = config["num_heads"]
        ffn_dim = config["ffn_dim"]
        head_dim = self.c_trunk // num_heads

        self.stem = config.get("stem", "cnn3")
        self.ape = config.get("ape", "none")
        self.rpe = config.get("rpe", "rope")
        self.use_rpb = self.rpe in ("rpb", "rope+rpb")
        self.use_rope = self.rpe in ("rope", "rope+rpb")
        kernel_size = {"cnn1": 1, "cnn3": 3, "cnn5": 5}[self.stem]
        self.conv_spatial = nn.Conv2d(num_bin_features, self.c_trunk,
                                      kernel_size=kernel_size, padding="same", bias=False)
        if self.ape == "d4":
            half = (pos_len - 1) // 2
            num_edge_positions = (half + 1) * (half + 2) // 2
            self.register_buffer("edge_index_map", build_edge_index_map(pos_len), persistent=False)
            self.pos_embed = nn.Embedding(num_edge_positions, self.c_trunk)
        elif self.ape == "per_pos":
            self.pos_embed = nn.Embedding(pos_len * pos_len, self.c_trunk)
        Linear = nn.Linear if use_fp8 else te.Linear
        self.linear_global = Linear(num_global_features, self.c_trunk, bias=False)

        # Stem normalization: per-component RMSNorm before summing
        self.stem_norm = config.get("stem_norm", False)
        if self.stem_norm:
            self.norm_spatial = RMSNormFP32(self.c_trunk, eps=1e-6)
            self.norm_global = RMSNormFP32(self.c_trunk, eps=1e-6)
            if self.ape in ("d4", "per_pos"):
                self.norm_ape = RMSNormFP32(self.c_trunk, eps=1e-6)

        if self.use_rpb:
            num_rpb_classes = pos_len * (pos_len + 1) // 2
            self.register_buffer("rpb_index_map", build_rpb_index_map(pos_len), persistent=False)
            self.rpb_tables = nn.ParameterList([
                nn.Parameter(torch.zeros(num_heads, num_rpb_classes))
                for _ in range(config["num_layers"])
            ])

        if self.use_rope:
            emb = precompute_freqs_cos_sin_2d(head_dim, pos_len)
            emb_expanded = torch.cat([emb, emb], dim=-1)
            self.register_buffer("rope_cos", emb_expanded.cos(), persistent=False)
            self.register_buffer("rope_sin", emb_expanded.sin(), persistent=False)

        # GAB: shared template MLP + per-layer mixers
        self.use_gab = config.get("use_gab", False)
        if self.use_gab:
            self.gab_template_mlp = GABTemplateMLP(
                num_templates=config.get("gab_num_templates", 16),
                num_fourier_features=config.get("gab_num_fourier_features", 8),
                mlp_hidden=config.get("gab_mlp_hidden", 64),
                pos_len=pos_len,
            )
            self.gab_mixers = nn.ModuleList([
                GABMixer(
                    self.c_trunk, num_heads,
                    config.get("gab_num_templates", 16),
                    config.get("gab_d1", 16),
                    config.get("gab_d2", 16),
                )
                for _ in range(config["num_layers"])
            ])

        self.blocks = nn.ModuleList([
            TransformerBlockTEDecomposed(
                c_main=self.c_trunk,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
            )
            for _ in range(config["num_layers"])
        ])
        self.norm_final = te.RMSNorm(self.c_trunk, eps=1e-6)

        num_scorebeliefs = config["num_scorebeliefs"]
        self.policy_head = PolicyHead(self.c_trunk, pos_len)
        self.value_head = ValueHead(self.c_trunk, num_scorebeliefs, pos_len, score_mode=score_mode)
        if not use_fp8:
            _replace_nn_linear_with_te(self.policy_head)
            _replace_nn_linear_with_te(self.value_head)

        self.moving_unowned_proportion_sum = 0.0
        self.moving_unowned_proportion_weight = 0.0

    def _run_trunk_impl(self, x):
        gab_templates = self.gab_template_mlp() if self.use_gab else None

        for i, block in enumerate(self.blocks):
            # GAB bias: use block's internal norm weight to compute x_norm
            attn_bias = None
            if self.use_gab:
                norm_weight = block.ln_qkv.layer_norm_weight
                x_norm = _rms_norm_fp32(x, norm_weight)
                attn_bias = self.gab_mixers[i](x_norm, gab_templates)

            if self.use_rpb:
                rpb = self.rpb_tables[i][:, self.rpb_index_map].unsqueeze(0).to(x.dtype)
                attn_bias = rpb if attn_bias is None else rpb + attn_bias

            if self.use_rope:
                x = block(x, self.rope_cos, self.rope_sin, attn_bias=attn_bias)
            else:
                x = block(x, attn_bias=attn_bias)

        return self.norm_final(x)

    def _forward_stem_impl(self, input_spatial, input_global):
        batch_size = input_spatial.shape[0]
        seq_len = self.pos_len * self.pos_len
        x_global = self.linear_global(input_global)
        x_spatial = self.conv_spatial(input_spatial)
        if self.stem_norm:
            x_spatial_nlc = x_spatial.view(batch_size, self.c_trunk, seq_len).permute(0, 2, 1)
            x_global_nlc = x_global.unsqueeze(1).expand(-1, seq_len, -1)
            x = self.norm_spatial(x_spatial_nlc) + self.norm_global(x_global_nlc)
        else:
            x = x_spatial + x_global.unsqueeze(-1).unsqueeze(-1)
            x = x.view(batch_size, self.c_trunk, seq_len).permute(0, 2, 1)
        if self.ape == "d4":
            ape = self.pos_embed(self.edge_index_map).to(dtype=x.dtype)
            x = x + (self.norm_ape(ape) if self.stem_norm else ape)
        elif self.ape == "per_pos":
            ape = self.pos_embed.weight.to(dtype=x.dtype)
            x = x + (self.norm_ape(ape) if self.stem_norm else ape)
        return x

    def forward_stem_for_onnx_export(self, input_spatial, input_global):
        return self._forward_stem_impl(input_spatial, input_global).float()

    def _forward_blocks_impl(self, x):
        return self._run_trunk_impl(x)

    def forward_blocks_for_onnx_export(self, input_stem):
        return self._forward_blocks_impl(input_stem).float()

    def forward_trunk_for_onnx_export(self, input_spatial, input_global):
        x = self._forward_stem_impl(input_spatial, input_global)
        return self._forward_blocks_impl(x).float()

    def forward(self, input_spatial, input_global):
        x = self._forward_stem_impl(input_spatial, input_global)
        x = self._forward_blocks_impl(x)

        out_policy = self.policy_head(x)
        (
            out_value, out_misc, out_moremisc,
            out_ownership, out_scoring, out_futurepos, out_seki,
            out_scorebelief,
        ) = self.value_head(x, input_global[:, -1:])

        return (
            out_policy.float(), out_value.float(), out_misc.float(), out_moremisc.float(),
            out_ownership.float(), out_scoring.float(), out_futurepos.float(), out_seki.float(),
            out_scorebelief.float(),
        )


# ---------------------------------------------------------------------------
# Checkpoint format detection and conversion
# ---------------------------------------------------------------------------
def detect_checkpoint_format(state_dict):
    """Detect checkpoint format: 'pt' (model.py) or 'te' (TransformerEngine)."""
    for key in state_dict:
        if ".layer.self_attention." in key:
            return "te"
    return "pt"


def convert_checkpoint_model_to_te(state_dict):
    """Convert model.py state_dict to TE (te.TransformerLayer) format."""
    new_sd = {}
    for key, value in state_dict.items():
        # GAB per-layer keys: blocks.X.gab_* → gab_mixers.X.gab_*
        if key.startswith("blocks.") and ".gab_" in key:
            new_sd[key.replace("blocks.", "gab_mixers.", 1)] = value
            continue
        if ".ffn_w1.weight" in key:
            block_prefix = key.rsplit(".ffn_w1.weight", 1)[0]
            wgate_key = block_prefix + ".ffn_wgate.weight"
            new_sd[block_prefix + ".layer.layernorm_mlp.fc1_weight"] = torch.cat([value, state_dict[wgate_key]], dim=0)
        elif ".ffn_wgate.weight" in key:
            continue
        elif ".ffn_w2.weight" in key:
            new_sd[key.replace(".ffn_w2.weight", ".layer.layernorm_mlp.fc2_weight")] = value
        elif ".norm1.norm.weight" in key:
            new_sd[key.replace(".norm1.norm.weight", ".layer.self_attention.layernorm_qkv.layer_norm_weight")] = value
        elif ".norm2.norm.weight" in key:
            new_sd[key.replace(".norm2.norm.weight", ".layer.layernorm_mlp.layer_norm_weight")] = value
        elif key == "norm_final.norm.weight":
            new_sd["norm_final.weight"] = value
        elif ".norm_final.norm.weight" in key:
            new_sd[key.replace(".norm_final.norm.weight", ".norm_final.weight")] = value
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


def convert_checkpoint_model_to_te_decomposed(state_dict):
    """Convert model.py state_dict to the export-only decomposed TE format."""
    new_sd = {}
    for key, value in state_dict.items():
        # GAB per-layer keys: blocks.X.gab_* → gab_mixers.X.gab_*
        if key.startswith("blocks.") and ".gab_" in key:
            new_sd[key.replace("blocks.", "gab_mixers.", 1)] = value
            continue
        if ".ffn_w1.weight" in key:
            block_prefix = key.rsplit(".ffn_w1.weight", 1)[0]
            wgate_key = block_prefix + ".ffn_wgate.weight"
            new_sd[block_prefix + ".ln_mlp.fc1_weight"] = torch.cat([value, state_dict[wgate_key]], dim=0)
        elif ".ffn_wgate.weight" in key:
            continue
        elif ".ffn_w2.weight" in key:
            new_sd[key.replace(".ffn_w2.weight", ".ln_mlp.fc2_weight")] = value
        elif ".norm1.norm.weight" in key:
            new_sd[key.replace(".norm1.norm.weight", ".ln_qkv.layer_norm_weight")] = value
        elif ".norm2.norm.weight" in key:
            new_sd[key.replace(".norm2.norm.weight", ".ln_mlp.layer_norm_weight")] = value
        elif key == "norm_final.norm.weight":
            new_sd["norm_final.weight"] = value
        elif ".norm_final.norm.weight" in key:
            new_sd[key.replace(".norm_final.norm.weight", ".norm_final.weight")] = value
        elif ".norm1.weight" in key:
            new_sd[key.replace(".norm1.weight", ".ln_qkv.layer_norm_weight")] = value
        elif ".norm2.weight" in key:
            new_sd[key.replace(".norm2.weight", ".ln_mlp.layer_norm_weight")] = value
        elif ".q_proj.weight" in key:
            block_prefix = key.rsplit(".q_proj.weight", 1)[0]
            qkv_weight = torch.cat([
                value,
                state_dict[block_prefix + ".k_proj.weight"],
                state_dict[block_prefix + ".v_proj.weight"],
            ], dim=0)
            new_sd[block_prefix + ".ln_qkv.weight"] = qkv_weight
        elif ".k_proj.weight" in key or ".v_proj.weight" in key:
            continue
        elif ".out_proj.weight" in key:
            new_sd[key.replace(".out_proj.weight", ".proj.weight")] = value
        else:
            new_sd[key] = value
    return new_sd


def convert_checkpoint_te_to_model(state_dict):
    """Convert TE (te.TransformerLayer) state_dict back to model.py format.

    Filters out TE-specific _extra_state keys.
    """
    new_sd = {}
    for key, value in state_dict.items():
        if "_extra_state" in key:
            continue
        # GAB per-layer keys: gab_mixers.X.gab_* → blocks.X.gab_*
        if key.startswith("gab_mixers."):
            new_sd[key.replace("gab_mixers.", "blocks.", 1)] = value
            continue
        if ".layer.layernorm_mlp.fc1_weight" in key:
            block_prefix = key.rsplit(".layer.layernorm_mlp.fc1_weight", 1)[0]
            half = value.shape[0] // 2
            new_sd[block_prefix + ".ffn_w1.weight"] = value[:half]
            new_sd[block_prefix + ".ffn_wgate.weight"] = value[half:]
        elif ".layer.layernorm_mlp.fc2_weight" in key:
            new_sd[key.replace(".layer.layernorm_mlp.fc2_weight", ".ffn_w2.weight")] = value
        elif ".layer.self_attention.layernorm_qkv.layer_norm_weight" in key:
            new_sd[key.replace(".layer.self_attention.layernorm_qkv.layer_norm_weight", ".norm1.norm.weight")] = value
        elif ".layer.self_attention.layernorm_qkv.layer_norm_bias" in key:
            # model.py uses nn.RMSNorm inside RMSNormFP32 and therefore has no bias parameter.
            continue
        elif ".layer.layernorm_mlp.layer_norm_weight" in key:
            new_sd[key.replace(".layer.layernorm_mlp.layer_norm_weight", ".norm2.norm.weight")] = value
        elif ".layer.layernorm_mlp.layer_norm_bias" in key:
            continue
        elif ".layer.self_attention.layernorm_qkv.query_weight" in key:
            new_sd[key.replace(".layer.self_attention.layernorm_qkv.query_weight", ".q_proj.weight")] = value
        elif ".layer.self_attention.layernorm_qkv.key_weight" in key:
            new_sd[key.replace(".layer.self_attention.layernorm_qkv.key_weight", ".k_proj.weight")] = value
        elif ".layer.self_attention.layernorm_qkv.value_weight" in key:
            new_sd[key.replace(".layer.self_attention.layernorm_qkv.value_weight", ".v_proj.weight")] = value
        elif ".layer.self_attention.proj.weight" in key:
            new_sd[key.replace(".layer.self_attention.proj.weight", ".out_proj.weight")] = value
        elif key == "norm_final.weight":
            new_sd["norm_final.norm.weight"] = value
        elif key.endswith(".norm_final.weight"):
            new_sd[key.replace(".norm_final.weight", ".norm_final.norm.weight")] = value
        else:
            new_sd[key] = value
    return new_sd
