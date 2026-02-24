#!/usr/bin/python3
"""
极简 Transformer 训练脚本 for KataGo.
- 只支持纯 Transformer 模型 (RoPE + GQA + SwiGLU)
- 早期融合 H,W 为序列维度，trunk 全程 NLC
- 1x1 Conv 全部替换为 Linear
- 使用 torch.amp + bf16
- AdamW 优化器
"""
import sys
import os
import argparse
import math
import time
import logging
import json
import glob
import numpy as np
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import modelconfigs
import data_processing_pytorch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXTRA_SCORE_DISTR_RADIUS = 60


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
class SoftPlusWithGradientFloor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, grad_floor, square):
        ctx.save_for_backward(x)
        ctx.grad_floor = grad_floor
        if square:
            return torch.square(F.softplus(0.5 * x))
        else:
            return F.softplus(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_x = grad_output * (ctx.grad_floor + (1.0 - ctx.grad_floor) / (1.0 + torch.exp(-x)))
        return grad_x, None, None


def cross_entropy(pred_logits, target_probs, dim):
    return -torch.sum(target_probs * F.log_softmax(pred_logits, dim=dim), dim=dim)


def huber_loss(x, y, delta):
    abs_diff = torch.abs(x - y)
    return torch.where(
        abs_diff > delta,
        0.5 * delta * delta + delta * (abs_diff - delta),
        0.5 * abs_diff * abs_diff,
    )


# ---------------------------------------------------------------------------
# 2D RoPE
# ---------------------------------------------------------------------------
def precompute_freqs_cos_sin_2d(dim: int, pos_len: int, theta: float = 100.0):
    assert dim % 4 == 0
    dim_half = dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, dim_half, 2).float() / dim_half))
    t = torch.arange(pos_len, dtype=torch.float32)
    grid_h, grid_w = torch.meshgrid(t, t, indexing="ij")
    emb_h = grid_h.unsqueeze(-1) * freqs
    emb_w = grid_w.unsqueeze(-1) * freqs
    emb = torch.cat([emb_h, emb_w], dim=-1).flatten(0, 1)
    emb = emb.repeat_interleave(2, dim=-1)
    return emb.cos(), emb.sin()


def apply_rotary_emb(xq, xk, cos, sin):
    def rotate_every_two(x):
        x = x.reshape(*x.shape[:-1], -1, 2)
        x0, x1 = x.unbind(dim=-1)
        return torch.stack([-x1, x0], dim=-1).flatten(-2)

    cos = cos.view(1, xq.shape[1], 1, xq.shape[-1])
    sin = sin.view(1, xq.shape[1], 1, xq.shape[-1])
    xq_out = xq * cos + rotate_every_two(xq) * sin
    xk_out = xk * cos + rotate_every_two(xk) * sin
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RMSNormFP32(nn.Module):
    """RMSNorm that always runs in float32 (disables autocast)."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.RMSNorm(dim, eps=eps)

    def forward(self, x):
        with torch.amp.autocast(x.device.type, enabled=False):
            return self.norm(x.float()).to(x.dtype)


# ---------------------------------------------------------------------------
# Transformer Block (NLC 格式, RoPE + GQA + SwiGLU + RMSNorm)
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, c_main: int, num_heads: int, num_kv_heads: int,
                 ffn_dim: int, pos_len: int, rope_theta: float = 100.0):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.n_rep = num_heads // num_kv_heads
        self.head_dim = c_main // num_heads
        self.ffn_dim = ffn_dim

        self.q_proj = nn.Linear(c_main, c_main, bias=False)
        self.k_proj = nn.Linear(c_main, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(c_main, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(c_main, c_main, bias=False)

        cos_cached, sin_cached = precompute_freqs_cos_sin_2d(self.head_dim, pos_len, rope_theta)
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

        # SwiGLU FFN
        self.ffn_w1 = nn.Linear(c_main, ffn_dim, bias=False)
        self.ffn_wgate = nn.Linear(c_main, ffn_dim, bias=False)
        self.ffn_w2 = nn.Linear(ffn_dim, c_main, bias=False)

        self.norm1 = RMSNormFP32(c_main, eps=1e-6)
        self.norm2 = RMSNormFP32(c_main, eps=1e-6)

    def forward(self, x, attn_mask=None):
        """
        x: (N, L, C)   attn_mask: (N, 1, 1, L) additive mask, 0 or -inf
        """
        B, L, C = x.shape
        xn = self.norm1(x)

        q = self.q_proj(xn).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(xn).view(B, L, self.num_kv_heads, self.head_dim)
        v = self.v_proj(xn).view(B, L, self.num_kv_heads, self.head_dim)

        q, k = apply_rotary_emb(q, k, self.cos_cached, self.sin_cached)

        q = q.permute(0, 2, 1, 3)  # (B, H, L, D)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        if self.n_rep > 1:
            k = k.unsqueeze(2).expand(B, self.num_kv_heads, self.n_rep, L, self.head_dim)
            k = k.reshape(B, self.num_heads, L, self.head_dim)
            v = v.unsqueeze(2).expand(B, self.num_kv_heads, self.n_rep, L, self.head_dim)
            v = v.reshape(B, self.num_heads, L, self.head_dim)

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, L, C)
        x = x + self.out_proj(attn_out)

        # SwiGLU FFN
        xn = self.norm2(x)
        x = x + self.ffn_w2(F.silu(self.ffn_w1(xn)) * self.ffn_wgate(xn))
        return x


# ---------------------------------------------------------------------------
# KataGPool (需要 NCHW 输入, 内部使用)
# ---------------------------------------------------------------------------
class KataGPool(nn.Module):
    def forward(self, x, mask, mask_sum_hw):
        """x: (N,C,H,W), mask: (N,1,H,W), mask_sum_hw: (N,1,1,1) -> (N,3C)"""
        x_fp32 = x.to(torch.float32)
        if mask is not None:
            sqrt_offset = torch.sqrt(mask_sum_hw) - 14.0
            layer_mean = torch.sum(x_fp32, dim=(2, 3), keepdim=True) / mask_sum_hw
            layer_max, _ = torch.max((x + (mask - 1.0)).view(x.shape[0], x.shape[1], -1).to(torch.float32), dim=2)
        else:
            sqrt_offset = (x.shape[2] * x.shape[3]) ** 0.5 - 14.0
            layer_mean = torch.mean(x_fp32, dim=(2, 3), keepdim=True)
            layer_max, _ = torch.max(x.view(x.shape[0], x.shape[1], -1).to(torch.float32), dim=2)
        layer_max = layer_max.view(x.shape[0], x.shape[1], 1, 1)
        out = torch.cat([layer_mean, layer_mean * (sqrt_offset / 10.0), layer_max], dim=1)
        return out.squeeze(-1).squeeze(-1)  # (N, 3C)


class KataValueHeadGPool(nn.Module):
    def forward(self, x, mask, mask_sum_hw):
        """x: (N,C,H,W), mask: (N,1,H,W), mask_sum_hw: (N,1,1,1) -> (N,3C)"""
        x_fp32 = x.to(torch.float32)
        if mask is not None:
            sqrt_offset = torch.sqrt(mask_sum_hw) - 14.0
            layer_mean = torch.sum(x_fp32, dim=(2, 3), keepdim=True) / mask_sum_hw
        else:
            sqrt_offset = (x.shape[2] * x.shape[3]) ** 0.5 - 14.0
            layer_mean = torch.mean(x_fp32, dim=(2, 3), keepdim=True)
        out = torch.cat([
            layer_mean,
            layer_mean * (sqrt_offset / 10.0),
            layer_mean * (sqrt_offset * sqrt_offset / 100.0 - 0.1),
        ], dim=1)
        return out.squeeze(-1).squeeze(-1)  # (N, 3C)


# ---------------------------------------------------------------------------
# PolicyHead (NLC 输入, 1x1 conv -> Linear)
# ---------------------------------------------------------------------------
class PolicyHead(nn.Module):
    def __init__(self, c_in, c_p1, c_g1, pos_len):
        super().__init__()
        self.pos_len = pos_len
        self.c_p1 = c_p1
        self.c_g1 = c_g1
        self.num_policy_outputs = 6

        # 合并 linear_p (c_in→c_p1) + linear_g (c_in→c_g1)
        self.linear_pg = nn.Linear(c_in, c_p1 + c_g1, bias=False)
        self.bias_g = nn.Parameter(torch.zeros(c_g1))           # 对齐 model_pytorch 的 biasg
        self.gpool = KataGPool()

        # gate (no bias) 和 pass (has bias) bias 不一致，不合并
        self.linear_gate = nn.Linear(3 * c_g1, c_p1, bias=False)
        self.linear_pass = nn.Linear(3 * c_g1, c_p1, bias=True)
        self.linear_pass2 = nn.Linear(c_p1, self.num_policy_outputs, bias=False)

        self.bias = nn.Parameter(torch.zeros(c_p1))
        self.linear_out = nn.Linear(c_p1, self.num_policy_outputs, bias=False)  # 替代 conv2p

    def forward(self, x_nlc, mask, mask_sum_hw):
        """x_nlc: (N,L,C), mask: (N,1,H,W) or None, mask_sum_hw: (N,1,1,1) or None"""
        N, L, _ = x_nlc.shape
        H = W = self.pos_len

        outp, outg = self.linear_pg(x_nlc).split([self.c_p1, self.c_g1], dim=-1)

        # bias + mask + activation (对齐 model_pytorch 的 biasg + actg)
        outg = outg + self.bias_g
        if mask is not None:
            outg = outg * mask.view(N, L, 1)
        outg = F.silu(outg)

        # Global pool 需要 NCHW
        outg_nchw = outg.permute(0, 2, 1).view(N, -1, H, W)
        outg_pooled = self.gpool(outg_nchw, mask, mask_sum_hw)  # (N, 3*c_g1)

        # pass logits + gate
        outpass = F.silu(self.linear_pass(outg_pooled))
        outpass = self.linear_pass2(outpass)
        gate = self.linear_gate(outg_pooled)  # (N, c_p1)
        outp = outp + gate.unsqueeze(1)
        outp = outp + self.bias
        if mask is not None:
            outp = outp * mask.view(N, L, 1)
        outp = F.silu(outp)
        outp = self.linear_out(outp)  # (N, L, num_policy)

        # (N, num_policy, L) 然后 mask 掉棋盘外
        outpolicy = outp.permute(0, 2, 1)
        if mask is not None:
            outpolicy = outpolicy - (1.0 - mask.view(N, 1, L)) * 5000.0
        # concat with pass: (N, num_policy, L+1)
        return torch.cat([outpolicy, outpass.unsqueeze(-1)], dim=2)


class SimplePolicyHeadPool(nn.Module):
    """Global mean pool -> single linear -> all policy logits."""
    def __init__(self, c_in, pos_len):
        super().__init__()
        self.pos_len = pos_len
        self.num_policy_outputs = 6
        self.linear = nn.Linear(c_in, (pos_len * pos_len + 1) * self.num_policy_outputs)

    def forward(self, x_nlc, mask, mask_sum_hw):
        N, L, _ = x_nlc.shape
        if mask is not None:
            pooled = (x_nlc * mask.view(N, L, 1)).sum(dim=1) / mask_sum_hw.view(N, 1)
        else:
            pooled = x_nlc.mean(dim=1)
        out = self.linear(pooled).view(N, self.num_policy_outputs, L + 1)
        if mask is not None:
            out[:, :, :L] = out[:, :, :L] - (1.0 - mask.view(N, 1, L)) * 5000.0
        return out


class SimplePolicyHeadLinear(nn.Module):
    """Per-position linear for board + global pool linear for pass."""
    def __init__(self, c_in, pos_len):
        super().__init__()
        self.pos_len = pos_len
        self.num_policy_outputs = 6
        self.linear_board = nn.Linear(c_in, self.num_policy_outputs)
        self.linear_pass = nn.Linear(c_in, self.num_policy_outputs)

    def forward(self, x_nlc, mask, mask_sum_hw):
        N, L, _ = x_nlc.shape
        board = self.linear_board(x_nlc).permute(0, 2, 1)  # (N, 6, L)
        if mask is not None:
            board = board - (1.0 - mask.view(N, 1, L)) * 5000.0
            pooled = (x_nlc * mask.view(N, L, 1)).sum(dim=1) / mask_sum_hw.view(N, 1)
        else:
            pooled = x_nlc.mean(dim=1)
        pass_logits = self.linear_pass(pooled)  # (N, 6)
        return torch.cat([board, pass_logits.unsqueeze(-1)], dim=2)  # (N, 6, L+1)


# ---------------------------------------------------------------------------
# ValueHead (NLC 输入, 1x1 conv -> Linear)
# ---------------------------------------------------------------------------
class ValueHead(nn.Module):
    def __init__(self, c_in, c_v1, c_v2, c_sv2, num_scorebeliefs, pos_len):
        super().__init__()
        self.pos_len = pos_len
        self.scorebelief_mid = pos_len * pos_len + EXTRA_SCORE_DISTR_RADIUS
        self.scorebelief_len = self.scorebelief_mid * 2
        self.num_scorebeliefs = num_scorebeliefs
        self.c_sv2 = c_sv2
        self.c_v1 = c_v1

        self.td_score_multiplier = 20.0
        self.scoremean_multiplier = 20.0
        self.scorestdev_multiplier = 20.0
        self.lead_multiplier = 20.0
        self.variance_time_multiplier = 40.0
        self.shortterm_value_error_multiplier = 0.25
        self.shortterm_score_error_multiplier = 150.0

        # 空间 -> 通道 (替代 conv1)
        self.linear_v1 = nn.Linear(c_in, c_v1, bias=False)
        self.bias_v1 = nn.Parameter(torch.zeros(c_v1))
        self.gpool = KataValueHeadGPool()

        self.linear2 = nn.Linear(3 * c_v1, c_v2, bias=True)

        # 合并 linear_value (c_v2→3) + linear_misc (c_v2→10) + linear_moremisc (c_v2→8)
        self.linear_vmm = nn.Linear(c_v2, 3 + 10 + 8, bias=True)

        # 合并 linear_ownership (c_v1→1) + linear_scoring (c_v1→1)
        self.linear_os = nn.Linear(c_v1, 2, bias=False)
        # 合并 linear_futurepos (c_in→2) + linear_seki (c_in→4)
        self.linear_fs = nn.Linear(c_in, 2 + 4, bias=False)

        # Score belief
        # 合并 linear_s2 (3*c_v1→c_sv2) + linear_smix (3*c_v1→num_scorebeliefs)
        self.linear_s2mix = nn.Linear(3 * c_v1, c_sv2 + num_scorebeliefs, bias=True)
        self.linear_s2off = nn.Linear(1, c_sv2, bias=False)
        self.linear_s2par = nn.Linear(1, c_sv2, bias=False)
        self.linear_s3 = nn.Linear(c_sv2, num_scorebeliefs, bias=True)

        self.register_buffer("score_belief_offset_vector", torch.tensor(
            [(float(i - self.scorebelief_mid) + 0.5) for i in range(self.scorebelief_len)],
            dtype=torch.float32,
        ), persistent=False)
        self.register_buffer("score_belief_offset_bias_vector", torch.tensor(
            [0.05 * (float(i - self.scorebelief_mid) + 0.5) for i in range(self.scorebelief_len)],
            dtype=torch.float32,
        ), persistent=False)
        self.register_buffer("score_belief_parity_vector", torch.tensor(
            [0.5 - float((i - self.scorebelief_mid) % 2) for i in range(self.scorebelief_len)],
            dtype=torch.float32,
        ), persistent=False)

    def forward(self, x_nlc, mask, mask_sum_hw, input_global):
        """
        x_nlc: (N,L,C) 经过 final norm + act
        mask: (N,1,H,W) or None
        input_global: (N, global_features)
        """
        N, L, _ = x_nlc.shape
        H = W = self.pos_len

        outv1 = self.linear_v1(x_nlc) + self.bias_v1  # (N, L, c_v1)
        if mask is not None:
            outv1 = outv1 * mask.view(N, L, 1)
        outv1 = F.silu(outv1)

        # Global pool
        outv1_nchw = outv1.permute(0, 2, 1).view(N, self.c_v1, H, W)
        pooled = self.gpool(outv1_nchw, mask, mask_sum_hw)  # (N, 3*c_v1)

        outv2 = F.silu(self.linear2(pooled))

        out_value, out_misc, out_moremisc = self.linear_vmm(outv2).split([3, 10, 8], dim=-1)

        # 空间输出
        out_ownership, out_scoring = self.linear_os(outv1).split([1, 1], dim=-1)
        out_futurepos, out_seki = self.linear_fs(x_nlc).split([2, 4], dim=-1)

        if mask is not None:
            mask_nlc = mask.view(N, L, 1)
            out_ownership = out_ownership * mask_nlc
            out_scoring = out_scoring * mask_nlc
            out_futurepos = out_futurepos * mask_nlc
            out_seki = out_seki * mask_nlc

        # 转回 NCHW 格式以匹配 loss 计算的期望
        out_ownership = out_ownership.permute(0, 2, 1).view(N, 1, H, W)
        out_scoring = out_scoring.permute(0, 2, 1).view(N, 1, H, W)
        out_futurepos = out_futurepos.permute(0, 2, 1).view(N, 2, H, W)
        out_seki = out_seki.permute(0, 2, 1).view(N, 4, H, W)

        # Score belief
        batch_size = N
        s2_out, outsmix = self.linear_s2mix(pooled).split([self.c_sv2, self.num_scorebeliefs], dim=-1)
        outsv2 = (
            s2_out.view(batch_size, 1, self.c_sv2)
            + self.linear_s2off(self.score_belief_offset_bias_vector.view(1, self.scorebelief_len, 1))
            + self.linear_s2par(
                (self.score_belief_parity_vector.view(1, self.scorebelief_len) * input_global[:, -1:])
                .view(batch_size, self.scorebelief_len, 1)
            )
        )
        outsv2 = F.silu(outsv2)
        outsv3 = self.linear_s3(outsv2)
        outsmix_logweights = F.log_softmax(outsmix, dim=1)
        out_scorebelief_logprobs = F.log_softmax(outsv3, dim=1)
        out_scorebelief_logprobs = torch.logsumexp(
            out_scorebelief_logprobs + outsmix_logweights.view(-1, 1, self.num_scorebeliefs), dim=2
        )

        return (
            out_value, out_misc, out_moremisc,
            out_ownership, out_scoring, out_futurepos, out_seki,
            out_scorebelief_logprobs,
        )


# ---------------------------------------------------------------------------
# Model 主体
# ---------------------------------------------------------------------------
class Model(nn.Module):
    def __init__(self, config: dict, pos_len: int):
        super().__init__()
        self.config = config
        self.pos_len = pos_len
        self.c_trunk = config["trunk_num_channels"]
        num_bin_features = modelconfigs.get_num_bin_input_features(config)
        num_global_features = modelconfigs.get_num_global_input_features(config)

        num_heads = config.get("transformer_heads", 4)
        num_kv_heads = config.get("transformer_kv_heads", num_heads)
        ffn_dim = config.get("transformer_ffn_channels", self.c_trunk * 2)
        rope_theta = config.get("rope_theta", 100.0)

        # Stem
        if config.get("initial_conv_1x1", False):
            self.conv_spatial = nn.Conv2d(num_bin_features, self.c_trunk, kernel_size=1, padding="same", bias=False)
        else:
            self.conv_spatial = nn.Conv2d(num_bin_features, self.c_trunk, kernel_size=3, padding="same", bias=False)
        self.linear_global = nn.Linear(num_global_features, self.c_trunk, bias=False)

        # Transformer blocks
        self.blocks = nn.ModuleList()
        for _ in config["block_kind"]:
            self.blocks.append(TransformerBlock(
                c_main=self.c_trunk,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                ffn_dim=ffn_dim,
                pos_len=pos_len,
                rope_theta=rope_theta,
            ))

        # Trunk final norm + act
        self.norm_final = RMSNormFP32(self.c_trunk, eps=1e-6)

        # Heads
        c_p1 = config["p1_num_channels"]
        c_g1 = config["g1_num_channels"]
        c_v1 = config["v1_num_channels"]
        c_v2 = config["v2_size"]
        c_sv2 = config["sbv2_num_channels"]
        num_scorebeliefs = config["num_scorebeliefs"]

        policy_head_type = config.get("policy_head_type", "full")
        if policy_head_type == "simple-pool":
            self.policy_head = SimplePolicyHeadPool(self.c_trunk, pos_len)
        elif policy_head_type == "simple-linear":
            self.policy_head = SimplePolicyHeadLinear(self.c_trunk, pos_len)
        else:
            self.policy_head = PolicyHead(self.c_trunk, c_p1, c_g1, pos_len)
        self.value_head = ValueHead(self.c_trunk, c_v1, c_v2, c_sv2, num_scorebeliefs, pos_len)

        # Seki 动态权重的移动平均状态
        self.moving_unowned_proportion_sum = 0.0
        self.moving_unowned_proportion_weight = 0.0

    def initialize(self, init_std=0.02):
        """Megatron-LM style initialization.
        - All weights: normal(0, init_std)
        - Output projections (out_proj, ffn_w2 in each block): normal(0, init_std / sqrt(2*N))
        - All biases: zero
        """
        num_blocks = len(self.blocks)
        output_std = init_std / math.sqrt(2.0 * num_blocks)

        for name, p in self.named_parameters():
            if p.dim() < 2:
                # bias, norm weight, scalar parameters -> zero for bias, skip for norm
                if "norm" not in name:
                    nn.init.zeros_(p)
            else:
                # Check if this is an output projection in a transformer block
                if ".out_proj." in name or ".ffn_w2." in name:
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

        # Mask
        mask = input_spatial[:, 0:1, :, :].contiguous()  # (N, 1, H, W)
        mask_sum_hw = torch.sum(mask, dim=(2, 3), keepdim=True)  # (N, 1, 1, 1)

        # Stem: NCHW -> NLC
        x_spatial = self.conv_spatial(input_spatial)         # (N, C, H, W)
        x_global = self.linear_global(input_global)          # (N, C)
        x = x_spatial + x_global.unsqueeze(-1).unsqueeze(-1) # (N, C, H, W)
        x = x.view(N, self.c_trunk, L).permute(0, 2, 1)     # (N, L, C)

        # Attention mask: (N, 1, 1, L) additive
        mask_flat = mask.view(N, 1, 1, L)
        attn_mask = torch.zeros_like(mask_flat, dtype=x.dtype)
        attn_mask.masked_fill_(mask_flat == 0, float("-inf"))

        # Trunk
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        # Final norm + act
        x = self.norm_final(x)
        x = F.silu(x)

        # Heads 在 fp32 下计算以保证数值稳定性
        x = x.float()
        input_global = input_global.float()

        out_policy = self.policy_head(x, mask, mask_sum_hw)
        (
            out_value, out_misc, out_moremisc,
            out_ownership, out_scoring, out_futurepos, out_seki,
            out_scorebelief,
        ) = self.value_head(x, mask, mask_sum_hw, input_global)

        return (
            out_policy, out_value, out_misc, out_moremisc,
            out_ownership, out_scoring, out_futurepos, out_seki,
            out_scorebelief,
        )

    def postprocess(self, outputs):
        (
            out_policy, out_value, out_misc, out_moremisc,
            out_ownership, out_scoring, out_futurepos, out_seki,
            out_scorebelief,
        ) = outputs

        policy_logits = out_policy
        value_logits = out_value
        td_value_logits = torch.stack(
            (out_misc[:, 4:7], out_misc[:, 7:10], out_moremisc[:, 2:5]), dim=1
        )
        pred_td_score = out_moremisc[:, 5:8] * self.value_head.td_score_multiplier
        ownership_pretanh = out_ownership
        pred_scoring = out_scoring
        futurepos_pretanh = out_futurepos
        seki_logits = out_seki
        pred_scoremean = out_misc[:, 0] * self.value_head.scoremean_multiplier
        pred_scorestdev = SoftPlusWithGradientFloor.apply(out_misc[:, 1], 0.05, False) * self.value_head.scorestdev_multiplier
        pred_lead = out_misc[:, 2] * self.value_head.lead_multiplier
        pred_variance_time = SoftPlusWithGradientFloor.apply(out_misc[:, 3], 0.05, False) * self.value_head.variance_time_multiplier

        pred_shortterm_value_error = SoftPlusWithGradientFloor.apply(out_moremisc[:, 0], 0.05, True) * self.value_head.shortterm_value_error_multiplier
        pred_shortterm_score_error = SoftPlusWithGradientFloor.apply(out_moremisc[:, 1], 0.05, True) * self.value_head.shortterm_score_error_multiplier

        scorebelief_logits = out_scorebelief

        return (
            policy_logits, value_logits, td_value_logits, pred_td_score,
            ownership_pretanh, pred_scoring, futurepos_pretanh, seki_logits,
            pred_scoremean, pred_scorestdev, pred_lead, pred_variance_time,
            pred_shortterm_value_error, pred_shortterm_score_error,
            scorebelief_logits,
        )


# ---------------------------------------------------------------------------
# Loss computation (aligned with metrics_pytorch.py)
# ---------------------------------------------------------------------------
def compute_loss(
    model, postprocessed, batch, pos_len, is_training,
    soft_policy_weight_scale=8.0,
    value_loss_scale=0.6,
    td_value_loss_scales=(0.6, 0.6, 0.6),
    seki_loss_scale=1.0,
    variance_time_loss_scale=1.0,
    disable_optimistic_policy=False,
):
    (
        policy_logits, value_logits, td_value_logits, pred_td_score,
        ownership_pretanh, pred_scoring, futurepos_pretanh, seki_logits,
        pred_scoremean, pred_scorestdev, pred_lead, pred_variance_time,
        pred_shortterm_value_error, pred_shortterm_score_error,
        scorebelief_logits,
    ) = postprocessed

    N = policy_logits.shape[0]
    pos_area = pos_len * pos_len

    input_binary = batch["binaryInputNCHW"]
    target_policy_ncmove = batch["policyTargetsNCMove"]
    target_global = batch["globalTargetsNC"]
    score_distr = batch["scoreDistrN"]
    target_value_nchw = batch["valueTargetsNCHW"]

    mask = input_binary[:, 0, :, :].contiguous()
    mask_sum_hw = torch.sum(mask, dim=(1, 2))

    H = W = pos_len
    policymask = torch.cat([mask.view(N, H * W), mask.new_ones(N, 1)], dim=1)

    # Targets
    target_policy_player = target_policy_ncmove[:, 0, :]
    target_policy_player = target_policy_player / torch.sum(target_policy_player, dim=1, keepdim=True)
    target_policy_opponent = target_policy_ncmove[:, 1, :]
    target_policy_opponent = target_policy_opponent / torch.sum(target_policy_opponent, dim=1, keepdim=True)

    # Soft policy
    target_policy_player_soft = (target_policy_player + 1e-7) * policymask
    target_policy_player_soft = torch.pow(target_policy_player_soft, 0.25)
    target_policy_player_soft = target_policy_player_soft / torch.sum(target_policy_player_soft, dim=1, keepdim=True)
    target_policy_opponent_soft = (target_policy_opponent + 1e-7) * policymask
    target_policy_opponent_soft = torch.pow(target_policy_opponent_soft, 0.25)
    target_policy_opponent_soft = target_policy_opponent_soft / torch.sum(target_policy_opponent_soft, dim=1, keepdim=True)

    target_weight_policy_player = target_global[:, 26]
    target_weight_policy_opponent = target_global[:, 28]
    target_value = target_global[:, 0:3]
    target_scoremean = target_global[:, 3]
    target_td_value = torch.stack(
        (target_global[:, 4:7], target_global[:, 8:11], target_global[:, 12:15]), dim=1
    )
    target_td_score = torch.cat(
        (target_global[:, 7:8], target_global[:, 11:12], target_global[:, 15:16]), dim=1
    )
    target_lead = target_global[:, 21]
    target_variance_time = target_global[:, 22]
    global_weight = target_global[:, 25]
    target_weight_ownership = target_global[:, 27]
    target_weight_lead = target_global[:, 29]
    target_weight_futurepos = target_global[:, 33]
    target_weight_scoring = target_global[:, 34]
    target_weight_value = 1.0 - target_global[:, 35]
    target_weight_td_value = 1.0 - target_global[:, 24]

    target_score_distribution = score_distr / 100.0
    target_ownership = target_value_nchw[:, 0, :, :]
    target_seki = target_value_nchw[:, 1, :, :]
    target_futurepos = target_value_nchw[:, 2:4, :, :]
    target_scoring = target_value_nchw[:, 4, :, :] / 120.0

    # --- Policy loss scales (对齐 metrics_pytorch.py) ---
    policy_opt_loss_scale = 0.93
    long_policy_opt_loss_scale = 0.1
    short_policy_opt_loss_scale = 0.2

    # --- Policy losses ---
    loss_policy_player = (global_weight * target_weight_policy_player * cross_entropy(
        policy_logits[:, 0, :], target_policy_player, dim=1
    )).sum()

    loss_policy_opponent = 0.15 * (global_weight * target_weight_policy_opponent * cross_entropy(
        policy_logits[:, 1, :], target_policy_opponent, dim=1
    )).sum()

    loss_policy_player_soft = (global_weight * target_weight_policy_player * cross_entropy(
        policy_logits[:, 2, :], target_policy_player_soft, dim=1
    )).sum()

    loss_policy_opponent_soft = 0.15 * (global_weight * target_weight_policy_opponent * cross_entropy(
        policy_logits[:, 3, :], target_policy_opponent_soft, dim=1
    )).sum()

    # --- Optimistic policy losses ---
    if disable_optimistic_policy:
        target_weight_longopt = target_weight_policy_player * 0.5
        loss_longoptimistic_policy = (global_weight * target_weight_longopt * cross_entropy(
            policy_logits[:, 4, :], target_policy_player, dim=1
        )).sum()
        target_weight_shortopt = target_weight_policy_player * 0.5
        loss_shortoptimistic_policy = (global_weight * target_weight_shortopt * cross_entropy(
            policy_logits[:, 5, :], target_policy_player, dim=1
        )).sum()
    else:
        # Long-term optimistic policy
        win_squared = torch.square(
            target_global[:, 0] + 0.5 * target_global[:, 2]
        )
        longterm_score_stdevs_excess = (
            target_global[:, 3] - pred_scoremean.detach()
        ) / torch.sqrt(torch.square(pred_scorestdev.detach()) + 0.25)
        target_weight_longopt = torch.clamp(
            win_squared + torch.sigmoid((longterm_score_stdevs_excess - 1.5) * 3.0),
            min=0.0, max=1.0,
        ) * target_weight_policy_player * target_weight_ownership
        loss_longoptimistic_policy = (global_weight * target_weight_longopt * cross_entropy(
            policy_logits[:, 4, :], target_policy_player, dim=1
        )).sum()

        # Short-term optimistic policy
        shortterm_value_actual = target_global[:, 12] - target_global[:, 13]
        shortterm_value_pred = torch.softmax(td_value_logits[:, 2, :].detach(), dim=1)
        shortterm_value_pred = shortterm_value_pred[:, 0] - shortterm_value_pred[:, 1]
        shortterm_value_stdevs_excess = (
            shortterm_value_actual - shortterm_value_pred
        ) / torch.sqrt(pred_shortterm_value_error.detach() + 0.0001)
        shortterm_score_stdevs_excess = (
            target_global[:, 15] - pred_td_score[:, 2].detach()
        ) / torch.sqrt(pred_shortterm_score_error.detach() + 0.25)
        target_weight_shortopt = torch.clamp(
            torch.sigmoid((shortterm_value_stdevs_excess - 1.5) * 3.0)
            + torch.sigmoid((shortterm_score_stdevs_excess - 1.5) * 3.0),
            min=0.0, max=1.0,
        ) * target_weight_policy_player * target_weight_ownership
        loss_shortoptimistic_policy = (global_weight * target_weight_shortopt * cross_entropy(
            policy_logits[:, 5, :], target_policy_player, dim=1
        )).sum()

    # --- Value losses ---
    loss_value = 1.20 * (global_weight * target_weight_value * cross_entropy(
        value_logits, target_value, dim=1
    )).sum()

    # TD value (分成 3 个独立项)
    td_loss_raw = cross_entropy(td_value_logits, target_td_value, dim=2) - cross_entropy(
        torch.log(target_td_value + 1e-30), target_td_value, dim=2
    )
    td_loss_weighted = 1.20 * global_weight.unsqueeze(1) * target_weight_td_value.unsqueeze(1) * td_loss_raw
    loss_td_value1 = td_loss_weighted[:, 0].sum()
    loss_td_value2 = td_loss_weighted[:, 1].sum()
    loss_td_value3 = td_loss_weighted[:, 2].sum()

    loss_td_score = 0.0004 * (global_weight * target_weight_ownership * torch.sum(
        huber_loss(pred_td_score, target_td_score, delta=12.0), dim=1
    )).sum()

    # --- Spatial losses ---
    # Ownership
    pred_own_logits = torch.cat([ownership_pretanh, -ownership_pretanh], dim=1).view(N, 2, pos_area)
    target_own_probs = torch.stack([(1.0 + target_ownership) / 2.0, (1.0 - target_ownership) / 2.0], dim=1).view(N, 2, pos_area)
    loss_ownership = 1.5 * (global_weight * target_weight_ownership * (
        torch.sum(cross_entropy(pred_own_logits, target_own_probs, dim=1) * mask.view(N, pos_area), dim=1) / mask_sum_hw
    )).sum()

    # Scoring
    loss_scoring_raw = torch.sum(torch.square(pred_scoring.squeeze(1) - target_scoring) * mask, dim=(1, 2)) / mask_sum_hw
    loss_scoring = (global_weight * target_weight_scoring * 4.0 * (torch.sqrt(loss_scoring_raw * 0.5 + 1.0) - 1.0)).sum()

    # Futurepos
    fp_loss = torch.square(torch.tanh(futurepos_pretanh) - target_futurepos) * mask.unsqueeze(1)
    fp_weight = torch.tensor([1.0, 0.25], device=fp_loss.device).view(1, 2, 1, 1)
    loss_futurepos = 0.25 * (global_weight * target_weight_futurepos * (
        torch.sum(fp_loss * fp_weight, dim=(1, 2, 3)) / torch.sqrt(mask_sum_hw)
    )).sum()

    # Seki (动态权重，对齐 metrics_pytorch.py)
    owned_target = torch.square(target_ownership)
    unowned_target = 1.0 - owned_target

    if is_training:
        unowned_proportion = torch.sum(unowned_target * mask, dim=(1, 2)) / (1.0 + mask_sum_hw)
        unowned_proportion = torch.mean(unowned_proportion * target_weight_ownership)
        model.moving_unowned_proportion_sum *= 0.998
        model.moving_unowned_proportion_weight *= 0.998
        model.moving_unowned_proportion_sum += unowned_proportion.item()
        model.moving_unowned_proportion_weight += 1.0
        moving_unowned_proportion = model.moving_unowned_proportion_sum / model.moving_unowned_proportion_weight
        seki_weight_scale = 8.0 * 0.005 / (0.005 + moving_unowned_proportion)
    else:
        seki_weight_scale = 7.0

    sign_pred = seki_logits[:, 0:3, :, :]
    sign_target = torch.stack([
        1.0 - torch.square(target_seki),
        F.relu(target_seki),
        F.relu(-target_seki),
    ], dim=1)
    loss_sign = torch.sum(cross_entropy(sign_pred, sign_target, dim=1) * mask, dim=(1, 2))
    neutral_pred = torch.stack([seki_logits[:, 3, :, :], torch.zeros_like(target_ownership)], dim=1)
    neutral_target = torch.stack([unowned_target, owned_target], dim=1)
    loss_neutral = torch.sum(cross_entropy(neutral_pred, neutral_target, dim=1) * mask, dim=(1, 2))
    loss_seki = (global_weight * seki_weight_scale * target_weight_ownership * (loss_sign + 0.5 * loss_neutral) / mask_sum_hw).sum()

    # --- Score belief losses ---
    loss_scoremean = 0.0015 * (global_weight * target_weight_ownership * huber_loss(
        pred_scoremean, target_scoremean, delta=12.0
    )).sum()

    pred_cdf = torch.cumsum(F.softmax(scorebelief_logits, dim=1), dim=1)
    target_cdf = torch.cumsum(target_score_distribution, dim=1)
    loss_sb_cdf = 0.020 * (global_weight * target_weight_ownership * torch.sum(
        torch.square(pred_cdf - target_cdf), dim=1
    )).sum()

    loss_sb_pdf = 0.020 * (global_weight * target_weight_ownership * cross_entropy(
        scorebelief_logits, target_score_distribution, dim=1
    )).sum()

    # Score stdev reg
    sb_probs = F.softmax(scorebelief_logits, dim=1)
    sb_offset = model.value_head.score_belief_offset_vector.view(1, -1)
    expected_score = torch.sum(sb_probs * sb_offset, dim=1, keepdim=True)
    stdev_of_belief = torch.sqrt(0.001 + torch.sum(sb_probs * torch.square(sb_offset - expected_score), dim=1))
    loss_scorestdev = 0.001 * (global_weight * huber_loss(pred_scorestdev, stdev_of_belief, delta=10.0)).sum()

    loss_lead = 0.0060 * (global_weight * target_weight_lead * huber_loss(
        pred_lead, target_lead, delta=8.0
    )).sum()

    loss_variance_time = 0.0003 * (global_weight * target_weight_ownership * huber_loss(
        pred_variance_time, target_variance_time + 1e-5, delta=50.0
    )).sum()

    # Shortterm error losses
    td_val_pred_probs = torch.softmax(td_value_logits[:, 2, :], dim=1)
    predvalue = (td_val_pred_probs[:, 0] - td_val_pred_probs[:, 1]).detach()
    realvalue = target_td_value[:, 2, 0] - target_td_value[:, 2, 1]
    sqerror_v = torch.square(predvalue - realvalue) + 1e-8
    loss_st_value_error = 2.0 * (global_weight * target_weight_ownership * huber_loss(
        pred_shortterm_value_error, sqerror_v, delta=0.4
    )).sum()

    predscore = pred_td_score[:, 2].detach()
    realscore = target_td_score[:, 2]
    sqerror_s = torch.square(predscore - realscore) + 1e-4
    loss_st_score_error = 0.00002 * (global_weight * target_weight_ownership * huber_loss(
        pred_shortterm_score_error, sqerror_s, delta=100.0
    )).sum()

    # --- Total loss (对齐 metrics_pytorch.py 的 loss_sum，再除以 N 取 mean) ---
    loss_sum = (
        loss_policy_player * policy_opt_loss_scale
        + loss_policy_opponent
        + loss_policy_player_soft * soft_policy_weight_scale
        + loss_policy_opponent_soft * soft_policy_weight_scale
        + loss_longoptimistic_policy * long_policy_opt_loss_scale
        + loss_shortoptimistic_policy * short_policy_opt_loss_scale
        + loss_value * value_loss_scale
        + loss_td_value1 * td_value_loss_scales[0]
        + loss_td_value2 * td_value_loss_scales[1]
        + loss_td_value3 * td_value_loss_scales[2]
        + loss_td_score
        + loss_ownership
        + loss_scoring * 0.25
        + loss_futurepos
        + loss_seki * seki_loss_scale
        + loss_scoremean
        + loss_sb_cdf
        + loss_sb_pdf
        + loss_scorestdev
        + loss_lead
        + loss_variance_time * variance_time_loss_scale
        + loss_st_value_error
        + loss_st_score_error
    ) / N

    # Accuracy
    with torch.no_grad():
        policy_acc1 = (global_weight * target_weight_policy_player * (
            torch.argmax(policy_logits[:, 0, :], dim=1) == torch.argmax(target_policy_player, dim=1)
        ).float()).sum()

    return loss_sum, {
        "loss": loss_sum.item(),
        "p0loss": loss_policy_player.item(),
        "p1loss": loss_policy_opponent.item(),
        "p0softloss": loss_policy_player_soft.item(),
        "p1softloss": loss_policy_opponent_soft.item(),
        "p0lopt": loss_longoptimistic_policy.item(),
        "p0sopt": loss_shortoptimistic_policy.item(),
        "vloss": loss_value.item(),
        "tdvloss1": loss_td_value1.item(),
        "tdvloss2": loss_td_value2.item(),
        "tdvloss3": loss_td_value3.item(),
        "tdsloss": loss_td_score.item(),
        "oloss": loss_ownership.item(),
        "sloss": loss_scoring.item(),
        "fploss": loss_futurepos.item(),
        "skloss": loss_seki.item(),
        "smloss": loss_scoremean.item(),
        "sbcdfloss": loss_sb_cdf.item(),
        "sbpdfloss": loss_sb_pdf.item(),
        "sdregloss": loss_scorestdev.item(),
        "leadloss": loss_lead.item(),
        "vtimeloss": loss_variance_time.item(),
        "evstloss": loss_st_value_error.item(),
        "esstloss": loss_st_score_error.item(),
        "pacc1": policy_acc1.item(),
        "wsum": global_weight.sum().item(),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Minimal Transformer training for KataGo")
    parser.add_argument("-traindir", required=True, help="Training output directory")
    parser.add_argument("-datadir", required=True, help="Data directory with train/ and val/ subdirs")
    parser.add_argument("-pos-len", type=int, default=19, help="Board size")
    parser.add_argument("-batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("-model-kind", type=str, default="b14c192h6tfrs-bng-silu", help="Model config name")
    parser.add_argument("-lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("-wd", type=float, default=0.1, help="Weight decay")
    parser.add_argument("-init-std", type=float, default=0.02, help="Init std for weights (Megatron-LM style)")
    parser.add_argument("-max-training-samples", type=int, default=100000000, help="Total training samples")
    parser.add_argument("-save-every-samples", type=int, default=1000000, help="Save checkpoint every N samples")
    parser.add_argument("-symmetry-type", type=str, default="xyt", help="Data symmetry type")
    parser.add_argument("-print-every", type=int, default=100, help="Print every N batches")
    parser.add_argument("-val-every-samples", type=int, default=1000000, help="Run validation every N samples")
    parser.add_argument("-warmup-samples", type=int, default=2000000, help="LR warmup samples")
    parser.add_argument("-enable-history-matrices", action="store_true", help="Enable history matrices (for Go)")
    parser.add_argument("-policy-head-type", type=str, default="full", choices=["full", "simple-pool", "simple-linear"], help="Policy head type")
    parser.add_argument("-initial-checkpoint", type=str, default=None, help="Initial checkpoint to load from")
    parser.add_argument("-no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("-soft-policy-weight-scale", type=float, default=8.0, help="Soft policy loss coeff")
    parser.add_argument("-value-loss-scale", type=float, default=0.6, help="Value loss coeff")
    parser.add_argument("-td-value-loss-scales", type=str, default="0.6,0.6,0.6", help="TD value loss coeffs")
    parser.add_argument("-seki-loss-scale", type=float, default=1.0, help="Seki loss coeff")
    parser.add_argument("-variance-time-loss-scale", type=float, default=1.0, help="Variance time loss coeff")
    parser.add_argument("-disable-optimistic-policy", action="store_true", help="Disable optimistic policy")
    args = parser.parse_args()

    # 解析 td_value_loss_scales
    td_value_loss_scales = [float(x) for x in args.td_value_loss_scales.split(",")]
    assert len(td_value_loss_scales) == 3

    # Logging
    os.makedirs(args.traindir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.traindir, "train.log"), mode="a"),
            logging.StreamHandler(),
        ],
    )
    logging.info(f"Args: {vars(args)}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logging.info(f"Device: {device}")
    if device.type == "cuda":
        logging.info(f"GPU: {torch.cuda.get_device_name()}")

    # AMP settings
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        amp_device = "cuda"
        amp_dtype = torch.bfloat16
        use_amp = True
    elif device.type == "mps":
        amp_device = "mps"
        amp_dtype = torch.bfloat16
        use_amp = True
    else:
        amp_device = "cpu"
        amp_dtype = torch.bfloat16
        use_amp = False  # CPU bf16 有兼容性问题，关闭 AMP

    # Model config
    model_config = modelconfigs.config_of_name[args.model_kind].copy()
    model_config["policy_head_type"] = args.policy_head_type
    logging.info(f"Model config: {json.dumps(model_config, indent=2, default=str)}")

    pos_len = args.pos_len
    batch_size = args.batch_size

    # Load or create model
    checkpoint_path = os.path.join(args.traindir, "checkpoint.ckpt")
    if os.path.exists(checkpoint_path):
        logging.info(f"Loading checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_config = state.get("config", model_config)
        model = Model(model_config, pos_len)
        model.load_state_dict(state["model"])
        model.moving_unowned_proportion_sum = state.get("moving_unowned_proportion_sum", 0.0)
        model.moving_unowned_proportion_weight = state.get("moving_unowned_proportion_weight", 0.0)
        global_step = state.get("global_step", 0)
        total_samples_trained = state.get("total_samples_trained", global_step * batch_size)
        logging.info(f"Resumed from step {global_step}, {total_samples_trained} samples")
    elif args.initial_checkpoint is not None:
        logging.info(f"Loading initial checkpoint: {args.initial_checkpoint}")
        state = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
        model_config = state.get("config", model_config)
        model = Model(model_config, pos_len)
        model.load_state_dict(state["model"])
        global_step = 0
        total_samples_trained = 0
    else:
        logging.info("Creating new model")
        model = Model(model_config, pos_len)
        model.initialize(init_std=args.init_std)
        logging.info(f"Initialized weights with std={args.init_std}, output_std={args.init_std / math.sqrt(2.0 * len(model.blocks)):.6f}")
        global_step = 0
        total_samples_trained = 0

    model.to(device)

    # torch.compile (MPS 后端不支持 inductor，自动关闭)
    if not args.no_compile and device.type != "mps":
        compiled_model = torch.compile(model, mode="default")
    else:
        compiled_model = model

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")

    # Optimizer: bias and norm parameters have no weight decay
    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() < 2 or "norm" in name:
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": args.wd},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=args.lr, betas=(0.9, 0.95))

    # Load optimizer state if resuming
    if os.path.exists(checkpoint_path):
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
            logging.info("Optimizer state loaded")

    # LR scheduler: linear warmup then cosine decay
    warmup_steps = args.warmup_samples // batch_size
    total_steps = args.max_training_samples // batch_size

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=global_step - 1 if global_step > 0 else -1)

    # Data
    train_dir = os.path.join(args.datadir, "train")
    val_dir = os.path.join(args.datadir, "val")

    # TensorBoard
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = os.path.join(args.traindir, "tb_logs")
        tb_writer = SummaryWriter(log_dir=tb_dir)
        logging.info(f"TensorBoard: {tb_dir}")
    except ImportError:
        tb_writer = None
        logging.info("TensorBoard not available")

    # Save helper
    def save_checkpoint():
        state_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": model_config,
            "global_step": global_step,
            "total_samples_trained": total_samples_trained,
            "moving_unowned_proportion_sum": model.moving_unowned_proportion_sum,
            "moving_unowned_proportion_weight": model.moving_unowned_proportion_weight,
        }
        path = os.path.join(args.traindir, "checkpoint.ckpt")
        torch.save(state_dict, path + ".tmp")
        os.replace(path + ".tmp", path)
        logging.info(f"Saved checkpoint at step {global_step}, {total_samples_trained} samples")

    # Metrics accumulation — all keys returned by compute_loss + count/grad_norm
    _metric_keys = [
        "loss", "p0loss", "p1loss", "p0softloss", "p1softloss",
        "p0lopt", "p0sopt",
        "vloss", "tdvloss1", "tdvloss2", "tdvloss3", "tdsloss",
        "oloss", "sloss", "fploss", "skloss",
        "smloss", "sbcdfloss", "sbpdfloss", "sdregloss",
        "leadloss", "vtimeloss", "evstloss", "esstloss",
        "pacc1", "wsum",
    ]
    running = {k: 0.0 for k in _metric_keys}
    running["count"] = 0
    running["grad_norm"] = 0.0

    def reset_running():
        for k in running:
            running[k] = 0.0

    # 需要除以 wsum 的指标（per-sample 指标）；其余除以 count（per-batch）
    _per_sample_keys = [k for k in _metric_keys if k not in ("loss", "wsum")]

    def print_metrics(elapsed):
        w = max(running["wsum"], 1e-10)
        bs = max(running["count"], 1)
        logging.info(
            f"step={global_step}, samples={total_samples_trained}, "
            f"time={elapsed:.1f}s, "
            f"lr={scheduler.get_last_lr()[0]:.2e}, "
            f"loss={running['loss'] / bs:.4f}, "
            f"p0loss={running['p0loss'] / w:.4f}, "
            f"vloss={running['vloss'] / w:.4f}, "
            f"oloss={running['oloss'] / w:.4f}, "
            f"skloss={running['skloss'] / w:.4f}, "
            f"pacc1={running['pacc1'] / w:.4f}"
        )
        if tb_writer is not None:
            tb_writer.add_scalar("train/loss", running["loss"] / bs, total_samples_trained)
            for k in _per_sample_keys:
                tb_writer.add_scalar(f"train/{k}", running[k] / w, total_samples_trained)
            tb_writer.add_scalar("train/lr", scheduler.get_last_lr()[0], total_samples_trained)
            tb_writer.add_scalar("train/grad_norm", running["grad_norm"] / bs, total_samples_trained)

    # Training
    logging.info("=" * 60)
    logging.info(f"Starting training: {total_samples_trained}/{args.max_training_samples} samples done")
    logging.info("=" * 60)

    last_save_samples = total_samples_trained
    last_val_samples = total_samples_trained
    reset_running()
    t0 = time.perf_counter()
    last_print_time = t0

    while total_samples_trained < args.max_training_samples:
        model.train()

        # Find training files
        train_files = sorted(glob.glob(os.path.join(train_dir, "*.npz")))
        if not train_files:
            logging.warning(f"No training files found in {train_dir}, waiting...")
            time.sleep(10)
            continue

        np.random.shuffle(train_files)

        for batch in data_processing_pytorch.read_npz_training_data(
            train_files,
            batch_size=batch_size,
            world_size=1,
            rank=0,
            pos_len=pos_len,
            device=device,
            symmetry_type=args.symmetry_type,
            include_meta=False,
            enable_history_matrices=args.enable_history_matrices,
            model_config=model_config,
        ):
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(amp_device, dtype=amp_dtype, enabled=use_amp):
                outputs = compiled_model(batch["binaryInputNCHW"], batch["globalInputNC"])

            # Postprocess in fp32
            postprocessed = model.postprocess(outputs)
            loss, metrics = compute_loss(
                model, postprocessed, batch, pos_len,
                is_training=True,
                soft_policy_weight_scale=args.soft_policy_weight_scale,
                value_loss_scale=args.value_loss_scale,
                td_value_loss_scales=td_value_loss_scales,
                seki_loss_scale=args.seki_loss_scale,
                variance_time_loss_scale=args.variance_time_loss_scale,
                disable_optimistic_policy=args.disable_optimistic_policy,
            )

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            total_samples_trained += batch_size

            # Accumulate
            for k in metrics:
                running[k] += metrics[k]
            running["grad_norm"] += grad_norm.item()
            running["count"] += 1

            if global_step % args.print_every == 0:
                t1 = time.perf_counter()
                print_metrics(t1 - last_print_time)
                reset_running()
                last_print_time = t1

            # Save checkpoint
            if total_samples_trained - last_save_samples >= args.save_every_samples:
                save_checkpoint()
                last_save_samples = total_samples_trained

            # Validation
            if total_samples_trained - last_val_samples >= args.val_every_samples:
                last_val_samples = total_samples_trained
                val_files = sorted(glob.glob(os.path.join(val_dir, "*.npz")))
                if val_files:
                    model.eval()
                    val_metrics = {k: 0.0 for k in _metric_keys}
                    val_metrics["count"] = 0
                    with torch.no_grad():
                        for vbatch in data_processing_pytorch.read_npz_training_data(
                            val_files[:3],
                            batch_size=batch_size,
                            world_size=1,
                            rank=0,
                            pos_len=pos_len,
                            device=device,
                            symmetry_type=None,
                            include_meta=False,
                            enable_history_matrices=args.enable_history_matrices,
                            model_config=model_config,
                        ):
                            with torch.amp.autocast(amp_device, dtype=amp_dtype, enabled=use_amp):
                                outputs = model(vbatch["binaryInputNCHW"], vbatch["globalInputNC"])
                            postprocessed = model.postprocess(outputs)
                            _, m = compute_loss(
                                model, postprocessed, vbatch, pos_len,
                                is_training=False,
                                soft_policy_weight_scale=args.soft_policy_weight_scale,
                                value_loss_scale=args.value_loss_scale,
                                td_value_loss_scales=td_value_loss_scales,
                                seki_loss_scale=args.seki_loss_scale,
                                variance_time_loss_scale=args.variance_time_loss_scale,
                                disable_optimistic_policy=args.disable_optimistic_policy,
                            )
                            for k in m:
                                val_metrics[k] += m[k]
                            val_metrics["count"] += 1

                    w = max(val_metrics["wsum"], 1e-10)
                    bs = max(val_metrics["count"], 1)
                    logging.info(
                        f"  VAL [{total_samples_trained} samples]: loss={val_metrics['loss'] / bs:.4f}, "
                        f"p0loss={val_metrics['p0loss'] / w:.4f}, "
                        f"vloss={val_metrics['vloss'] / w:.4f}, "
                        f"oloss={val_metrics['oloss'] / w:.4f}, "
                        f"skloss={val_metrics['skloss'] / w:.4f}, "
                        f"pacc1={val_metrics['pacc1'] / w:.4f}"
                    )
                    if tb_writer is not None:
                        tb_writer.add_scalar("val/loss", val_metrics["loss"] / bs, total_samples_trained)
                        for k in _per_sample_keys:
                            tb_writer.add_scalar(f"val/{k}", val_metrics[k] / w, total_samples_trained)
                    model.train()

            if total_samples_trained >= args.max_training_samples:
                break

    # Final save
    save_checkpoint()
    logging.info(f"Training complete: {total_samples_trained} samples, {global_step} steps")
    if tb_writer is not None:
        tb_writer.close()


if __name__ == "__main__":
    main()
