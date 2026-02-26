#!/usr/bin/python3
"""
极简 Transformer 训练脚本 for KataGo.
- 只支持纯 Transformer 模型 (RoPE + GQA + SwiGLU)
- 早期融合 H,W 为序列维度，trunk 全程 NLC
- 1x1 Conv 全部替换为 Linear
- 使用 torch.amp + bf16
- AdamW + Muon 优化器（Muon 用于 trunk 权重，AdamW 用于其余参数）
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

import torch
import torch.nn as nn
import torch.nn.functional as F

import modelconfigs
import data_processing_pytorch


# ---------------------------------------------------------------------------
# Muon 优化器辅助
# ---------------------------------------------------------------------------
# Newton-Schulz 迭代系数，用于矩阵正交化（zeropower）
_POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

# Newton-Schulz 系数，用于矩阵逆 4 次根（Shampoo 预条件）
_NS_COEFFS_R4_SCALED = (
    (3.7745392156862745, -9.830711636812923, 7.211935063687831),
    (1.7744313725490195, -0.5323686439402083, 0.05420935725061334),
    (1.4744509803921568, -0.5384714581368423, 0.10138210476839715),
    (1.3786764705882353, -0.5094735805293277, 0.13074301029260285),
)

@torch.compile
def polar_express(G):
    """Newton-Schulz 迭代实现矩阵正交化，固定 5 步。"""
    assert G.ndim == 2
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.mT

    # 范数归一化
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-7)

    # 5 步 Newton-Schulz 迭代
    for a, b, c in _POLAR_EXPRESS_COEFFS:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    # 恢复转置
    if G.size(0) > G.size(1):
        X = X.mT
    return X

class MuonOptimizer:
    """Muon 优化器：动量 + Newton-Schulz 正交化。"""

    def __init__(self, named_params, lr_multiplier, momentum, wd, scale_mode="moonlight", device="cuda"):
        self.named_params = named_params  # {name: param}
        self.lr_multiplier = lr_multiplier
        self.momentum = momentum
        self.wd = wd
        self.scale_mode = scale_mode
        self._device = device
        self.last_update_rms = 0.0
        self.states = {name: self._init_state(p) for name, p in named_params.items()}

    @staticmethod
    def _init_state(p):
        return {"momentum": torch.zeros_like(p)}

    def step(self, base_lr):
        muon_lr = base_lr * self.lr_multiplier
        rms_sum = torch.tensor(0.0, device=self._device)
        rms_cnt = 0
        with torch.no_grad():
            for name, p in self.named_params.items():
                if p.grad is None:
                    continue
                assert p.grad.ndim in (2, 4), f"Muon 只支持 2D/4D 参数，got ndim={p.grad.ndim}"
                state = self.states[name]
                grad = p.grad
                original_shape = grad.shape

                # 动量累积
                state["momentum"].mul_(self.momentum).add_(grad)
                update = state["momentum"]
                if update.ndim == 4:
                    update = update.view(update.size(0), -1)

                # Newton-Schulz 正交化 + 缩放
                update = polar_express(update)
                if self.scale_mode == "moonlight":
                    update = update * max(update.size()) ** 0.5
                else:  # mup
                    update = update * max(1, update.size(0) / update.size(1)) ** 0.5
                update = update.view(original_shape)

                # 累积 update RMS
                rms_sum += update.norm() * self.lr_multiplier / update.numel() ** 0.5
                rms_cnt += 1

                # 权重衰减 + 参数更新
                p.mul_(1 - base_lr * self.wd)
                p.add_(update.to(p.dtype), alpha=-muon_lr)

        self.last_update_rms = (rms_sum / rms_cnt).item() if rms_cnt > 0 else 0.0

    def state_dict(self):
        return {name: {k: v.cpu() for k, v in s.items()} for name, s in self.states.items()}

    def load_state_dict(self, saved, device):
        for name, tensors in saved.items():
            if name in self.states:
                for k, v in tensors.items():
                    self.states[name][k].copy_(v.to(device))


@torch.compile
def inv_quarter_sandwich(L, M, R):
    """Newton-Schulz 迭代计算 L^{-1/4} @ M @ R^{-1/4}，固定 4 步，全 fp32。"""
    assert L.ndim == 2 and M.ndim == 2 and R.ndim == 2
    eps = 1e-4
    M = M.float()

    m = L.size(-1)
    n = R.size(-1)
    I_L = torch.eye(m, device=L.device)
    I_R = torch.eye(n, device=L.device)

    # Frobenius 范数归一化
    tL = torch.sqrt((L * L.mT).sum())
    tR = torch.sqrt((R * R.mT).sum())
    L = L / tL + eps * I_L
    R = R / tR + eps * I_R

    # 4 步 Newton-Schulz 迭代
    for a, b, c in _NS_COEFFS_R4_SCALED:
        L2 = L @ L
        WL = a * I_L + b * L + c * L2

        R2 = R @ R
        WR = a * I_R + b * R + c * R2

        M = WL @ M @ WR

        # r=4: L ← L W_L^4, R ← R W_R^4
        WL4 = (WL @ WL) @ (WL @ WL)
        WR4 = (WR @ WR) @ (WR @ WR)
        L = L @ WL4
        R = R @ WR4

    # 反归一化: 乘回 tL^{-1/4}, tR^{-1/4}
    M = M * (tL ** (-0.25)) * (tR ** (-0.25))
    return M


class ShampooOptimizer:
    """Shampoo 优化器：L/R 预条件矩阵的 EMA + 矩阵逆根。"""

    def __init__(self, named_params, lr_multiplier, momentum, wd, beta2=0.999, device="cuda"):
        self.named_params = named_params  # {name: param}
        self.lr_multiplier = lr_multiplier
        self.momentum = momentum
        self.wd = wd
        self.beta2 = beta2
        self.step_count = 0
        self._device = device
        self.last_precond_rms = 0.0
        self.states = {name: self._init_state(p, device) for name, p in named_params.items()}

    @staticmethod
    def _init_state(p, device):
        if p.ndim >= 2:
            m, n = p.shape[0], p.shape[1:].numel()
        else:
            m, n = p.shape[0], 1
        return {
            "momentum": torch.zeros_like(p),
            "L": torch.zeros(m, m, dtype=torch.float32, device=device),
            "R": torch.zeros(n, n, dtype=torch.float32, device=device),
        }

    def step(self, base_lr):
        self.step_count += 1
        shampoo_lr = base_lr * self.lr_multiplier
        bias_corr1 = 1 - self.momentum ** self.step_count   # 一阶矩偏差校正
        bias_corr2 = 1 - self.beta2 ** self.step_count      # 二阶矩偏差校正
        rms_sum = torch.tensor(0.0, device=self._device)
        rms_cnt = 0
        with torch.no_grad():
            for name, p in self.named_params.items():
                if p.grad is None:
                    continue
                assert p.grad.ndim in (2, 4), f"Shampoo 只支持 2D/4D 参数，got ndim={p.grad.ndim}"
                state = self.states[name]
                grad = p.grad
                original_shape = grad.shape
                if grad.ndim == 4:
                    grad_2d = grad.view(grad.size(0), -1)
                else:
                    grad_2d = grad

                # 一阶矩 EMA
                state["momentum"].lerp_(grad, 1 - self.momentum)
                momentum_2d = state["momentum"]
                if momentum_2d.ndim == 4:
                    momentum_2d = momentum_2d.view(momentum_2d.size(0), -1)
                momentum_2d_hat = momentum_2d / bias_corr1

                # 二阶矩 EMA（L/R 预条件矩阵）
                state["L"].lerp_(grad_2d @ grad_2d.mT, 1 - self.beta2)
                state["R"].lerp_(grad_2d.mT @ grad_2d, 1 - self.beta2)

                # 预条件：L^{-1/4} @ m @ R^{-1/4}
                precond = inv_quarter_sandwich(
                    state["L"] / bias_corr2, momentum_2d_hat, state["R"] / bias_corr2,
                )

                # 累积 precond RMS
                rms_sum += precond.norm() * self.lr_multiplier / precond.numel() ** 0.5
                rms_cnt += 1
                precond = precond.view(original_shape)

                # 权重衰减 + 参数更新
                p.mul_(1 - base_lr * self.wd)
                p.add_(precond.to(p.dtype), alpha=-shampoo_lr)

        self.last_precond_rms = (rms_sum / rms_cnt).item() if rms_cnt > 0 else 0.0

    def state_dict(self):
        result = {name: {k: v.cpu() for k, v in s.items()} for name, s in self.states.items()}
        result["__step_count__"] = self.step_count
        return result

    def load_state_dict(self, saved, device):
        self.step_count = saved.get("__step_count__", 0)
        for name, tensors in saved.items():
            if name == "__step_count__":
                continue
            if name in self.states:
                for k, v in tensors.items():
                    self.states[name][k].copy_(v.to(device))


class AdamOptimizer:
    """独立 Adam 优化器（AdamW 风格解耦权重衰减），跟踪 update RMS。"""

    def __init__(self, named_params, wd, beta1=0.9, beta2=0.95, eps=1e-8, device="cuda"):
        self.named_params = named_params  # {name: param}
        self.wd = wd
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.step_count = 0
        self._device = device
        self.last_update_rms = 0.0
        self.states = {name: self._init_state(p) for name, p in named_params.items()}

    @staticmethod
    def _init_state(p):
        return {
            "m": torch.zeros_like(p),
            "v": torch.zeros_like(p),
        }

    def step(self, base_lr):
        self.step_count += 1
        bias_corr1 = 1 - self.beta1 ** self.step_count
        bias_corr2 = 1 - self.beta2 ** self.step_count
        rms_sum = torch.tensor(0.0, device=self._device)
        rms_cnt = 0
        with torch.no_grad():
            for name, p in self.named_params.items():
                if p.grad is None:
                    continue
                state = self.states[name]
                grad = p.grad

                # 一阶矩 EMA
                state["m"].lerp_(grad, 1 - self.beta1)
                # 二阶矩 EMA
                state["v"].lerp_(grad * grad, 1 - self.beta2)

                # 偏差校正
                m_hat = state["m"] / bias_corr1
                v_hat = state["v"] / bias_corr2

                # Adam update
                update = m_hat / (v_hat.sqrt() + self.eps)

                # 累积 update RMS（全 tensor 操作，避免 CUDA 同步）
                rms_sum += update.norm() / update.numel() ** 0.5
                rms_cnt += 1

                # 权重衰减 + 参数更新
                p.mul_(1 - base_lr * self.wd)
                p.add_(update, alpha=-base_lr)

        self.last_update_rms = (rms_sum / rms_cnt).item() if rms_cnt > 0 else 0.0

    def state_dict(self):
        result = {name: {k: v.cpu() for k, v in s.items()} for name, s in self.states.items()}
        result["__step_count__"] = self.step_count
        return result

    def load_state_dict(self, saved, device):
        self.step_count = saved.get("__step_count__", 0)
        for name, tensors in saved.items():
            if name == "__step_count__":
                continue
            if name in self.states:
                for k, v in tensors.items():
                    self.states[name][k].copy_(v.to(device))


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
EXTRA_SCORE_DISTR_RADIUS = 60


# ---------------------------------------------------------------------------
# 辅助函数
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
    """始终在 float32 下运行的 RMSNorm（禁用 autocast）。"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.RMSNorm(dim, eps=eps)

    def forward(self, x):
        with torch.amp.autocast(x.device.type, enabled=False):
            return self.norm(x.float()).to(x.dtype)


# ---------------------------------------------------------------------------
# Transformer Block（NLC 格式，RoPE + GQA + SwiGLU + RMSNorm）
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
        x: (N, L, C)   attn_mask: (N, 1, 1, L) 加性 mask，0 或 -inf
        """
        B, L, C = x.shape
        x_normed = self.norm1(x)

        q = self.q_proj(x_normed).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x_normed).view(B, L, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x_normed).view(B, L, self.num_kv_heads, self.head_dim)

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
        x_normed = self.norm2(x)
        x = x + self.ffn_w2(F.silu(self.ffn_w1(x_normed)) * self.ffn_wgate(x_normed))
        return x


# ---------------------------------------------------------------------------
# PolicyHead（NLC 输入）
# ---------------------------------------------------------------------------
class PolicyHead(nn.Module):
    """逐位置投影（棋盘落子）+ 全局池化投影（pass）。"""
    def __init__(self, c_in, pos_len):
        super().__init__()
        self.pos_len = pos_len
        self.num_policy_outputs = 6
        self.linear_board = nn.Linear(c_in, self.num_policy_outputs, bias=False)
        self.linear_pass = nn.Linear(c_in, self.num_policy_outputs, bias=False)

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
# ValueHead（NLC 输入，逐位置 + mean-pool 投影）
# ---------------------------------------------------------------------------
class ValueHead(nn.Module):
    def __init__(self, c_in, c_sv2, num_scorebeliefs, pos_len, score_mode="mixop"):
        super().__init__()
        self.pos_len = pos_len
        self.scorebelief_mid = pos_len * pos_len + EXTRA_SCORE_DISTR_RADIUS
        self.scorebelief_len = self.scorebelief_mid * 2
        self.num_scorebeliefs = num_scorebeliefs
        self.c_sv2 = c_sv2
        self.score_mode = score_mode

        # 逐位置: ownership(1) + scoring(1) + futurepos(2) + seki(4)
        # 全局（mean-pool）: value(3) + misc(10) + moremisc(8)
        self.n_spatial = 1 + 1 + 2 + 4  # 8
        self.n_global = 3 + 10 + 8      # 21
        self.linear_sv = nn.Linear(c_in, self.n_spatial + self.n_global, bias=False)

        # 分数信念头
        if score_mode == "simple":
            self.linear_s_simple = nn.Linear(c_in, self.scorebelief_len, bias=False)
        elif score_mode == "mix":
            self.linear_s_mix = nn.Linear(c_in, self.scorebelief_len * num_scorebeliefs + num_scorebeliefs, bias=False)
        elif score_mode == "mixop":
            self.linear_s_mix = nn.Linear(c_in, self.scorebelief_len * num_scorebeliefs + num_scorebeliefs, bias=False)
            self.linear_s2off = nn.Linear(1, num_scorebeliefs, bias=False)
            self.linear_s2par = nn.Linear(1, num_scorebeliefs, bias=False)

        self.register_buffer("score_belief_offset_vector", torch.tensor(
            [(float(i - self.scorebelief_mid) + 0.5) for i in range(self.scorebelief_len)],
            dtype=torch.float32,
        ), persistent=False)
        if score_mode == "mixop":
            self.register_buffer("score_belief_offset_bias_vector", torch.tensor(
                [0.05 * (float(i - self.scorebelief_mid) + 0.5) for i in range(self.scorebelief_len)],
                dtype=torch.float32,
            ), persistent=False)
            self.register_buffer("score_belief_parity_vector", torch.tensor(
                [0.5 - float((i - self.scorebelief_mid) % 2) for i in range(self.scorebelief_len)],
                dtype=torch.float32,
            ), persistent=False)

    def forward(self, x_nlc, mask, mask_sum_hw, score_parity):
        N, L, _ = x_nlc.shape
        H = W = self.pos_len

        # 合并投影：逐位置特征 + 全局特征
        spatial_global = self.linear_sv(x_nlc)  # (N, L, n_spatial + n_global)
        spatial, global_feats = spatial_global.split([self.n_spatial, self.n_global], dim=-1)

        # 逐位置输出：ownership / scoring / futurepos / seki
        if mask is not None:
            spatial = spatial * mask.view(N, L, 1)
        spatial = spatial.permute(0, 2, 1).view(N, self.n_spatial, H, W)
        out_ownership, out_scoring, out_futurepos, out_seki = spatial.split([1, 1, 2, 4], dim=1)

        # 全局输出：mean-pool → value / misc / moremisc
        if mask is not None:
            global_feats = global_feats * mask.view(N, L, 1)
            global_feats = global_feats.sum(dim=1) / mask_sum_hw.view(N, 1)
        else:
            global_feats = global_feats.mean(dim=1)
        out_value, out_misc, out_moremisc = global_feats.split([3, 10, 8], dim=-1)

        # 分数信念：先 mean-pool 再投影
        if mask is not None:
            pooled_s = (x_nlc * mask.view(N, L, 1)).sum(dim=1) / mask_sum_hw.view(N, 1)
        else:
            pooled_s = x_nlc.mean(dim=1)
        if self.score_mode == "simple":
            out_scorebelief_logprobs = F.log_softmax(self.linear_s_simple(pooled_s), dim=1)
        elif self.score_mode in ("mix", "mixop"):
            score_proj = self.linear_s_mix(pooled_s)
            belief_logits, mix_logits = score_proj.split(
                [self.scorebelief_len * self.num_scorebeliefs, self.num_scorebeliefs], dim=-1
            )
            belief_logits = belief_logits.view(N, self.scorebelief_len, self.num_scorebeliefs)
            if self.score_mode == "mixop":
                belief_logits = (
                    belief_logits
                    + self.linear_s2off(self.score_belief_offset_bias_vector.view(1, self.scorebelief_len, 1))
                    + self.linear_s2par(
                        (self.score_belief_parity_vector.view(1, self.scorebelief_len) * score_parity)
                        .view(N, self.scorebelief_len, 1)
                    )
                )
            mix_log_weights = F.log_softmax(mix_logits, dim=1)
            out_scorebelief_logprobs = F.log_softmax(belief_logits, dim=1)
            with torch.amp.autocast(out_scorebelief_logprobs.device.type, enabled=False):
                out_scorebelief_logprobs = torch.logsumexp(
                    out_scorebelief_logprobs.float() + mix_log_weights.float().view(-1, 1, self.num_scorebeliefs), dim=2
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
    def __init__(self, config: dict, pos_len: int, score_mode: str = "mixop"):
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

        # 输入编码（Stem）
        if config.get("initial_conv_1x1", False):
            self.conv_spatial = nn.Conv2d(num_bin_features, self.c_trunk, kernel_size=1, padding="same", bias=False)
        else:
            self.conv_spatial = nn.Conv2d(num_bin_features, self.c_trunk, kernel_size=3, padding="same", bias=False)
        self.linear_global = nn.Linear(num_global_features, self.c_trunk, bias=False)

        # Transformer 块
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

        # Trunk 最终归一化
        self.norm_final = RMSNormFP32(self.c_trunk, eps=1e-6)

        # 输出头
        c_sv2 = config["sbv2_num_channels"]
        num_scorebeliefs = config["num_scorebeliefs"]

        self.policy_head = PolicyHead(self.c_trunk, pos_len)
        self.value_head = ValueHead(self.c_trunk, c_sv2, num_scorebeliefs, pos_len, score_mode=score_mode)

        # Seki 动态权重的移动平均状态
        self.moving_unowned_proportion_sum = 0.0
        self.moving_unowned_proportion_weight = 0.0

    def initialize(self, init_std=0.02):
        """Megatron-LM 风格初始化:
        - 所有权重: normal(0, init_std)
        - 输出投影 (out_proj, ffn_w2): normal(0, init_std / sqrt(2*N_blocks))
        - 所有偏置: 零
        """
        num_blocks = len(self.blocks)
        output_std = init_std / math.sqrt(2.0 * num_blocks)

        for name, p in self.named_parameters():
            if p.dim() < 2:
                # bias, norm weight, scalar parameters -> zero for bias, skip for norm
                if "norm" not in name:
                    nn.init.zeros_(p)
            else:
                # Transformer block 的输出投影使用更小的初始化标准差
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

        # 掩码：第 0 通道表示合法位置
        mask = input_spatial[:, 0:1, :, :].contiguous()  # (N, 1, H, W)
        mask_sum_hw = torch.sum(mask, dim=(2, 3), keepdim=True)  # (N, 1, 1, 1)

        # Stem: NCHW → NLC
        x_spatial = self.conv_spatial(input_spatial)         # (N, C, H, W)
        x_global = self.linear_global(input_global)          # (N, C)
        x = x_spatial + x_global.unsqueeze(-1).unsqueeze(-1) # (N, C, H, W)
        x = x.view(N, self.c_trunk, L).permute(0, 2, 1)     # (N, L, C)

        # 注意力掩码: (N, 1, 1, L) 加性
        mask_flat = mask.view(N, 1, 1, L)
        attn_mask = torch.zeros_like(mask_flat, dtype=x.dtype)
        attn_mask.masked_fill_(mask_flat == 0, float("-inf"))

        # Trunk: 逐层 Transformer 前向
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        # 最终归一化
        x = self.norm_final(x)

        # 输出头（在 autocast 下运行，输出转 float 保证后续数值稳定）
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
# 损失计算（对齐 metrics_pytorch.py）
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

    # 目标分布
    target_policy_player = target_policy_ncmove[:, 0, :]
    target_policy_player = target_policy_player / torch.sum(target_policy_player, dim=1, keepdim=True)
    target_policy_opponent = target_policy_ncmove[:, 1, :]
    target_policy_opponent = target_policy_opponent / torch.sum(target_policy_opponent, dim=1, keepdim=True)

    # 软策略目标（0.25 次幂平滑）
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

    # --- 策略损失系数（对齐 metrics_pytorch.py）---
    policy_opt_loss_scale = 0.93
    long_policy_opt_loss_scale = 0.1
    short_policy_opt_loss_scale = 0.2

    # --- 策略损失 ---
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

    # --- 乐观策略损失 ---
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
        # 长期乐观策略
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

        # 短期乐观策略
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

    # --- 价值损失 ---
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
        F.huber_loss(pred_td_score, target_td_score, reduction="none", delta=12.0), dim=1
    )).sum()

    # --- 空间损失 ---
    # 所有权
    pred_own_logits = torch.cat([ownership_pretanh, -ownership_pretanh], dim=1).view(N, 2, pos_area)
    target_own_probs = torch.stack([(1.0 + target_ownership) / 2.0, (1.0 - target_ownership) / 2.0], dim=1).view(N, 2, pos_area)
    loss_ownership = 1.5 * (global_weight * target_weight_ownership * (
        torch.sum(cross_entropy(pred_own_logits, target_own_probs, dim=1) * mask.view(N, pos_area), dim=1) / mask_sum_hw
    )).sum()

    # 计分
    loss_scoring_raw = torch.sum(torch.square(pred_scoring.squeeze(1) - target_scoring) * mask, dim=(1, 2)) / mask_sum_hw
    loss_scoring = (global_weight * target_weight_scoring * 4.0 * (torch.sqrt(loss_scoring_raw * 0.5 + 1.0) - 1.0)).sum()

    # 未来落子位置
    fp_loss = torch.square(torch.tanh(futurepos_pretanh) - target_futurepos) * mask.unsqueeze(1)
    fp_weight = torch.tensor([1.0, 0.25], device=fp_loss.device).view(1, 2, 1, 1)
    loss_futurepos = 0.25 * (global_weight * target_weight_futurepos * (
        torch.sum(fp_loss * fp_weight, dim=(1, 2, 3)) / torch.sqrt(mask_sum_hw)
    )).sum()

    # Seki（动态权重，对齐 metrics_pytorch.py）
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

    # --- 分数信念损失 ---
    loss_scoremean = 0.0015 * (global_weight * target_weight_ownership * F.huber_loss(
        pred_scoremean, target_scoremean, reduction="none", delta=12.0
    )).sum()

    pred_cdf = torch.cumsum(F.softmax(scorebelief_logits, dim=1), dim=1)
    target_cdf = torch.cumsum(target_score_distribution, dim=1)
    loss_sb_cdf = 0.020 * (global_weight * target_weight_ownership * torch.sum(
        torch.square(pred_cdf - target_cdf), dim=1
    )).sum()

    loss_sb_pdf = 0.020 * (global_weight * target_weight_ownership * cross_entropy(
        scorebelief_logits, target_score_distribution, dim=1
    )).sum()

    # 分数标准差正则化
    score_belief_probs = F.softmax(scorebelief_logits, dim=1)
    score_belief_offsets = model.value_head.score_belief_offset_vector.view(1, -1)
    expected_score = torch.sum(score_belief_probs * score_belief_offsets, dim=1, keepdim=True)
    stdev_of_belief = torch.sqrt(0.001 + torch.sum(score_belief_probs * torch.square(score_belief_offsets - expected_score), dim=1))
    loss_scorestdev = 0.001 * (global_weight * F.huber_loss(pred_scorestdev, stdev_of_belief, reduction="none", delta=10.0)).sum()

    loss_lead = 0.0060 * (global_weight * target_weight_lead * F.huber_loss(
        pred_lead, target_lead, reduction="none", delta=8.0
    )).sum()

    loss_variance_time = 0.0003 * (global_weight * target_weight_ownership * F.huber_loss(
        pred_variance_time, target_variance_time + 1e-5, reduction="none", delta=50.0
    )).sum()

    # 短期误差损失
    td_val_pred_probs = torch.softmax(td_value_logits[:, 2, :], dim=1)
    predvalue = (td_val_pred_probs[:, 0] - td_val_pred_probs[:, 1]).detach()
    realvalue = target_td_value[:, 2, 0] - target_td_value[:, 2, 1]
    sqerror_v = torch.square(predvalue - realvalue) + 1e-8
    loss_st_value_error = 2.0 * (global_weight * target_weight_ownership * F.huber_loss(
        pred_shortterm_value_error, sqerror_v, reduction="none", delta=0.4
    )).sum()

    predscore = pred_td_score[:, 2].detach()
    realscore = target_td_score[:, 2]
    sqerror_s = torch.square(predscore - realscore) + 1e-4
    loss_st_score_error = 0.00002 * (global_weight * target_weight_ownership * F.huber_loss(
        pred_shortterm_score_error, sqerror_s, reduction="none", delta=100.0
    )).sum()

    # --- 总损失（对齐 metrics_pytorch.py 的 loss_sum，除以 N 取 mean）---
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

    # 准确率
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
# 训练主循环
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
    parser.add_argument("-muon-scope", type=str, default="off", choices=["all", "blocks", "off"],
                        help="Muon scope: all=all 2D non-norm params, blocks=only blocks.* params, off=pure AdamW")
    parser.add_argument("-muon-momentum", type=float, default=0.95, help="Muon momentum beta")
    parser.add_argument("-muon-lr-multiplier", type=float, default=0.2, help="Muon LR multiplier over base lr")
    parser.add_argument("-muon-scale", type=str, default="moonlight", choices=["moonlight", "mup"],
                        help="Muon update scale: moonlight=sqrt(max(m,n)), mup=sqrt(max(1,m/n))")
    parser.add_argument("-shampoo-scope", type=str, default="off", choices=["all", "blocks", "off"],
                        help="Shampoo scope: all=all 2D non-norm params, blocks=only blocks.* params, off=disabled")
    parser.add_argument("-shampoo-lr-multiplier", type=float, default=30.0, help="Shampoo LR multiplier over base lr")
    parser.add_argument("-shampoo-momentum", type=float, default=0.9, help="Shampoo momentum beta")
    parser.add_argument("-shampoo-beta2", type=float, default=0.95, help="Shampoo L/R EMA coefficient")
    parser.add_argument("-init-std", type=float, default=0.02, help="Init std for weights (Megatron-LM style)")
    parser.add_argument("-max-training-samples", type=int, default=100000000, help="Total training samples")
    parser.add_argument("-save-every-samples", type=int, default=1000000, help="Save checkpoint every N samples")
    parser.add_argument("-symmetry-type", type=str, default="xyt", help="Data symmetry type")
    parser.add_argument("-print-every", type=int, default=100, help="Print every N batches")
    parser.add_argument("-val-every-samples", type=int, default=1000000, help="Run validation every N samples")
    parser.add_argument("-warmup-samples", type=int, default=2000000, help="LR warmup samples")
    parser.add_argument("-enable-history-matrices", action="store_true", help="Enable history matrices (for Go)")
    parser.add_argument("-initial-checkpoint", type=str, default=None, help="Initial checkpoint to load from")
    parser.add_argument("-no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("-soft-policy-weight-scale", type=float, default=8.0, help="Soft policy loss coeff")
    parser.add_argument("-value-loss-scale", type=float, default=0.6, help="Value loss coeff")
    parser.add_argument("-td-value-loss-scales", type=str, default="0.6,0.6,0.6", help="TD value loss coeffs")
    parser.add_argument("-seki-loss-scale", type=float, default=1.0, help="Seki loss coeff")
    parser.add_argument("-variance-time-loss-scale", type=float, default=1.0, help="Variance time loss coeff")
    parser.add_argument("-disable-optimistic-policy", action="store_true", help="Disable optimistic policy")
    parser.add_argument("-prefetch-batches", type=int, default=20, help="Prefetch queue depth (0=off, 2=recommended)")
    parser.add_argument("-score-mode", type=str, default="simple", choices=["mixop", "mix", "simple"],
                        help="Score belief head mode: mixop=linear+offset/parity+MoS, mix=linear+MoS, simple=single linear")
    args = parser.parse_args()

    # 互斥检查：muon 和 shampoo 不能同时启用
    if args.muon_scope != "off" and args.shampoo_scope != "off":
        parser.error("muon-scope and shampoo-scope cannot both be enabled. Set one to 'off'.")

    # 解析 td_value_loss_scales
    td_value_loss_scales = [float(x) for x in args.td_value_loss_scales.split(",")]
    assert len(td_value_loss_scales) == 3

    # 日志配置
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

    # AMP 设置
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

    # 模型配置
    model_config = modelconfigs.config_of_name[args.model_kind].copy()
    logging.info(f"Model config: {json.dumps(model_config, indent=2, default=str)}")

    pos_len = args.pos_len
    batch_size = args.batch_size

    # 加载或创建模型
    checkpoint_path = os.path.join(args.traindir, "checkpoint.ckpt")
    if os.path.exists(checkpoint_path):
        logging.info(f"Loading checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_config = state.get("config", model_config)
        model = Model(model_config, pos_len, score_mode=args.score_mode)
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
        model = Model(model_config, pos_len, score_mode=args.score_mode)
        model.load_state_dict(state["model"])
        global_step = 0
        total_samples_trained = 0
    else:
        logging.info("Creating new model")
        model = Model(model_config, pos_len, score_mode=args.score_mode)
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

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")

    # 优化器：将参数分为 Muon / Shampoo / Adam 组
    muon_params = {}   # name -> param
    shampoo_params = {}  # name -> param
    adam_params = {}   # name -> param (2D+ with weight decay)
    no_decay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() < 2 or "norm" in name:
            no_decay_params.append(p)
        elif args.muon_scope == "all":
            muon_params[name] = p
        elif args.muon_scope == "blocks" and "blocks." in name:
            muon_params[name] = p
        elif args.shampoo_scope == "all":
            shampoo_params[name] = p
        elif args.shampoo_scope == "blocks" and "blocks." in name:
            shampoo_params[name] = p
        else:
            adam_params[name] = p

    logging.info(f"Muon params: {sum(p.numel() for p in muon_params.values()):,}, "
                 f"Shampoo params: {sum(p.numel() for p in shampoo_params.values()):,}, "
                 f"Adam decay: {sum(p.numel() for p in adam_params.values()):,}, "
                 f"AdamW no-decay: {sum(p.numel() for p in no_decay_params):,}")

    # torch.optim.AdamW 仅处理 no_decay 参数（1D bias/norm）
    optimizer = torch.optim.AdamW([
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=args.lr, betas=(0.9, 0.95))

    # 独立优化器实例
    muon_opt = MuonOptimizer(
        muon_params, lr_multiplier=args.muon_lr_multiplier,
        momentum=args.muon_momentum, wd=args.wd, scale_mode=args.muon_scale, device=device,
    ) if muon_params else None
    shampoo_opt = ShampooOptimizer(
        shampoo_params, lr_multiplier=args.shampoo_lr_multiplier,
        momentum=args.shampoo_momentum, wd=args.wd, beta2=args.shampoo_beta2, device=device,
    ) if shampoo_params else None
    adam_opt = AdamOptimizer(
        adam_params, wd=args.wd, beta1=0.9, beta2=0.95, device=device,
    ) if adam_params else None

    # 恢复优化器状态
    if os.path.exists(checkpoint_path):
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
            logging.info("Optimizer state loaded")
        if "muon_state" in state and muon_opt is not None:
            muon_opt.load_state_dict(state["muon_state"], device)
            logging.info("Muon state loaded")
        elif "muon_bufs" in state and muon_opt is not None:
            # 向后兼容：旧版 checkpoint 格式
            for k, v in state["muon_bufs"].items():
                if k in muon_opt.states:
                    muon_opt.states[k]["momentum"].copy_(v.to(device))
            logging.info("Muon momentum buffers loaded (legacy format)")
        if "shampoo_state" in state and shampoo_opt is not None:
            shampoo_opt.load_state_dict(state["shampoo_state"], device)
            logging.info("Shampoo state loaded")
        if "adam_state" in state and adam_opt is not None:
            adam_opt.load_state_dict(state["adam_state"], device)
            logging.info("Adam state loaded")

    # 学习率调度：线性预热 + 余弦衰减
    warmup_steps = args.warmup_samples // batch_size
    total_steps = args.max_training_samples // batch_size

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=global_step - 1 if global_step > 0 else -1)

    # 数据目录
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

    # 保存 checkpoint
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
        if muon_opt is not None:
            state_dict["muon_state"] = muon_opt.state_dict()
        if shampoo_opt is not None:
            state_dict["shampoo_state"] = shampoo_opt.state_dict()
        if adam_opt is not None:
            state_dict["adam_state"] = adam_opt.state_dict()
        path = os.path.join(args.traindir, "checkpoint.ckpt")
        torch.save(state_dict, path + ".tmp")
        os.replace(path + ".tmp", path)
        logging.info(f"Saved checkpoint at step {global_step}, {total_samples_trained} samples")

    # 指标累积 — compute_loss 返回的所有 key + count/grad_norm
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
    running["muon_update_rms"] = 0.0
    running["shampoo_precond_rms"] = 0.0
    running["adam_update_rms"] = 0.0

    def reset_running():
        for k in running:
            running[k] = 0.0

    # 按 wsum 归一化的指标（per-sample）；其余按 count 归一化（per-batch）
    _per_sample_keys = [k for k in _metric_keys if k not in ("loss", "wsum")]

    def print_metrics(elapsed):
        weight_sum = max(running["wsum"], 1e-10)
        batch_count = max(running["count"], 1)
        logging.info(
            f"step={global_step}, samples={total_samples_trained}, "
            f"time={elapsed:.1f}s, "
            f"lr={scheduler.get_last_lr()[0]:.2e}, "
            f"loss={running['loss'] / batch_count:.4f}, "
            f"p0loss={running['p0loss'] / weight_sum:.4f}, "
            f"vloss={running['vloss'] / weight_sum:.4f}, "
            f"oloss={running['oloss'] / weight_sum:.4f}, "
            f"skloss={running['skloss'] / weight_sum:.4f}, "
            f"pacc1={running['pacc1'] / weight_sum:.4f}"
        )
        if tb_writer is not None:
            tb_writer.add_scalar("train/loss", running["loss"] / batch_count, total_samples_trained)
            for k in _per_sample_keys:
                tb_writer.add_scalar(f"train/{k}", running[k] / weight_sum, total_samples_trained)
            tb_writer.add_scalar("train/lr", scheduler.get_last_lr()[0], total_samples_trained)
            tb_writer.add_scalar("train/grad_norm", running["grad_norm"] / batch_count, total_samples_trained)
            if muon_opt is not None:
                tb_writer.add_scalar("train/muon_update_rms", running["muon_update_rms"] / batch_count, total_samples_trained)
            if shampoo_opt is not None:
                tb_writer.add_scalar("train/shampoo_precond_rms", running["shampoo_precond_rms"] / batch_count, total_samples_trained)
            if adam_opt is not None:
                tb_writer.add_scalar("train/adam_update_rms", running["adam_update_rms"] / batch_count, total_samples_trained)

    # 开始训练
    logging.info("=" * 60)
    logging.info(f"Starting training: {total_samples_trained}/{args.max_training_samples} samples done")
    logging.info("=" * 60)

    last_save_samples = total_samples_trained
    last_val_samples = total_samples_trained
    reset_running()
    time_start = time.perf_counter()
    last_print_time = time_start

    while total_samples_trained < args.max_training_samples:
        model.train()

        # 查找训练文件
        train_files = sorted(glob.glob(os.path.join(train_dir, "*.npz")))
        if not train_files:
            logging.warning(f"No training files found in {train_dir}, waiting...")
            time.sleep(10)
            continue

        np.random.shuffle(train_files)

        use_pin_memory = args.prefetch_batches > 0 and device.type == "cuda"
        train_gen = data_processing_pytorch.read_npz_training_data(
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
            use_pin_memory=use_pin_memory,
        )
        for batch in data_processing_pytorch.prefetch_generator(train_gen, args.prefetch_batches):
            for p in model.parameters():
                p.grad = None

            with torch.amp.autocast(amp_device, dtype=amp_dtype, enabled=use_amp):
                outputs = compiled_model(batch["binaryInputNCHW"], batch["globalInputNC"])

            # fp32 后处理
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

            # Muon / Shampoo / Adam 更新步
            base_lr = scheduler.get_last_lr()[0]
            if muon_opt is not None:
                muon_opt.step(base_lr)
            if shampoo_opt is not None:
                shampoo_opt.step(base_lr)
            if adam_opt is not None:
                adam_opt.step(base_lr)

            global_step += 1
            total_samples_trained += batch_size

            # 累积指标
            for k in metrics:
                running[k] += metrics[k]
            running["grad_norm"] += grad_norm.item()
            if muon_opt is not None:
                running["muon_update_rms"] += muon_opt.last_update_rms
            if shampoo_opt is not None:
                running["shampoo_precond_rms"] += shampoo_opt.last_precond_rms
            if adam_opt is not None:
                running["adam_update_rms"] += adam_opt.last_update_rms
            running["count"] += 1

            if global_step % args.print_every == 0:
                time_now = time.perf_counter()
                print_metrics(time_now - last_print_time)
                reset_running()
                last_print_time = time_now

            # 定期保存
            if total_samples_trained - last_save_samples >= args.save_every_samples:
                save_checkpoint()
                last_save_samples = total_samples_trained

            # 验证
            if total_samples_trained - last_val_samples >= args.val_every_samples:
                last_val_samples = total_samples_trained
                val_files = sorted(glob.glob(os.path.join(val_dir, "*.npz")))
                if val_files:
                    model.eval()
                    val_metrics = {k: 0.0 for k in _metric_keys}
                    val_metrics["count"] = 0
                    with torch.no_grad():
                        val_gen = data_processing_pytorch.read_npz_training_data(
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
                            use_pin_memory=use_pin_memory,
                        )
                        for val_batch in data_processing_pytorch.prefetch_generator(val_gen, args.prefetch_batches):
                            with torch.amp.autocast(amp_device, dtype=amp_dtype, enabled=use_amp):
                                outputs = model(val_batch["binaryInputNCHW"], val_batch["globalInputNC"])
                            postprocessed = model.postprocess(outputs)
                            _, batch_metrics = compute_loss(
                                model, postprocessed, val_batch, pos_len,
                                is_training=False,
                                soft_policy_weight_scale=args.soft_policy_weight_scale,
                                value_loss_scale=args.value_loss_scale,
                                td_value_loss_scales=td_value_loss_scales,
                                seki_loss_scale=args.seki_loss_scale,
                                variance_time_loss_scale=args.variance_time_loss_scale,
                                disable_optimistic_policy=args.disable_optimistic_policy,
                            )
                            for k in batch_metrics:
                                val_metrics[k] += batch_metrics[k]
                            val_metrics["count"] += 1

                    weight_sum = max(val_metrics["wsum"], 1e-10)
                    batch_count = max(val_metrics["count"], 1)
                    logging.info(
                        f"  VAL [{total_samples_trained} samples]: loss={val_metrics['loss'] / batch_count:.4f}, "
                        f"p0loss={val_metrics['p0loss'] / weight_sum:.4f}, "
                        f"vloss={val_metrics['vloss'] / weight_sum:.4f}, "
                        f"oloss={val_metrics['oloss'] / weight_sum:.4f}, "
                        f"skloss={val_metrics['skloss'] / weight_sum:.4f}, "
                        f"pacc1={val_metrics['pacc1'] / weight_sum:.4f}"
                    )
                    if tb_writer is not None:
                        tb_writer.add_scalar("val/loss", val_metrics["loss"] / batch_count, total_samples_trained)
                        for k in _per_sample_keys:
                            tb_writer.add_scalar(f"val/{k}", val_metrics[k] / weight_sum, total_samples_trained)
                    model.train()

            if total_samples_trained >= args.max_training_samples:
                break

    # 最终保存
    save_checkpoint()
    logging.info(f"Training complete: {total_samples_trained} samples, {global_step} steps")
    if tb_writer is not None:
        tb_writer.close()


if __name__ == "__main__":
    main()
