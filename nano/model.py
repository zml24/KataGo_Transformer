"""Transformer model architecture for KataGo nano training."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import get_num_bin_input_features, get_num_global_input_features

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXTRA_SCORE_DISTR_RADIUS = 60


# ---------------------------------------------------------------------------
# Helpers
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
    """RMSNorm that always runs in float32 (autocast disabled)."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.RMSNorm(dim, eps=eps)

    def forward(self, x):
        with torch.amp.autocast(x.device.type, enabled=False):
            return self.norm(x.float()).to(x.dtype)


# ---------------------------------------------------------------------------
# Transformer Block (NLC format, RoPE + MHA + SwiGLU + RMSNorm)
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, c_main: int, num_heads: int,
                 ffn_dim: int, pos_len: int, rope_theta: float = 100.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = c_main // num_heads
        self.ffn_dim = ffn_dim

        self.q_proj = nn.Linear(c_main, c_main, bias=False)
        self.k_proj = nn.Linear(c_main, c_main, bias=False)
        self.v_proj = nn.Linear(c_main, c_main, bias=False)
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

    def forward(self, x, attn_mask):
        """
        x: (N, L, C)
        attn_mask: (N, 1, 1, L) additive mask
        """
        B, L, C = x.shape
        x_normed = self.norm1(x)

        q = self.q_proj(x_normed).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x_normed).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x_normed).view(B, L, self.num_heads, self.head_dim)

        q, k = apply_rotary_emb(q, k, self.cos_cached, self.sin_cached)

        # SDPA: (B, H, S, D)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, L, C)
        x = x + self.out_proj(attn_out)

        # SwiGLU FFN
        x_normed = self.norm2(x)
        x = x + self.ffn_w2(F.silu(self.ffn_w1(x_normed)) * self.ffn_wgate(x_normed))
        return x


# ---------------------------------------------------------------------------
# PolicyHead (NLC input)
# ---------------------------------------------------------------------------
class PolicyHead(nn.Module):
    """Per-position projection (board moves) + global pooling projection (pass)."""
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
# ValueHead (NLC input, per-position + mean-pool projection)
# ---------------------------------------------------------------------------
class ValueHead(nn.Module):
    def __init__(self, c_in, num_scorebeliefs, pos_len, score_mode="mixop"):
        super().__init__()
        self.pos_len = pos_len
        self.scorebelief_mid = pos_len * pos_len + EXTRA_SCORE_DISTR_RADIUS
        self.scorebelief_len = self.scorebelief_mid * 2
        self.num_scorebeliefs = num_scorebeliefs
        self.score_mode = score_mode

        # Per-position: ownership(1) + scoring(1) + futurepos(2) + seki(4)
        # Global (mean-pool): value(3) + misc(10) + moremisc(8)
        self.n_spatial = 1 + 1 + 2 + 4  # 8
        self.n_global = 3 + 10 + 8      # 21
        self.linear_sv = nn.Linear(c_in, self.n_spatial + self.n_global, bias=False)

        # Score belief head
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

        spatial_global = self.linear_sv(x_nlc)
        spatial, global_feats = spatial_global.split([self.n_spatial, self.n_global], dim=-1)

        if mask is not None:
            spatial = spatial * mask.view(N, L, 1)
        spatial = spatial.permute(0, 2, 1).view(N, self.n_spatial, H, W)
        out_ownership, out_scoring, out_futurepos, out_seki = spatial.split([1, 1, 2, 4], dim=1)

        if mask is not None:
            global_feats = global_feats * mask.view(N, L, 1)
            global_feats = global_feats.sum(dim=1) / mask_sum_hw.view(N, 1)
        else:
            global_feats = global_feats.mean(dim=1)
        out_value, out_misc, out_moremisc = global_feats.split([3, 10, 8], dim=-1)

        # Score belief: mean-pool then project
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
# Model
# ---------------------------------------------------------------------------
class Model(nn.Module):
    def __init__(self, config: dict, pos_len: int, score_mode: str = "mixop"):
        super().__init__()
        self.config = config
        self.pos_len = pos_len
        self.c_trunk = config["hidden_size"]
        num_bin_features = get_num_bin_input_features(config)
        num_global_features = get_num_global_input_features(config)

        num_heads = config["num_heads"]
        ffn_dim = config["ffn_dim"]

        # Stem
        self.conv_spatial = nn.Conv2d(num_bin_features, self.c_trunk, kernel_size=3, padding="same", bias=False)
        self.linear_global = nn.Linear(num_global_features, self.c_trunk, bias=False)

        # Transformer blocks
        self.blocks = nn.ModuleList()
        for _ in range(config["num_layers"]):
            self.blocks.append(TransformerBlock(
                c_main=self.c_trunk,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                pos_len=pos_len,
            ))

        # Final normalization
        self.norm_final = RMSNormFP32(self.c_trunk, eps=1e-6)

        # Output heads
        num_scorebeliefs = config["num_scorebeliefs"]

        self.policy_head = PolicyHead(self.c_trunk, pos_len)
        self.value_head = ValueHead(self.c_trunk, num_scorebeliefs, pos_len, score_mode=score_mode)

        # Seki dynamic weight moving average state
        self.moving_unowned_proportion_sum = 0.0
        self.moving_unowned_proportion_weight = 0.0

    def initialize(self, init_std=0.02):
        """Megatron-LM style initialization."""
        num_blocks = len(self.blocks)
        output_std = init_std / math.sqrt(2.0 * num_blocks)

        for name, p in self.named_parameters():
            if p.dim() < 2:
                if "norm" not in name:
                    nn.init.zeros_(p)
            else:
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

        mask = input_spatial[:, 0:1, :, :].contiguous()
        mask_sum_hw = torch.sum(mask, dim=(2, 3), keepdim=True)

        # Stem: NCHW -> NLC
        x_spatial = self.conv_spatial(input_spatial)
        x_global = self.linear_global(input_global)
        x = x_spatial + x_global.unsqueeze(-1).unsqueeze(-1)
        x = x.view(N, self.c_trunk, L).permute(0, 2, 1)

        # Attention mask
        mask_flat = mask.view(N, 1, 1, L)
        attn_mask = torch.zeros_like(mask_flat, dtype=x.dtype)
        attn_mask.masked_fill_(mask_flat == 0, float("-inf"))

        # Trunk
        for block in self.blocks:
            x = block(x, attn_mask)

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
