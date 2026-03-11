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
def build_edge_index_map(pos_len: int) -> torch.Tensor:
    """Precompute edge-distance index map for absolute position encoding.

    For each position (r, c), compute sorted edge-distance pair (a, b) where
    a = min(dist_r, dist_c), b = max(dist_r, dist_c), then map to a unique
    index via b*(b+1)/2 + a. This gives 55 equivalence classes for 19x19
    (0 <= a <= b <= 9) with full D_4 invariance.

    Returns: LongTensor of shape (pos_len * pos_len,)
    """
    coords = torch.arange(pos_len)
    edge_dist = torch.min(coords, pos_len - 1 - coords)  # 0 to (pos_len-1)//2
    grid_r, grid_c = torch.meshgrid(edge_dist, edge_dist, indexing="ij")
    a = torch.min(grid_r, grid_c)
    b = torch.max(grid_r, grid_c)
    return (b * (b + 1) // 2 + a).flatten().long()


def build_rpb_index_map(pos_len: int) -> torch.Tensor:
    """Precompute pairwise relative position index map for RPB.

    For each pair of positions (i, j) on the board, compute:
        dx = row_i - row_j, dy = col_i - col_j
        a = min(|dx|, |dy|), b = max(|dx|, |dy|)
        index = b*(b+1)/2 + a

    For 19x19 board: 0 <= a <= b <= 18, giving 190 equivalence classes.

    Returns: LongTensor of shape (L, L) where L = pos_len * pos_len
    """
    L = pos_len * pos_len
    positions = torch.arange(L)
    rows = positions // pos_len
    cols = positions % pos_len
    dx = (rows.unsqueeze(1) - rows.unsqueeze(0)).abs()
    dy = (cols.unsqueeze(1) - cols.unsqueeze(0)).abs()
    a = torch.min(dx, dy)
    b = torch.max(dx, dy)
    return (b * (b + 1) // 2 + a).long()


def precompute_freqs_cos_sin_2d(dim: int, pos_len: int, theta: float = 100.0):
    assert dim % 4 == 0
    dim_half = dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, dim_half, 2).float() / dim_half))
    t = torch.arange(pos_len, dtype=torch.float32)
    grid_h, grid_w = torch.meshgrid(t, t, indexing="ij")
    emb_h = grid_h.unsqueeze(-1) * freqs
    emb_w = grid_w.unsqueeze(-1) * freqs
    emb = torch.cat([emb_h, emb_w], dim=-1).flatten(0, 1)
    return emb.reshape(pos_len * pos_len, 1, 1, dim_half)


def apply_rotary_emb(xq, xk, cos, sin):
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    cos = cos.view(1, xq.shape[1], 1, xq.shape[-1])
    sin = sin.view(1, xq.shape[1], 1, xq.shape[-1])
    xq_out = xq * cos + rotate_half(xq) * sin
    xk_out = xk * cos + rotate_half(xk) * sin
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
# GAB (Geometric Attention Bias)
# ---------------------------------------------------------------------------
def compute_gab_fourier_features(dr, dc, freqs):
    """Compute Fourier features for relative (dr, dc) offsets.
    dr, dc: (...) float tensors of row/col offsets
    freqs: (F,) learnable frequency parameters
    Returns: (..., 8*F)
    """
    dr_f = dr.unsqueeze(-1) * freqs   # (..., F)
    dc_f = dc.unsqueeze(-1) * freqs
    dp_f = (dr + dc).unsqueeze(-1) * freqs
    dm_f = (dr - dc).unsqueeze(-1) * freqs
    features = torch.stack([
        torch.sin(dr_f), torch.cos(dr_f),
        torch.sin(dc_f), torch.cos(dc_f),
        torch.sin(dp_f), torch.cos(dp_f),
        torch.sin(dm_f), torch.cos(dm_f),
    ], dim=-1)  # (..., F, 8)
    return features.flatten(-2)  # (..., 8*F)


class GABTemplateMLP(nn.Module):
    """Shared MLP mapping relative (dr, dc) offsets to T template values.
    Computed once per forward pass and shared across all transformer blocks.
    """
    def __init__(self, num_templates, num_fourier_features, mlp_hidden, pos_len):
        super().__init__()
        assert num_fourier_features >= 2
        fourier_dim = 8 * num_fourier_features

        init_freqs = torch.exp(torch.linspace(
            math.log(1.0), math.log(1.0 / 50.0), num_fourier_features
        ))
        self.gab_freqs = nn.Parameter(init_freqs)

        self.linear1 = nn.Linear(fourier_dim, mlp_hidden)
        self.linear2 = nn.Linear(mlp_hidden, num_templates)

        S = pos_len * pos_len
        idx = torch.arange(S)
        rows, cols = idx // pos_len, idx % pos_len
        self.register_buffer("offset_dr", (rows.unsqueeze(1) - rows.unsqueeze(0)).float(), persistent=False)
        self.register_buffer("offset_dc", (cols.unsqueeze(1) - cols.unsqueeze(0)).float(), persistent=False)

    def forward(self):
        """Returns: (S, S, T) templates for all position pairs."""
        feats = compute_gab_fourier_features(self.offset_dr, self.offset_dc, self.gab_freqs)
        return self.linear2(F.gelu(self.linear1(feats)))


# ---------------------------------------------------------------------------
# Transformer Block (NLC format, RoPE + MHA + SwiGLU + RMSNorm)
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, c_main: int, num_heads: int, ffn_dim: int,
                 gab_num_templates: int = 0, gab_d1: int = 16, gab_d2: int = 16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = c_main // num_heads
        self.ffn_dim = ffn_dim
        self.use_gab = gab_num_templates > 0

        self.q_proj = nn.Linear(c_main, c_main, bias=False)
        self.k_proj = nn.Linear(c_main, c_main, bias=False)
        self.v_proj = nn.Linear(c_main, c_main, bias=False)
        self.out_proj = nn.Linear(c_main, c_main, bias=False)

        # SwiGLU FFN
        self.ffn_w1 = nn.Linear(c_main, ffn_dim, bias=False)
        self.ffn_wgate = nn.Linear(c_main, ffn_dim, bias=False)
        self.ffn_w2 = nn.Linear(ffn_dim, c_main, bias=False)

        self.norm1 = RMSNormFP32(c_main, eps=1e-6)
        self.norm2 = RMSNormFP32(c_main, eps=1e-6)

        if self.use_gab:
            self.gab_num_templates = gab_num_templates
            self.gab_proj1 = nn.Linear(c_main, gab_d1, bias=False)
            self.gab_proj2 = nn.Linear(gab_d1, gab_d2, bias=False)
            self.gab_norm1 = nn.LayerNorm(gab_d2)
            self.gab_proj3 = nn.Linear(gab_d2, num_heads * gab_num_templates, bias=False)
            self.gab_norm2 = nn.LayerNorm(num_heads * gab_num_templates)

    def _compute_gab_bias(self, x_norm, gab_templates):
        """Compute GAB attention bias from board state and shared templates.
        x_norm: (B, S, C) normalized token representations
        gab_templates: (S, S, T) shared templates from GABTemplateMLP
        Returns: (B, H, S, S) attention bias
        """
        B = x_norm.shape[0]
        pooled = self.gab_proj1(x_norm).mean(dim=1)       # (B, d1)
        z = F.gelu(self.gab_proj2(pooled))                 # (B, d2)
        z = self.gab_norm1(z)
        z = F.gelu(self.gab_proj3(z))                      # (B, H*T)
        z = self.gab_norm2(z)
        z = z.view(B, self.num_heads, self.gab_num_templates)  # (B, H, T)
        return torch.einsum("bhd,std->bhst", z, gab_templates)

    def forward(self, x, rope_cos=None, rope_sin=None, attn_bias=None, gab_templates=None):
        """
        x: (N, L, C)
        rope_cos, rope_sin: (L, head_dim) precomputed RoPE embeddings (None when using RPB)
        attn_bias: (1, H, L, L) relative position bias (None when using RoPE)
        gab_templates: (S, S, T) shared GAB templates (None when not using GAB)
        """
        B, L, C = x.shape
        x_normed = self.norm1(x)

        if self.use_gab and gab_templates is not None:
            gab_bias = self._compute_gab_bias(x_normed, gab_templates)
            attn_bias = gab_bias if attn_bias is None else attn_bias + gab_bias

        q = self.q_proj(x_normed).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x_normed).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x_normed).view(B, L, self.num_heads, self.head_dim)

        if rope_cos is not None:
            q, k = apply_rotary_emb(q, k, rope_cos, rope_sin)

        # SDPA: (B, H, S, D)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=0.0)
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

    def forward(self, x_nlc):
        N, L, _ = x_nlc.shape
        board = self.linear_board(x_nlc).permute(0, 2, 1)  # (N, 6, L)
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

    def forward(self, x_nlc, score_parity):
        N, L, _ = x_nlc.shape
        H = W = self.pos_len

        spatial_global = self.linear_sv(x_nlc)
        spatial, global_feats = spatial_global.split([self.n_spatial, self.n_global], dim=-1)

        spatial = spatial.permute(0, 2, 1).view(N, self.n_spatial, H, W)
        out_ownership, out_scoring, out_futurepos, out_seki = spatial.split([1, 1, 2, 4], dim=1)

        global_feats = global_feats.mean(dim=1)
        out_value, out_misc, out_moremisc = global_feats.split([3, 10, 8], dim=-1)

        # Score belief: mean-pool then project
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
        self.linear_global = nn.Linear(num_global_features, self.c_trunk, bias=False)

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

        # RPE: precompute RoPE embeddings once for the whole model (rotate_half style)
        if self.use_rope:
            emb = precompute_freqs_cos_sin_2d(head_dim, pos_len)           # (L, 1, 1, dim_half)
            emb_expanded = torch.cat([emb, emb], dim=-1)                   # (L, 1, 1, dim)
            self.register_buffer("rope_cos", emb_expanded.cos(), persistent=False)
            self.register_buffer("rope_sin", emb_expanded.sin(), persistent=False)

        # GAB: shared template MLP
        self.use_gab = config.get("use_gab", False)
        if self.use_gab:
            self.gab_template_mlp = GABTemplateMLP(
                num_templates=config.get("gab_num_templates", 16),
                num_fourier_features=config.get("gab_num_fourier_features", 8),
                mlp_hidden=config.get("gab_mlp_hidden", 64),
                pos_len=pos_len,
            )

        # Transformer blocks
        gab_kwargs = dict(
            gab_num_templates=config.get("gab_num_templates", 16),
            gab_d1=config.get("gab_d1", 16),
            gab_d2=config.get("gab_d2", 16),
        ) if self.use_gab else {}
        self.blocks = nn.ModuleList()
        for _ in range(config["num_layers"]):
            self.blocks.append(TransformerBlock(
                c_main=self.c_trunk,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                **gab_kwargs,
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

    def initialize(self, init_std=0.02, use_fan_in_init=True, stem_init_aligned=False):
        """Weight initialization.

        When use_fan_in_init=True: Megatron-LM style, Linear/Conv std = 1/sqrt(fan_in).
        When use_fan_in_init=False: all Linear/Conv layers use fixed init_std.
        In both modes, output layers (out_proj, ffn_w2) additionally scale by 1/sqrt(2*num_blocks).
        init_std is always used for non-linear/conv parameters (APE, RPB, etc.).
        When stem_init_aligned=True: override stem (conv_spatial, linear_global) weight std to
        init_std/sqrt(fan_in), aligning their output variance with APE's init_std.
        """
        num_blocks = len(self.blocks)

        for name, p in self.named_parameters():
            if p.dim() < 2:
                if "norm" not in name:
                    nn.init.zeros_(p)
            else:
                if use_fan_in_init:
                    std = 1.0 / math.sqrt(p[0].numel())  # 1/sqrt(fan_in)
                else:
                    std = init_std
                if ".out_proj." in name or ".ffn_w2." in name:
                    std = std / math.sqrt(2.0 * num_blocks)
                nn.init.normal_(p, mean=0.0, std=std)

        if self.ape in ("d4", "per_pos"):
            nn.init.normal_(self.pos_embed.weight, mean=0.0, std=init_std)
        if self.use_rpb:
            for table in self.rpb_tables:
                nn.init.zeros_(table)
        if self.use_gab:
            # Restore geometric initialization for GAB frequencies (zeroed by loop above)
            num_ff = self.config.get("gab_num_fourier_features", 8)
            with torch.no_grad():
                self.gab_template_mlp.gab_freqs.copy_(
                    torch.exp(torch.linspace(math.log(1.0), math.log(1.0 / 50.0), num_ff))
                )

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

    def _forward_trunk_impl(self, input_spatial, input_global):
        x = self._forward_stem_impl(input_spatial, input_global)
        return self._forward_blocks_impl(x)

    def _forward_blocks_impl(self, x):
        gab_templates = self.gab_template_mlp() if self.use_gab else None

        # Trunk
        for i, block in enumerate(self.blocks):
            attn_bias = None
            if self.use_rpb:
                attn_bias = self.rpb_tables[i][:, self.rpb_index_map].unsqueeze(0).to(x.dtype)  # (1, H, L, L)
            if self.use_rope:
                x = block(x, self.rope_cos, self.rope_sin, attn_bias=attn_bias, gab_templates=gab_templates)
            else:
                x = block(x, attn_bias=attn_bias, gab_templates=gab_templates)

        return self.norm_final(x)

    def _forward_stem_impl(self, input_spatial, input_global):
        N = input_spatial.shape[0]
        H = W = self.pos_len
        L = H * W

        # Stem: NCHW -> NLC
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

    def forward_blocks_for_onnx_export(self, input_stem):
        return self._forward_blocks_impl(input_stem).float()

    def forward_trunk_for_onnx_export(self, input_spatial, input_global):
        return self._forward_trunk_impl(input_spatial, input_global).float()

    def forward(self, input_spatial, input_global):
        """
        input_spatial: (N, C_bin, H, W)
        input_global:  (N, C_global)
        """
        x = self._forward_trunk_impl(input_spatial, input_global)

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
