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
    return emb.reshape(pos_len * pos_len, 1, 1, dim_half)


def apply_rotary_emb(xq, xk, cos, sin):
    """Apply rotary position embedding (computed in FP32 for numerical stability).
    cos, sin: (L, 1, 1, D) for standard RoPE or (1, L, H, D) for learnable RoPE.
    Both are reshaped to (1, L, ?, D) via view for broadcasting with (B, L, H, D).
    """
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    orig_dtype = xq.dtype
    with torch.amp.autocast(xq.device.type, enabled=False):
        xq, xk = xq.float(), xk.float()
        cos = cos.float().view(1, xq.shape[1], -1, xq.shape[-1])
        sin = sin.float().view(1, xq.shape[1], -1, xq.shape[-1])
        xq_out = xq * cos + rotate_half(xq) * sin
        xk_out = xk * cos + rotate_half(xk) * sin
    return xq_out.to(orig_dtype), xk_out.to(orig_dtype)


class RMSNormFP32(nn.Module):
    """RMSNorm that always runs in float32 (disables autocast)."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.RMSNorm(dim, eps=eps)

    def forward(self, x):
        with torch.amp.autocast(x.device.type, enabled=False):
            return self.norm(x.float()).to(x.dtype)


class ZeroCenteredRMSNormFP32(nn.Module):
    """Zero-Centered RMSNorm in FP32.

    Weight initialized to 0; forward uses (1 + weight) * rms_norm(x).
    Weight decay pushes weight toward 0, i.e. gamma toward 1.
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        with torch.amp.autocast(x.device.type, enabled=False):
            x_f32 = x.float()
            mean_sq = (x_f32 * x_f32).mean(-1, keepdim=True)
            inv_rms = torch.rsqrt(mean_sq + self.eps)
            return ((1.0 + self.weight.float()) * (x_f32 * inv_rms)).to(x.dtype)


# ---------------------------------------------------------------------------
# Attention Residuals (Full depth attention)
# ---------------------------------------------------------------------------
def attn_res(states, proj, norm):
    """Full depth attention over all previous layer outputs.
    states: list of N tensors, each [B, T, D]
    proj: Linear(D, 1, bias=False) — learnable pseudo-query vector
    norm: RMSNormFP32(D) — key normalization
    """
    V = torch.stack(states, dim=2)                         # [B, T, N, D]
    K = norm(V)                                             # [B, T, N, D]
    B, T, N, D = V.shape
    q = proj.weight.view(1, 1, 1, D).expand(B, T, 1, D)   # [B, T, 1, D]
    # Treat T as num_heads, N (depth) as seq_len; SDPA fuses QK^T + softmax + @V
    h = F.scaled_dot_product_attention(q, K, V, scale=1.0, dropout_p=0.0)
    return h.squeeze(2)                                     # [B, T, D]


# ---------------------------------------------------------------------------
# GAB (Global Average Biasing) — input-independent positional templates
# ---------------------------------------------------------------------------

def compute_gab_fourier_features(dr, dc, freqs):
    """Compute Fourier features for relative (dr, dc) offsets.
    dr, dc: (S, S) float tensors of row/col offsets
    freqs: (F,) tensor of learnable frequencies
    Returns: (S, S, 8*F)
    """
    f = freqs.view(1, 1, -1)                    # (1, 1, F)
    dr_u = dr.unsqueeze(-1)                      # (S, S, 1)
    dc_u = dc.unsqueeze(-1)
    dr_plus_dc = (dr + dc).unsqueeze(-1)
    dr_minus_dc = (dr - dc).unsqueeze(-1)
    return torch.cat([
        torch.sin(f * dr_u), torch.cos(f * dr_u),
        torch.sin(f * dc_u), torch.cos(f * dc_u),
        torch.sin(f * dr_plus_dc), torch.cos(f * dr_plus_dc),
        torch.sin(f * dr_minus_dc), torch.cos(f * dr_minus_dc),
    ], dim=-1)


class GABTemplateMLP(nn.Module):
    """Shared MLP that maps relative (dr, dc) offsets to T template values.
    Computed once and shared across all GAB-enabled transformer blocks.
    """
    def __init__(self, gab_num_templates, gab_num_fourier_features, gab_mlp_hidden, pos_len):
        super().__init__()
        self.gab_num_templates = gab_num_templates
        assert gab_num_fourier_features >= 2
        fourier_input_dim = 8 * gab_num_fourier_features

        init_freqs = torch.exp(torch.linspace(
            math.log(1.0), math.log(1.0 / 50.0), gab_num_fourier_features))
        self.gab_freqs = nn.Parameter(init_freqs)

        self.linear1 = nn.Linear(fourier_input_dim, gab_mlp_hidden)
        self.linear2 = nn.Linear(gab_mlp_hidden, gab_num_templates)

        S = pos_len * pos_len
        s_idx = torch.arange(S)
        s_r, s_c = s_idx // pos_len, s_idx % pos_len
        offset_dr = (s_r.unsqueeze(1) - s_r.unsqueeze(0)).float()
        offset_dc = (s_c.unsqueeze(1) - s_c.unsqueeze(0)).float()
        self.register_buffer("offset_dr", offset_dr, persistent=False)
        self.register_buffer("offset_dc", offset_dc, persistent=False)

    def forward(self, seq_len):
        """Returns: (seq_len, seq_len, T) templates, pre-scaled by 1/sqrt(T)."""
        dr = self.offset_dr[:seq_len, :seq_len]
        dc = self.offset_dc[:seq_len, :seq_len]
        fourier_feats = compute_gab_fourier_features(dr, dc, self.gab_freqs)
        x = F.silu(self.linear1(fourier_feats))
        x = self.linear2(x)
        return x * (1.0 / math.sqrt(self.gab_num_templates))


# ---------------------------------------------------------------------------
# TAB (Trunk Average Biasing) — input-dependent factored attention bias
# ---------------------------------------------------------------------------

def tab_rotate(z, cos_a, sin_a):
    """Apply complex rotation to z.
    z: (*, 2, c_z, H, W) where dim -4 is [real, imag]
    cos_a, sin_a: broadcastable to (*, 1, c_z, H, W)
    """
    r = z[..., 0:1, :, :, :]
    i = z[..., 1:2, :, :, :]
    new_r = r * cos_a - i * sin_a
    new_i = r * sin_a + i * cos_a
    return torch.cat([new_r, new_i], dim=-4)


class ComplexConv2d(nn.Module):
    """2D convolution enforcing complex multiplication structure.
    Input: (*, 2*c_in, H, W), Output: (*, 2*c_out, H, W).
    """
    def __init__(self, c_in, c_out=None, kernel_size=1, dilation=1):
        super().__init__()
        if c_out is None:
            c_out = c_in
        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.real_kernel = nn.Parameter(torch.empty(c_out, c_in, kernel_size, kernel_size))
        self.imag_kernel = nn.Parameter(torch.empty(c_out, c_in, kernel_size, kernel_size))

    def forward(self, x):
        top = torch.cat([self.real_kernel, -self.imag_kernel], dim=1)
        bot = torch.cat([self.imag_kernel, self.real_kernel], dim=1)
        kernel = torch.cat([top, bot], dim=0)
        padding = self.dilation * (self.kernel_size // 2)
        return F.conv2d(x, kernel, padding=padding, dilation=self.dilation)


class TABEquivariantBlock(nn.Module):
    """One equivariant residual block for TAB with complex convolutions."""
    def __init__(self, c_z, dilation):
        super().__init__()
        self.conv1 = ComplexConv2d(c_z, kernel_size=3, dilation=dilation)
        self.conv2 = ComplexConv2d(c_z, kernel_size=3, dilation=1)
        self.c_z = c_z

    def forward(self, z, cos_a, sin_a, block_idx):
        zskip = z
        z = z * (1.0 / math.sqrt(block_idx + 1))
        z = F.silu(z)
        z = tab_rotate(z, cos_a, sin_a)
        z = z.reshape(z.shape[0], 2 * self.c_z, z.shape[3], z.shape[4])
        z = self.conv1(z)
        z = z.reshape(z.shape[0], 2, self.c_z, z.shape[2], z.shape[3])
        z = tab_rotate(z, cos_a, -sin_a)
        z = F.silu(z)
        z = tab_rotate(z, cos_a, sin_a)
        z = z.reshape(z.shape[0], 2 * self.c_z, z.shape[3], z.shape[4])
        z = self.conv2(z)
        z = z.reshape(z.shape[0], 2, self.c_z, z.shape[2], z.shape[3])
        z = tab_rotate(z, cos_a, -sin_a)
        return z + zskip


class TABModule(nn.Module):
    """TAB module: per-frequency independent c_z-channel complex convnet.
    Produces factored keys/queries shared across all transformer blocks.
    """
    def __init__(self, trunk_channels, tab_c_z, tab_num_templates, tab_num_freqs,
                 tab_num_blocks, tab_dilation, pos_len):
        super().__init__()
        self.tab_c_z = tab_c_z
        self.tab_num_freqs = tab_num_freqs
        self.tab_num_templates = tab_num_templates

        self.input_proj = nn.Conv2d(trunk_channels, 2 * tab_num_freqs * tab_c_z,
                                    kernel_size=1, bias=False)

        log_lo = math.log(1.0 / 50.0)
        log_hi = math.log(1.0)
        init_freqs = torch.exp(torch.empty(tab_num_freqs, 2).uniform_(log_lo, log_hi))
        init_freqs = init_freqs * (torch.randint(0, 2, (tab_num_freqs, 2)) * 2 - 1).float()
        self.rope_freqs = nn.Parameter(init_freqs)

        self.blocks = nn.ModuleList()
        for _ in range(tab_num_blocks):
            self.blocks.append(TABEquivariantBlock(tab_c_z, tab_dilation))

        self.key_proj = ComplexConv2d(tab_c_z, 1, kernel_size=1)
        self.query_proj = ComplexConv2d(tab_c_z, tab_num_templates, kernel_size=1)

    def forward(self, x, mask):
        """
        x: (N, C, H, W) trunk stem output
        mask: (N, 1, H, W) or None
        Returns: keys (N, 2*F, 1, S), queries (N, 2*F, T, S), pre-scaled
        """
        N, C, H, W = x.shape
        S = H * W
        num_F = self.tab_num_freqs
        T = self.tab_num_templates
        c_z = self.tab_c_z

        z = self.input_proj(x)                  # (N, 2*F*c_z, H, W)
        z = z.view(N, num_F, 2, c_z, H, W)

        gy = torch.arange(H, device=x.device, dtype=x.dtype)
        gx = torch.arange(W, device=x.device, dtype=x.dtype)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
        angles = (self.rope_freqs[:, 0:1].unsqueeze(-1) * grid_x.unsqueeze(0) +
                  self.rope_freqs[:, 1:2].unsqueeze(-1) * grid_y.unsqueeze(0))
        cos_a = torch.cos(angles).view(1, num_F, 1, 1, H, W)
        sin_a = torch.sin(angles).view(1, num_F, 1, 1, H, W)

        if mask is not None:
            z = z * mask.view(N, 1, 1, 1, H, W)

        z = z.reshape(N * num_F, 2, c_z, H, W)
        cos_a_b = cos_a.expand(N, num_F, 1, 1, H, W).reshape(N * num_F, 1, 1, H, W)
        sin_a_b = sin_a.expand(N, num_F, 1, 1, H, W).reshape(N * num_F, 1, 1, H, W)

        for idx, block in enumerate(self.blocks):
            z = block(z, cos_a_b, sin_a_b, idx)

        z = z * (1.0 / math.sqrt(len(self.blocks) + 1))
        z = F.silu(z)
        z = tab_rotate(z, cos_a_b, sin_a_b)
        z_flat = z.reshape(N * num_F, 2 * c_z, H, W)

        keys = self.key_proj(z_flat).view(N, 2 * num_F, 1, S)
        queries = self.query_proj(z_flat).view(N, 2 * num_F, T, S)
        return keys / math.sqrt(num_F), queries / math.sqrt(T)


class ComplexDepthwiseConv2d(nn.Module):
    """Depthwise 2D complex convolution (per-channel, no cross-channel mixing).
    Input/Output: (*, 2*c, H, W) laid out as [re_0..re_{c-1}, im_0..im_{c-1}].
    """
    def __init__(self, c, kernel_size=3, dilation=1):
        super().__init__()
        self.c = c
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.real_kernel = nn.Parameter(torch.empty(c, 1, kernel_size, kernel_size))
        self.imag_kernel = nn.Parameter(torch.empty(c, 1, kernel_size, kernel_size))

    def forward(self, x):
        padding = self.dilation * (self.kernel_size // 2)
        x_re = x[..., :self.c, :, :]
        x_im = x[..., self.c:, :, :]
        x_ri = torch.cat([x_re, x_im], dim=-3)
        k_ri = torch.cat([self.real_kernel, self.imag_kernel], dim=0)
        conv1 = F.conv2d(x_ri, k_ri, padding=padding, dilation=self.dilation, groups=2 * self.c)
        k_neg_ir = torch.cat([-self.imag_kernel, self.real_kernel], dim=0)
        conv2 = F.conv2d(x_ri, k_neg_ir, padding=padding, dilation=self.dilation, groups=2 * self.c)
        out_re = conv1[..., :self.c, :, :] - conv1[..., self.c:, :, :]
        out_im = conv2[..., self.c:, :, :] - conv2[..., :self.c, :, :]
        return torch.cat([out_re, out_im], dim=-3)


class FrequencyMixingTABBlock(nn.Module):
    """Residual block for frequency-mixing TAB.
    Depthwise convs per-frequency in rotated frame, 1x1 convs mix across frequencies.
    """
    def __init__(self, c_z, dilation):
        super().__init__()
        self.c_z = c_z
        self.dw_conv1 = ComplexDepthwiseConv2d(c_z, kernel_size=3, dilation=dilation)
        self.mix1 = nn.Conv2d(2 * c_z, 2 * c_z, kernel_size=1, bias=False)
        self.dw_conv2 = ComplexDepthwiseConv2d(c_z, kernel_size=3, dilation=1)
        self.mix2 = nn.Conv2d(2 * c_z, 2 * c_z, kernel_size=1, bias=False)

    def forward(self, z, cos_a, sin_a, block_idx):
        N, _, c_z, H, W = z.shape
        zskip = z
        z = z * (1.0 / math.sqrt(block_idx + 1))
        z = F.silu(z)
        z = tab_rotate(z, cos_a, sin_a)
        z_flat = z.reshape(N, 2 * c_z, H, W)
        z_flat = self.dw_conv1(z_flat)
        z = z_flat.view(N, 2, c_z, H, W)
        z = tab_rotate(z, cos_a, -sin_a)
        z_flat = z.reshape(N, 2 * c_z, H, W)
        z_flat = self.mix1(z_flat)
        z = z_flat.view(N, 2, c_z, H, W)
        z = F.silu(z)
        z = tab_rotate(z, cos_a, sin_a)
        z_flat = z.reshape(N, 2 * c_z, H, W)
        z_flat = self.dw_conv2(z_flat)
        z = z_flat.view(N, 2, c_z, H, W)
        z = tab_rotate(z, cos_a, -sin_a)
        z_flat = z.reshape(N, 2 * c_z, H, W)
        z_flat = self.mix2(z_flat)
        z = z_flat.view(N, 2, c_z, H, W)
        return z + zskip


class FrequencyMixingTABModule(nn.Module):
    """TAB module with frequency mixing. c_z IS the number of frequencies.
    Depthwise convs for spatial mixing in rotated frame, 1x1 convs for frequency mixing.
    """
    def __init__(self, trunk_channels, tab_c_z, tab_num_templates,
                 tab_num_blocks, tab_dilation, pos_len):
        super().__init__()
        self.tab_c_z = tab_c_z
        self.tab_num_templates = tab_num_templates

        self.input_proj = nn.Conv2d(trunk_channels, 2 * tab_c_z, kernel_size=1, bias=False)

        log_lo = math.log(1.0 / 50.0)
        log_hi = math.log(1.0)
        init_freqs = torch.exp(torch.empty(tab_c_z, 2).uniform_(log_lo, log_hi))
        init_freqs = init_freqs * (torch.randint(0, 2, (tab_c_z, 2)) * 2 - 1).float()
        self.rope_freqs = nn.Parameter(init_freqs)

        self.blocks = nn.ModuleList()
        for _ in range(tab_num_blocks):
            self.blocks.append(FrequencyMixingTABBlock(tab_c_z, tab_dilation))

        self.key_proj = nn.Conv2d(2 * tab_c_z, 2 * tab_c_z, kernel_size=1, bias=False)
        self.query_proj = nn.Conv2d(2 * tab_c_z, 2 * tab_c_z * tab_num_templates,
                                    kernel_size=1, bias=False)

    def forward(self, x, mask):
        """
        x: (N, C, H, W) trunk stem output
        mask: (N, 1, H, W) or None
        Returns: keys (N, 2*c_z, 1, S), queries (N, 2*c_z, T, S), pre-scaled
        """
        N, C, H, W = x.shape
        S = H * W
        c_z = self.tab_c_z
        T = self.tab_num_templates

        z = self.input_proj(x)
        if mask is not None:
            z = z * mask
        z = z.view(N, 2, c_z, H, W)

        gy = torch.arange(H, device=x.device, dtype=x.dtype)
        gx = torch.arange(W, device=x.device, dtype=x.dtype)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
        angles = (self.rope_freqs[:, 0:1].unsqueeze(-1) * grid_x.unsqueeze(0) +
                  self.rope_freqs[:, 1:2].unsqueeze(-1) * grid_y.unsqueeze(0))
        cos_a = torch.cos(angles).view(1, 1, c_z, H, W)
        sin_a = torch.sin(angles).view(1, 1, c_z, H, W)

        for idx, block in enumerate(self.blocks):
            z = block(z, cos_a, sin_a, idx)

        z = z * (1.0 / math.sqrt(len(self.blocks) + 1))
        z = F.silu(z)
        z_flat = z.reshape(N, 2 * c_z, H, W)

        cos_a_out = cos_a.view(1, c_z, 1, 1, H, W).expand(N, c_z, 1, 1, H, W).reshape(
            N * c_z, 1, 1, H, W)
        sin_a_out = sin_a.view(1, c_z, 1, 1, H, W).expand(N, c_z, 1, 1, H, W).reshape(
            N * c_z, 1, 1, H, W)

        keys = self.key_proj(z_flat)                       # (N, 2*c_z, H, W)
        keys = keys.view(N * c_z, 2, 1, H, W)
        keys = tab_rotate(keys, cos_a_out, sin_a_out)
        keys = keys.reshape(N, 2 * c_z, 1, S)

        queries = self.query_proj(z_flat)                  # (N, 2*c_z*T, H, W)
        queries = queries.view(N * c_z, 2, T, H, W)
        queries = tab_rotate(queries, cos_a_out, sin_a_out)
        queries = queries.reshape(N, 2 * c_z, T, S)

        return keys / math.sqrt(c_z), queries / math.sqrt(T)


# ---------------------------------------------------------------------------
# Transformer Block (NLC format, RoPE + MHA + SwiGLU + RMSNorm)
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, c_main: int, num_heads: int, ffn_dim: int,
                 use_attn_res: bool = False, is_first_block: bool = False,
                 use_gated_attn: bool = False, norm_fp32: bool = True,
                 zero_centered_norm: bool = False,
                 use_gab: bool = False, use_tab: bool = False,
                 gab_num_templates: int = 0, tab_num_templates: int = 0,
                 gab_d1: int = 64, gab_d2: int = 64,
                 tab_num_extra_dims: int = 0,
                 learnable_rope: bool = False, pos_len: int = 0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = c_main // num_heads
        self.ffn_dim = ffn_dim
        self.use_gab = use_gab
        self.use_tab = use_tab

        # Learnable RoPE: per-head 2D frequency parameters
        self.learnable_rope = learnable_rope
        if learnable_rope:
            assert self.head_dim % 2 == 0, f"Head dim must be even for learnable RoPE, got {self.head_dim}"
            P = self.head_dim // 2
            self.pos_len = pos_len
            log_lo = math.log(1.0 / 50.0)
            log_hi = math.log(1.0)
            init_freqs = torch.exp(torch.empty(num_heads, P, 2).uniform_(log_lo, log_hi))
            init_freqs = init_freqs * (torch.randint(0, 2, (num_heads, P, 2)) * 2 - 1).float()
            self.rope_freqs = nn.Parameter(init_freqs)  # (num_heads, P, 2)

        self.q_proj = nn.Linear(c_main, c_main, bias=False)
        self.k_proj = nn.Linear(c_main, c_main, bias=False)
        self.v_proj = nn.Linear(c_main, c_main, bias=False)
        self.out_proj = nn.Linear(c_main, c_main, bias=False)

        # SwiGLU FFN
        self.ffn_w1 = nn.Linear(c_main, ffn_dim, bias=False)
        self.ffn_wgate = nn.Linear(c_main, ffn_dim, bias=False)
        self.ffn_w2 = nn.Linear(ffn_dim, c_main, bias=False)

        if zero_centered_norm:
            NormClass = ZeroCenteredRMSNormFP32
        elif norm_fp32:
            NormClass = RMSNormFP32
        else:
            NormClass = nn.RMSNorm
        self.norm1 = NormClass(c_main, eps=1e-6)
        self.norm2 = NormClass(c_main, eps=1e-6)

        self.use_gated_attn = use_gated_attn
        if use_gated_attn:
            self.attn_gate_proj = nn.Linear(c_main, c_main, bias=False)

        # GAB/TAB per-block projection layers
        if use_gab or use_tab:
            self.gab_num_templates = gab_num_templates
            self.tab_num_templates = tab_num_templates
            self.tab_num_extra_dims = tab_num_extra_dims
            total_num_weights = gab_num_templates + tab_num_templates
            self.total_num_weights = total_num_weights
            self.gab_proj1 = nn.Linear(c_main, gab_d1, bias=False)
            self.gab_proj2 = nn.Linear(gab_d1, gab_d2, bias=False)
            self.gab_norm1 = NormClass(gab_d2, eps=1e-6)
            self.gab_proj3 = nn.Linear(gab_d2, num_heads * total_num_weights, bias=False)
            self.gab_norm2 = NormClass(num_heads * total_num_weights, eps=1e-6)

        self.use_attn_res = use_attn_res
        # First block: only stem in history, pre-attention depth attention is a no-op
        self.skip_first_attn_res = is_first_block and use_attn_res
        if use_attn_res:
            if not is_first_block:
                self.attn_res_proj = nn.Linear(c_main, 1, bias=False)
                self.attn_res_norm = NormClass(c_main)
            self.mlp_res_proj = nn.Linear(c_main, 1, bias=False)
            self.mlp_res_norm = NormClass(c_main)

    def _compute_learnable_rope(self, L, device):
        """Compute per-head cos/sin from learnable 2D frequencies.
        Returns cos, sin each of shape (1, L, H, D) in doubled format,
        compatible with apply_rotary_emb's rotate_half pattern.
        """
        idx = torch.arange(L, device=device)
        s_x = (idx % self.pos_len).float()   # col
        s_y = (idx // self.pos_len).float()   # row
        omega_x = self.rope_freqs[:, :, 0]    # (H, P)
        omega_y = self.rope_freqs[:, :, 1]    # (H, P)
        # angles[h, l, p] = s_x[l] * omega_x[h, p] + s_y[l] * omega_y[h, p]
        angles = (s_x.unsqueeze(0).unsqueeze(-1) * omega_x.unsqueeze(1)
                + s_y.unsqueeze(0).unsqueeze(-1) * omega_y.unsqueeze(1))  # (H, L, P)
        # Convert to doubled format (1, L, H, D) matching standard RoPE's rotate_half layout
        angles = angles.permute(1, 0, 2)                   # (L, H, P)
        angles = torch.cat([angles, angles], dim=-1)        # (L, H, D)
        return angles.unsqueeze(0).cos(), angles.unsqueeze(0).sin()  # (1, L, H, D)

    def _compute_gab_bias(self, x_norm, mask_flat, gab_tab_shared):
        """Compute attention bias from GAB templates and/or TAB factored K/Q.
        x_norm: (B, S, C) normalized token representations
        mask_flat: (B, S) float or None
        gab_tab_shared: dict with precomputed templates/keys/queries
        Returns: (template_bias, extra_kq) where
            template_bias: (B, H, S, S) materialized attention bias, or None
            extra_kq: (extra_k, extra_q) each (B, H, S, D_extra), or None
        """
        B, S, _ = x_norm.shape

        y = self.gab_proj1(x_norm)                              # (B, S, d1)
        if mask_flat is not None:
            mask_expanded = mask_flat.unsqueeze(-1)              # (B, S, 1)
            pooled = (y * mask_expanded).sum(dim=1) / mask_flat.sum(dim=1, keepdim=True)
        else:
            pooled = y.mean(dim=1)                              # (B, d1)

        z = F.silu(self.gab_proj2(pooled))                      # (B, d2)
        z = self.gab_norm1(z)
        z = F.silu(self.gab_proj3(z))                           # (B, H*total)
        z = self.gab_norm2(z)
        z = z.view(B, self.num_heads, self.total_num_weights)   # (B, H, W_total)

        bias = None
        extra_k_parts = []
        extra_q_parts = []
        idx = 0

        if self.use_gab:
            z_gab = z[:, :, idx:idx + self.gab_num_templates]
            idx += self.gab_num_templates
            gab_templates = gab_tab_shared["gab_templates"]     # (S, S, T)
            bias = torch.einsum("bhd,std->bhst", z_gab, gab_templates)

        if self.use_tab:
            z_tab = z[:, :, idx:idx + self.tab_num_templates]   # (B, H, T)
            idx += self.tab_num_templates
            tab_keys = gab_tab_shared["tab_keys"]               # (N, 2*F, 1, S)
            tab_queries = gab_tab_shared["tab_queries"]         # (N, 2*F, T, S)
            mixed_q = torch.einsum("bht,bfts->bhfs", z_tab, tab_queries)
            extra_q_parts.append(mixed_q.permute(0, 1, 3, 2))  # (B, H, S, 2*F)
            tab_keys_flat = tab_keys.squeeze(2).permute(0, 2, 1)  # (B, S, 2*F)
            tab_keys_flat = tab_keys_flat.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            extra_k_parts.append(tab_keys_flat)                 # (B, H, S, 2*F)

        extra_kq = None
        if extra_k_parts:
            extra_k = torch.cat(extra_k_parts, dim=-1)
            extra_q = torch.cat(extra_q_parts, dim=-1)
            extra_kq = (extra_k, extra_q)

        return bias, extra_kq

    def forward(self, x, rope_cos, rope_sin, attn_mask=None,
                mask_flat=None, gab_tab_shared=None):
        """
        x: (N, L, C)
        rope_cos, rope_sin: (L, 1, 1, head_dim) precomputed RoPE embeddings
        attn_mask: optional (N, 1, 1, L) additive mask, 0 for valid, -inf for padding
        mask_flat: optional (N, L) float for GAB masked pooling
        gab_tab_shared: optional dict with precomputed GAB/TAB data
        """
        B, L, C = x.shape
        x_normed = self.norm1(x)

        q = self.q_proj(x_normed).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x_normed).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x_normed).view(B, L, self.num_heads, self.head_dim)

        if self.learnable_rope:
            cos, sin = self._compute_learnable_rope(L, x.device)
            q, k = apply_rotary_emb(q, k, cos, sin)
        else:
            q, k = apply_rotary_emb(q, k, rope_cos, rope_sin)

        # SDPA: (B, H, S, D)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # GAB/TAB: compute attention bias and extra K/Q
        template_bias = None
        extra_kq = None
        if (self.use_gab or self.use_tab) and gab_tab_shared is not None:
            template_bias, extra_kq = self._compute_gab_bias(
                x_normed, mask_flat, gab_tab_shared)

        if template_bias is not None:
            attn_mask = (attn_mask + template_bias) if attn_mask is not None else template_bias

        scale = None
        if extra_kq is not None:
            extra_k, extra_q = extra_kq
            q = q * (1.0 / math.sqrt(self.head_dim))
            scale = 1.0
            q = torch.cat([q, extra_q], dim=-1)
            k = torch.cat([k, extra_k], dim=-1)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, scale=scale)
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, L, C)
        if self.use_gated_attn:
            attn_out = torch.sigmoid(self.attn_gate_proj(x_normed)) * attn_out
        x = x + self.out_proj(attn_out)

        # SwiGLU FFN:
        # keep the gate product in FP32 for numerical stability, but let W2
        # follow the ambient autocast / AMP precision.
        x_normed = self.norm2(x)
        w1_out = F.silu(self.ffn_w1(x_normed))
        wgate_out = self.ffn_wgate(x_normed)
        with torch.amp.autocast(x.device.type, enabled=False):
            ffn_hidden = (w1_out.float() * wgate_out.float()).to(x.dtype)
        x = x + self.ffn_w2(ffn_hidden)
        return x

    def forward_attn_res(self, states, rope_cos, rope_sin,
                         attn_mask=None, mask_flat=None, gab_tab_shared=None):
        """Forward with Attention Residuals: depth attention replaces standard residual.

        states: mutable list of all previous sub-layer outputs. Each entry is an
                individual sub-layer output (not accumulated). This method appends
                the attn output and mlp output as separate entries.
        """
        # 1. Depth attention before self-attention
        if not self.skip_first_attn_res:
            h = attn_res(states, self.attn_res_proj, self.attn_res_norm)
        else:
            h = states[-1]

        # 2. Self-attention
        B, L, C = h.shape
        h_normed = self.norm1(h)
        q = self.q_proj(h_normed).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(h_normed).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(h_normed).view(B, L, self.num_heads, self.head_dim)
        if self.learnable_rope:
            cos, sin = self._compute_learnable_rope(L, h.device)
            q, k = apply_rotary_emb(q, k, cos, sin)
        else:
            q, k = apply_rotary_emb(q, k, rope_cos, rope_sin)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # GAB/TAB
        template_bias = None
        extra_kq = None
        if (self.use_gab or self.use_tab) and gab_tab_shared is not None:
            template_bias, extra_kq = self._compute_gab_bias(
                h_normed, mask_flat, gab_tab_shared)

        if template_bias is not None:
            attn_mask = (attn_mask + template_bias) if attn_mask is not None else template_bias

        scale = None
        if extra_kq is not None:
            extra_k, extra_q = extra_kq
            q = q * (1.0 / math.sqrt(self.head_dim))
            scale = 1.0
            q = torch.cat([q, extra_q], dim=-1)
            k = torch.cat([k, extra_k], dim=-1)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, scale=scale)
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, L, C)
        if self.use_gated_attn:
            attn_out = torch.sigmoid(self.attn_gate_proj(h_normed)) * attn_out
        attn_out = self.out_proj(attn_out)

        # 3. Record attn output in history
        states.append(attn_out)

        # 4. Depth attention before FFN (now sees attn output too)
        h = attn_res(states, self.mlp_res_proj, self.mlp_res_norm)

        # 5. FFN
        h_normed = self.norm2(h)
        w1_out = F.silu(self.ffn_w1(h_normed))
        wgate_out = self.ffn_wgate(h_normed)
        with torch.amp.autocast(h.device.type, enabled=False):
            ffn_hidden = (w1_out.float() * wgate_out.float()).to(h.dtype)
        mlp_out = self.ffn_w2(ffn_hidden)

        # 6. Record mlp output in history
        states.append(mlp_out)


# ---------------------------------------------------------------------------
# PolicyHead (NLC input)
# ---------------------------------------------------------------------------
class PolicyHead(nn.Module):
    """Per-position projection (board moves) + global pooling projection (pass)."""
    def __init__(self, c_in, pos_len, head_bias=False):
        super().__init__()
        self.pos_len = pos_len
        self.num_policy_outputs = 6
        self.linear_board = nn.Linear(c_in, self.num_policy_outputs, bias=head_bias)
        self.linear_pass = nn.Linear(c_in, self.num_policy_outputs, bias=head_bias)

    def forward(self, x_nlc, mask=None):
        """
        x_nlc: (N, L, C)
        mask: optional (N, L) float, 1=valid, 0=padding
        """
        N, L, _ = x_nlc.shape
        board = self.linear_board(x_nlc).permute(0, 2, 1)  # (N, 6, L)
        if mask is not None:
            # Mask-aware global average pooling for pass logits
            mask_expanded = mask.unsqueeze(-1)  # (N, L, 1)
            pooled = (x_nlc * mask_expanded).sum(dim=1) / mask.sum(dim=1, keepdim=True)
            # Mask out invalid board positions with large negative number
            board = board - (1.0 - mask.unsqueeze(1)) * 5000.0  # (N, 6, L)
        else:
            pooled = x_nlc.mean(dim=1)
        pass_logits = self.linear_pass(pooled)  # (N, 6)
        return torch.cat([board, pass_logits.unsqueeze(-1)], dim=2)  # (N, 6, L+1)


# ---------------------------------------------------------------------------
# ValueHead (NLC input, per-position + mean-pool projection)
# ---------------------------------------------------------------------------
class ValueHead(nn.Module):
    def __init__(self, c_in, num_scorebeliefs, pos_len, score_mode="mixop", head_bias=False):
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
        self.linear_sv = nn.Linear(c_in, self.n_spatial + self.n_global, bias=head_bias)

        # Score belief head
        if score_mode == "simple":
            self.linear_s_simple = nn.Linear(c_in, self.scorebelief_len, bias=head_bias)
        elif score_mode == "mix":
            self.linear_s_mix = nn.Linear(c_in, self.scorebelief_len * num_scorebeliefs + num_scorebeliefs, bias=head_bias)
        elif score_mode == "mixop":
            self.linear_s_mix = nn.Linear(c_in, self.scorebelief_len * num_scorebeliefs + num_scorebeliefs, bias=head_bias)
            self.linear_s2off = nn.Linear(1, num_scorebeliefs, bias=head_bias)
            self.linear_s2par = nn.Linear(1, num_scorebeliefs, bias=head_bias)

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

    def forward(self, x_nlc, score_parity, mask=None):
        """
        x_nlc: (N, L, C)
        score_parity: (N, 1)
        mask: optional (N, L) float, 1=valid, 0=padding
        """
        N, L, _ = x_nlc.shape
        H = W = self.pos_len

        spatial_global = self.linear_sv(x_nlc)
        spatial, global_feats = spatial_global.split([self.n_spatial, self.n_global], dim=-1)

        spatial = spatial.permute(0, 2, 1).view(N, self.n_spatial, H, W)
        if mask is not None:
            spatial = spatial * mask.view(N, 1, H, W)
        out_ownership, out_scoring, out_futurepos, out_seki = spatial.split([1, 1, 2, 4], dim=1)

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)  # (N, L, 1)
            mask_sum = mask.sum(dim=1, keepdim=True)  # (N, 1)
            global_feats = (global_feats * mask_expanded).sum(dim=1) / mask_sum
        else:
            global_feats = global_feats.mean(dim=1)
        out_value, out_misc, out_moremisc = global_feats.split([3, 10, 8], dim=-1)

        # Score belief: mask-aware mean-pool then project
        if mask is not None:
            pooled_s = (x_nlc * mask_expanded).sum(dim=1) / mask_sum
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
            out_scorebelief_logprobs = torch.logsumexp(
                out_scorebelief_logprobs + mix_log_weights.view(-1, 1, self.num_scorebeliefs), dim=2
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
    def __init__(self, config: dict, pos_len: int, score_mode: str = "mixop", varlen: bool = False,
                 attn_res: bool = False, gated_attn: bool = False, head_bias: bool = False,
                 norm_fp32: bool = True, zero_centered_norm: bool = False,
                 use_gab: bool = False, use_tab: bool = False,
                 learnable_rope: bool = False):
        super().__init__()
        self.config = config
        self.pos_len = pos_len
        self.varlen = varlen
        self.attn_res = attn_res
        self.gated_attn = gated_attn
        self.zero_centered_norm = zero_centered_norm
        self.use_gab = use_gab
        self.use_tab = use_tab
        self.c_trunk = config["hidden_size"]
        num_bin_features = get_num_bin_input_features(config)
        num_global_features = get_num_global_input_features(config)

        num_heads = config["num_heads"]
        ffn_dim = config["ffn_dim"]
        head_dim = self.c_trunk // num_heads

        # Stem
        self.stem = config.get("stem", "cnn3")
        dw_kernels = {"dw19": 19, "dw37": 37}
        if self.stem in dw_kernels:
            self.conv_spatial = nn.Conv2d(num_bin_features, self.c_trunk,
                                          kernel_size=1, bias=False)
            self.conv_dw = nn.Conv2d(self.c_trunk, self.c_trunk,
                                     kernel_size=dw_kernels[self.stem], padding="same",
                                     groups=self.c_trunk, bias=False)
        else:
            kernel_size = {"cnn1": 1, "cnn3": 3, "cnn5": 5}[self.stem]
            self.conv_spatial = nn.Conv2d(num_bin_features, self.c_trunk,
                                          kernel_size=kernel_size, padding="same", bias=False)
            self.conv_dw = None
        self.linear_global = nn.Linear(num_global_features, self.c_trunk, bias=False)

        # RoPE: fixed precomputed (default) or learnable per-head frequencies
        self.learnable_rope = learnable_rope
        if not learnable_rope:
            emb = precompute_freqs_cos_sin_2d(head_dim, pos_len)
            emb_expanded = torch.cat([emb, emb], dim=-1)
            self.register_buffer("rope_cos", emb_expanded.cos(), persistent=False)
            self.register_buffer("rope_sin", emb_expanded.sin(), persistent=False)
        else:
            self.rope_cos = None
            self.rope_sin = None

        # GAB shared module
        if use_gab:
            self.gab_template_mlp = GABTemplateMLP(
                config["gab_num_templates"], config["gab_num_fourier_features"],
                config["gab_mlp_hidden"], pos_len)
        else:
            self.gab_template_mlp = None

        # TAB shared module
        tab_num_extra_dims = 0
        if use_tab:
            tab_use_fm = config.get("tab_use_frequency_mixing", False)
            if tab_use_fm:
                self.tab_module = FrequencyMixingTABModule(
                    self.c_trunk, config["tab_c_z"], config["tab_num_templates"],
                    config["tab_num_blocks"], config["tab_dilation"], pos_len)
                tab_num_extra_dims = 2 * config["tab_c_z"]
            else:
                self.tab_module = TABModule(
                    self.c_trunk, config["tab_c_z"], config["tab_num_templates"],
                    config["tab_num_freqs"], config["tab_num_blocks"],
                    config["tab_dilation"], pos_len)
                tab_num_extra_dims = 2 * config["tab_num_freqs"]
        else:
            self.tab_module = None

        # Transformer blocks
        self.blocks = nn.ModuleList()
        for i in range(config["num_layers"]):
            self.blocks.append(TransformerBlock(
                c_main=self.c_trunk,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                use_attn_res=attn_res,
                is_first_block=(i == 0),
                use_gated_attn=gated_attn,
                norm_fp32=norm_fp32,
                zero_centered_norm=zero_centered_norm,
                use_gab=use_gab,
                use_tab=use_tab,
                gab_num_templates=config.get("gab_num_templates", 0),
                tab_num_templates=config.get("tab_num_templates", 0),
                gab_d1=config.get("gab_d1", 64),
                gab_d2=config.get("gab_d2", 64),
                tab_num_extra_dims=tab_num_extra_dims,
                learnable_rope=learnable_rope,
                pos_len=pos_len,
            ))

        # Final normalization
        if zero_centered_norm:
            NormClass = ZeroCenteredRMSNormFP32
        elif norm_fp32:
            NormClass = RMSNormFP32
        else:
            NormClass = nn.RMSNorm
        self.norm_final = NormClass(self.c_trunk, eps=1e-6)

        # Final depth attention for attn_res (aggregates all sub-layer outputs)
        if attn_res:
            self.final_attn_res_proj = nn.Linear(self.c_trunk, 1, bias=False)
            self.final_attn_res_norm = NormClass(self.c_trunk, eps=1e-6)

        # Output heads
        num_scorebeliefs = config["num_scorebeliefs"]

        self.policy_head = PolicyHead(self.c_trunk, pos_len, head_bias=head_bias)
        self.value_head = ValueHead(self.c_trunk, num_scorebeliefs, pos_len, score_mode=score_mode, head_bias=head_bias)

        # Seki dynamic weight moving average state
        self.moving_unowned_proportion_sum = 0.0
        self.moving_unowned_proportion_weight = 0.0

    def initialize(self, init_std=0.02):
        """Weight initialization.

        All Linear/Conv layers use fixed init_std.
        Output layers (out_proj, ffn_w2) additionally scale by 1/sqrt(2*num_blocks).
        GAB/TAB learnable frequencies keep their geometric/random initialization.
        """
        num_blocks = len(self.blocks)

        for name, p in self.named_parameters():
            # Skip GAB/TAB learnable frequencies — keep their init
            if "gab_freqs" in name or "rope_freqs" in name:
                continue
            if p.dim() < 2:
                if "norm" not in name:
                    nn.init.zeros_(p)
            else:
                if "attn_res_proj" in name or "mlp_res_proj" in name:
                    nn.init.zeros_(p)
                else:
                    std = init_std
                    if ".out_proj." in name or ".ffn_w2." in name:
                        std = std / math.sqrt(2.0 * num_blocks)
                    nn.init.normal_(p, mean=0.0, std=std)

    def fuse_zero_centered_norm(self):
        """Fuse zero-centered norm: replace ZeroCenteredRMSNormFP32 with RMSNormFP32.

        Each module's weight is replaced by weight + 1, producing standard RMSNorm behavior.
        """
        for name, module in list(self.named_modules()):
            if isinstance(module, ZeroCenteredRMSNormFP32):
                new_norm = RMSNormFP32(module.weight.shape[0], eps=module.eps)
                new_norm.norm.weight.data.copy_(module.weight.data + 1.0)
                parts = name.split(".")
                parent = self
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], new_norm)
        self.zero_centered_norm = False

    def _forward_trunk_impl(self, input_spatial, input_global):
        x, attn_mask, mask_flat = self._forward_stem_impl(input_spatial, input_global)

        # Precompute shared GAB/TAB data
        gab_tab_shared = {}
        if self.gab_template_mlp is not None:
            gab_tab_shared["gab_templates"] = self.gab_template_mlp(x.shape[1])
        if self.tab_module is not None:
            mask_nchw = input_spatial[:, 0:1, :, :].contiguous() if self.varlen else None
            keys, queries = self.tab_module(self._stem_nchw, mask_nchw)
            gab_tab_shared["tab_keys"] = keys
            gab_tab_shared["tab_queries"] = queries
            self._stem_nchw = None  # release reference

        x = self._forward_blocks_impl(x, attn_mask=attn_mask, mask_flat=mask_flat,
                                       gab_tab_shared=gab_tab_shared if gab_tab_shared else None)
        return x, mask_flat

    def _forward_blocks_impl(self, x, attn_mask=None, mask_flat=None, gab_tab_shared=None):
        if self.attn_res:
            return self._forward_blocks_attn_res_impl(
                x, attn_mask=attn_mask, mask_flat=mask_flat, gab_tab_shared=gab_tab_shared)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin, attn_mask=attn_mask,
                      mask_flat=mask_flat, gab_tab_shared=gab_tab_shared)
        return self.norm_final(x)

    def _forward_blocks_attn_res_impl(self, x, attn_mask=None, mask_flat=None,
                                       gab_tab_shared=None):
        states = [x]  # stem output as first state
        for block in self.blocks:
            block.forward_attn_res(
                states, self.rope_cos, self.rope_sin,
                attn_mask=attn_mask, mask_flat=mask_flat, gab_tab_shared=gab_tab_shared,
            )
        # Final depth attention to aggregate all sub-layer outputs
        h = attn_res(states, self.final_attn_res_proj, self.final_attn_res_norm)
        return self.norm_final(h)

    def _forward_stem_impl(self, input_spatial, input_global):
        N = input_spatial.shape[0]
        H = W = self.pos_len
        L = H * W

        # Extract mask from channel 0 when varlen is enabled
        if self.varlen:
            mask = input_spatial[:, 0:1, :, :].contiguous()  # (N, 1, H, W)
            mask_flat = mask.view(N, L)  # (N, L)
        else:
            mask_flat = None

        # Stem: NCHW -> NLC
        x_global = self.linear_global(input_global)
        x_spatial = self.conv_spatial(input_spatial)
        if self.conv_dw is not None:
            x_spatial = self.conv_dw(x_spatial)
        stem_nchw = x_spatial + x_global.unsqueeze(-1).unsqueeze(-1)

        # Save NCHW output for TAB module (needs spatial structure)
        if self.tab_module is not None:
            self._stem_nchw = stem_nchw
        else:
            self._stem_nchw = None

        x = stem_nchw.view(N, self.c_trunk, L).permute(0, 2, 1)

        # Additive attention mask in x.dtype (fp16/bf16 under autocast)
        if self.varlen:
            attn_mask = torch.zeros(N, 1, 1, L, device=x.device, dtype=x.dtype)
            attn_mask.masked_fill_(mask_flat.view(N, 1, 1, L) == 0, float('-inf'))
        else:
            attn_mask = None

        return x, attn_mask, mask_flat

    def forward_stem_for_onnx_export(self, input_spatial, input_global):
        x, attn_mask, mask_flat = self._forward_stem_impl(input_spatial, input_global)
        if self.varlen:
            return x.float(), mask_flat.float()
        return x.float()

    def forward_blocks_for_onnx_export(self, input_stem, mask_flat=None):
        if self.varlen and mask_flat is not None:
            N, L = mask_flat.shape
            attn_mask = torch.zeros(N, 1, 1, L, device=mask_flat.device, dtype=input_stem.dtype)
            attn_mask.masked_fill_(mask_flat.view(N, 1, 1, L) == 0, float('-inf'))
            return self._forward_blocks_impl(input_stem, attn_mask=attn_mask).float()
        return self._forward_blocks_impl(input_stem).float()

    def forward_trunk_for_onnx_export(self, input_spatial, input_global):
        x, mask_flat = self._forward_trunk_impl(input_spatial, input_global)
        if self.varlen:
            return x.float(), mask_flat.float()
        return x.float()

    def forward(self, input_spatial, input_global):
        """
        input_spatial: (N, C_bin, H, W)
        input_global:  (N, C_global)
        """
        x, mask_flat = self._forward_trunk_impl(input_spatial, input_global)

        # Output heads in fp32.
        with torch.amp.autocast(x.device.type, enabled=False):
            x_fp32 = x.float()
            out_policy = self.policy_head(x_fp32, mask=mask_flat)
            (
                out_value, out_misc, out_moremisc,
                out_ownership, out_scoring, out_futurepos, out_seki,
                out_scorebelief,
            ) = self.value_head(x_fp32, input_global[:, -1:].float(), mask=mask_flat)

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
