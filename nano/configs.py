"""Minimal transformer model configurations for KataGo nano training."""

from typing import Dict, Any

ModelConfig = Dict[str, Any]


def get_version(config: ModelConfig):
    return config["version"]


def get_num_bin_input_features(config: ModelConfig):
    version = get_version(config)
    if version == 10 or version == 11 or version == 12 or version == 13 or version == 14 or version == 15:
        return 22
    elif version == 101 or version == 102:
        return 22
    else:
        assert False


def get_num_global_input_features(config: ModelConfig):
    version = get_version(config)
    if version == 10 or version == 11 or version == 12 or version == 13 or version == 14 or version == 15:
        return 19
    elif version == 101 or version == 102:
        return 39
    else:
        assert False


def make_config(num_layers, hidden_size, num_heads, ffn_dim=None, num_scorebeliefs=8, version=15,
                stem="cnn3", ape="none", rpe="rope", use_gab=False):
    """Create a model config from minimal parameters.

    Args:
        num_layers: Number of transformer blocks.
        hidden_size: Hidden dimension (trunk channels).
        num_heads: Number of attention heads.
        ffn_dim: SwiGLU FFN intermediate dimension. Default: hidden_size * 8 // 3.
        num_scorebeliefs: Number of score belief mixtures. Default: 8.
        version: Data format version. Default: 15.
        stem: Stem convolution kernel size. "cnn1" (1x1), "cnn3" (3x3), "cnn5" (5x5).
        ape: Absolute position encoding. "none" (disabled), "d4" (D4-symmetric
            edge-distance embedding, 55 classes for 19x19), "per_pos" (independent
            embedding per board position, 361 for 19x19).
        rpe: Relative position encoding. "rope" (2D RoPE on Q,K),
            "rpb" (per-layer per-head scalar bias on attention logits),
            "rope+rpb" (both RoPE and RPB simultaneously).
        use_gab: Enable Geometric Attention Bias (GAB). Adds learned position-dependent
            bias to attention logits via shared Fourier templates + per-layer mixing.
            GAB hyperparams (gab_d1, gab_d2, gab_num_templates, gab_num_fourier_features,
            gab_mlp_hidden) can be overridden by setting them in the returned config dict.
    """
    if ffn_dim is None:
        ffn_dim = hidden_size * 8 // 3
    return {
        "version": version,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "ffn_dim": ffn_dim,
        "num_scorebeliefs": num_scorebeliefs,
        "stem": stem,
        "ape": ape,
        "rpe": rpe,
        "use_gab": use_gab,
    }


def _migrate_use_ape(val):
    """Convert old bool use_ape to new string ape format."""
    if isinstance(val, bool):
        return "d4" if val else "none"
    return val

def migrate_config(old: ModelConfig) -> ModelConfig:
    """Convert old-format config (with trunk_num_channels etc.) to new minimal format."""
    if "hidden_size" in old:
        old = dict(old)
        # Migrate old bool use_ape → string ape
        if "use_ape" in old:
            old["ape"] = _migrate_use_ape(old.pop("use_ape"))
        # Migrate old ape field (legacy "cnn"/"ape-stem" format) → stem + ape
        if "ape" in old and old["ape"] in ("cnn", "ape-stem"):
            ape = old.pop("ape")
            _APE_MAP = {
                "cnn": ("cnn3", "none"),
                "ape-stem": ("cnn1", "d4"),
            }
            stem, ape_mode = _APE_MAP[ape]
            old.setdefault("stem", stem)
            old.setdefault("ape", ape_mode)
        # Migrate old pos_enc → stem + ape + rpe
        if "pos_enc" in old:
            pos_enc = old.pop("pos_enc")
            _POS_ENC_MAP = {
                "rope": ("cnn3", "none", "rope"),
                "ape-stem": ("cnn1", "d4", "rope"),
                "rpb": ("cnn3", "none", "rpb"),
            }
            stem, ape_mode, rpe = _POS_ENC_MAP[pos_enc]
            old.setdefault("stem", stem)
            old.setdefault("ape", ape_mode)
            old.setdefault("rpe", rpe)
        else:
            old.setdefault("stem", "cnn3")
            old.setdefault("ape", "none")
            old.setdefault("rpe", "rope")
        return old
    return make_config(
        num_layers=len(old["block_kind"]),
        hidden_size=old["trunk_num_channels"],
        num_heads=old["transformer_heads"],
        ffn_dim=old.get("transformer_ffn_channels", old["trunk_num_channels"] * 8 // 3),
        num_scorebeliefs=old.get("num_scorebeliefs", 8),
        version=old.get("version", 15),
        stem=old.get("stem", "cnn3"),
        ape=_migrate_use_ape(old.get("use_ape", old.get("ape", "none"))),
        rpe=old.get("rpe", "rope"),
        use_gab=old.get("use_gab", False),
    )


# ---------------------------------------------------------------------------
# Predefined model configs
# ---------------------------------------------------------------------------

# ~5M params — ViT-Ti
b12c192 = make_config(12, 192, 3, ffn_dim=512)

# ~22M params — ViT-S
b12c384 = make_config(12, 384, 6, ffn_dim=1024)
b12c384_gab = make_config(12, 384, 6, ffn_dim=1024, use_gab=True)

# ~90M params — ViT-B
b12c768 = make_config(12, 768, 12, ffn_dim=2048)
b24c512 = make_config(24, 512, 8, ffn_dim=1536)

# ~330M params — ViT-L
b24c1024 = make_config(24, 1024, 16, ffn_dim=3072)

config_of_name = {
    "b12c192": b12c192,
    "b12c384": b12c384,
    "b12c384_gab": b12c384_gab,
    "b24c512": b24c512,
    "b12c768": b12c768,
    "b24c1024": b24c1024,
}
