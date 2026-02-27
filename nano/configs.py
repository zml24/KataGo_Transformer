"""Transformer-only model configurations for KataGo nano training."""

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


# ---------------------------------------------------------------------------
# Transformer configs (RoPE + SwiGLU)
# ---------------------------------------------------------------------------

# ~1.3M params — tiny, for debugging
b11c96h3tfrs = {
    "version": 15,
    "norm_kind": "fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels": 96,
    "mid_num_channels": 96,
    "gpool_num_channels": 32,
    "transformer_ffn_channels": 256,
    "transformer_heads": 3,
    "transformer_kv_heads": 3,
    "use_attention_pool": False,
    "num_attention_pool_heads": 4,
    "block_kind": [["rconv%d" % (i+1), "transformerropesg"] for i in range(11)],
    "p1_num_channels": 32,
    "g1_num_channels": 32,
    "v1_num_channels": 32,
    "sbv2_num_channels": 48,
    "num_scorebeliefs": 4,
    "v2_size": 64,
}

# ~6M params
b14c192h6tfrs = {
    "version": 15,
    "norm_kind": "fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels": 192,
    "mid_num_channels": 192,
    "gpool_num_channels": 32,
    "transformer_ffn_channels": 512,
    "transformer_heads": 6,
    "transformer_kv_heads": 6,
    "use_attention_pool": False,
    "num_attention_pool_heads": 4,
    "block_kind": [["rconv%d" % (i+1), "transformerropesg"] for i in range(14)],
    "p1_num_channels": 32,
    "g1_num_channels": 32,
    "v1_num_channels": 32,
    "sbv2_num_channels": 80,
    "num_scorebeliefs": 8,
    "v2_size": 96,
}

# ~20M params
b12c384h12tfrs = {
    "version": 15,
    "norm_kind": "fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels": 384,
    "mid_num_channels": 384,
    "gpool_num_channels": 64,
    "transformer_ffn_channels": 1024,
    "transformer_heads": 12,
    "transformer_kv_heads": 12,
    "use_attention_pool": False,
    "num_attention_pool_heads": 4,
    "block_kind": [["rconv%d" % (i+1), "transformerropesg"] for i in range(12)],
    "p1_num_channels": 48,
    "g1_num_channels": 48,
    "v1_num_channels": 96,
    "sbv2_num_channels": 112,
    "num_scorebeliefs": 8,
    "v2_size": 128,
}

# ~85M params — train_simple.sh default
b12c768h12tfrs = {
    "version": 15,
    "norm_kind": "fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels": 768,
    "mid_num_channels": 768,
    "gpool_num_channels": 128,
    "transformer_ffn_channels": 2048,
    "transformer_heads": 12,
    "transformer_kv_heads": 12,
    "use_attention_pool": False,
    "num_attention_pool_heads": 4,
    "block_kind": [["rconv%d" % (i+1), "transformerropesg"] for i in range(12)],
    "p1_num_channels": 64,
    "g1_num_channels": 64,
    "v1_num_channels": 128,
    "sbv2_num_channels": 128,
    "num_scorebeliefs": 8,
    "v2_size": 144,
}

# ~300M params
b24c1024h16tfrs = {
    "version": 15,
    "norm_kind": "fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels": 1024,
    "mid_num_channels": 1024,
    "gpool_num_channels": 128,
    "transformer_ffn_channels": 3072,
    "transformer_heads": 16,
    "transformer_kv_heads": 16,
    "use_attention_pool": False,
    "num_attention_pool_heads": 4,
    "block_kind": [["rconv%d" % (i+1), "transformerropesg"] for i in range(24)],
    "p1_num_channels": 64,
    "g1_num_channels": 64,
    "v1_num_channels": 128,
    "sbv2_num_channels": 128,
    "num_scorebeliefs": 8,
    "v2_size": 144,
}

# ---------------------------------------------------------------------------
# config_of_name: same access pattern as the original modelconfigs.py
# ---------------------------------------------------------------------------
config_of_name = {
    "b11c96h3tfrs": b11c96h3tfrs,
    "b14c192h6tfrs": b14c192h6tfrs,
    "b12c384h12tfrs": b12c384h12tfrs,
    "b12c768h12tfrs": b12c768h12tfrs,
    "b24c1024h16tfrs": b24c1024h16tfrs,
}
