#!/usr/bin/env python3
"""Export KataGo nano model to ONNX format.

Usage:
    python export_onnx.py --checkpoint /path/to/checkpoint.ckpt
    python export_onnx.py --checkpoint /path/to/checkpoint.ckpt --output model.onnx --verify
    python export_onnx.py --checkpoint /path/to/checkpoint.ckpt --method te-official --device cuda
"""

import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

from configs import get_num_bin_input_features, get_num_global_input_features, migrate_config
from model import Model


# ---------------------------------------------------------------------------
# Patch nn.RMSNorm.forward so that the legacy exporter sees only basic math ops
# instead of aten::rms_norm, which is still problematic in some ONNX paths.
# ---------------------------------------------------------------------------
_original_rms_norm_forward = None
if hasattr(nn, "RMSNorm"):
    _original_rms_norm_forward = nn.RMSNorm.forward

    def _manual_rms_norm_forward(self, x):
        x_f32 = x.float()
        mean_sq = (x_f32 * x_f32).mean(-1, keepdim=True)
        inv_rms = torch.rsqrt(mean_sq + torch.tensor(self.eps, dtype=x_f32.dtype, device=x_f32.device))
        return (self.weight * (x_f32 * inv_rms)).type_as(x)

    nn.RMSNorm.forward = _manual_rms_norm_forward


INPUT_NAMES = ["input_spatial", "input_global"]
BLOCKS_INPUT_NAMES = ["input_stem"]
DEFAULT_ONNX_OPSET = 25
FULL_OUTPUT_NAMES = [
    "out_policy",       # (N, 6, L+1)
    "out_value",        # (N, 3)
    "out_misc",         # (N, 10)
    "out_moremisc",     # (N, 8)
    "out_ownership",    # (N, 1, H, W)
    "out_scoring",      # (N, 1, H, W)
    "out_futurepos",    # (N, 2, H, W)
    "out_seki",         # (N, 4, H, W)
    "out_scorebelief",  # (N, scorebelief_len)
]
STEM_OUTPUT_NAMES = [
    "out_stem",         # (N, L, C)
]
BLOCKS_OUTPUT_NAMES = [
    "out_blocks",       # (N, L, C)
]
TRUNK_OUTPUT_NAMES = [
    "out_trunk",        # (N, L, C)
]


def _input_names(export_scope):
    if export_scope == "blocks":
        return BLOCKS_INPUT_NAMES
    return INPUT_NAMES


def _output_names(export_scope):
    if export_scope == "full":
        return FULL_OUTPUT_NAMES
    if export_scope == "stem":
        return STEM_OUTPUT_NAMES
    if export_scope == "blocks":
        return BLOCKS_OUTPUT_NAMES
    if export_scope == "trunk":
        return TRUNK_OUTPUT_NAMES
    raise ValueError(f"Unsupported export scope: {export_scope}")


class ExportWrapper(nn.Module):
    def __init__(self, model, export_scope="full"):
        super().__init__()
        self.model = model
        self.export_scope = export_scope
        self._export_scope = export_scope
        self._export_input_names = _input_names(export_scope)
        self._export_output_names = _output_names(export_scope)

    def forward(self, input0, input1=None):
        if self.export_scope == "stem":
            export_stem = getattr(self.model, "forward_stem_for_onnx_export", None)
            if export_stem is None:
                raise RuntimeError(
                    f"Model {type(self.model).__name__} does not implement forward_stem_for_onnx_export()"
                )
            return (export_stem(input0, input1),)
        if self.export_scope == "blocks":
            export_blocks = getattr(self.model, "forward_blocks_for_onnx_export", None)
            if export_blocks is None:
                raise RuntimeError(
                    f"Model {type(self.model).__name__} does not implement forward_blocks_for_onnx_export()"
                )
            return (export_blocks(input0),)
        if self.export_scope == "trunk":
            export_trunk = getattr(self.model, "forward_trunk_for_onnx_export", None)
            if export_trunk is None:
                raise RuntimeError(
                    f"Model {type(self.model).__name__} does not implement forward_trunk_for_onnx_export()"
                )
            return (export_trunk(input0, input1),)
        export_forward = getattr(self.model, "forward_for_onnx_export", None)
        if export_forward is None:
            return self.model(input0, input1)
        return export_forward(input0, input1)


def _load_checkpoint(args):
    print(f"Loading checkpoint: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = migrate_config(state["config"])
    print(f"Model config: {config}")
    print(f"pos_len={args.pos_len}, score_mode={args.score_mode}, method={args.method}")
    return state, config


def _resolve_output_path(args):
    output_path = args.output
    if output_path is None:
        output_path = os.path.join(os.path.dirname(args.checkpoint), "model.onnx")
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    return output_path


def _resolve_model_state(state, use_ema):
    model_state = dict(state["model"])
    if not use_ema:
        return model_state

    ema_shadow = state.get("ema_shadow")
    if ema_shadow is None:
        print("ERROR: --use-ema specified but checkpoint has no ema_shadow state")
        sys.exit(1)

    for name, tensor in ema_shadow.items():
        if name in model_state:
            model_state[name] = tensor
    print(f"Using EMA weights ({len(ema_shadow)} parameters)")
    return model_state


def _looks_like_te_checkpoint(model_state):
    return any(".layer.self_attention." in key for key in model_state)


def _make_dummy_inputs(config, pos_len, device, batch_size=1):
    num_bin = get_num_bin_input_features(config)
    num_global = get_num_global_input_features(config)
    input_spatial = torch.randn(batch_size, num_bin, pos_len, pos_len, device=device)
    input_global = torch.randn(batch_size, num_global, device=device)
    return input_spatial, input_global


def _make_blocks_input(model, input_spatial, input_global):
    with torch.inference_mode():
        return model.forward_stem_for_onnx_export(input_spatial, input_global)


def _resolve_export_inputs(model, export_scope, input_spatial, input_global):
    if export_scope == "blocks":
        input_stem = _make_blocks_input(model, input_spatial, input_global)
        return (input_stem,), input_stem, None
    return (input_spatial, input_global), input_spatial, input_global


def _legacy_dynamic_axes():
    dynamic_axes = {"input_spatial": {0: "batch"}, "input_global": {0: "batch"}}
    for name in FULL_OUTPUT_NAMES:
        dynamic_axes[name] = {0: "batch"}
    return dynamic_axes


def _dynamic_axes_for_scope(export_scope):
    if export_scope == "full":
        return _legacy_dynamic_axes()
    dynamic_axes = {_output_names(export_scope)[0]: {0: "batch"}}
    for name in _input_names(export_scope):
        dynamic_axes[name] = {0: "batch"}
    return dynamic_axes


def _te_dynamic_shapes(export_scope):
    batch = torch.export.Dim("batch")
    if export_scope == "blocks":
        return ({0: batch},)
    return (
        {0: batch},
        {0: batch},
    )


def _print_param_count(model):
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")


def convert_trt_fp8_to_standard_qdq(onnx_path):
    """Replace trt::TRT_FP8QuantizeLinear/DequantizeLinear with standard ONNX QuantizeLinear/DequantizeLinear.

    TE's translation table emits custom ``trt::TRT_FP8*`` ops that trigger a
    Myelin compiler bug in TRT 10.15.  Standard ONNX Q/DQ ops with FP8 types
    are handled by TRT's native quantization path and avoid the crash.

    Besides renaming the ops, this also fixes the quantized tensor element
    types from UINT8 (TE's container type) to FLOAT8E4M3FN so that TRT
    recognises them as FP8 rather than INT8.
    """
    import onnx
    from onnx import TensorProto

    model = onnx.load(onnx_path)

    # Pass 1: rename ops and collect quantized tensor names.
    fp8_tensor_names = set()
    converted = 0
    for node in model.graph.node:
        if node.domain == "trt" and node.op_type == "TRT_FP8QuantizeLinear":
            node.domain = ""
            node.op_type = "QuantizeLinear"
            fp8_tensor_names.update(node.output)
            converted += 1
        elif node.domain == "trt" and node.op_type == "TRT_FP8DequantizeLinear":
            node.domain = ""
            node.op_type = "DequantizeLinear"
            converted += 1

    # Pass 2: fix element types from UINT8 -> FLOAT8E4M3FN in value_info.
    type_fixed = 0
    for vi in model.graph.value_info:
        if vi.name in fp8_tensor_names and vi.type.tensor_type.elem_type == TensorProto.UINT8:
            vi.type.tensor_type.elem_type = TensorProto.FLOAT8E4M3FN
            type_fixed += 1

    if converted > 0:
        onnx.save(model, onnx_path, save_as_external_data=True, all_tensors_to_one_file=True,
                  location=os.path.basename(onnx_path) + ".data")
        print(f"Converted {converted} trt::TRT_FP8 ops to standard QuantizeLinear/DequantizeLinear")
        if type_fixed > 0:
            print(f"  Updated {type_fixed} tensor types from UINT8 to FLOAT8E4M3FN")
    return converted


def _collect_export_artifacts(output_path):
    artifacts = []
    for path in (output_path, output_path + ".data"):
        if os.path.exists(path):
            artifacts.append(path)
    return artifacts


def _save_summary(output_path):
    artifacts = _collect_export_artifacts(output_path)
    if not artifacts:
        print(f"Saved: {output_path}")
        return

    output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    if len(artifacts) == 1:
        print(f"Saved: {output_path} ({output_size_mb:.1f} MB)")
        return

    total_size_mb = sum(os.path.getsize(path) for path in artifacts) / (1024 * 1024)
    print(f"Saved: {output_path} ({output_size_mb:.1f} MB, total with external data {total_size_mb:.1f} MB)")
    for path in artifacts[1:]:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  External data: {path} ({size_mb:.1f} MB)")


def _resolve_te_support():
    try:
        import transformer_engine.pytorch as te
    except ImportError as exc:
        raise RuntimeError(
            "Transformer Engine is required for TE-based export. "
            "Install transformer-engine[pytorch] on a CUDA machine first. "
            f"Original import error: {exc}"
        ) from exc

    try:
        from transformer_engine.pytorch.export import te_translation_table
    except ImportError as exc:
        raise RuntimeError(
            "This Transformer Engine build does not expose "
            f"transformer_engine.pytorch.export.te_translation_table. Original import error: {exc}"
        ) from exc

    te_onnx_export = getattr(te, "onnx_export", None)
    if te_onnx_export is None:
        try:
            from transformer_engine.pytorch import onnx_export as te_onnx_export
        except ImportError as exc:
            raise RuntimeError("Unable to locate Transformer Engine ONNX export context manager.") from exc

    return te, te_onnx_export, te_translation_table


def _resolve_te_device(device_arg):
    if device_arg is not None:
        device = torch.device(device_arg)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("TE-based export requires a CUDA device, for example --device cuda or --device cuda:0.")
    return device


def _resolve_te_autocast_config(args):
    config = {
        "enabled": False,
        "recipe": None,
        "description": "disabled",
    }
    if not getattr(args, "use_fp8", False):
        return config

    recipe_name = getattr(args, "fp8_recipe", "float8-current-scaling")
    if recipe_name != "float8-current-scaling":
        raise RuntimeError(f"Unsupported --fp8-recipe value: {recipe_name}")

    try:
        from transformer_engine.common.recipe import Float8CurrentScaling, Format
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import Transformer Engine FP8 recipe support for TE-based export. "
            f"Original import error: {exc}"
        ) from exc

    config["enabled"] = True
    config["recipe"] = Float8CurrentScaling(fp8_format=Format.HYBRID)
    config["description"] = "Float8CurrentScaling(HYBRID)"
    return config


def _te_autocast_ctx(te, autocast_config):
    kwargs = {"enabled": autocast_config["enabled"]}
    if autocast_config["recipe"] is not None:
        kwargs["recipe"] = autocast_config["recipe"]
    return te.autocast(**kwargs)


def _te_fp8_exec_shape_is_valid(*shape_dims):
    if not shape_dims:
        return False
    return math.prod(shape_dims[:-1]) % 8 == 0 and shape_dims[-1] % 16 == 0


def _fp8_aligned_batch_size(pos_len, hidden_size):
    """Return the minimum batch size that satisfies TE FP8 alignment requirements.

    TE FP8 requires prod(shape[:-1]) % 8 == 0 and shape[-1] % 16 == 0.
    For activations shaped (batch, seq_len, hidden_size) where seq_len = pos_len^2,
    we need batch * pos_len^2 to be divisible by 8.
    """
    if hidden_size % 16 != 0:
        return None
    seq_len = pos_len * pos_len
    return 8 // math.gcd(seq_len, 8)


def _maybe_disable_te_fp8_for_export(autocast_config, *, batch_size, seq_len, hidden_size, export_label):
    if not autocast_config["enabled"]:
        return autocast_config

    logical_shape = (int(batch_size), int(seq_len), int(hidden_size))
    flattened_shape = (math.prod(logical_shape[:-1]), logical_shape[-1])
    if _te_fp8_exec_shape_is_valid(*logical_shape):
        return autocast_config

    print(
        f"WARNING: requested --use-fp8 for {export_label}, but the export trace uses TE activation shape "
        f"{list(flattened_shape)} from logical shape {list(logical_shape)}. "
        "Transformer Engine FP8 requires prod(shape[:-1]) % 8 == 0 and shape[-1] % 16 == 0. "
        "Falling back to non-FP8 TE autocast for this export. "
        "Use a trace shape where batch * pos_len^2 is divisible by 8 to keep FP8 enabled."
    )

    downgraded = dict(autocast_config)
    downgraded["enabled"] = False
    downgraded["recipe"] = None
    downgraded["description"] = "disabled (export trace shape is not FP8-aligned)"
    return downgraded


def _validate_te_load_result(load_result):
    missing = [key for key in load_result.missing_keys if "_extra_state" not in key]
    unexpected = [key for key in load_result.unexpected_keys if "_extra_state" not in key]
    if missing or unexpected:
        print("ERROR: failed to load TE checkpoint cleanly for official ONNX export")
        if missing:
            print(f"  missing keys: {missing}")
        if unexpected:
            print(f"  unexpected keys: {unexpected}")
        sys.exit(1)


def _make_legacy_fallback_args(args):
    fallback_args = argparse.Namespace(**vars(args))
    fallback_args.method = "legacy"
    fallback_args.use_te = True
    return fallback_args


def _make_te_decomposed_fallback_args(args):
    fallback_args = argparse.Namespace(**vars(args))
    fallback_args.method = "te-decomposed"
    return fallback_args


def _export_legacy(args, state, config):
    model_state = _resolve_model_state(state, args.use_ema)
    should_try_te_conversion = args.use_te or _looks_like_te_checkpoint(model_state)
    if should_try_te_conversion:
        try:
            from model_te import detect_checkpoint_format, convert_checkpoint_te_to_model
        except ImportError as exc:
            if _looks_like_te_checkpoint(model_state):
                print("ERROR: legacy export detected a TE checkpoint but could not import model_te for conversion")
            else:
                print("ERROR: --use-te requires Transformer Engine and model_te.py dependencies to be installed")
            raise SystemExit(1) from exc
        if detect_checkpoint_format(model_state) == "te":
            print("Converting TE checkpoint to model.py format for legacy ONNX export")
            model_state = convert_checkpoint_te_to_model(model_state)

    model = Model(config, args.pos_len, score_mode=args.score_mode)
    model.load_state_dict(model_state)
    model.eval()
    _print_param_count(model)

    input_spatial, input_global = _make_dummy_inputs(config, args.pos_len, device="cpu")
    output_path = _resolve_output_path(args)
    opset_version = DEFAULT_ONNX_OPSET if args.opset is None else args.opset
    output_names = _output_names(args.export_scope)
    wrapper = ExportWrapper(model, args.export_scope).eval()
    export_inputs, verify_input0, verify_input1 = _resolve_export_inputs(
        model, args.export_scope, input_spatial, input_global
    )

    print(f"Exporting ONNX with legacy exporter (opset {opset_version}) ...")
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            export_inputs,
            output_path,
            input_names=_input_names(args.export_scope),
            output_names=output_names,
            dynamic_axes=_dynamic_axes_for_scope(args.export_scope),
            opset_version=opset_version,
            do_constant_folding=True,
            dynamo=False,
        )

    _save_summary(output_path)
    return output_path, wrapper, verify_input0, verify_input1


def _export_te_official(args, state, config):
    te, te_onnx_export, te_translation_table = _resolve_te_support()
    from model_te import Model as TEModel, convert_checkpoint_model_to_te, detect_checkpoint_format

    device = _resolve_te_device(args.device)
    autocast_config = _resolve_te_autocast_config(args)
    model_state = _resolve_model_state(state, args.use_ema)
    if detect_checkpoint_format(model_state) == "pt":
        print("Converting model.py checkpoint to TE format for official TE ONNX export")
        model_state = convert_checkpoint_model_to_te(model_state)

    model = TEModel(config, args.pos_len, score_mode=args.score_mode, use_fp8=args.use_fp8)
    load_result = model.load_state_dict(model_state, strict=False)
    _validate_te_load_result(load_result)
    model.eval()
    model.to(device)
    _print_param_count(model)

    batch_size = 1
    if autocast_config["enabled"]:
        aligned = _fp8_aligned_batch_size(args.pos_len, config["hidden_size"])
        if aligned is not None and aligned > 1:
            batch_size = aligned
            print(
                f"Using batch_size={batch_size} for FP8-aligned export trace "
                f"(batch * pos_len^2 = {batch_size * args.pos_len * args.pos_len}, divisible by 8)"
            )
    input_spatial, input_global = _make_dummy_inputs(config, args.pos_len, device=device, batch_size=batch_size)
    autocast_config = _maybe_disable_te_fp8_for_export(
        autocast_config,
        batch_size=input_spatial.shape[0],
        seq_len=args.pos_len * args.pos_len,
        hidden_size=config["hidden_size"],
        export_label="official TE export",
    )
    output_names = _output_names(args.export_scope)
    wrapper = ExportWrapper(model, args.export_scope).eval()
    output_path = _resolve_output_path(args)
    export_inputs, verify_input0, verify_input1 = _resolve_export_inputs(
        model, args.export_scope, input_spatial, input_global
    )

    export_kwargs = {
        "input_names": _input_names(args.export_scope),
        "output_names": output_names,
        "dynamo": True,
        "fallback": False,
        "custom_translation_table": te_translation_table,
    }
    if args.dynamic_batch:
        export_kwargs["dynamic_shapes"] = _te_dynamic_shapes(args.export_scope)
    if args.opset is not None:
        export_kwargs["opset_version"] = args.opset

    print("Running one TE eager forward pass before export ...")
    print(f"Using TE autocast for official export: {autocast_config['description']}")
    with torch.no_grad(), _te_autocast_ctx(te, autocast_config):
        wrapper(*export_inputs)

    opset_desc = f"opset {DEFAULT_ONNX_OPSET if args.opset is None else args.opset}"
    print(f"Exporting ONNX with Transformer Engine official exporter ({opset_desc}) ...")
    with torch.no_grad(), _te_autocast_ctx(te, autocast_config):
        with te_onnx_export(enabled=True):
            torch.onnx.export(
                wrapper,
                export_inputs,
                output_path,
                **export_kwargs,
            )

    _save_summary(output_path)
    return output_path, wrapper, verify_input0, verify_input1


def _export_te_decomposed(args, state, config):
    te, te_onnx_export, te_translation_table = _resolve_te_support()
    from model_te import (
        ModelDecomposedExport,
        convert_checkpoint_model_to_te_decomposed,
        convert_checkpoint_te_to_model,
        detect_checkpoint_format,
    )

    device = _resolve_te_device(args.device)
    autocast_config = _resolve_te_autocast_config(args)
    model_state = _resolve_model_state(state, args.use_ema)
    if detect_checkpoint_format(model_state) == "te":
        print("Converting TE checkpoint to model.py format for decomposed TE ONNX export")
        model_state = convert_checkpoint_te_to_model(model_state)

    model_state = convert_checkpoint_model_to_te_decomposed(model_state)
    model = ModelDecomposedExport(config, args.pos_len, score_mode=args.score_mode, use_fp8=args.use_fp8)
    load_result = model.load_state_dict(model_state, strict=False)
    _validate_te_load_result(load_result)
    model.eval()
    model.to(device)
    _print_param_count(model)

    batch_size = 1
    if autocast_config["enabled"]:
        aligned = _fp8_aligned_batch_size(args.pos_len, config["hidden_size"])
        if aligned is not None and aligned > 1:
            batch_size = aligned
            print(
                f"Using batch_size={batch_size} for FP8-aligned export trace "
                f"(batch * pos_len^2 = {batch_size * args.pos_len * args.pos_len}, divisible by 8)"
            )
    input_spatial, input_global = _make_dummy_inputs(config, args.pos_len, device=device, batch_size=batch_size)
    autocast_config = _maybe_disable_te_fp8_for_export(
        autocast_config,
        batch_size=input_spatial.shape[0],
        seq_len=args.pos_len * args.pos_len,
        hidden_size=config["hidden_size"],
        export_label="decomposed TE export",
    )
    output_names = _output_names(args.export_scope)
    wrapper = ExportWrapper(model, args.export_scope).eval()
    output_path = _resolve_output_path(args)
    export_inputs, verify_input0, verify_input1 = _resolve_export_inputs(
        model, args.export_scope, input_spatial, input_global
    )

    export_kwargs = {
        "input_names": _input_names(args.export_scope),
        "output_names": output_names,
        "dynamo": True,
        "fallback": False,
        "custom_translation_table": te_translation_table,
    }
    if args.dynamic_batch:
        export_kwargs["dynamic_shapes"] = _te_dynamic_shapes(args.export_scope)
    if args.opset is not None:
        export_kwargs["opset_version"] = args.opset

    print("Running one decomposed TE eager forward pass before export ...")
    print(f"Using TE autocast for decomposed export: {autocast_config['description']}")
    with torch.no_grad(), _te_autocast_ctx(te, autocast_config):
        wrapper(*export_inputs)

    opset_desc = f"opset {DEFAULT_ONNX_OPSET if args.opset is None else args.opset}"
    print(f"Exporting ONNX with decomposed Transformer Engine exporter ({opset_desc}) ...")
    with torch.no_grad(), _te_autocast_ctx(te, autocast_config):
        with te_onnx_export(enabled=True):
            torch.onnx.export(
                wrapper,
                export_inputs,
                output_path,
                **export_kwargs,
            )

    _save_summary(output_path)
    return output_path, wrapper, verify_input0, verify_input1


def export(args):
    state, config = _load_checkpoint(args)
    if args.method == "legacy":
        return _export_legacy(args, state, config)
    if args.method == "te-decomposed":
        try:
            return _export_te_decomposed(args, state, config)
        except RuntimeError as exc:
            if getattr(args, "fallback_to_legacy_on_te_export_error", False):
                print("\nWARNING: te-decomposed export failed, falling back to legacy export.")
                print(f"  original error: {exc}")
                return _export_legacy(_make_legacy_fallback_args(args), state, config)
            raise
    if args.method == "te-official":
        try:
            return _export_te_official(args, state, config)
        except RuntimeError as exc:
            if getattr(args, "fallback_to_te_decomposed_on_te_export_error", False):
                print("\nWARNING: te-official export failed, falling back to decomposed TE export.")
                print(f"  original error: {exc}")
                try:
                    return _export_te_decomposed(_make_te_decomposed_fallback_args(args), state, config)
                except RuntimeError as decomposed_exc:
                    if getattr(args, "fallback_to_legacy_on_te_export_error", False):
                        print("\nWARNING: te-decomposed export failed, falling back to legacy export.")
                        print(f"  original error: {decomposed_exc}")
                        return _export_legacy(_make_legacy_fallback_args(args), state, config)
                    raise RuntimeError(
                        "Both te-official and te-decomposed exports failed.\n"
                        f"te-official error: {exc}\n"
                        f"te-decomposed error: {decomposed_exc}"
                    ) from decomposed_exc
            if getattr(args, "fallback_to_legacy_on_te_export_error", False):
                print("\nWARNING: te-official export failed, falling back to legacy export.")
                print(f"  original error: {exc}")
                return _export_legacy(_make_legacy_fallback_args(args), state, config)
            raise
    raise ValueError(f"Unsupported export method: {args.method}")


def verify(onnx_path, model, input_spatial, input_global, provider="CPUExecutionProvider", atol=1e-5, rtol=1e-5):
    import onnxruntime as ort

    # Restore original RMSNorm so PyTorch inference matches training behavior exactly.
    if _original_rms_norm_forward is not None:
        nn.RMSNorm.forward = _original_rms_norm_forward

    print(f"\nVerifying with onnxruntime ({provider}) ...")
    sess = ort.InferenceSession(onnx_path, providers=[provider])
    input_names = getattr(model, "_export_input_names", INPUT_NAMES)

    with torch.inference_mode():
        if input_global is None:
            pt_outputs = model(input_spatial)
        else:
            pt_outputs = model(input_spatial, input_global)

    ort_inputs = {input_names[0]: input_spatial.detach().cpu().numpy()}
    if input_global is not None:
        ort_inputs[input_names[1]] = input_global.detach().cpu().numpy()
    ort_outputs = sess.run(None, ort_inputs)

    output_names = getattr(model, "_export_output_names", FULL_OUTPUT_NAMES)
    all_close = True
    for i, name in enumerate(output_names):
        pt_arr = pt_outputs[i].detach().float().cpu().numpy()
        ort_arr = ort_outputs[i]
        max_diff = np.max(np.abs(pt_arr - ort_arr))
        ok = np.allclose(pt_arr, ort_arr, atol=atol, rtol=rtol)
        status = "OK" if ok else "MISMATCH"
        print(f"  {name:20s} shape={str(pt_arr.shape):20s} max_diff={max_diff:.2e}  {status}")
        if not ok:
            all_close = False

    if all_close:
        print("All outputs match!")
    else:
        print("WARNING: some outputs have significant numerical differences")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Export KataGo nano model to ONNX")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.ckpt")
    parser.add_argument("--output", default=None, help="Output .onnx path (default: <checkpoint_dir>/model.onnx)")
    parser.add_argument("--method", type=str, default="legacy", choices=["legacy", "te-official", "te-decomposed"],
                        help="Export method: legacy=model.py path, te-official=Transformer Engine official ONNX export, te-decomposed=TE modules with manual RoPE")
    parser.add_argument("--device", default=None,
                        help="Torch device for te-official export (default: cuda if available)")
    parser.add_argument("--pos-len", type=int, default=19, help="Board size (default: 19)")
    parser.add_argument("--score-mode", type=str, default="simple",
                        choices=["mixop", "mix", "simple"], help="Score belief head mode")
    parser.add_argument("--export-scope", type=str, default="full", choices=["full", "stem", "blocks", "trunk"],
                        help="Export the full model, stem-only, blocks-only, or stem+blocks (default: full)")
    parser.add_argument("--opset", type=int, default=DEFAULT_ONNX_OPSET,
                        help=f"ONNX opset version (default: {DEFAULT_ONNX_OPSET})")
    parser.add_argument("--dynamic-batch", action="store_true",
                        help="Enable dynamic batch shapes for te-official export (disabled by default to match the official example)")
    parser.add_argument("--verify", action="store_true", help="Verify exported model with onnxruntime")
    parser.add_argument("--ort-provider", type=str, default="CPUExecutionProvider",
                        help="onnxruntime provider used by --verify (default: CPUExecutionProvider)")
    parser.add_argument("--fallback-to-te-decomposed-on-te-export-error", action="store_true",
                        help="If te-official export fails, retry with a decomposed TE export path that applies RoPE via plain PyTorch ops")
    parser.add_argument("--fallback-to-legacy-on-te-export-error", action="store_true",
                        help="If TE-based export fails, fall back to legacy export by converting the checkpoint to model.py format")
    parser.add_argument("--use-fp8", action="store_true",
                        help="TE-based exporters only: enable TE FP8 autocast during eager warmup and ONNX export")
    parser.add_argument("--fp8-recipe", type=str, default="float8-current-scaling",
                        choices=["float8-current-scaling"],
                        help="FP8 recipe used with --use-fp8 (default: float8-current-scaling)")
    parser.add_argument("--use-te", action="store_true",
                        help="Legacy exporter only: force conversion of a TE checkpoint before export (normally auto-detected)")
    parser.add_argument("--use-ema", action="store_true",
                        help="Export EMA shadow weights instead of training weights")
    args = parser.parse_args()

    try:
        onnx_path, model, input_spatial, input_global = export(args)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    if args.verify:
        if args.method == "legacy":
            verify(onnx_path, model, input_spatial, input_global, provider=args.ort_provider, atol=1e-5, rtol=1e-5)
        else:
            verify(onnx_path, model, input_spatial, input_global, provider=args.ort_provider, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    main()
