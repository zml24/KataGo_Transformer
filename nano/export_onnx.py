#!/usr/bin/env python3
"""Export KataGo nano model to ONNX format.

Usage:
    python export_onnx.py --checkpoint /path/to/checkpoint.ckpt
    python export_onnx.py --checkpoint /path/to/checkpoint.ckpt --output model.onnx --verify
    python export_onnx.py --checkpoint /path/to/checkpoint.ckpt --method te-official --device cuda
"""

import argparse
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
OUTPUT_NAMES = [
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


class TEOfficialExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_spatial, input_global):
        export_forward = getattr(self.model, "forward_for_onnx_export", None)
        if export_forward is None:
            return self.model(input_spatial, input_global)
        return export_forward(input_spatial, input_global)


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


def _make_dummy_inputs(config, pos_len, device):
    num_bin = get_num_bin_input_features(config)
    num_global = get_num_global_input_features(config)
    input_spatial = torch.randn(1, num_bin, pos_len, pos_len, device=device)
    input_global = torch.randn(1, num_global, device=device)
    return input_spatial, input_global


def _legacy_dynamic_axes():
    dynamic_axes = {"input_spatial": {0: "batch"}, "input_global": {0: "batch"}}
    for name in OUTPUT_NAMES:
        dynamic_axes[name] = {0: "batch"}
    return dynamic_axes


def _te_dynamic_shapes():
    batch = torch.export.Dim("batch")
    return (
        {0: batch},
        {0: batch},
    )


def _print_param_count(model):
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")


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
    opset_version = 17 if args.opset is None else args.opset

    print(f"Exporting ONNX with legacy exporter (opset {opset_version}) ...")
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (input_spatial, input_global),
            output_path,
            input_names=INPUT_NAMES,
            output_names=OUTPUT_NAMES,
            dynamic_axes=_legacy_dynamic_axes(),
            opset_version=opset_version,
            do_constant_folding=True,
            dynamo=False,
        )

    _save_summary(output_path)
    return output_path, model, input_spatial, input_global


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

    input_spatial, input_global = _make_dummy_inputs(config, args.pos_len, device=device)
    wrapper = TEOfficialExportWrapper(model).eval()
    output_path = _resolve_output_path(args)

    export_kwargs = {
        "input_names": INPUT_NAMES,
        "output_names": OUTPUT_NAMES,
        "dynamo": True,
        "fallback": False,
        "custom_translation_table": te_translation_table,
    }
    if args.dynamic_batch:
        export_kwargs["dynamic_shapes"] = _te_dynamic_shapes()
    if args.opset is not None:
        export_kwargs["opset_version"] = args.opset

    print("Running one TE eager forward pass before export ...")
    print(f"Using TE autocast for official export: {autocast_config['description']}")
    with torch.no_grad(), _te_autocast_ctx(te, autocast_config):
        wrapper(input_spatial, input_global)

    opset_desc = f"opset {args.opset}" if args.opset is not None else "PyTorch default opset"
    print(f"Exporting ONNX with Transformer Engine official exporter ({opset_desc}) ...")
    with torch.no_grad(), _te_autocast_ctx(te, autocast_config):
        with te_onnx_export(enabled=True):
            torch.onnx.export(
                wrapper,
                (input_spatial, input_global),
                output_path,
                **export_kwargs,
            )

    _save_summary(output_path)
    return output_path, wrapper, input_spatial, input_global


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

    input_spatial, input_global = _make_dummy_inputs(config, args.pos_len, device=device)
    wrapper = TEOfficialExportWrapper(model).eval()
    output_path = _resolve_output_path(args)

    export_kwargs = {
        "input_names": INPUT_NAMES,
        "output_names": OUTPUT_NAMES,
        "dynamo": True,
        "fallback": False,
        "custom_translation_table": te_translation_table,
    }
    if args.dynamic_batch:
        export_kwargs["dynamic_shapes"] = _te_dynamic_shapes()
    if args.opset is not None:
        export_kwargs["opset_version"] = args.opset

    print("Running one decomposed TE eager forward pass before export ...")
    print(f"Using TE autocast for decomposed export: {autocast_config['description']}")
    with torch.no_grad(), _te_autocast_ctx(te, autocast_config):
        wrapper(input_spatial, input_global)

    opset_desc = f"opset {args.opset}" if args.opset is not None else "PyTorch default opset"
    print(f"Exporting ONNX with decomposed Transformer Engine exporter ({opset_desc}) ...")
    with torch.no_grad(), _te_autocast_ctx(te, autocast_config):
        with te_onnx_export(enabled=True):
            torch.onnx.export(
                wrapper,
                (input_spatial, input_global),
                output_path,
                **export_kwargs,
            )

    _save_summary(output_path)
    return output_path, wrapper, input_spatial, input_global


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

    with torch.inference_mode():
        pt_outputs = model(input_spatial, input_global)

    ort_inputs = {
        "input_spatial": input_spatial.detach().cpu().numpy(),
        "input_global": input_global.detach().cpu().numpy(),
    }
    ort_outputs = sess.run(None, ort_inputs)

    all_close = True
    for i, name in enumerate(OUTPUT_NAMES):
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
    parser.add_argument("--opset", type=int, default=None,
                        help="ONNX opset version (default: legacy=17, te-official=PyTorch default)")
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
