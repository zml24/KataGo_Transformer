#!/usr/bin/env python3
"""Generate a random TransformerEngine checkpoint and export ONNX in multiple modes.

Usage:
    python test/validate_te_official_export.py
    python test/validate_te_official_export.py --config b24c1024 --verify-onnxruntime
"""

import argparse
import os
import random
import shutil
import subprocess
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import configs
from export_onnx import export, verify

DEFAULT_MODES = ["te-official", "te-decomposed", "legacy"]


def _mode_slug(mode):
    return mode.replace("-", "_")


def _resolve_mode_artifact_path(base_path, output_dir, config_name, mode, ext, single_mode):
    mode_suffix = _mode_slug(mode)
    if base_path is None:
        return os.path.join(output_dir, f"{config_name}_{mode_suffix}{ext}")

    root, current_ext = os.path.splitext(base_path)
    if not current_ext:
        root = base_path
        current_ext = ext
    if single_mode:
        return root + current_ext
    return f"{root}_{mode_suffix}{current_ext}"


def _make_export_args(args, checkpoint_path, onnx_path, method, enable_nested_fallbacks):
    return argparse.Namespace(
        checkpoint=checkpoint_path,
        output=onnx_path,
        method=method,
        device=args.device,
        pos_len=args.pos_len,
        score_mode=args.score_mode,
        opset=args.opset,
        dynamic_batch=args.dynamic_batch,
        verify=False,
        ort_provider=args.ort_provider,
        fallback_to_te_decomposed_on_te_export_error=(
            enable_nested_fallbacks and args.fallback_to_te_decomposed_on_te_export_error
        ),
        fallback_to_legacy_on_te_export_error=(
            enable_nested_fallbacks and args.fallback_to_legacy_on_te_export_error
        ),
        use_te=(method == "legacy"),
        use_ema=False,
    )


def _save_random_te_checkpoint(args, checkpoint_path):
    try:
        from model_te import Model as TEModel
    except ImportError as exc:
        print(
            "ERROR: failed to import model_te / transformer_engine.pytorch for TE validation.\n"
            f"Original import error: {exc}\n"
            "Hint: top-level `import transformer_engine` is not enough. "
            "This script needs `import transformer_engine.pytorch as te` to work."
        )
        raise SystemExit(1) from exc

    model_config = configs.config_of_name[args.config]
    print(f"Config: {args.config} -> {model_config}")
    print(f"Seed: {args.seed}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = TEModel(model_config, args.pos_len, score_mode=args.score_mode)
    model.initialize(init_std=args.init_std)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters (TE): {num_params:,}")

    torch.save({"model": model.state_dict(), "config": model_config}, checkpoint_path)
    print(f"Saved random TE checkpoint: {checkpoint_path}")

    return model_config


def _maybe_run_trtexec(args, onnx_path, engine_path, model_config, mode):
    if args.skip_trtexec:
        print(f"Skipping TensorRT build validation for {mode} because --skip-trtexec was set")
        return

    trtexec_path = shutil.which(args.trtexec_bin)
    if trtexec_path is None:
        print(f"Skipping TensorRT build validation for {mode} because {args.trtexec_bin!r} was not found in PATH")
        return

    num_bin = configs.get_num_bin_input_features(model_config)
    num_global = configs.get_num_global_input_features(model_config)
    batch = args.batch_size
    shapes = (
        f"input_spatial:{batch}x{num_bin}x{args.pos_len}x{args.pos_len},"
        f"input_global:{batch}x{num_global}"
    )
    cmd = [
        trtexec_path,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--minShapes={shapes}",
        f"--optShapes={shapes}",
        f"--maxShapes={shapes}",
        "--skipInference",
    ]

    print(f"Running TensorRT build validation for {mode}:")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"TensorRT engine saved to: {engine_path}")


def _verify_export(args, mode, onnx_path, model, input_spatial, input_global):
    if not args.verify_onnxruntime:
        return

    atol = 1e-5 if mode == "legacy" else 1e-4
    rtol = 1e-5 if mode == "legacy" else 1e-4
    verify(
        onnx_path,
        model,
        input_spatial,
        input_global,
        provider=args.ort_provider,
        atol=atol,
        rtol=rtol,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate a random TE checkpoint and export ONNX with multiple methods"
    )
    parser.add_argument("--config", default="b24c1024", choices=list(configs.config_of_name.keys()),
                        help="Model config to generate (default: b24c1024)")
    parser.add_argument("--output-dir", default=os.path.join(ROOT, "test", "models"),
                        help="Directory for generated checkpoint/onnx/engine files")
    parser.add_argument("--checkpoint", default=None,
                        help="Output checkpoint path (default: <output-dir>/<config>_te_random.ckpt)")
    parser.add_argument("--output", default=None,
                        help="Output ONNX path. With multiple modes, mode suffixes are appended automatically")
    parser.add_argument("--engine", default=None,
                        help="TensorRT engine path. With multiple modes, mode suffixes are appended automatically")
    parser.add_argument("--modes", nargs="+", choices=DEFAULT_MODES, default=DEFAULT_MODES,
                        help="Export modes to run in order (default: te-official te-decomposed legacy)")
    parser.add_argument("--device", default="cuda",
                        help="Torch device for official TE export (default: cuda)")
    parser.add_argument("--pos-len", type=int, default=19, help="Board size (default: 19)")
    parser.add_argument("--score-mode", type=str, default="simple",
                        choices=["mixop", "mix", "simple"], help="Score belief head mode")
    parser.add_argument("--opset", type=int, default=None,
                        help="ONNX opset version passed to export_onnx.py (default: PyTorch exporter default)")
    parser.add_argument("--dynamic-batch", action="store_true",
                        help="Enable dynamic batch shapes during te-official export")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed (default: 1234)")
    parser.add_argument("--init-std", type=float, default=0.02,
                        help="Initialization std passed to model.initialize() (default: 0.02)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="TensorRT validation batch size (default: 1)")
    parser.add_argument("--verify-onnxruntime", action="store_true",
                        help="Also compare PyTorch and ONNX outputs with onnxruntime")
    parser.add_argument("--ort-provider", default="CPUExecutionProvider",
                        help="onnxruntime provider used by --verify-onnxruntime")
    parser.add_argument("--trtexec-bin", default="trtexec",
                        help="trtexec binary name or absolute path (default: trtexec)")
    parser.add_argument("--skip-trtexec", action="store_true",
                        help="Do not run TensorRT build validation")
    parser.add_argument("--fallback-to-te-decomposed-on-te-export-error", action="store_true",
                        help="If the official TE export fails, retry with a decomposed TE export path before considering legacy export")
    parser.add_argument("--fallback-to-legacy-on-te-export-error", action="store_true",
                        help="If TE-based export fails, fall back to legacy export so TensorRT validation can continue")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = args.checkpoint or os.path.join(args.output_dir, f"{args.config}_te_random.ckpt")
    model_config = _save_random_te_checkpoint(args, checkpoint_path)
    enable_nested_fallbacks = len(args.modes) == 1

    results = []
    for mode in args.modes:
        onnx_path = _resolve_mode_artifact_path(
            args.output, args.output_dir, args.config, mode, ".onnx", single_mode=enable_nested_fallbacks
        )
        engine_path = _resolve_mode_artifact_path(
            args.engine, args.output_dir, args.config, mode, ".plan", single_mode=enable_nested_fallbacks
        )

        print(f"\n=== Export mode: {mode} ===")
        export_args = _make_export_args(args, checkpoint_path, onnx_path, mode, enable_nested_fallbacks)
        try:
            onnx_path, model, input_spatial, input_global = export(export_args)
            _verify_export(args, mode, onnx_path, model, input_spatial, input_global)
            _maybe_run_trtexec(args, onnx_path, engine_path, model_config, mode)
            results.append((mode, "OK", onnx_path))
        except RuntimeError as exc:
            print(f"ERROR: mode {mode} failed")
            print(exc)
            results.append((mode, "FAILED", str(exc)))

    print("\nExport summary:")
    any_success = False
    for mode, status, detail in results:
        print(f"  {mode:14s} {status:7s} {detail}")
        if status == "OK":
            any_success = True

    if not any_success:
        raise SystemExit(1)

    print("Validation flow finished")


if __name__ == "__main__":
    main()
