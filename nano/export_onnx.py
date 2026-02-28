#!/usr/bin/env python3
"""Export KataGo nano model to ONNX format.

Usage:
    python export_onnx.py --checkpoint /path/to/checkpoint.ckpt
    python export_onnx.py --checkpoint /path/to/checkpoint.ckpt --output model.onnx --verify
"""

import argparse
import os
import sys

import numpy as np
import torch

from configs import get_num_bin_input_features, get_num_global_input_features, migrate_config
from model import Model


# ---------------------------------------------------------------------------
# Patch nn.RMSNorm.forward so that ONNX export sees only basic math ops
# instead of the unsupported aten::rms_norm operator.
# ---------------------------------------------------------------------------
_original_rms_norm_forward = None
if hasattr(torch.nn, "RMSNorm"):
    _original_rms_norm_forward = torch.nn.RMSNorm.forward

    def _manual_rms_norm_forward(self, x):
        x_f32 = x.float()
        mean_sq = (x_f32 * x_f32).mean(-1, keepdim=True)
        inv_rms = torch.rsqrt(mean_sq + torch.tensor(self.eps, dtype=x_f32.dtype, device=x_f32.device))
        return (self.weight * (x_f32 * inv_rms)).type_as(x)

    torch.nn.RMSNorm.forward = _manual_rms_norm_forward


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


def export(args):
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = migrate_config(state["config"])
    print(f"Model config: {config}")
    print(f"pos_len={args.pos_len}, score_mode={args.score_mode}")

    # Build model — always export via model.py's Model for ONNX compatibility
    model_state = state["model"]
    if args.use_te:
        from model_te import detect_checkpoint_format, convert_checkpoint_te_to_model
        if detect_checkpoint_format(model_state) == "te":
            print("Converting TE checkpoint to model.py format for ONNX export")
            model_state = convert_checkpoint_te_to_model(model_state)
    model = Model(config, args.pos_len, score_mode=args.score_mode)
    model.load_state_dict(model_state)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    # Dummy inputs — mask channel (channel 0) set to 1.0 so all positions are valid,
    # avoiding -inf propagation in attention masking during tracing.
    num_bin = get_num_bin_input_features(config)
    num_global = get_num_global_input_features(config)
    H = W = args.pos_len

    input_spatial = torch.randn(1, num_bin, H, W)
    input_spatial[:, 0, :, :] = 1.0
    input_global = torch.randn(1, num_global)

    # Output path
    output_path = args.output
    if output_path is None:
        output_path = os.path.join(os.path.dirname(args.checkpoint), "model.onnx")

    # Dynamic axes: batch dimension only
    dynamic_axes = {"input_spatial": {0: "batch"}, "input_global": {0: "batch"}}
    for name in OUTPUT_NAMES:
        dynamic_axes[name] = {0: "batch"}

    # Export
    print(f"Exporting ONNX (opset {args.opset}) ...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (input_spatial, input_global),
            output_path,
            input_names=["input_spatial", "input_global"],
            output_names=OUTPUT_NAMES,
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            do_constant_folding=True,
        )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved: {output_path} ({size_mb:.1f} MB)")
    return output_path, model, input_spatial, input_global


def verify(onnx_path, model, input_spatial, input_global):
    import onnxruntime as ort

    # Restore original RMSNorm so PyTorch inference matches training behavior exactly
    if _original_rms_norm_forward is not None:
        torch.nn.RMSNorm.forward = _original_rms_norm_forward

    print("\nVerifying with onnxruntime ...")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    with torch.no_grad():
        pt_outputs = model(input_spatial, input_global)

    ort_inputs = {
        "input_spatial": input_spatial.numpy(),
        "input_global": input_global.numpy(),
    }
    ort_outputs = sess.run(None, ort_inputs)

    all_close = True
    for i, name in enumerate(OUTPUT_NAMES):
        pt_arr = pt_outputs[i].numpy()
        ort_arr = ort_outputs[i]
        max_diff = np.max(np.abs(pt_arr - ort_arr))
        ok = np.allclose(pt_arr, ort_arr, atol=1e-5)
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
    parser.add_argument("--pos-len", type=int, default=19, help="Board size (default: 19)")
    parser.add_argument("--score-mode", type=str, default="simple",
                        choices=["mixop", "mix", "simple"], help="Score belief head mode")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    parser.add_argument("--verify", action="store_true", help="Verify exported model with onnxruntime")
    parser.add_argument("--use-te", action="store_true",
                        help="Checkpoint is from TE model — convert weights to model.py format before export")
    args = parser.parse_args()

    onnx_path, model, input_spatial, input_global = export(args)

    if args.verify:
        verify(onnx_path, model, input_spatial, input_global)


if __name__ == "__main__":
    main()
