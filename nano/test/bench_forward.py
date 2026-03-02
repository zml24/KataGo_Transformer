#!/usr/bin/env python3
"""Forward-only MFU benchmark for KataGo nano models.

Measures pure forward pass throughput with constant data pinned on GPU.
No backward pass, no loss computation, no optimizer.

Usage:
    cd nano && python test/bench_forward.py
    python test/bench_forward.py --model-kind b12c192 --batch-size 512
    python test/bench_forward.py --no-compile
"""

import sys
import os
import argparse
import time

# Allow imports from parent nano/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

import configs
from losses import estimate_forward_flops, get_gpu_peak_tflops


def main():
    parser = argparse.ArgumentParser(description="Forward-only MFU benchmark")
    parser.add_argument("--model-kind", type=str, default="b24c1024",
                        choices=list(configs.config_of_name.keys()),
                        help="Model config preset (default: b24c1024)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pos-len", type=int, default=19)
    parser.add_argument("--warmup", type=int, default=20,
                        help="Number of warmup iterations (default: 20)")
    parser.add_argument("--iters", type=int, default=100,
                        help="Number of timed iterations (default: 100)")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile")
    parser.add_argument("--use-te", action="store_true",
                        help="Use TransformerEngine model (model_te.py)")
    parser.add_argument("--te-compile", type=str, default="safe", choices=["safe", "full"],
                        help="TE + torch.compile strategy: safe=skip TE trunk tracing, full=compile full model")
    parser.add_argument("--use-fp8", action="store_true",
                        help="Enable FP8 inference (requires --use-te and Hopper/Ada GPU)")
    parser.add_argument("--fp8-recipe", type=str, default="delayed", choices=["delayed", "current", "block"],
                        help="FP8 recipe: delayed=DelayedScaling, current=Float8CurrentScaling, block=Float8BlockScaling")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is required for this benchmark"
    device = torch.device("cuda")

    if args.use_fp8:
        assert args.use_te, "--use-fp8 requires --use-te"

    # Model setup
    if args.use_te:
        from model_te import Model
        model = Model(
            configs.config_of_name[args.model_kind], args.pos_len,
            te_compile=args.te_compile,
        )
    else:
        from model import Model
        model = Model(configs.config_of_name[args.model_kind], args.pos_len)

    model_config = configs.config_of_name[args.model_kind]
    model.initialize()
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())

    if not args.no_compile:
        compiled_model = torch.compile(model, mode="default")
    else:
        compiled_model = model

    # FLOPs and GPU info
    forward_flops = estimate_forward_flops(model_config, args.pos_len)
    gpu_peak_tflops = get_gpu_peak_tflops(device)
    gpu_name = torch.cuda.get_device_name(device)

    # Print config
    print("=" * 60)
    print(f"Forward-only MFU Benchmark")
    print("=" * 60)
    print(f"Model:          {args.model_kind}")
    print(f"  layers={model_config['num_layers']}, hidden={model_config['hidden_size']}, "
          f"heads={model_config['num_heads']}, ffn={model_config['ffn_dim']}")
    print(f"  params:       {total_params:,}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Board:          {args.pos_len}x{args.pos_len}")
    print(f"torch.compile:  {'OFF' if args.no_compile else 'ON'}")
    print(f"TransformerEngine: {'ON' if args.use_te else 'OFF'}")
    if args.use_te:
        print(f"TE compile:     {args.te_compile}")
    print(f"FP8:            {'ON' if args.use_fp8 else 'OFF'}")
    print(f"FLOPs/sample:   {forward_flops/1e9:.2f} GFLOPs")
    print(f"GPU:            {gpu_name}")
    print(f"GPU BF16 peak:  {gpu_peak_tflops:.1f} TFLOPS")
    print(f"Warmup iters:   {args.warmup}")
    print(f"Timed iters:    {args.iters}")
    print("=" * 60)

    # Constant data on GPU — no transfer overhead
    num_bin_features = configs.get_num_bin_input_features(model_config)
    num_global_features = configs.get_num_global_input_features(model_config)

    input_spatial = torch.zeros(args.batch_size, num_bin_features, args.pos_len, args.pos_len,
                                dtype=torch.float32, device=device)
    # Channel 0 is the on-board mask: all 1 for full-board positions
    input_spatial[:, 0, :, :] = 1.0
    input_spatial[:, 1:9, :, :] = torch.randint(0, 2, (args.batch_size, 8, args.pos_len, args.pos_len),
                                                  dtype=torch.float32, device=device)
    input_global = torch.randn(args.batch_size, num_global_features, dtype=torch.float32, device=device)

    # FP8 context
    import contextlib
    if args.use_fp8:
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import DelayedScaling, Float8CurrentScaling, Float8BlockScaling, Format
        if args.fp8_recipe == "delayed":
            fp8_recipe = DelayedScaling(
                fp8_format=Format.HYBRID,
                amax_history_len=1024,
                amax_compute_algo="max",
            )
        elif args.fp8_recipe == "current":
            fp8_recipe = Float8CurrentScaling(fp8_format=Format.HYBRID)
        elif args.fp8_recipe == "block":
            fp8_recipe = Float8BlockScaling()
        fp8_ctx_fn = lambda: te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe)
    else:
        fp8_ctx_fn = contextlib.nullcontext

    # Warmup (includes torch.compile tracing)
    print(f"\nWarming up ({args.warmup} iters)...", flush=True)
    with torch.inference_mode():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            with fp8_ctx_fn():
                for i in range(args.warmup):
                    _ = compiled_model(input_spatial, input_global)
    torch.cuda.synchronize()
    print("Warmup done.")

    # Timed benchmark
    print(f"Running benchmark ({args.iters} iters)...", flush=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            with fp8_ctx_fn():
                for i in range(args.iters):
                    _ = compiled_model(input_spatial, input_global)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    elapsed = t1 - t0

    # Results
    samples_per_sec = args.iters * args.batch_size / elapsed
    achieved_tflops = samples_per_sec * forward_flops / 1e12
    mfu = achieved_tflops / gpu_peak_tflops * 100.0 if gpu_peak_tflops > 0 else 0.0

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Total time:     {elapsed:.2f}s ({elapsed/args.iters*1000:.1f} ms/iter)")
    print(f"Samples/sec:    {samples_per_sec:.0f}")
    print(f"Achieved:       {achieved_tflops:.1f} TFLOPS")
    print(f"MFU:            {mfu:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
