#!/usr/bin/python3
"""
Minimal Transformer training script for KataGo (nano version).
Self-contained — only depends on modules within nano/.
"""
import sys
import os
import argparse
import math
import time
import logging
import json
import glob
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed
import torch.multiprocessing
from torch.nn.parallel import DistributedDataParallel
import atexit

import configs
import data as data_processing
from model import Model
from optimizers import MuonOptimizer, ShampooOptimizer
from losses import compute_loss, postprocess_and_loss_core, _METRIC_KEYS, estimate_forward_flops, get_gpu_peak_tflops


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------
def multiprocessing_setup(rank: int, world_size: int, master_port: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = f'{master_port}'
    logging.info(f"Running torch.distributed.init_process_group, rank={rank}, world_size={world_size}")
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    logging.info(f"Returned from init_process_group, rank={rank}")

def multiprocessing_cleanup():
    torch.distributed.destroy_process_group()


# ---------------------------------------------------------------------------
# Training main loop
# ---------------------------------------------------------------------------
def main(rank, world_size, args, multi_gpu_device_ids):
    # Parse td_value_loss_scales
    td_value_loss_scales = [float(x) for x in args.td_value_loss_scales.split(",")]
    assert len(td_value_loss_scales) == 3

    # Logging
    os.makedirs(args.traindir, exist_ok=True)
    logging.root.handlers = []
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(args.traindir, f"train{rank}.log"), mode="a"),
                logging.StreamHandler(),
            ],
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(args.traindir, f"train{rank}.log"), mode="a"),
            ],
        )
    logging.info(f"Args: {vars(args)}")

    # DDP init
    if world_size > 1:
        multiprocessing_setup(rank, world_size, args.master_port)
        atexit.register(multiprocessing_cleanup)

    # Device selection
    if torch.cuda.is_available():
        my_gpu_id = multi_gpu_device_ids[rank]
        torch.cuda.set_device(my_gpu_id)
        device = torch.device("cuda", my_gpu_id)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logging.info(f"Device: {device}")
    if device.type == "cuda":
        logging.info(f"GPU: {torch.cuda.get_device_name()}")

    # AMP setup
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        amp_device = "cuda"
        amp_dtype = torch.bfloat16
        use_amp = True
    elif device.type == "mps":
        amp_device = "mps"
        amp_dtype = torch.bfloat16
        use_amp = True
    else:
        amp_device = "cpu"
        amp_dtype = torch.bfloat16
        use_amp = False

    # Model config
    model_config = configs.config_of_name[args.model_kind].copy()
    logging.info(f"Model config: {json.dumps(model_config, indent=2, default=str)}")

    pos_len = args.pos_len
    batch_size = args.batch_size

    # Load or create model
    checkpoint_path = os.path.join(args.traindir, "checkpoint.ckpt")
    if os.path.exists(checkpoint_path):
        logging.info(f"Loading checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_config = state.get("config", model_config)
        model = Model(model_config, pos_len, score_mode=args.score_mode, attn_backend=args.attn_backend)
        model.load_state_dict(state["model"])
        model.moving_unowned_proportion_sum = state.get("moving_unowned_proportion_sum", 0.0)
        model.moving_unowned_proportion_weight = state.get("moving_unowned_proportion_weight", 0.0)
        global_step = state.get("global_step", 0)
        total_samples_trained = state.get("total_samples_trained", global_step * batch_size)
        logging.info(f"Resumed from step {global_step}, {total_samples_trained} samples")
    elif args.initial_checkpoint is not None:
        logging.info(f"Loading initial checkpoint: {args.initial_checkpoint}")
        state = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
        model_config = state.get("config", model_config)
        model = Model(model_config, pos_len, score_mode=args.score_mode, attn_backend=args.attn_backend)
        model.load_state_dict(state["model"])
        global_step = 0
        total_samples_trained = 0
    else:
        logging.info("Creating new model")
        model = Model(model_config, pos_len, score_mode=args.score_mode, attn_backend=args.attn_backend)
        model.initialize(init_std=args.init_std)
        logging.info(f"Initialized weights with std={args.init_std}, output_std={args.init_std / math.sqrt(2.0 * len(model.blocks)):.6f}")
        global_step = 0
        total_samples_trained = 0

    model.to(device)

    # torch.compile
    if not args.no_compile and device.type != "mps":
        compiled_model = torch.compile(model, mode="default")
        compiled_loss_fn = torch.compile(postprocess_and_loss_core, mode="reduce-overhead")
    else:
        compiled_model = model
        compiled_loss_fn = postprocess_and_loss_core

    # DDP wrapper
    if world_size > 1:
        ddp_model = DistributedDataParallel(compiled_model, device_ids=[device])
    else:
        ddp_model = compiled_model

    # Parameter stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")

    # Optimizer: split params into Muon / Shampoo / Adam groups
    muon_params = {}
    shampoo_params = {}
    adam_params = {}
    no_decay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() < 2 or "norm" in name:
            no_decay_params.append(p)
        elif args.muon_scope == "all":
            muon_params[name] = p
        elif args.muon_scope == "blocks" and "blocks." in name:
            muon_params[name] = p
        elif args.shampoo_scope == "all":
            shampoo_params[name] = p
        elif args.shampoo_scope == "blocks" and "blocks." in name:
            shampoo_params[name] = p
        else:
            adam_params[name] = p

    logging.info(f"Muon params: {sum(p.numel() for p in muon_params.values()):,}, "
                 f"Shampoo params: {sum(p.numel() for p in shampoo_params.values()):,}, "
                 f"Adam decay: {sum(p.numel() for p in adam_params.values()):,}, "
                 f"AdamW no-decay: {sum(p.numel() for p in no_decay_params):,}")

    # FLOPs estimation
    forward_flops = estimate_forward_flops(model_config, pos_len)
    train_flops_per_sample = 3 * forward_flops
    gpu_peak_tflops = get_gpu_peak_tflops(device)
    logging.info(f"FLOPs/sample (fwd): {forward_flops/1e9:.2f}G, (train): {train_flops_per_sample/1e9:.2f}G")
    if gpu_peak_tflops > 0:
        logging.info(f"GPU BF16 peak: {gpu_peak_tflops:.1f} TFLOPS")

    # torch.optim.AdamW for no_decay params (1D bias/norm) + adam_params (2D with decay)
    adam_param_groups = [
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    if adam_params:
        adam_param_groups.append({"params": list(adam_params.values()), "weight_decay": args.wd})
    optimizer = torch.optim.AdamW(adam_param_groups, lr=args.lr, betas=(0.9, 0.95), fused=True)

    # Independent optimizer instances
    muon_opt = MuonOptimizer(
        muon_params, lr_multiplier=args.muon_lr_multiplier,
        momentum=args.muon_momentum, wd=args.wd, scale_mode=args.muon_scale, device=device,
    ) if muon_params else None
    shampoo_opt = ShampooOptimizer(
        shampoo_params, lr_multiplier=args.shampoo_lr_multiplier,
        momentum=args.shampoo_momentum, wd=args.wd, beta2=args.shampoo_beta2, device=device,
    ) if shampoo_params else None

    # Restore optimizer state
    if os.path.exists(checkpoint_path):
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
            logging.info("Optimizer state loaded")
        if "muon_state" in state and muon_opt is not None:
            muon_opt.load_state_dict(state["muon_state"], device)
            logging.info("Muon state loaded")
        elif "muon_bufs" in state and muon_opt is not None:
            for k, v in state["muon_bufs"].items():
                if k in muon_opt.states:
                    muon_opt.states[k]["momentum"].copy_(v.to(device))
            logging.info("Muon momentum buffers loaded (legacy format)")
        if "shampoo_state" in state and shampoo_opt is not None:
            shampoo_opt.load_state_dict(state["shampoo_state"], device)
            logging.info("Shampoo state loaded")
    # LR schedule: linear warmup + cosine decay
    warmup_steps = args.warmup_samples // batch_size
    total_steps = args.max_training_samples // batch_size

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=global_step - 1 if global_step > 0 else -1)

    # Data directories
    train_dir = os.path.join(args.datadir, "train")
    val_dir = os.path.join(args.datadir, "val")

    # TensorBoard (rank 0 only)
    tb_writer = None
    if rank == 0:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = os.path.join(args.traindir, "tb_logs")
            tb_writer = SummaryWriter(log_dir=tb_dir)
            logging.info(f"TensorBoard: {tb_dir}")
        except ImportError:
            logging.info("TensorBoard not available")

    # Save checkpoint
    def save_checkpoint():
        state_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": model_config,
            "global_step": global_step,
            "total_samples_trained": total_samples_trained,
            "moving_unowned_proportion_sum": model.moving_unowned_proportion_sum,
            "moving_unowned_proportion_weight": model.moving_unowned_proportion_weight,
        }
        if muon_opt is not None:
            state_dict["muon_state"] = muon_opt.state_dict()
        if shampoo_opt is not None:
            state_dict["shampoo_state"] = shampoo_opt.state_dict()
        path = os.path.join(args.traindir, "checkpoint.ckpt")
        torch.save(state_dict, path + ".tmp")
        os.replace(path + ".tmp", path)
        logging.info(f"Saved checkpoint at step {global_step}, {total_samples_trained} samples")

    # Metrics accumulation
    _metric_keys = [
        "loss", "p0loss", "p1loss", "p0softloss", "p1softloss",
        "p0lopt", "p0sopt",
        "vloss", "tdvloss1", "tdvloss2", "tdvloss3", "tdsloss",
        "oloss", "sloss", "fploss", "skloss",
        "smloss", "sbcdfloss", "sbpdfloss", "sdregloss",
        "leadloss", "vtimeloss", "evstloss", "esstloss",
        "pacc1", "wsum",
    ]
    running = {k: 0.0 for k in _metric_keys}
    running["count"] = 0
    running["grad_norm"] = 0.0
    running["muon_update_rms"] = 0.0
    running["shampoo_precond_rms"] = 0.0

    def reset_running():
        for k in running:
            running[k] = 0.0

    _per_sample_keys = [k for k in _metric_keys if k not in ("loss", "wsum")]

    def print_metrics(elapsed):
        weight_sum = max(running["wsum"], 1e-10)
        batch_count = max(running["count"], 1)
        samples_per_sec = batch_count * batch_size * world_size / elapsed
        achieved_tflops_per_gpu = samples_per_sec * train_flops_per_sample / (world_size * 1e12)
        mfu = achieved_tflops_per_gpu / gpu_peak_tflops * 100.0 if gpu_peak_tflops > 0 else 0.0
        logging.info(
            f"step={global_step}, samples={total_samples_trained}, "
            f"time={elapsed:.1f}s, "
            f"lr={scheduler.get_last_lr()[0]:.2e}, "
            f"loss={running['loss'] / batch_count:.4f}, "
            f"p0loss={running['p0loss'] / weight_sum:.4f}, "
            f"vloss={running['vloss'] / weight_sum:.4f}, "
            f"oloss={running['oloss'] / weight_sum:.4f}, "
            f"skloss={running['skloss'] / weight_sum:.4f}, "
            f"pacc1={running['pacc1'] / weight_sum:.4f}, "
            f"sps={samples_per_sec:.0f}, MFU={mfu:.1f}%"
        )
        if tb_writer is not None:
            tb_writer.add_scalar("train/loss", running["loss"] / batch_count, total_samples_trained)
            for k in _per_sample_keys:
                tb_writer.add_scalar(f"train/{k}", running[k] / weight_sum, total_samples_trained)
            tb_writer.add_scalar("train/lr", scheduler.get_last_lr()[0], total_samples_trained)
            tb_writer.add_scalar("train/grad_norm", running["grad_norm"] / batch_count, total_samples_trained)
            if muon_opt is not None:
                tb_writer.add_scalar("train/muon_update_rms", running["muon_update_rms"] / batch_count, total_samples_trained)
            if shampoo_opt is not None:
                tb_writer.add_scalar("train/shampoo_precond_rms", running["shampoo_precond_rms"] / batch_count, total_samples_trained)
            tokens_per_sec = samples_per_sec * pos_len * pos_len
            tb_writer.add_scalar("perf/samples_per_sec", samples_per_sec, total_samples_trained)
            tb_writer.add_scalar("perf/tokens_per_sec", tokens_per_sec, total_samples_trained)
            tb_writer.add_scalar("perf/achieved_tflops_per_gpu", achieved_tflops_per_gpu, total_samples_trained)
            tb_writer.add_scalar("perf/mfu", mfu, total_samples_trained)

    # Start training
    logging.info("=" * 60)
    logging.info(f"Starting training: {total_samples_trained}/{args.max_training_samples} samples done")
    logging.info("=" * 60)

    last_save_samples = total_samples_trained
    last_val_samples = total_samples_trained
    reset_running()
    time_start = time.perf_counter()
    last_print_time = time_start

    while total_samples_trained < args.max_training_samples:
        model.train()

        # Find training files
        train_files = sorted(glob.glob(os.path.join(train_dir, "*.npz")))
        if not train_files:
            logging.warning(f"No training files found in {train_dir}, waiting...")
            time.sleep(10)
            continue

        np.random.shuffle(train_files)

        use_pin_memory = args.prefetch_batches > 0 and device.type == "cuda"
        train_gen = data_processing.read_npz_training_data(
            train_files,
            batch_size=batch_size,
            world_size=world_size,
            rank=rank,
            pos_len=pos_len,
            device=device,
            symmetry_type=args.symmetry_type,
            include_meta=False,
            enable_history_matrices=args.enable_history_matrices,
            model_config=model_config,
            use_pin_memory=use_pin_memory,
        )
        for batch in data_processing.prefetch_generator(train_gen, args.prefetch_batches):
            for p in model.parameters():
                p.grad = None

            with torch.amp.autocast(amp_device, dtype=amp_dtype, enabled=use_amp):
                outputs = ddp_model(batch["binaryInputNCHW"], batch["globalInputNC"])

            # Compiled postprocess + loss (seki moving average computed inside as tensor ops)
            moving_sum_t = torch.tensor(model.moving_unowned_proportion_sum, device=device)
            moving_weight_t = torch.tensor(model.moving_unowned_proportion_weight, device=device)
            loss, metrics_stack, new_moving_sum, new_moving_weight = compiled_loss_fn(
                outputs, model.value_head.score_belief_offset_vector,
                batch["binaryInputNCHW"], batch["policyTargetsNCMove"],
                batch["globalTargetsNC"], batch["scoreDistrN"], batch["valueTargetsNCHW"],
                pos_len, moving_sum_t, moving_weight_t, True,
                soft_policy_weight_scale=args.soft_policy_weight_scale,
                value_loss_scale=args.value_loss_scale,
                td_value_loss_scales=td_value_loss_scales,
                seki_loss_scale=args.seki_loss_scale,
                variance_time_loss_scale=args.variance_time_loss_scale,
                disable_optimistic_policy=args.disable_optimistic_policy,
            )

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Write back seki moving average state
            model.moving_unowned_proportion_sum = new_moving_sum.item()
            model.moving_unowned_proportion_weight = new_moving_weight.item()

            # Muon / Shampoo / Adam update
            base_lr = scheduler.get_last_lr()[0]
            if muon_opt is not None:
                muon_opt.step(base_lr)
            if shampoo_opt is not None:
                shampoo_opt.step(base_lr)
            global_step += 1
            total_samples_trained += batch_size * world_size

            # Accumulate metrics (single CUDA sync via .tolist())
            metrics = dict(zip(_METRIC_KEYS, metrics_stack.tolist()))
            for k in metrics:
                running[k] += metrics[k]
            running["grad_norm"] += grad_norm.item()
            if muon_opt is not None:
                running["muon_update_rms"] += muon_opt.last_update_rms
            if shampoo_opt is not None:
                running["shampoo_precond_rms"] += shampoo_opt.last_precond_rms
            running["count"] += 1

            if rank == 0 and global_step % args.print_every == 0:
                time_now = time.perf_counter()
                print_metrics(time_now - last_print_time)
                reset_running()
                last_print_time = time_now

            # Periodic save (rank 0 only)
            if rank == 0 and total_samples_trained - last_save_samples >= args.save_every_samples:
                save_checkpoint()
                last_save_samples = total_samples_trained

            # Validation (rank 0 only)
            if rank == 0 and total_samples_trained - last_val_samples >= args.val_every_samples:
                last_val_samples = total_samples_trained
                val_files = sorted(glob.glob(os.path.join(val_dir, "*.npz")))
                if val_files:
                    model.eval()
                    val_metrics = {k: 0.0 for k in _metric_keys}
                    val_metrics["count"] = 0
                    with torch.no_grad():
                        val_gen = data_processing.read_npz_training_data(
                            val_files[:3],
                            batch_size=batch_size,
                            world_size=1,
                            rank=0,
                            pos_len=pos_len,
                            device=device,
                            symmetry_type=None,
                            include_meta=False,
                            enable_history_matrices=args.enable_history_matrices,
                            model_config=model_config,
                            use_pin_memory=use_pin_memory,
                        )
                        for val_batch in data_processing.prefetch_generator(val_gen, args.prefetch_batches):
                            with torch.amp.autocast(amp_device, dtype=amp_dtype, enabled=use_amp):
                                outputs = model(val_batch["binaryInputNCHW"], val_batch["globalInputNC"])
                            _, batch_metrics = compute_loss(
                                model, outputs, val_batch, pos_len,
                                is_training=False,
                                soft_policy_weight_scale=args.soft_policy_weight_scale,
                                value_loss_scale=args.value_loss_scale,
                                td_value_loss_scales=td_value_loss_scales,
                                seki_loss_scale=args.seki_loss_scale,
                                variance_time_loss_scale=args.variance_time_loss_scale,
                                disable_optimistic_policy=args.disable_optimistic_policy,
                            )
                            for k in batch_metrics:
                                val_metrics[k] += batch_metrics[k]
                            val_metrics["count"] += 1

                    weight_sum = max(val_metrics["wsum"], 1e-10)
                    batch_count = max(val_metrics["count"], 1)
                    logging.info(
                        f"  VAL [{total_samples_trained} samples]: loss={val_metrics['loss'] / batch_count:.4f}, "
                        f"p0loss={val_metrics['p0loss'] / weight_sum:.4f}, "
                        f"vloss={val_metrics['vloss'] / weight_sum:.4f}, "
                        f"oloss={val_metrics['oloss'] / weight_sum:.4f}, "
                        f"skloss={val_metrics['skloss'] / weight_sum:.4f}, "
                        f"pacc1={val_metrics['pacc1'] / weight_sum:.4f}"
                    )
                    if tb_writer is not None:
                        tb_writer.add_scalar("val/loss", val_metrics["loss"] / batch_count, total_samples_trained)
                        for k in _per_sample_keys:
                            tb_writer.add_scalar(f"val/{k}", val_metrics[k] / weight_sum, total_samples_trained)
                    model.train()

            if total_samples_trained >= args.max_training_samples:
                break

    # Final save (rank 0 only)
    if rank == 0:
        save_checkpoint()
    logging.info(f"Training complete: {total_samples_trained} samples, {global_step} steps")
    if tb_writer is not None:
        tb_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal Transformer training for KataGo (nano)")
    parser.add_argument("--traindir", required=True, help="Training output directory")
    parser.add_argument("--datadir", required=True, help="Data directory with train/ and val/ subdirs")
    parser.add_argument("--pos-len", type=int, default=19, help="Board size")
    parser.add_argument("--batch-size", type=int, default=256, help="Per-GPU batch size")
    parser.add_argument("--model-kind", type=str, default="b14c192h6tfrs", help="Model config name")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--wd", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--muon-scope", type=str, default="off", choices=["all", "blocks", "off"],
                        help="Muon scope: all=all 2D non-norm params, blocks=only blocks.* params, off=pure AdamW")
    parser.add_argument("--muon-momentum", type=float, default=0.95, help="Muon momentum beta")
    parser.add_argument("--muon-lr-multiplier", type=float, default=0.2, help="Muon LR multiplier over base lr")
    parser.add_argument("--muon-scale", type=str, default="moonlight", choices=["moonlight", "mup"],
                        help="Muon update scale: moonlight=sqrt(max(m,n)), mup=sqrt(max(1,m/n))")
    parser.add_argument("--shampoo-scope", type=str, default="off", choices=["all", "blocks", "off"],
                        help="Shampoo scope: all=all 2D non-norm params, blocks=only blocks.* params, off=disabled")
    parser.add_argument("--shampoo-lr-multiplier", type=float, default=2.0, help="Shampoo LR multiplier over base lr")
    parser.add_argument("--shampoo-momentum", type=float, default=0.9, help="Shampoo momentum beta")
    parser.add_argument("--shampoo-beta2", type=float, default=0.95, help="Shampoo L/R EMA coefficient")
    parser.add_argument("--init-std", type=float, default=0.02, help="Init std for weights (Megatron-LM style)")
    parser.add_argument("--max-training-samples", type=int, default=100000000, help="Total training samples")
    parser.add_argument("--save-every-samples", type=int, default=1000000, help="Save checkpoint every N samples")
    parser.add_argument("--symmetry-type", type=str, default="xyt", help="Data symmetry type")
    parser.add_argument("--print-every", type=int, default=100, help="Print every N batches")
    parser.add_argument("--val-every-samples", type=int, default=1000000, help="Run validation every N samples")
    parser.add_argument("--warmup-samples", type=int, default=2000000, help="LR warmup samples")
    parser.add_argument("--enable-history-matrices", action="store_true", help="Enable history matrices (for Go)")
    parser.add_argument("--initial-checkpoint", type=str, default=None, help="Initial checkpoint to load from")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--soft-policy-weight-scale", type=float, default=8.0, help="Soft policy loss coeff")
    parser.add_argument("--value-loss-scale", type=float, default=0.6, help="Value loss coeff")
    parser.add_argument("--td-value-loss-scales", type=str, default="0.6,0.6,0.6", help="TD value loss coeffs")
    parser.add_argument("--seki-loss-scale", type=float, default=1.0, help="Seki loss coeff")
    parser.add_argument("--variance-time-loss-scale", type=float, default=1.0, help="Variance time loss coeff")
    parser.add_argument("--disable-optimistic-policy", action="store_true", help="Disable optimistic policy")
    parser.add_argument("--multi-gpus", type=str, default=None, help="Comma-separated GPU device ids for DDP (e.g. 0,1,2,3)")
    parser.add_argument("--master-port", type=int, default=23456, help="Localhost port for DDP communication")
    parser.add_argument("--prefetch-batches", type=int, default=20, help="Prefetch queue depth (0=off)")
    parser.add_argument("--score-mode", type=str, default="simple", choices=["mixop", "mix", "simple"],
                        help="Score belief head mode: mixop=linear+offset/parity+MoS, mix=linear+MoS, simple=single linear")
    parser.add_argument("--attn-backend", type=str, default="sdpa", choices=["sdpa", "flex"],
                        help="Attention backend: sdpa=F.scaled_dot_product_attention, flex=FlexAttention")
    args = parser.parse_args()

    # Mutual exclusion check
    if args.muon_scope != "off" and args.shampoo_scope != "off":
        parser.error("muon-scope and shampoo-scope cannot both be enabled. Set one to 'off'.")

    # Parse multi-gpus
    multi_gpu_device_ids = []
    if args.multi_gpus is not None:
        for piece in args.multi_gpus.split(","):
            piece = piece.strip()
            multi_gpu_device_ids.append(int(piece))
    else:
        multi_gpu_device_ids = [0]

    num_gpus_used = len(multi_gpu_device_ids)

    if num_gpus_used > 1:
        torch.multiprocessing.set_start_method("spawn")
        torch.multiprocessing.spawn(
            main,
            nprocs=num_gpus_used,
            args=(num_gpus_used, args, multi_gpu_device_ids),
        )
    else:
        main(0, 1, args, multi_gpu_device_ids)
