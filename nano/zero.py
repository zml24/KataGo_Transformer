"""ZeRO Stage 1: optimizer state partitioning across GPUs."""

import logging

import torch
import torch.distributed as dist
from optimizers import MuonOptimizer, ShampooOptimizer


# ---------------------------------------------------------------------------
# Cost estimation for load-balanced partitioning
# ---------------------------------------------------------------------------
def _muon_cost(p):
    """Estimate compute cost of Muon's polar_express (5 NS iterations)."""
    if p.ndim == 4:
        m, n = p.shape[0], p.shape[1] * p.shape[2] * p.shape[3]
    else:
        m, n = p.shape
    lo, hi = min(m, n), max(m, n)
    # 5 NS iters, each ~3 matmuls of shape (lo, lo) @ (lo, hi)
    return 5 * lo * lo * (2 * hi + lo)


def _shampoo_cost(p):
    """Estimate compute cost of Shampoo's inv_quarter_sandwich (4 NS iters + L/R EMA)."""
    if p.ndim >= 2:
        m, n = p.shape[0], p.shape[1:].numel()
    else:
        m, n = p.shape[0], 1
    # 4 NS iters: each does ~8 matmuls for L (m^3) and R (n^3) plus sandwich (m^2*n + m*n^2)
    # Plus L/R EMA: grad @ grad.T (m^2*n) + grad.T @ grad (m*n^2)
    return 4 * (8 * m**3 + 8 * n**3 + m*m*n + m*n*n) + (m*m*n + m*n*n)


# ---------------------------------------------------------------------------
# LPT (Longest Processing Time) greedy partition
# ---------------------------------------------------------------------------
def _lpt_partition(named_params, cost_fn, world_size):
    """Partition named_params across ranks using LPT greedy algorithm.

    Returns: list of dicts, one per rank. Each dict is {name: param}.
    All ranks compute the same result deterministically.
    """
    items = [(cost_fn(p), name, p) for name, p in named_params.items()]
    # Sort by cost descending, break ties by name for determinism
    items.sort(key=lambda x: (-x[0], x[1]))

    partitions = [{} for _ in range(world_size)]
    loads = [0] * world_size

    for cost, name, p in items:
        # Assign to rank with smallest current load; break ties by rank index
        target = min(range(world_size), key=lambda r: (loads[r], r))
        partitions[target][name] = p
        loads[target] += cost

    return partitions


def _numel_partition(named_params, world_size):
    """Partition named_params by numel using LPT."""
    return _lpt_partition(named_params, lambda p: p.numel(), world_size)


# ---------------------------------------------------------------------------
# Coalesced broadcast: each owner rank broadcasts its params in one flat tensor
# ---------------------------------------------------------------------------
def _coalesced_broadcast(partitions, rank, world_size):
    """Broadcast updated parameters from each owner rank to all others.

    partitions: list of dicts [{name: param}, ...], one per rank.
    """
    for src_rank in range(world_size):
        params = list(partitions[src_rank].values())
        if not params:
            continue
        # Group by (device, dtype) to avoid implicit dtype promotion in torch.cat.
        buckets = {}
        for p in params:
            key = (p.device, p.dtype)
            if key not in buckets:
                buckets[key] = []
            buckets[key].append(p)

        for (dev, dtype), bucket in buckets.items():
            if rank == src_rank:
                with torch.no_grad():
                    flat = torch.cat([p.detach().reshape(-1) for p in bucket], dim=0).contiguous()
            else:
                total_numel = sum(p.numel() for p in bucket)
                flat = torch.empty(total_numel, dtype=dtype, device=dev)
            dist.broadcast(flat, src=src_rank)
            if rank != src_rank:
                # Unpack back into param tensors.
                with torch.no_grad():
                    offset = 0
                    for p in bucket:
                        numel = p.numel()
                        p.copy_(flat[offset:offset + numel].reshape_as(p))
                        offset += numel


def sync_zero_params(optimizers, rank, world_size):
    """Sync parameters once for multiple ZeRO optimizer wrappers.

    Args:
        optimizers: iterable of ZeRO optimizer wrappers (None entries allowed).
        rank: local rank.
        world_size: total number of ranks.
    """
    merged = [{} for _ in range(world_size)]
    for opt in optimizers:
        if opt is None:
            continue
        parts = opt.partitions
        if len(parts) != world_size:
            raise ValueError("Inconsistent world_size across ZeRO optimizers")
        for r in range(world_size):
            for name, p in parts[r].items():
                if name in merged[r]:
                    raise ValueError(f"Parameter '{name}' appears in multiple ZeRO optimizers on rank {r}")
                merged[r][name] = p

    _coalesced_broadcast(merged, rank, world_size)


# ---------------------------------------------------------------------------
# ZeROAdamW
# ---------------------------------------------------------------------------
class ZeROAdamW:
    """ZeRO Stage 1 wrapper for AdamW (handles both decay and no-decay params)."""

    def __init__(self, adam_params, no_decay_params, lr, betas, wd, device, rank, world_size):
        """
        adam_params: dict {name: param} for weight-decay params
        no_decay_params: dict {name: param} for no-decay params (1D bias/norm)
        """
        self.rank = rank
        self.world_size = world_size

        # Merge all params for partitioning
        all_params = {}
        self._no_decay_names = set(no_decay_params.keys())
        for name, p in no_decay_params.items():
            all_params[name] = p
        for name, p in adam_params.items():
            all_params[name] = p

        self._all_params = all_params
        self.partitions = _numel_partition(all_params, world_size)

        # Build param groups for this rank's partition
        my_params = self.partitions[rank]
        my_decay = [p for name, p in my_params.items() if name not in self._no_decay_names]
        my_no_decay = [p for name, p in my_params.items() if name in self._no_decay_names]

        param_groups = []
        if my_no_decay:
            param_groups.append({"params": my_no_decay, "weight_decay": 0.0})
        if my_decay:
            param_groups.append({"params": my_decay, "weight_decay": wd})

        # Fallback: if this rank has no params (shouldn't happen with 30+ params),
        # create a dummy group so scheduler works
        if not param_groups:
            param_groups.append({"params": [], "weight_decay": 0.0})

        self._optimizer = torch.optim.AdamW(
            param_groups, lr=lr, betas=betas, fused=(device.type == "cuda"),
        )

        # Log partition info
        my_numel = sum(p.numel() for p in my_params.values())
        total_numel = sum(p.numel() for p in all_params.values())
        logging.info(
            f"ZeROAdamW rank {rank}: {len(my_params)}/{len(all_params)} params, "
            f"{my_numel:,}/{total_numel:,} elements "
            f"({100*my_numel/max(total_numel,1):.1f}%)"
        )

    @property
    def optimizer(self):
        """Expose internal optimizer for LR scheduler."""
        return self._optimizer

    def step(self, sync=True):
        """Run optimizer step on local partition.

        If sync=True (default), broadcast updated params immediately.
        """
        self._optimizer.step()
        if sync:
            _coalesced_broadcast(self.partitions, self.rank, self.world_size)

    def gather_state_for_save(self):
        """Collective operation: gather optimizer state to rank 0 as name-based dict.

        All ranks must call this. Returns the gathered state on rank 0, None on others.
        """
        # Build local name-based state
        my_params = self.partitions[self.rank]
        local_state = {}
        opt_state = self._optimizer.state

        for name, p in my_params.items():
            if p in opt_state:
                s = opt_state[p]
                local_state[name] = {k: v.cpu() if torch.is_tensor(v) else v for k, v in s.items()}

        # Gather to rank 0
        gathered = [None] * self.world_size if self.rank == 0 else None
        dist.gather_object(local_state, gathered, dst=0)

        if self.rank == 0:
            merged = {}
            for rank_state in gathered:
                merged.update(rank_state)
            return merged
        return None

    def load_state_distributed(self, saved_state, device):
        """Load name-based optimizer state into this rank's partition.

        saved_state: the full name-based state dict (same on all ranks).
        Handles the case where AdamW state is not yet initialized (lazy init).
        """
        opt_state = self._optimizer.state
        my_params = self.partitions[self.rank]

        for name, p in my_params.items():
            if name not in saved_state:
                continue
            saved_s = saved_state[name]
            if not isinstance(saved_s, dict):
                continue
            if p not in opt_state:
                # AdamW lazily initializes state on first step();
                # create the entry so we can populate it from the checkpoint.
                opt_state[p] = {}
            for k, v in saved_s.items():
                if torch.is_tensor(v):
                    if k in opt_state[p]:
                        opt_state[p][k].copy_(v.to(device))
                    else:
                        opt_state[p][k] = v.clone().to(device)
                else:
                    opt_state[p][k] = v


# ---------------------------------------------------------------------------
# ZeROMuon
# ---------------------------------------------------------------------------
class ZeROMuon:
    """ZeRO Stage 1 wrapper for MuonOptimizer."""

    def __init__(self, named_params, lr_multiplier, momentum, wd, scale_mode, device, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self._all_params = named_params
        self.partitions = _lpt_partition(named_params, _muon_cost, world_size)

        my_params = self.partitions[rank]
        self._local_numel = sum(p.numel() for p in my_params.values())
        self._local_opt = MuonOptimizer(
            my_params, lr_multiplier=lr_multiplier, momentum=momentum,
            wd=wd, scale_mode=scale_mode, device=device,
        ) if my_params else None

        self.last_update_rms = 0.0

        total_numel = sum(p.numel() for p in named_params.values())
        my_cost = sum(_muon_cost(p) for p in my_params.values())
        total_cost = sum(_muon_cost(p) for p in named_params.values())
        logging.info(
            f"ZeROMuon rank {rank}: {len(my_params)}/{len(named_params)} params, "
            f"{self._local_numel:,}/{total_numel:,} elements, "
            f"cost {my_cost/max(total_cost,1)*100:.1f}%"
        )

    def step(self, base_lr, sync=True):
        if self._local_opt is not None:
            self._local_opt.step(base_lr)
            self.last_update_rms = self._local_opt.last_update_rms
        else:
            self.last_update_rms = 0.0

        if sync:
            _coalesced_broadcast(self.partitions, self.rank, self.world_size)

        # Weighted all-reduce: global_rms = sqrt(sum(rms_i^2 * numel_i) / sum(numel_i))
        dev = next(iter(self._all_params.values())).device
        buf = torch.tensor(
            [self.last_update_rms ** 2 * self._local_numel, float(self._local_numel)],
            dtype=torch.float64, device=dev,
        )
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        self.last_update_rms = (buf[0] / max(buf[1].item(), 1)).sqrt().item()

    def state_dict(self):
        """Return name-based state dict (local partition only, for gather)."""
        if self._local_opt is not None:
            return self._local_opt.state_dict()
        return {}

    def gather_state_for_save(self):
        """Collective: gather Muon state to rank 0."""
        local_state = self.state_dict()
        gathered = [None] * self.world_size if self.rank == 0 else None
        dist.gather_object(local_state, gathered, dst=0)
        if self.rank == 0:
            merged = {}
            for rank_state in gathered:
                merged.update(rank_state)
            return merged
        return None

    def load_state_distributed(self, saved_state, device):
        """Load name-based state into local partition."""
        if self._local_opt is not None:
            self._local_opt.load_state_dict(saved_state, device)


# ---------------------------------------------------------------------------
# ZeROShampoo
# ---------------------------------------------------------------------------
class ZeROShampoo:
    """ZeRO Stage 1 wrapper for ShampooOptimizer."""

    def __init__(self, named_params, lr_multiplier, momentum, wd, beta2, device, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self._all_params = named_params
        self.partitions = _lpt_partition(named_params, _shampoo_cost, world_size)

        my_params = self.partitions[rank]
        self._local_numel = sum(p.numel() for p in my_params.values())
        self._local_opt = ShampooOptimizer(
            my_params, lr_multiplier=lr_multiplier, momentum=momentum,
            wd=wd, beta2=beta2, device=device,
        ) if my_params else None

        self.last_precond_rms = 0.0

        total_numel = sum(p.numel() for p in named_params.values())
        my_cost = sum(_shampoo_cost(p) for p in my_params.values())
        total_cost = sum(_shampoo_cost(p) for p in named_params.values())
        logging.info(
            f"ZeROShampoo rank {rank}: {len(my_params)}/{len(named_params)} params, "
            f"{self._local_numel:,}/{total_numel:,} elements, "
            f"cost {my_cost/max(total_cost,1)*100:.1f}%"
        )

    def step(self, base_lr, sync=True):
        if self._local_opt is not None:
            self._local_opt.step(base_lr)
            self.last_precond_rms = self._local_opt.last_precond_rms
        else:
            self.last_precond_rms = 0.0

        if sync:
            _coalesced_broadcast(self.partitions, self.rank, self.world_size)

        # Weighted all-reduce: global_rms = sqrt(sum(rms_i^2 * numel_i) / sum(numel_i))
        dev = next(iter(self._all_params.values())).device
        buf = torch.tensor(
            [self.last_precond_rms ** 2 * self._local_numel, float(self._local_numel)],
            dtype=torch.float64, device=dev,
        )
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        self.last_precond_rms = (buf[0] / max(buf[1].item(), 1)).sqrt().item()

    def state_dict(self):
        if self._local_opt is not None:
            return self._local_opt.state_dict()
        return {}

    def gather_state_for_save(self):
        """Collective: gather Shampoo state to rank 0."""
        local_state = self.state_dict()
        gathered = [None] * self.world_size if self.rank == 0 else None
        dist.gather_object(local_state, gathered, dst=0)
        if self.rank == 0:
            merged = {}
            for rank_state in gathered:
                merged.update(rank_state)
            return merged
        return None

    def load_state_distributed(self, saved_state, device):
        if self._local_opt is not None:
            self._local_opt.load_state_dict(saved_state, device)
