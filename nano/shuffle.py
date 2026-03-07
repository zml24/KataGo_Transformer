#!/usr/bin/env python3
"""Shuffle NPZ training data for KataGo nano training.

Simplified version of train/shuffle.py: no windowing/tapering, just full shuffle
of all data using the Shardify + Sequential-Merge-Repack two-stage approach.

Output guarantees:
  - Every output file has exactly --rows-per-file rows, except train's last file.
  - Val: all files exactly --rows-per-file rows. Remainder (< rows_per_file) moves
    to train (not discarded). Val is processed first.
  - Train: all files exactly --rows-per-file rows, except the last file which holds
    whatever is left (including val's remainder).
  - Set --rows-per-file to a power of 2 so it's divisible by batch_size * world_size.

Usage:
    python3 shuffle.py <input-dirs...> \\
        --num-processes 8 \\
        [--rows-per-file 131072] \\
        --split "train:0.00:0.95:/out/train:/tmp/train" \\
        --split "val:0.95:1.00:/out/val:/tmp/val"
"""

import argparse
import hashlib
import itertools
import json
import multiprocessing
import os
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass, field

import numpy as np

REQUIRED_KEYS = [
    "binaryInputNCHWPacked",
    "globalInputNC",
    "policyTargetsNCMove",
    "globalTargetsNC",
    "scoreDistrN",
    "valueTargetsNCHW",
]
OPTIONAL_KEYS = []

POS_LEN = 19
PACKED_BYTES = (POS_LEN * POS_LEN + 7) // 8  # 46


def format_duration(seconds):
    """Format seconds as a compact human-readable duration."""
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


class ProgressLogger:
    """Periodic progress logger with throughput and ETA."""

    def __init__(self, desc, total, unit, interval_sec=30.0):
        self.desc = desc
        self.total = max(0, int(total))
        self.unit = unit
        self.interval_sec = max(1.0, float(interval_sec))
        self.t0 = time.time()
        self.last_report_time = self.t0

    def maybe_report(self, completed, extra=""):
        now = time.time()
        if completed < self.total and now - self.last_report_time < self.interval_sec:
            return
        self._report(completed, now, extra)

    def final_report(self, completed, extra=""):
        self._report(completed, time.time(), extra)

    def _report(self, completed, now, extra):
        elapsed = max(1e-9, now - self.t0)
        rate = completed / elapsed
        if self.total > 0:
            pct = 100.0 * completed / self.total
            if completed > 0 and completed < self.total:
                eta = (self.total - completed) / rate
                eta_text = f", ETA {format_duration(eta)}"
            else:
                eta_text = ""
            total_text = f"/{self.total}"
            pct_text = f" ({pct:.1f}%)"
        else:
            total_text = ""
            pct_text = ""
            eta_text = ""

        rate_text = f", {rate:.2f} {self.unit}/s" if rate > 0 else ""
        extra_text = f", {extra}" if extra else ""
        print(f"  Progress [{self.desc}]: {completed}{total_text} {self.unit}{pct_text}"
              f"{rate_text}{eta_text}{extra_text}", flush=True)
        self.last_report_time = now


@dataclass
class SplitConfig:
    name: str
    md5_lbound: float
    md5_ubound: float
    out_dir: str
    tmp_dir: str
    file_rows: list = field(default_factory=list)


def scan_file(args):
    """Get row count + optional board size check in a single pass.

    Args:
        args: (filename, board_size) where board_size may be None.

    Returns:
        (filename, num_rows, ok) where ok is True if the file passes the
        board size check (or if no check is requested).
    """
    filename, board_size = args
    try:
        npheaders = get_numpy_npz_headers(filename)
    except (PermissionError, zipfile.BadZipFile) as e:
        print(f"WARNING: {e}: {filename}")
        return (filename, None, False)
    if npheaders is None or len(npheaders) == 0:
        return (filename, None, False)

    num_rows = None
    for key in ["binaryInputNCHWPacked", "binaryInputNCHWPacked.npy"]:
        if key in npheaders:
            num_rows = npheaders[key][0][0]
            break
    if num_rows is None:
        return (filename, None, False)

    if board_size is None:
        return (filename, num_rows, True)

    # Board size check
    try:
        with np.load(filename) as npz:
            packed = npz["binaryInputNCHWPacked"]
            if packed.shape[2] != PACKED_BYTES:
                return (filename, num_rows, False)
            ch0 = np.unpackbits(packed[0:1, 0:1, :], axis=2)
            ch0 = ch0[0, 0, :POS_LEN * POS_LEN].reshape(POS_LEN, POS_LEN)
            ok = (int(ch0[:board_size, :board_size].sum()) == board_size * board_size
                  and int(ch0.sum()) == board_size * board_size)
            return (filename, num_rows, ok)
    except Exception:
        return (filename, num_rows, False)


def get_numpy_npz_headers(filename):
    """Read NPZ headers without loading array data."""
    with zipfile.ZipFile(filename) as z:
        npzheaders = {}
        for subfilename in z.namelist():
            npyfile = z.open(subfilename)
            try:
                version = np.lib.format.read_magic(npyfile)
            except ValueError:
                print(f"WARNING: bad array in {filename}: {subfilename}")
                return None
            (shape, is_fortran, dtype) = np.lib.format._read_array_header(npyfile, version)
            npzheaders[subfilename] = (shape, is_fortran, dtype)
        return npzheaders


def get_header_entry(npheaders, key):
    """Get NPZ header entry, allowing for the '.npy' suffix inside zip files."""
    if key in npheaders:
        return npheaders[key]
    key_npy = f"{key}.npy"
    if key_npy in npheaders:
        return npheaders[key_npy]
    return None


def estimate_required_bytes_per_row(filename):
    """Estimate bytes/row for the arrays this script actually reads."""
    try:
        npheaders = get_numpy_npz_headers(filename)
    except (PermissionError, zipfile.BadZipFile):
        return None

    if not npheaders:
        return None

    packed_header = get_header_entry(npheaders, "binaryInputNCHWPacked")
    if packed_header is None:
        return None

    num_rows = packed_header[0][0]
    if num_rows <= 0:
        return None

    total_bytes = 0
    for key in REQUIRED_KEYS + OPTIONAL_KEYS:
        entry = get_header_entry(npheaders, key)
        if entry is None:
            if key in REQUIRED_KEYS:
                return None
            continue
        shape, _, dtype = entry
        itemsize = np.dtype(dtype).itemsize
        num_items = 1
        for dim in shape:
            num_items *= dim
        total_bytes += itemsize * num_items

    return total_bytes / num_rows


def format_bytes(num_bytes):
    """Format bytes as a human-readable binary size."""
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0


def get_total_memory_bytes():
    """Best-effort total system RAM detection."""
    page_names = ("SC_PAGE_SIZE", "SC_PAGESIZE")
    for page_name in page_names:
        try:
            page_size = os.sysconf(page_name)
            phys_pages = os.sysconf("SC_PHYS_PAGES")
            return int(page_size) * int(phys_pages)
        except (AttributeError, OSError, ValueError):
            continue
    return None


def log_memory_estimates(sample_file, worker_group_size, rows_per_file,
                         shard_processes, merge_processes):
    """Print rough memory estimates for current shuffle settings."""
    bytes_per_row = estimate_required_bytes_per_row(sample_file)
    if bytes_per_row is None:
        print("Memory estimate: unavailable (could not read sample NPZ headers)", flush=True)
        return

    shard_raw = bytes_per_row * worker_group_size
    merge_raw = bytes_per_row * rows_per_file

    # Rough peak multipliers from the current algorithm:
    # shardify ~= loaded file arrays + concatenated arrays + shuffled copy
    # merge ~= buffered arrays + concatenated chunk + shuffled output chunk
    shard_peak_per_worker = shard_raw * 3.25
    merge_peak_per_worker = merge_raw * 4.0
    shard_peak_total = shard_peak_per_worker * shard_processes
    merge_peak_total = merge_peak_per_worker * merge_processes

    print("Memory estimate (rough, for required arrays only):", flush=True)
    print(f"  Sample file: {sample_file}", flush=True)
    print(f"  Bytes per row: {bytes_per_row:.0f} ({format_bytes(bytes_per_row)})", flush=True)
    print(f"  Shardify per worker: ~{format_bytes(shard_peak_per_worker)} "
          f"(worker_group_size={worker_group_size})", flush=True)
    print(f"  Shardify total: ~{format_bytes(shard_peak_total)} "
          f"({shard_processes} workers)", flush=True)
    print(f"  Merge per worker: ~{format_bytes(merge_peak_per_worker)} "
          f"(rows_per_file={rows_per_file})", flush=True)
    print(f"  Merge total: ~{format_bytes(merge_peak_total)} "
          f"({merge_processes} workers)", flush=True)

    total_mem = get_total_memory_bytes()
    if total_mem is not None:
        print(f"  Host RAM: {format_bytes(total_mem)}", flush=True)
        if shard_peak_total > total_mem * 0.70:
            print("WARNING: shardify memory estimate exceeds 70% of host RAM. "
                  "Reduce --worker-group-size and/or --shard-processes.", flush=True)
        if merge_peak_total > total_mem * 0.70:
            print("WARNING: merge memory estimate exceeds 70% of host RAM. "
                  "Reduce --merge-processes and/or --rows-per-file.", flush=True)


def md5_hash_float(s):
    """Hash a string to a float in [0, 1) using MD5."""
    return int("0x" + hashlib.md5(s.encode("utf-8")).hexdigest()[:13], 16) / 2**52


def load_npz_arrays(filename):
    """Load all relevant arrays from an NPZ file. Returns dict or None."""
    try:
        with np.load(filename) as npz:
            data = {}
            for key in REQUIRED_KEYS:
                if key not in npz:
                    print(f"WARNING: missing key {key} in {filename}")
                    return None
                data[key] = npz[key]
            for key in OPTIONAL_KEYS:
                if key in npz:
                    data[key] = npz[key]
            return data
    except Exception as e:
        print(f"WARNING: error loading {filename}: {e}")
        return None


def joint_shuffle(arrs, n=None):
    """Jointly shuffle a list of arrays along axis 0, optionally taking first n."""
    total = len(arrs[0])
    perm = np.random.permutation(total)
    if n is not None:
        perm = perm[:n]
    return [arr[perm] for arr in arrs]


def shardify(input_idx, file_group, num_out_files, out_tmp_dirs,
             compress_shards=False, shard_chunk_size=16384):
    """Load a group of files, shuffle, and distribute rows to shard temp dirs."""
    np.random.seed([int.from_bytes(os.urandom(4), byteorder="little") for _ in range(4)])

    # Load and concatenate all files in the group
    all_data = {}
    for fpath in file_group:
        data = load_npz_arrays(fpath)
        if data is None:
            continue
        for key, arr in data.items():
            if key not in all_data:
                all_data[key] = []
            all_data[key].append(arr)

    if not all_data or "binaryInputNCHWPacked" not in all_data:
        return np.zeros(num_out_files, dtype=np.int64)

    merged = {key: np.concatenate(arrs) for key, arrs in all_data.items()}
    num_rows = merged["binaryInputNCHWPacked"].shape[0]

    # Shuffle
    keys = list(merged.keys())
    arrays = [merged[k] for k in keys]
    arrays = joint_shuffle(arrays)
    merged = dict(zip(keys, arrays))

    # Distribute to shards by chunk: assign each chunk of rows to one bucket.
    # With chunk_size=16K and ~80K rows/group, this produces ~5 shards instead of ~num_out_files.
    num_chunks = (num_rows + shard_chunk_size - 1) // shard_chunk_size
    chunk_assignments = np.random.randint(num_out_files, size=num_chunks)
    # Expand chunk assignments to per-row assignments
    assignments = np.repeat(chunk_assignments, shard_chunk_size)[:num_rows]
    counts = np.bincount(assignments, minlength=num_out_files)

    # Group rows by bucket and write
    order = np.argsort(assignments, kind="stable")
    save_fn = np.savez_compressed if compress_shards else np.savez
    offset = 0
    for out_idx in range(num_out_files):
        n = counts[out_idx]
        if n == 0:
            continue
        idx = order[offset:offset + n]
        shard = {key: merged[key][idx] for key in keys}
        out_path = os.path.join(out_tmp_dirs[out_idx], f"{input_idx}.npz")
        save_fn(out_path, **shard)
        offset += n

    return counts


def shardify_star(args):
    """Helper for imap_unordered with shardify."""
    return shardify(*args)


def sequential_merge_repack(num_shards, out_tmp_dirs, out_dir, rows_per_file,
                            keep_remainder=False, extra_rows=None):
    """Sequentially read all shards, buffer rows, and write exact-sized output files.

    Args:
        num_shards: Number of sharding worker groups (= shard files per tmp dir).
        out_tmp_dirs: List of shard temp directories.
        out_dir: Final output directory.
        rows_per_file: Exact number of rows per output file.
        keep_remainder: If True, don't write the last partial chunk; return it instead.
        extra_rows: Optional dict of arrays to inject into buffer as initial data.

    Returns:
        (written_files, remainder)
        - written_files: list of (filepath, num_rows)
        - remainder: dict of arrays (< rows_per_file) or None
    """
    np.random.seed([int.from_bytes(os.urandom(4), byteorder="little") for _ in range(5)])

    buffer = {}  # key -> list of arrays
    buffer_rows = 0
    out_file_idx = 0
    written_files = []

    def flush_buffer(n_rows):
        nonlocal buffer, buffer_rows, out_file_idx

        merged = {key: np.concatenate(arrs) for key, arrs in buffer.items()}
        keys = list(merged.keys())

        take = {key: merged[key][:n_rows] for key in keys}
        remain = {key: merged[key][n_rows:] for key in keys}

        # Shuffle the chunk before writing
        arrays = joint_shuffle([take[k] for k in keys])
        take = dict(zip(keys, arrays))

        out_path = os.path.join(out_dir, f"data{out_file_idx}.npz")
        np.savez_compressed(out_path, **take)
        written_files.append((out_path, n_rows))
        out_file_idx += 1

        remain_rows = remain[keys[0]].shape[0]
        if remain_rows > 0:
            buffer = {key: [arr] for key, arr in remain.items()}
        else:
            buffer = {}
        buffer_rows = remain_rows

    # Inject extra rows (e.g. val remainder) into buffer
    if extra_rows is not None:
        n = extra_rows["binaryInputNCHWPacked"].shape[0]
        if n > 0:
            for key, arr in extra_rows.items():
                buffer[key] = [arr]
            buffer_rows = n
            print(f"  Injected {n} extra rows into buffer", flush=True)

    # Collect all shard file paths and randomize read order
    all_shard_files = []
    for tmp_dir in out_tmp_dirs:
        for shard_idx in range(num_shards):
            shard_path = os.path.join(tmp_dir, f"{shard_idx}.npz")
            if os.path.exists(shard_path):
                all_shard_files.append(shard_path)
    np.random.shuffle(all_shard_files)

    for shard_path in all_shard_files:
        try:
            with np.load(shard_path) as npz:
                for key in npz.keys():
                    if key not in buffer:
                        buffer[key] = []
                    buffer[key].append(npz[key])
                buffer_rows += npz["binaryInputNCHWPacked"].shape[0]
        except Exception as e:
            print(f"WARNING: error reading shard {shard_path}: {e}")
            continue

        # Flush full files as buffer accumulates
        while buffer_rows >= rows_per_file:
            flush_buffer(rows_per_file)

    # Handle final remainder
    remainder = None
    if buffer_rows > 0:
        if keep_remainder:
            remainder = {key: np.concatenate(arrs) for key, arrs in buffer.items()}
            print(f"  Remainder: {buffer_rows} rows -> train", flush=True)
        else:
            flush_buffer(buffer_rows)

    return written_files, remainder


def merge_one_bucket(bucket_idx, tmp_dir, num_shards, out_dir,
                     out_file_idx, rows_per_file, remainder_path,
                     extra_rows_path=None, progress_interval_sec=30.0):
    """Merge all shard files in one bucket, write output files, save remainder.

    Uses streaming buffer: loads one shard at a time and flushes when the buffer
    reaches rows_per_file, so memory stays O(rows_per_file + single_shard_size)
    instead of O(entire_bucket_size).

    Args:
        bucket_idx: Index of this bucket (for logging).
        tmp_dir: The temp directory for this bucket (e.g., tmp.shuf3/).
        num_shards: Number of shard files to look for (0..num_shards-1).
        out_dir: Final output directory.
        out_file_idx: Starting output file index for this bucket.
        rows_per_file: Exact number of rows per output file.
        remainder_path: Path to save remainder as NPZ (if any leftover rows).
        extra_rows_path: Optional path to NPZ file with extra rows to inject.

    Returns:
        (bucket_idx, written_files, remainder_rows)
        - written_files: list of (filepath, num_rows)
        - remainder_rows: int, number of leftover rows saved to remainder_path
    """
    np.random.seed([int.from_bytes(os.urandom(4), byteorder="little") for _ in range(5)])

    buffer = {}  # key -> list of arrays
    buffer_rows = 0
    file_idx = out_file_idx
    written_files = []

    def flush_buffer(n_rows):
        nonlocal buffer, buffer_rows, file_idx

        merged = {key: np.concatenate(arrs) for key, arrs in buffer.items()}
        keys = list(merged.keys())

        take = {key: merged[key][:n_rows] for key in keys}
        remain = {key: merged[key][n_rows:] for key in keys}

        arrays = joint_shuffle([take[k] for k in keys])
        take = dict(zip(keys, arrays))

        out_path = os.path.join(out_dir, f"data{file_idx}.npz")
        np.savez_compressed(out_path, **take)
        written_files.append((out_path, n_rows))
        file_idx += 1

        remain_rows = remain[keys[0]].shape[0]
        if remain_rows > 0:
            buffer = {key: [arr] for key, arr in remain.items()}
        else:
            buffer = {}
        buffer_rows = remain_rows

    # Inject extra rows (e.g. val remainder) if provided
    if extra_rows_path is not None and os.path.exists(extra_rows_path):
        with np.load(extra_rows_path) as npz:
            for key in npz.keys():
                buffer[key] = [npz[key]]
            buffer_rows = npz["binaryInputNCHWPacked"].shape[0]
            print(f"  Bucket {bucket_idx}: injected {buffer_rows} extra rows", flush=True)

    # Read all shard files in this bucket (randomized order)
    shard_files = []
    for shard_idx in range(num_shards):
        path = os.path.join(tmp_dir, f"{shard_idx}.npz")
        if os.path.exists(path):
            shard_files.append(path)
    np.random.shuffle(shard_files)
    total_shard_files = len(shard_files)
    last_progress_time = time.time()

    for processed_shards, path in enumerate(shard_files, start=1):
        try:
            with np.load(path) as npz:
                for key in npz.keys():
                    if key not in buffer:
                        buffer[key] = []
                    buffer[key].append(npz[key])
                buffer_rows += npz["binaryInputNCHWPacked"].shape[0]
        except Exception as e:
            print(f"WARNING: error reading shard {path}: {e}")
            continue

        # Flush as buffer accumulates
        while buffer_rows >= rows_per_file:
            flush_buffer(rows_per_file)

        now = time.time()
        if processed_shards < total_shard_files and now - last_progress_time >= progress_interval_sec:
            pct = 100.0 * processed_shards / total_shard_files if total_shard_files > 0 else 100.0
            print(f"  Bucket {bucket_idx}: read {processed_shards}/{total_shard_files} shard files "
                  f"({pct:.1f}%), wrote {len(written_files)} files, buffered {buffer_rows} rows",
                  flush=True)
            last_progress_time = now

    # Save remainder to temp file (avoid pickling large arrays through IPC)
    if buffer_rows > 0:
        remainder = {key: np.concatenate(arrs) for key, arrs in buffer.items()}
        np.savez(remainder_path, **remainder)  # uncompressed, temp file

    print(f"  Bucket {bucket_idx}: finished {total_shard_files}/{total_shard_files} shard files, "
          f"wrote {len(written_files)} files, remainder {buffer_rows} rows", flush=True)
    return bucket_idx, written_files, buffer_rows


def merge_one_bucket_star(args):
    """Helper for imap_unordered with merge_one_bucket."""
    return merge_one_bucket(*args)


def parallel_merge_repack(num_shards, out_tmp_dirs, out_dir, rows_per_file,
                          num_processes, keep_remainder=False,
                          extra_rows=None, tmp_base_dir=None,
                          bucket_row_counts=None, progress_interval_sec=30.0):
    """Parallel merge: each bucket processed independently by a worker.

    Args:
        num_shards: Number of shard files per bucket (= num worker groups).
        out_tmp_dirs: List of bucket temp directories.
        out_dir: Final output directory.
        rows_per_file: Exact number of rows per output file.
        num_processes: Number of parallel workers.
        keep_remainder: If True, don't write the last partial chunk; return it.
        extra_rows: Optional dict of arrays to inject (e.g. val remainder for train).
        tmp_base_dir: Temp directory for remainder files and extra_rows file.
        bucket_row_counts: Pre-computed list of row counts per bucket (from shardify).

    Returns:
        (written_files, remainder) -- same contract as sequential_merge_repack
    """
    num_buckets = len(out_tmp_dirs)

    # Use pre-computed bucket row counts from shardify (skip header scanning)
    if bucket_row_counts is None:
        bucket_row_counts = []
        for tmp_dir in out_tmp_dirs:
            total = 0
            for shard_idx in range(num_shards):
                path = os.path.join(tmp_dir, f"{shard_idx}.npz")
                if os.path.exists(path):
                    try:
                        headers = get_numpy_npz_headers(path)
                        if headers:
                            for key in headers:
                                if key in ("binaryInputNCHWPacked",
                                           "binaryInputNCHWPacked.npy"):
                                    total += headers[key][0][0]
                                    break
                    except Exception:
                        pass
            bucket_row_counts.append(total)

    # Save extra_rows to temp file for IPC
    extra_rows_path = None
    extra_count = 0
    if extra_rows is not None:
        extra_count = extra_rows["binaryInputNCHWPacked"].shape[0]
        if extra_count > 0:
            extra_rows_path = os.path.join(tmp_base_dir, "extra_rows.npz")
            np.savez(extra_rows_path, **extra_rows)

    # --- Phase 2: Pre-allocate output file indices ---
    # Add extra_rows to bucket 0
    effective_counts = list(bucket_row_counts)
    if extra_count > 0:
        effective_counts[0] += extra_count

    out_file_starts = []
    cumulative = 0
    for count in effective_counts:
        out_file_starts.append(cumulative)
        cumulative += count // rows_per_file

    # --- Phase 3: Parallel merge ---
    remainder_dir = os.path.join(tmp_base_dir, "tmp.remainders")
    os.makedirs(remainder_dir, exist_ok=True)

    tasks = []
    for bucket_idx, tmp_dir in enumerate(out_tmp_dirs):
        remainder_path = os.path.join(remainder_dir, f"rem_{bucket_idx}.npz")
        inject_path = extra_rows_path if bucket_idx == 0 else None
        tasks.append((
            bucket_idx, tmp_dir, num_shards, out_dir,
            out_file_starts[bucket_idx], rows_per_file,
            remainder_path, inject_path, progress_interval_sec,
        ))

    with multiprocessing.Pool(num_processes) as pool:
        results = []
        progress = ProgressLogger(
            desc="merge buckets",
            total=len(tasks),
            unit="buckets",
            interval_sec=progress_interval_sec,
        )
        total_written_files = 0
        for completed_buckets, result in enumerate(pool.imap_unordered(merge_one_bucket_star, tasks), start=1):
            bucket_idx, written_files, rem_rows = result
            results.append(result)
            total_written_files += len(written_files)
            bucket_rows = bucket_row_counts[bucket_idx] if bucket_idx < len(bucket_row_counts) else 0
            progress.maybe_report(
                completed_buckets,
                extra=f"bucket {bucket_idx} done, files_written={total_written_files}, "
                      f"bucket_rows={bucket_rows}",
            )

    # --- Phase 4: Collect results and handle remainders ---
    all_written = []
    all_remainder_paths = []
    total_remainder_rows = 0

    for bucket_idx, written_files, rem_rows in results:
        all_written.extend(written_files)
        if rem_rows > 0:
            rem_path = os.path.join(remainder_dir, f"rem_{bucket_idx}.npz")
            all_remainder_paths.append(rem_path)
            total_remainder_rows += rem_rows

    # Determine next file index
    next_file_idx = cumulative  # from the pre-allocation

    # Merge all remainders
    remainder = None
    if total_remainder_rows > 0:
        combined = {}
        for rem_path in all_remainder_paths:
            try:
                with np.load(rem_path) as npz:
                    for key in npz.keys():
                        if key not in combined:
                            combined[key] = []
                        combined[key].append(npz[key])
            except Exception as e:
                print(f"WARNING: error reading remainder {rem_path}: {e}")

        if combined:
            merged_rem = {key: np.concatenate(arrs) for key, arrs in combined.items()}
            keys = list(merged_rem.keys())
            total_rem = merged_rem[keys[0]].shape[0]

            # Produce full files from merged remainders
            offset = 0
            while offset + rows_per_file <= total_rem:
                chunk = {key: merged_rem[key][offset:offset + rows_per_file] for key in keys}
                arrays = joint_shuffle([chunk[k] for k in keys])
                chunk = dict(zip(keys, arrays))
                out_path = os.path.join(out_dir, f"data{next_file_idx}.npz")
                np.savez_compressed(out_path, **chunk)
                all_written.append((out_path, rows_per_file))
                next_file_idx += 1
                offset += rows_per_file

            # Final remainder
            final_rem_rows = total_rem - offset
            if final_rem_rows > 0:
                if keep_remainder:
                    remainder = {key: merged_rem[key][offset:] for key in keys}
                    print(f"  Remainder: {final_rem_rows} rows -> train", flush=True)
                else:
                    chunk = {key: merged_rem[key][offset:] for key in keys}
                    arrays = joint_shuffle([chunk[k] for k in keys])
                    chunk = dict(zip(keys, arrays))
                    out_path = os.path.join(out_dir, f"data{next_file_idx}.npz")
                    np.savez_compressed(out_path, **chunk)
                    all_written.append((out_path, final_rem_rows))

    # Cleanup temp files
    shutil.rmtree(remainder_dir, ignore_errors=True)
    if extra_rows_path and os.path.exists(extra_rows_path):
        os.remove(extra_rows_path)

    # Sort by file path for consistent ordering
    all_written.sort(key=lambda x: x[0])

    return all_written, remainder


class Timer:
    def __init__(self, desc):
        self.desc = desc

    def __enter__(self):
        print(f"Beginning: {self.desc}", flush=True)
        self.t0 = time.time()
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.t0
        print(f"Finished: {self.desc} in {elapsed:.1f}s", flush=True)


def md5_filter(file_rows, lbound, ubound):
    """Filter file_rows by MD5 hash of basename into [lbound, ubound)."""
    filtered = []
    for filename, num_rows in file_rows:
        h = md5_hash_float(os.path.basename(filename))
        if h < lbound or h >= ubound:
            continue
        filtered.append((filename, num_rows))
    return filtered


def select_val_files_fixed(file_rows, val_num_files, rows_per_file):
    """Randomly select files for val to reach val_num_files * rows_per_file rows.

    Args:
        file_rows: List of (filename, num_rows) for all valid files.
        val_num_files: Target number of val output files.
        rows_per_file: Rows per output file.

    Returns:
        (val_file_rows, train_file_rows) — disjoint partition of file_rows.
    """
    target_rows = val_num_files * rows_per_file
    total_rows = sum(nr for _, nr in file_rows)

    if total_rows < target_rows:
        print(f"WARNING: total rows ({total_rows}) < val target ({target_rows}). "
              f"Using all data for val.", flush=True)

    shuffled = list(file_rows)
    np.random.shuffle(shuffled)

    val_file_rows = []
    train_file_rows = []
    accumulated = 0
    for filename, num_rows in shuffled:
        if accumulated < target_rows:
            val_file_rows.append((filename, num_rows))
            accumulated += num_rows
        else:
            train_file_rows.append((filename, num_rows))

    return val_file_rows, train_file_rows


def process_split(split, shard_processes, merge_processes, rows_per_file, worker_group_size,
                  num_buckets=None, shard_chunk_size=16384,
                  keep_remainder=False, extra_rows=None, compress_shards=False,
                  progress_interval_sec=30.0):
    """Process a single split: shardify + parallel merge-repack + index.json.

    Args:
        split: SplitConfig with file_rows populated.
        shard_processes: Number of parallel workers for shardify.
        merge_processes: Number of parallel workers for merge/repack.
        rows_per_file: Exact rows per output file.
        worker_group_size: Target rows per sharding worker group.
        keep_remainder: If True, return leftover rows instead of writing them.
        extra_rows: Optional dict of arrays to inject (e.g. val remainder for train).
        progress_interval_sec: Seconds between progress updates.

    Returns:
        remainder: dict of arrays (< rows_per_file) or None.
    """
    file_rows = split.file_rows
    total_rows = sum(nr for _, nr in file_rows)
    extra_count = extra_rows["binaryInputNCHWPacked"].shape[0] if extra_rows is not None else 0

    print(f"\n{'='*60}", flush=True)
    print(f"Processing split '{split.name}': {len(file_rows)} files, {total_rows} rows", flush=True)
    if extra_count > 0:
        print(f"  + {extra_count} extra rows from val remainder", flush=True)
    print(f"  out_dir: {split.out_dir}", flush=True)
    print(f"  tmp_dir: {split.tmp_dir}", flush=True)

    if not file_rows and extra_count == 0:
        print(f"  No data for split '{split.name}', skipping.", flush=True)
        return None

    os.makedirs(split.out_dir)

    num_output_files = max(1, round(total_rows / rows_per_file)) if total_rows > 0 else 1
    if num_buckets is not None:
        num_shards_buckets = min(num_output_files, max(1, num_buckets))
    else:
        num_shards_buckets = min(num_output_files, max(1, merge_processes))
    out_tmp_dirs = [os.path.join(split.tmp_dir, f"tmp.shuf{i}") for i in range(num_shards_buckets)]

    print(f"  Intermediate shard buckets: {num_shards_buckets}", flush=True)
    print(f"  Rows per file: {rows_per_file}", flush=True)
    print(f"  Shard workers: {shard_processes}", flush=True)
    print(f"  Merge workers: {merge_processes}", flush=True)

    # Clean and create tmp dirs
    for d in out_tmp_dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    # Group files for sharding
    np.random.seed()
    shuffled = list(file_rows)
    np.random.shuffle(shuffled)

    groups = []
    group = []
    group_rows = 0
    for filename, num_rows in shuffled:
        group.append(filename)
        group_rows += num_rows
        if group_rows >= worker_group_size:
            groups.append(group)
            group = []
            group_rows = 0
    if group:
        groups.append(group)

    num_worker_groups = len(groups)
    print(f"  Grouped into {num_worker_groups} worker groups", flush=True)

    # Stage 1: Shardify (parallel)
    bucket_row_counts = np.zeros(num_shards_buckets, dtype=np.int64)
    if groups:
        with multiprocessing.Pool(shard_processes) as pool:
            with Timer(f"Sharding ({split.name})"):
                progress = ProgressLogger(
                    desc=f"sharding {split.name}",
                    total=num_worker_groups,
                    unit="groups",
                    interval_sec=progress_interval_sec,
                )
                total_sharded = 0
                shard_tasks = [
                    (idx, groups[idx], num_shards_buckets, out_tmp_dirs,
                     compress_shards, shard_chunk_size)
                    for idx in range(num_worker_groups)
                ]
                for completed_groups, counts in enumerate(pool.imap_unordered(shardify_star, shard_tasks), start=1):
                    bucket_row_counts += counts
                    total_sharded += int(counts.sum())
                    progress.maybe_report(
                        completed_groups,
                        extra=f"rows={total_sharded}",
                    )
                print(f"  Sharded {total_sharded} rows", flush=True)
    else:
        num_worker_groups = 0
        total_sharded = 0

    # Stage 2: Parallel merge + repack
    with Timer(f"Parallel merge+repack ({split.name})"):
        written_files, remainder = parallel_merge_repack(
            num_shards=num_worker_groups,
            out_tmp_dirs=out_tmp_dirs,
            out_dir=split.out_dir,
            rows_per_file=rows_per_file,
            num_processes=merge_processes,
            keep_remainder=keep_remainder,
            extra_rows=extra_rows,
            tmp_base_dir=split.tmp_dir,
            bucket_row_counts=bucket_row_counts.tolist(),
            progress_interval_sec=progress_interval_sec,
        )

    # Write index.json
    index_entries = []
    total_written = 0
    for fname, nr in written_files:
        index_entries.append({"name": os.path.basename(fname), "num_rows": nr})
        total_written += nr

    index = {
        "files": index_entries,
        "total_rows": total_written,
        "rows_per_file": rows_per_file,
    }
    index_path = os.path.join(split.out_dir, "index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    # Summary
    if index_entries:
        row_counts = [e["num_rows"] for e in index_entries]
        if len(set(row_counts)) == 1:
            print(f"  {len(index_entries)} files, all {row_counts[0]} rows", flush=True)
        else:
            full_count = sum(1 for r in row_counts[:-1] if r == row_counts[0])
            print(f"  {full_count} full files ({row_counts[0]} rows each), "
                  f"last file: {row_counts[-1]} rows", flush=True)

    remainder_count = remainder["binaryInputNCHWPacked"].shape[0] if remainder else 0
    total_input = total_sharded + extra_count
    discarded = total_input - total_written - remainder_count
    print(f"  Total rows written: {total_written}" +
          (f" (discarded: {discarded})" if discarded > 0 else ""), flush=True)
    print(f"  Index written to: {index_path}", flush=True)

    # Cleanup tmp dirs
    for d in out_tmp_dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
    print(f"  Temp dirs cleaned up for '{split.name}'.", flush=True)

    return remainder


def parse_split(s):
    """Parse a --split argument string 'name:md5_lo:md5_hi:out_dir:tmp_dir'."""
    parts = s.split(":")
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            f"--split requires 5 colon-separated fields "
            f"(name:md5_lo:md5_hi:out_dir:tmp_dir), got {len(parts)}: '{s}'"
        )
    name, md5_lo, md5_hi, out_dir, tmp_dir = parts
    try:
        md5_lo = float(md5_lo)
        md5_hi = float(md5_hi)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"md5_lo and md5_hi must be floats, got '{parts[1]}' and '{parts[2]}'"
        )
    return SplitConfig(
        name=name,
        md5_lbound=md5_lo,
        md5_ubound=md5_hi,
        out_dir=out_dir,
        tmp_dir=tmp_dir,
    )


def main():
    parser = argparse.ArgumentParser(description="Shuffle NPZ training data for nano.")
    parser.add_argument("dirs", nargs="+", help="Input directories containing NPZ files")
    parser.add_argument("--num-processes", type=int, required=True, help="Number of parallel workers")
    parser.add_argument("--shard-processes", type=int, default=None,
                        help="Parallel workers for shardify only (default: --num-processes)")
    parser.add_argument("--merge-processes", type=int, default=None,
                        help="Parallel workers for merge/repack only (default: --num-processes)")
    parser.add_argument("--rows-per-file", type=int, default=131072,
                        help="Exact rows per output file (default: 131072). "
                             "Use a power of 2 so it divides evenly by batch_size * world_size.")
    parser.add_argument("--worker-group-size", type=int, default=80000,
                        help="Target rows per sharding worker group (default: 80000)")
    parser.add_argument("--filter-board-size", type=int, default=None,
                        help="Keep only files matching this board size (e.g. 19 for 19x19)")
    parser.add_argument("--split", type=parse_split, action="append", dest="splits",
                        metavar="name:md5_lo:md5_hi:out_dir:tmp_dir",
                        help="Define a split (repeatable). Format: name:md5_lo:md5_hi:out_dir:tmp_dir")
    parser.add_argument("--val-num-files", type=int, default=None,
                        help="Fixed number of val output files (val total = N * rows-per-file). "
                             "Overrides MD5 bounds for val split.")
    parser.add_argument("--num-buckets", type=int, default=None,
                        help="Number of intermediate shard buckets (default: merge-processes)")
    parser.add_argument("--shard-chunk-size", type=int, default=16384,
                        help="Rows per chunk when distributing to buckets (default: 16384). "
                             "Larger = fewer shard files, slightly less uniform bucket sizes.")
    parser.add_argument("--compress-shards", action="store_true", default=False,
                        help="Compress intermediate shard files (saves tmp disk space, slower)")
    parser.add_argument("--progress-interval-sec", type=float, default=30.0,
                        help="Seconds between progress updates during sharding/merge (default: 30)")
    args = parser.parse_args()

    if not args.splits:
        print("ERROR: at least one --split is required.", file=sys.stderr)
        sys.exit(1)

    if args.val_num_files is not None:
        if args.val_num_files <= 0:
            print("ERROR: --val-num-files must be > 0.", file=sys.stderr)
            sys.exit(1)
        split_names = {s.name for s in args.splits}
        if "val" not in split_names or "train" not in split_names:
            print("ERROR: --val-num-files requires both 'val' and 'train' splits.", file=sys.stderr)
            sys.exit(1)

    # Pre-check: fail fast if any output directory already exists
    for split in args.splits:
        if os.path.exists(split.out_dir):
            print(f"ERROR: Output directory already exists: {split.out_dir}", file=sys.stderr)
            sys.exit(1)

    num_processes = args.num_processes
    shard_processes = args.shard_processes if args.shard_processes is not None else num_processes
    merge_processes = args.merge_processes if args.merge_processes is not None else num_processes
    rows_per_file = args.rows_per_file
    worker_group_size = args.worker_group_size
    num_buckets = args.num_buckets if args.num_buckets is not None else merge_processes
    shard_chunk_size = args.shard_chunk_size
    progress_interval_sec = args.progress_interval_sec

    # --- Stage 1: Find all NPZ files ---
    all_files = []
    with Timer("Finding files"):
        for d in args.dirs:
            for root, dirs, files in os.walk(d, followlinks=True):
                for f in files:
                    if f.endswith(".npz"):
                        all_files.append(os.path.join(root, f))
    print(f"Found {len(all_files)} NPZ files", flush=True)

    if not all_files:
        print("No files found, exiting.")
        sys.exit(0)

    # --- Stage 2: Scan files (row counts + optional board size filter) ---
    board_size = args.filter_board_size
    file_rows = []
    total_rows = 0
    bad_files = 0
    filtered_by_board = 0
    if board_size is not None:
        scan_desc = f"Scanning files (row counts + {board_size}x{board_size} filter)"
    else:
        scan_desc = "Scanning files (row counts)"
    with Timer(scan_desc):
        with multiprocessing.Pool(num_processes) as pool:
            progress = ProgressLogger(
                desc="scan files",
                total=len(all_files),
                unit="files",
                interval_sec=progress_interval_sec,
            )
            scan_iter = zip(all_files, itertools.repeat(board_size))
            for scanned_files, (filename, num_rows, ok) in enumerate(
                pool.imap_unordered(scan_file, scan_iter, chunksize=64),
                start=1,
            ):
                if num_rows is None or num_rows <= 0:
                    bad_files += 1
                elif not ok:
                    filtered_by_board += 1
                else:
                    file_rows.append((filename, num_rows))
                    total_rows += num_rows

                progress.maybe_report(
                    scanned_files,
                    extra=f"valid={len(file_rows)}, bad={bad_files}, "
                          f"filtered={filtered_by_board}, rows={total_rows}",
                )

    print(f"Valid files: {len(file_rows)}, bad/empty: {bad_files}", flush=True)
    if board_size is not None:
        print(f"Filtered by board size ({board_size}x{board_size}): {filtered_by_board} files removed", flush=True)
    print(f"Total rows: {total_rows}", flush=True)
    if file_rows:
        log_memory_estimates(
            sample_file=file_rows[0][0],
            worker_group_size=worker_group_size,
            rows_per_file=rows_per_file,
            shard_processes=shard_processes,
            merge_processes=merge_processes,
        )

    if total_rows == 0:
        print("No rows found, exiting.")
        sys.exit(0)

    # --- Stage 3: Split files into val/train ---
    splits_by_name = {s.name: s for s in args.splits}

    if args.val_num_files is not None:
        # Fixed val size mode: randomly select files for val
        val_file_rows, train_file_rows = select_val_files_fixed(
            file_rows, args.val_num_files, rows_per_file,
        )
        splits_by_name["val"].file_rows = val_file_rows
        splits_by_name["train"].file_rows = train_file_rows

        val_rows = sum(nr for _, nr in val_file_rows)
        train_rows = sum(nr for _, nr in train_file_rows)
        print(f"Split 'val': {len(val_file_rows)}/{len(file_rows)} files, "
              f"{val_rows}/{total_rows} rows "
              f"(fixed {args.val_num_files} output files)", flush=True)
        print(f"Split 'train': {len(train_file_rows)}/{len(file_rows)} files, "
              f"{train_rows}/{total_rows} rows", flush=True)

        # Other splits still use MD5
        for split in args.splits:
            if split.name not in ("train", "val"):
                split.file_rows = md5_filter(file_rows, split.md5_lbound, split.md5_ubound)
                split_rows = sum(nr for _, nr in split.file_rows)
                print(f"Split '{split.name}': {len(split.file_rows)}/{len(file_rows)} files, "
                      f"{split_rows}/{total_rows} rows "
                      f"(MD5 [{split.md5_lbound:.2f}, {split.md5_ubound:.2f}))", flush=True)
    else:
        # Default: MD5 hash-based split
        for split in args.splits:
            split.file_rows = md5_filter(file_rows, split.md5_lbound, split.md5_ubound)
            split_rows = sum(nr for _, nr in split.file_rows)
            print(f"Split '{split.name}': {len(split.file_rows)}/{len(file_rows)} files, "
                  f"{split_rows}/{total_rows} rows "
                  f"(MD5 [{split.md5_lbound:.2f}, {split.md5_ubound:.2f}))", flush=True)

    # --- Stage 4: Process splits (val first, then train) ---
    # Val is processed first so its remainder rows can be added to train.
    val_split = splits_by_name.get("val")
    train_split = splits_by_name.get("train")

    compress_shards = args.compress_shards

    val_remainder = None
    if val_split is not None:
        val_remainder = process_split(
            val_split, shard_processes, merge_processes, rows_per_file, worker_group_size,
            num_buckets=num_buckets, shard_chunk_size=shard_chunk_size,
            keep_remainder=True, compress_shards=compress_shards,
            progress_interval_sec=progress_interval_sec,
        )

    if train_split is not None:
        process_split(
            train_split, shard_processes, merge_processes, rows_per_file, worker_group_size,
            num_buckets=num_buckets, shard_chunk_size=shard_chunk_size,
            extra_rows=val_remainder, compress_shards=compress_shards,
            progress_interval_sec=progress_interval_sec,
        )

    # Process any other splits (neither train nor val)
    for split in args.splits:
        if split.name not in ("train", "val"):
            process_split(split, shard_processes, merge_processes, rows_per_file, worker_group_size,
                          num_buckets=num_buckets, shard_chunk_size=shard_chunk_size,
                          compress_shards=compress_shards,
                          progress_interval_sec=progress_interval_sec)

    print(f"\nAll splits done.", flush=True)


if __name__ == "__main__":
    main()
