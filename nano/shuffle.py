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
OPTIONAL_KEYS = [
    "qValueTargetsNCMove",
    "metadataInputNC",
]

POS_LEN = 19
PACKED_BYTES = (POS_LEN * POS_LEN + 7) // 8  # 46


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


def shardify(input_idx, file_group, num_out_files, out_tmp_dirs):
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
        return 0

    merged = {key: np.concatenate(arrs) for key, arrs in all_data.items()}
    num_rows = merged["binaryInputNCHWPacked"].shape[0]

    # Shuffle
    keys = list(merged.keys())
    arrays = [merged[k] for k in keys]
    arrays = joint_shuffle(arrays)
    merged = dict(zip(keys, arrays))

    # Distribute to shards
    assignments = np.random.randint(num_out_files, size=num_rows)
    counts = np.bincount(assignments, minlength=num_out_files)
    cumsum = np.cumsum(counts)

    for out_idx in range(num_out_files):
        start = cumsum[out_idx] - counts[out_idx]
        stop = cumsum[out_idx]
        if stop <= start:
            continue
        shard = {key: merged[key][start:stop] for key in keys}
        out_path = os.path.join(out_tmp_dirs[out_idx], f"{input_idx}.npz")
        np.savez_compressed(out_path, **shard)

    return num_rows


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


def process_split(split, num_processes, rows_per_file, worker_group_size,
                  keep_remainder=False, extra_rows=None):
    """Process a single split: shardify + sequential merge-repack + index.json.

    Args:
        split: SplitConfig with file_rows populated.
        num_processes: Number of parallel workers for shardify.
        rows_per_file: Exact rows per output file.
        worker_group_size: Target rows per sharding worker group.
        keep_remainder: If True, return leftover rows instead of writing them.
        extra_rows: Optional dict of arrays to inject (e.g. val remainder for train).

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

    # Number of intermediate shard buckets (only affects shardify parallelism)
    num_shards_buckets = max(1, round(total_rows / rows_per_file)) if total_rows > 0 else 1
    out_tmp_dirs = [os.path.join(split.tmp_dir, f"tmp.shuf{i}") for i in range(num_shards_buckets)]

    print(f"  Intermediate shard buckets: {num_shards_buckets}", flush=True)
    print(f"  Rows per file: {rows_per_file}", flush=True)

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
    if groups:
        with multiprocessing.Pool(num_processes) as pool:
            with Timer(f"Sharding ({split.name})"):
                shard_results = pool.starmap(
                    shardify,
                    [(idx, groups[idx], num_shards_buckets, out_tmp_dirs)
                     for idx in range(num_worker_groups)],
                )
                total_sharded = sum(shard_results)
                print(f"  Sharded {total_sharded} rows", flush=True)
    else:
        num_worker_groups = 0
        total_sharded = 0

    # Stage 2: Sequential merge + repack
    with Timer(f"Merge+repack ({split.name})"):
        written_files, remainder = sequential_merge_repack(
            num_shards=num_worker_groups,
            out_tmp_dirs=out_tmp_dirs,
            out_dir=split.out_dir,
            rows_per_file=rows_per_file,
            keep_remainder=keep_remainder,
            extra_rows=extra_rows,
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
    args = parser.parse_args()

    if not args.splits:
        print("ERROR: at least one --split is required.", file=sys.stderr)
        sys.exit(1)

    # Pre-check: fail fast if any output directory already exists
    for split in args.splits:
        if os.path.exists(split.out_dir):
            print(f"ERROR: Output directory already exists: {split.out_dir}", file=sys.stderr)
            sys.exit(1)

    num_processes = args.num_processes
    rows_per_file = args.rows_per_file
    worker_group_size = args.worker_group_size

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
    with Timer("Scanning files (row counts" + (f" + {board_size}x{board_size} filter)" if board_size else ")")):
        with multiprocessing.Pool(num_processes) as pool:
            results = pool.map(scan_file, [(f, board_size) for f in all_files], chunksize=64)

    file_rows = []
    total_rows = 0
    bad_files = 0
    filtered_by_board = 0
    for filename, num_rows, ok in results:
        if num_rows is None or num_rows <= 0:
            bad_files += 1
            continue
        if not ok:
            filtered_by_board += 1
            continue
        file_rows.append((filename, num_rows))
        total_rows += num_rows

    print(f"Valid files: {len(file_rows)}, bad/empty: {bad_files}", flush=True)
    if board_size is not None:
        print(f"Filtered by board size ({board_size}x{board_size}): {filtered_by_board} files removed", flush=True)
    print(f"Total rows: {total_rows}", flush=True)

    if total_rows == 0:
        print("No rows found, exiting.")
        sys.exit(0)

    # --- Stage 3: MD5 split ---
    for split in args.splits:
        split.file_rows = md5_filter(file_rows, split.md5_lbound, split.md5_ubound)
        split_rows = sum(nr for _, nr in split.file_rows)
        print(f"Split '{split.name}': {len(split.file_rows)}/{len(file_rows)} files, "
              f"{split_rows}/{total_rows} rows "
              f"(MD5 [{split.md5_lbound:.2f}, {split.md5_ubound:.2f}))", flush=True)

    # --- Stage 4: Process splits (val first, then train) ---
    # Val is processed first so its remainder rows can be added to train.
    splits_by_name = {s.name: s for s in args.splits}
    val_split = splits_by_name.get("val")
    train_split = splits_by_name.get("train")

    val_remainder = None
    if val_split is not None:
        val_remainder = process_split(
            val_split, num_processes, rows_per_file, worker_group_size,
            keep_remainder=True,
        )

    if train_split is not None:
        process_split(
            train_split, num_processes, rows_per_file, worker_group_size,
            extra_rows=val_remainder,
        )

    # Process any other splits (neither train nor val)
    for split in args.splits:
        if split.name not in ("train", "val"):
            process_split(split, num_processes, rows_per_file, worker_group_size)

    print(f"\nAll splits done.", flush=True)


if __name__ == "__main__":
    main()
