#!/usr/bin/env python3
"""Shuffle NPZ training data for KataGo nano training.

Simplified version of train/shuffle.py: no windowing/tapering, just full shuffle
of all data using the Shardify + Merge two-stage approach.

Usage:
    python3 shuffle.py <input-dirs...> \\
        --out-dir <dir> --tmp-dir <dir> \\
        --num-processes 8 --batch-size 1024 \\
        [--approx-rows-per-out-file 70000] \\
        [--md5-lbound 0.00] [--md5-ubound 0.95]
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


def check_board_size(args):
    """Check if a file's data matches the required board size.

    Each file is one game (one board size), so checking the first row suffices.
    Returns (filename, num_rows, match).
    """
    filename, num_rows, board_size = args
    try:
        with np.load(filename) as npz:
            packed = npz["binaryInputNCHWPacked"]
            if packed.shape[2] != PACKED_BYTES:
                return (filename, num_rows, False)
            # Unpack channel 0 of first row and reshape to (POS_LEN, POS_LEN)
            ch0 = np.unpackbits(packed[0:1, 0:1, :], axis=2)
            ch0 = ch0[0, 0, :POS_LEN * POS_LEN].reshape(POS_LEN, POS_LEN)
            # Board occupies top-left board_size x board_size, rest is padding
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


def compute_num_rows(filename):
    """Get row count from NPZ file by reading only the zip header."""
    try:
        npheaders = get_numpy_npz_headers(filename)
    except (PermissionError, zipfile.BadZipFile) as e:
        print(f"WARNING: {e}: {filename}")
        return (filename, None)
    if npheaders is None or len(npheaders) == 0:
        return (filename, None)

    for key in ["binaryInputNCHWPacked", "binaryInputNCHWPacked.npy"]:
        if key in npheaders:
            return (filename, npheaders[key][0][0])
    return (filename, None)


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


def merge_shards(out_file, num_shards, tmp_dir, batch_size):
    """Merge shard files into a single shuffled output file.

    Returns: number of rows written (truncated to batch_size multiple).
    """
    np.random.seed([int.from_bytes(os.urandom(4), byteorder="little") for _ in range(5)])

    all_data = {}
    for shard_idx in range(num_shards):
        shard_path = os.path.join(tmp_dir, f"{shard_idx}.npz")
        if not os.path.exists(shard_path):
            continue
        try:
            with np.load(shard_path) as npz:
                for key in npz.keys():
                    if key not in all_data:
                        all_data[key] = []
                    all_data[key].append(npz[key])
        except Exception as e:
            print(f"WARNING: error reading shard {shard_path}: {e}")

    if not all_data:
        return 0

    merged = {key: np.concatenate(arrs) for key, arrs in all_data.items()}
    num_rows = merged["binaryInputNCHWPacked"].shape[0]

    # Shuffle
    keys = list(merged.keys())
    arrays = [merged[k] for k in keys]
    arrays = joint_shuffle(arrays)
    merged = dict(zip(keys, arrays))

    # Truncate to batch_size multiple
    num_keep = (num_rows // batch_size) * batch_size
    if num_keep == 0:
        return 0

    out = {key: merged[key][:num_keep] for key in keys}
    np.savez_compressed(out_file, **out)
    return num_keep


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


def main():
    parser = argparse.ArgumentParser(description="Shuffle NPZ training data for nano.")
    parser.add_argument("dirs", nargs="+", help="Input directories containing NPZ files")
    parser.add_argument("--out-dir", required=True, help="Output directory for shuffled files")
    parser.add_argument("--tmp-dir", required=True, help="Temporary directory for shards")
    parser.add_argument("--num-processes", type=int, required=True, help="Number of parallel workers")
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size (output rows are a multiple of this)")
    parser.add_argument("--approx-rows-per-out-file", type=int, default=70000,
                        help="Target rows per output file (default: 70000)")
    parser.add_argument("--worker-group-size", type=int, default=80000,
                        help="Target rows per sharding worker group (default: 80000)")
    parser.add_argument("--md5-lbound", type=float, default=None,
                        help="Include only files with MD5 hash >= this (for train/val split)")
    parser.add_argument("--md5-ubound", type=float, default=None,
                        help="Include only files with MD5 hash < this (for train/val split)")
    parser.add_argument("--filter-board-size", type=int, default=None,
                        help="Keep only files matching this board size (e.g. 19 for 19x19)")
    args = parser.parse_args()

    out_dir = args.out_dir
    tmp_dir = args.tmp_dir
    num_processes = args.num_processes
    batch_size = args.batch_size
    approx_rows_per_out_file = args.approx_rows_per_out_file
    worker_group_size = args.worker_group_size

    # --- Step 1: Find all NPZ files ---
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

    # --- Step 2: Compute row counts ---
    with Timer("Computing row counts"):
        with multiprocessing.Pool(num_processes) as pool:
            results = pool.map(compute_num_rows, all_files, chunksize=64)

    file_rows = []  # (filename, num_rows)
    total_rows = 0
    bad_files = 0
    for filename, num_rows in results:
        if num_rows is None or num_rows <= 0:
            bad_files += 1
            continue
        file_rows.append((filename, num_rows))
        total_rows += num_rows

    print(f"Valid files: {len(file_rows)}, bad/empty: {bad_files}", flush=True)
    print(f"Total rows: {total_rows}", flush=True)

    if total_rows == 0:
        print("No rows found, exiting.")
        sys.exit(0)

    # --- Step 3: MD5 filtering ---
    if args.md5_lbound is not None or args.md5_ubound is not None:
        filtered = []
        filtered_rows = 0
        for filename, num_rows in file_rows:
            h = md5_hash_float(os.path.basename(filename))
            if args.md5_lbound is not None and h < args.md5_lbound:
                continue
            if args.md5_ubound is not None and h >= args.md5_ubound:
                continue
            filtered.append((filename, num_rows))
            filtered_rows += num_rows
        print(f"MD5 filter: {len(filtered)}/{len(file_rows)} files, "
              f"{filtered_rows}/{total_rows} rows", flush=True)
        file_rows = filtered
        total_rows = filtered_rows

    if not file_rows:
        print("No files after MD5 filtering, exiting.")
        sys.exit(0)

    # --- Step 3b: Board size filtering ---
    if args.filter_board_size is not None:
        bs = args.filter_board_size
        with Timer(f"Filtering for {bs}x{bs} boards"):
            with multiprocessing.Pool(num_processes) as pool:
                check_results = pool.map(
                    check_board_size,
                    [(f, nr, bs) for f, nr in file_rows],
                    chunksize=64,
                )
        filtered = [(f, nr) for f, nr, ok in check_results if ok]
        filtered_rows = sum(nr for _, nr in filtered)
        print(f"Board size filter ({bs}x{bs}): {len(filtered)}/{len(file_rows)} files, "
              f"{filtered_rows}/{total_rows} rows", flush=True)
        file_rows = filtered
        total_rows = filtered_rows

    if not file_rows:
        print("No files after filtering, exiting.")
        sys.exit(0)

    # --- Step 4: Setup output ---
    if os.path.exists(out_dir):
        raise RuntimeError(f"Output directory already exists: {out_dir}")
    os.makedirs(out_dir)

    num_out_files = max(1, round(total_rows / approx_rows_per_out_file))
    out_files = [os.path.join(out_dir, f"data{i}.npz") for i in range(num_out_files)]
    out_tmp_dirs = [os.path.join(tmp_dir, f"tmp.shuf{i}") for i in range(num_out_files)]

    print(f"Will produce {num_out_files} output files", flush=True)

    # Clean and create tmp dirs
    for d in out_tmp_dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    # --- Step 5: Group files for sharding ---
    np.random.seed()
    np.random.shuffle(file_rows)

    groups = []
    group = []
    group_rows = 0
    for filename, num_rows in file_rows:
        group.append(filename)
        group_rows += num_rows
        if group_rows >= worker_group_size:
            groups.append(group)
            group = []
            group_rows = 0
    if group:
        groups.append(group)

    print(f"Grouped {len(file_rows)} files into {len(groups)} worker groups", flush=True)

    # --- Step 6: Shardify + Merge ---
    with multiprocessing.Pool(num_processes) as pool:
        with Timer("Sharding"):
            shard_results = pool.starmap(
                shardify,
                [(idx, groups[idx], num_out_files, out_tmp_dirs) for idx in range(len(groups))],
            )
            total_sharded = sum(shard_results)
            print(f"  Sharded {total_sharded} rows", flush=True)

        num_shards = len(groups)
        with Timer("Merging"):
            merge_results = pool.starmap(
                merge_shards,
                [(out_files[idx], num_shards, out_tmp_dirs[idx], batch_size)
                 for idx in range(num_out_files)],
            )

    # --- Step 7: Write index.json ---
    index_entries = []
    total_written = 0
    for fname, num_rows in zip(out_files, merge_results):
        if num_rows > 0:
            index_entries.append({"name": os.path.basename(fname), "num_rows": num_rows})
            total_written += num_rows

    index = {
        "files": index_entries,
        "total_rows": total_written,
        "batch_size": batch_size,
    }
    index_path = os.path.join(out_dir, "index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nOutput files and row counts:", flush=True)
    for entry in index_entries:
        print(f"  {entry['name']}: {entry['num_rows']}", flush=True)
    print(f"Total rows written: {total_written}", flush=True)
    print(f"Index written to: {index_path}", flush=True)

    # --- Step 8: Cleanup ---
    for d in out_tmp_dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
    print("Temp dirs cleaned up.", flush=True)


if __name__ == "__main__":
    main()
