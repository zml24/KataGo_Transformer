#!/usr/bin/env python3
"""
Download KataGo training data from katagoarchive.org for use with KataGo_Transformer.

Usage:
    # Download all kata1 training data to ./data/selfplay/
    python download_training_data.py

    # Download to a custom directory
    python download_training_data.py --output-dir /path/to/data/selfplay

    # Download only the latest N files (most recent first)
    python download_training_data.py --latest 100

    # Download specific range by index
    python download_training_data.py --start 0 --end 50

    # Use more parallel downloads
    python download_training_data.py --workers 8

    # Dry run - just list files without downloading
    python download_training_data.py --dry-run

After downloading, run the shuffle script to prepare data for training:
    cd train
    bash shuffle.sh ../data ../data ../data/tmp 8 384
"""

import argparse
import hashlib
import html.parser
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ARCHIVE_BASE_URL = "https://katagoarchive.org/kata1/trainingdata"
INDEX_URL = f"{ARCHIVE_BASE_URL}/index.html"

# Older runs
OLDER_RUNS = {
    "g170": "https://katagoarchive.org/g170/selfplay",
    "g104": "https://katagoarchive.org/g104",
    "g65": "https://katagoarchive.org/g65",
}

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) KataGo-Training-Downloader/1.0"


class LinkExtractor(html.parser.HTMLParser):
    """Extract all href links from an HTML page."""

    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for attr_name, attr_value in attrs:
                if attr_name == "href" and attr_value:
                    self.links.append(attr_value)


def make_request(url, retries=4, timeout=30):
    """Make an HTTP request with retries and exponential backoff."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** (attempt + 1)
            print(f"  Retry {attempt + 1}/{retries - 1} for {url} after {wait}s: {e}")
            time.sleep(wait)


def fetch_index(url):
    """Fetch and parse an HTML index page to extract file links."""
    print(f"Fetching index: {url}")
    resp = make_request(url)
    content = resp.read().decode("utf-8", errors="replace")

    parser = LinkExtractor()
    parser.feed(content)

    return parser.links


def discover_tgz_files(url):
    """Discover all .tgz files from the index page."""
    links = fetch_index(url)
    tgz_files = []
    for link in links:
        if link.endswith(".tgz") or link.endswith(".tar.gz") or link.endswith(".npz"):
            # Handle relative and absolute URLs
            if link.startswith("http://") or link.startswith("https://"):
                tgz_files.append(link)
            elif link.startswith("/"):
                tgz_files.append(f"https://katagoarchive.org{link}")
            else:
                base = url.rsplit("/", 1)[0]
                tgz_files.append(f"{base}/{link}")
    return tgz_files


def discover_subdirectories(url):
    """Discover subdirectories from index page (for recursive crawling)."""
    links = fetch_index(url)
    subdirs = []
    for link in links:
        # Skip parent directory links and non-directory links
        if link in ("../", ".", "..", "/"):
            continue
        if link.endswith("/") and not link.startswith("http"):
            base = url.rsplit("/", 1)[0] if not url.endswith("/") else url.rstrip("/")
            subdirs.append(f"{base}/{link}")
        elif link.endswith("/") and link.startswith("http"):
            subdirs.append(link)
    return subdirs


def discover_all_files(base_url, recursive=True):
    """Recursively discover all downloadable files."""
    all_files = []

    # First try the direct index
    try:
        files = discover_tgz_files(base_url)
        all_files.extend(files)
    except Exception as e:
        print(f"  Warning: Could not fetch {base_url}: {e}")

    # If recursive, also check subdirectories
    if recursive:
        try:
            subdirs = discover_subdirectories(base_url)
            for subdir in subdirs:
                try:
                    sub_files = discover_tgz_files(subdir)
                    all_files.extend(sub_files)
                    print(f"  Found {len(sub_files)} files in {subdir}")
                except Exception as e:
                    print(f"  Warning: Could not fetch {subdir}: {e}")
        except Exception:
            pass

    return all_files


def get_file_size(url):
    """Get file size via HEAD request."""
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
        resp = urllib.request.urlopen(req, timeout=15)
        size = resp.headers.get("Content-Length")
        return int(size) if size else None
    except Exception:
        return None


def download_file(url, output_path, chunk_size=1024 * 1024):
    """Download a file with resume support."""
    tmp_path = output_path + ".downloading"
    downloaded = 0

    # Resume support
    if os.path.exists(tmp_path):
        downloaded = os.path.getsize(tmp_path)

    # If final file already exists, skip
    if os.path.exists(output_path):
        return output_path, True, "already exists"

    headers = {"User-Agent": USER_AGENT}
    if downloaded > 0:
        headers["Range"] = f"bytes={downloaded}-"

    req = urllib.request.Request(url, headers=headers)

    try:
        resp = urllib.request.urlopen(req, timeout=60)
        total_size = resp.headers.get("Content-Length")
        if total_size:
            total_size = int(total_size) + downloaded

        mode = "ab" if downloaded > 0 else "wb"
        with open(tmp_path, mode) as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

        # Rename to final path
        os.rename(tmp_path, output_path)
        return output_path, True, f"{downloaded / 1024 / 1024:.1f} MB"

    except Exception as e:
        return output_path, False, str(e)


def extract_tgz(tgz_path, extract_dir):
    """Extract a .tgz file into the target directory."""
    try:
        with tarfile.open(tgz_path, "r:gz") as tar:
            # Safety: check for path traversal
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    print(f"  WARNING: Skipping unsafe path in {tgz_path}: {member.name}")
                    continue
            tar.extractall(path=extract_dir, filter="data")
        return True
    except (tarfile.TarError, AttributeError):
        # filter="data" requires Python 3.12+; fallback for older versions
        try:
            with tarfile.open(tgz_path, "r:gz") as tar:
                safe_members = []
                for member in tar.getmembers():
                    if member.name.startswith("/") or ".." in member.name:
                        continue
                    safe_members.append(member)
                tar.extractall(path=extract_dir, members=safe_members)
            return True
        except Exception as e:
            print(f"  ERROR extracting {tgz_path}: {e}")
            return False


def format_size(size_bytes):
    """Format bytes to human readable string."""
    if size_bytes is None:
        return "unknown size"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def download_and_extract_one(url, download_dir, selfplay_dir, keep_archives):
    """Download one file and extract it. Returns (url, success, message)."""
    filename = url.rsplit("/", 1)[-1]
    download_path = os.path.join(download_dir, filename)

    # Download
    _, success, msg = download_file(url, download_path)
    if not success:
        return url, False, f"download failed: {msg}"

    # If it's a .tgz or .tar.gz, extract it
    if filename.endswith(".tgz") or filename.endswith(".tar.gz"):
        ok = extract_tgz(download_path, selfplay_dir)
        if not ok:
            return url, False, "extraction failed"
        if not keep_archives:
            os.remove(download_path)
        return url, True, f"downloaded and extracted ({msg})"

    # If it's already an .npz file, just move it
    elif filename.endswith(".npz"):
        target = os.path.join(selfplay_dir, filename)
        if not os.path.exists(target):
            shutil.move(download_path, target)
        return url, True, f"downloaded ({msg})"

    return url, True, f"downloaded ({msg})"


def main():
    parser = argparse.ArgumentParser(
        description="Download KataGo training data from katagoarchive.org",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Download all kata1 data
  %(prog)s --latest 100                 # Download only latest 100 files
  %(prog)s --run g170                   # Download g170 run data
  %(prog)s --dry-run                    # List files without downloading
  %(prog)s --keep-archives              # Keep .tgz files after extraction
  %(prog)s --workers 8                  # Use 8 parallel downloads

After downloading, prepare data for training:
  cd train && bash shuffle.sh ../data ../data ../data/tmp 8 384
        """,
    )
    parser.add_argument(
        "--output-dir",
        default="data/selfplay",
        help="Output directory for extracted NPZ files (default: data/selfplay)",
    )
    parser.add_argument(
        "--download-dir",
        default=None,
        help="Directory to store downloaded archives (default: <output-dir>/../downloads)",
    )
    parser.add_argument(
        "--run",
        default="kata1",
        choices=["kata1", "g170", "g104", "g65"],
        help="Which training run to download (default: kata1)",
    )
    parser.add_argument(
        "--latest",
        type=int,
        default=None,
        help="Only download the latest N files (most recent by listing order)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Start index for file range to download",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End index (exclusive) for file range to download",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel download workers (default: 4)",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep .tgz files after extraction (default: delete after extracting)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list files, do not download",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Override the archive index URL",
    )

    args = parser.parse_args()

    # Resolve output directory relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.output_dir):
        output_dir = os.path.join(script_dir, args.output_dir)
    else:
        output_dir = args.output_dir

    if args.download_dir:
        download_dir = args.download_dir
    else:
        download_dir = os.path.join(os.path.dirname(output_dir), "downloads")

    # Determine base URL
    if args.url:
        base_url = args.url
    elif args.run == "kata1":
        base_url = ARCHIVE_BASE_URL
    else:
        base_url = OLDER_RUNS[args.run]

    index_url = base_url if base_url.endswith(".html") else f"{base_url}/index.html"

    print(f"=== KataGo Training Data Downloader ===")
    print(f"Run:          {args.run}")
    print(f"Index URL:    {index_url}")
    print(f"Output dir:   {output_dir}")
    print(f"Download dir: {download_dir}")
    print(f"Workers:      {args.workers}")
    print()

    # Discover files
    print("Discovering available files...")
    try:
        all_files = discover_all_files(base_url, recursive=True)
        # Also try index.html directly
        if not all_files:
            all_files = discover_all_files(index_url, recursive=True)
    except Exception as e:
        print(f"ERROR: Could not fetch file listing from {index_url}")
        print(f"  {e}")
        print()
        print("If the website is down or blocking requests, you can try:")
        print("  1. Download manually from a browser at:")
        print(f"     {index_url}")
        print("  2. Use wget:")
        print(f"     wget -r -np -nd -A '*.tgz' -P {download_dir} {base_url}/")
        print("  3. Use a mirror or alternative URL with --url")
        sys.exit(1)

    if not all_files:
        print("No downloadable files found at the index page.")
        print()
        print("The archive may use a different page structure. Try downloading with wget:")
        print(f"  wget -r -np -nd -A '*.tgz' -P {download_dir} {base_url}/")
        print()
        print("Or try with curl to inspect the page:")
        print(f"  curl -L '{index_url}'")
        sys.exit(1)

    # Deduplicate
    all_files = list(dict.fromkeys(all_files))

    print(f"Found {len(all_files)} files")

    # Apply range filters
    if args.latest:
        all_files = all_files[-args.latest:]
        print(f"  Selected latest {len(all_files)} files")
    elif args.start is not None or args.end is not None:
        start = args.start or 0
        end = args.end or len(all_files)
        all_files = all_files[start:end]
        print(f"  Selected files [{start}:{end}], {len(all_files)} files")

    if args.dry_run:
        print()
        print("Files that would be downloaded:")
        for i, url in enumerate(all_files):
            print(f"  [{i:4d}] {url}")
        print(f"\nTotal: {len(all_files)} files")
        return

    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(download_dir, exist_ok=True)

    # Download and extract
    print(f"\nStarting download of {len(all_files)} files with {args.workers} workers...")
    print()

    success_count = 0
    fail_count = 0
    skip_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for url in all_files:
            future = executor.submit(
                download_and_extract_one,
                url,
                download_dir,
                output_dir,
                args.keep_archives,
            )
            futures[future] = url

        for future in as_completed(futures):
            url = futures[future]
            filename = url.rsplit("/", 1)[-1]
            try:
                _, success, msg = future.result()
                if success:
                    if "already exists" in msg:
                        skip_count += 1
                        print(f"  SKIP  {filename} ({msg})")
                    else:
                        success_count += 1
                        print(f"  OK    {filename} ({msg})")
                else:
                    fail_count += 1
                    print(f"  FAIL  {filename} ({msg})")
            except Exception as e:
                fail_count += 1
                print(f"  ERROR {filename}: {e}")

    print()
    print(f"=== Download Complete ===")
    print(f"  Success:  {success_count}")
    print(f"  Skipped:  {skip_count}")
    print(f"  Failed:   {fail_count}")
    print()

    # Count NPZ files in output
    npz_count = 0
    for root, dirs, files in os.walk(output_dir):
        npz_count += sum(1 for f in files if f.endswith(".npz"))
    print(f"Total NPZ files in {output_dir}: {npz_count}")

    if npz_count > 0:
        print()
        print("Next steps - shuffle data for training:")
        print(f"  cd {os.path.join(script_dir, 'train')}")
        data_dir = os.path.dirname(output_dir)
        print(f"  bash shuffle.sh {data_dir} {data_dir} {data_dir}/tmp 8 384")


if __name__ == "__main__":
    main()
