#!/usr/bin/env python3
"""
Download KataGo training data from katagoarchive.org for use with KataGo_Transformer.

The archive (S3 bucket: katago-public) hosts training data as .tgz archives,
each containing .npz selfplay data files. This script downloads and extracts
them into the directory structure expected by shuffle.sh and train scripts.

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
import html.parser
import os
import re
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

# S3 bucket: katago-public, region: us-east-1
# Served via S3 static website hosting at katagoarchive.org
# Bucket does NOT allow ListObjects API - must use index.html pages

ARCHIVE_URLS = {
    "kata1": "https://katagoarchive.org/kata1/trainingdata/index.html",
    "g170": "https://katagoarchive.org/g170/selfplay/index.html",
    "g104": "https://katagoarchive.org/g104/index.html",
    "g65": "https://katagoarchive.org/g65/index.html",
}

USER_AGENT = "Mozilla/5.0 (compatible; KataGo-Training-Downloader/1.0)"


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


def make_request(url, retries=4, timeout=120):
    """Make an HTTP request with retries and exponential backoff."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error = None
    for attempt in range(retries):
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            last_error = e
            if attempt == retries - 1:
                raise
            wait = 2 ** (attempt + 1)
            print(f"  Retry {attempt + 1}/{retries} for {os.path.basename(url)} "
                  f"after {wait}s: {e}")
            time.sleep(wait)


def fetch_index(index_url):
    """Fetch an S3 static website index.html and extract all links.

    The katagoarchive.org site uses S3 static website hosting.
    Bare directory URLs return 403; only index.html URLs work.
    The index page may be large (chunked transfer), so we use a generous timeout.
    """
    print(f"Fetching index: {index_url}")
    resp = make_request(index_url, timeout=180)

    # Read in chunks to handle large chunked-transfer responses
    chunks = []
    while True:
        chunk = resp.read(1024 * 1024)  # 1MB at a time
        if not chunk:
            break
        chunks.append(chunk)
    content = b"".join(chunks).decode("utf-8", errors="replace")
    print(f"  Received {len(content)} bytes")

    parser = LinkExtractor()
    parser.feed(content)
    return parser.links


def resolve_url(link, index_url):
    """Resolve a relative link against the index page URL."""
    if link.startswith("http://") or link.startswith("https://"):
        return link
    # Get the base URL (directory containing index.html)
    base = index_url.rsplit("/", 1)[0]
    if link.startswith("/"):
        # Absolute path - resolve against origin
        from urllib.parse import urlparse
        parsed = urlparse(index_url)
        return f"{parsed.scheme}://{parsed.netloc}{link}"
    return f"{base}/{link}"


def discover_files(index_url, extensions=(".tgz", ".tar.gz", ".npz")):
    """Discover all downloadable files from an index.html page."""
    links = fetch_index(index_url)

    files = []
    for link in links:
        if any(link.endswith(ext) for ext in extensions):
            files.append(resolve_url(link, index_url))

    # Also discover subdirectory index pages and recurse
    subdirs = []
    for link in links:
        if link in ("../", ".", "..", "/", "index.html"):
            continue
        if link.endswith("/"):
            subdir_url = resolve_url(link, index_url)
            if not subdir_url.endswith("index.html"):
                subdir_url = subdir_url.rstrip("/") + "/index.html"
            subdirs.append(subdir_url)

    for subdir_url in subdirs:
        try:
            sub_links = fetch_index(subdir_url)
            for link in sub_links:
                if any(link.endswith(ext) for ext in extensions):
                    files.append(resolve_url(link, subdir_url))
            print(f"  Found files in subdirectory: {subdir_url}")
        except Exception as e:
            print(f"  Warning: Could not fetch {subdir_url}: {e}")

    return files


def download_file(url, output_path, retries=4, chunk_size=1024 * 1024):
    """Download a file with resume support and retries."""
    tmp_path = output_path + ".part"

    # If final file already exists, skip
    if os.path.exists(output_path):
        return output_path, True, "already exists"

    downloaded = 0
    # Resume from partial download
    if os.path.exists(tmp_path):
        downloaded = os.path.getsize(tmp_path)

    last_error = None
    for attempt in range(retries):
        try:
            headers = {"User-Agent": USER_AGENT}
            if downloaded > 0:
                headers["Range"] = f"bytes={downloaded}-"

            req = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(req, timeout=120)

            # Check if server supports range requests
            if downloaded > 0 and resp.status != 206:
                # Server doesn't support range; start over
                downloaded = 0

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
            size_str = f"{downloaded / 1024 / 1024:.1f} MB"
            return output_path, True, size_str

        except Exception as e:
            last_error = e
            if attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                time.sleep(wait)

    return output_path, False, str(last_error)


def extract_tgz(tgz_path, extract_dir):
    """Extract a .tgz file, filtering out unsafe paths."""
    try:
        with tarfile.open(tgz_path, "r:gz") as tar:
            safe_members = []
            for member in tar.getmembers():
                # Skip absolute paths and path traversal
                if member.name.startswith("/") or ".." in member.name:
                    continue
                safe_members.append(member)
            tar.extractall(path=extract_dir, members=safe_members)
        return True
    except Exception as e:
        print(f"  ERROR extracting {os.path.basename(tgz_path)}: {e}")
        return False


def download_and_extract_one(url, download_dir, selfplay_dir, keep_archives):
    """Download one file and extract it. Returns (url, success, message)."""
    filename = url.rsplit("/", 1)[-1]
    download_path = os.path.join(download_dir, filename)

    # Download
    _, success, msg = download_file(url, download_path)
    if not success:
        return url, False, f"download failed: {msg}"

    # Extract .tgz / .tar.gz files
    if filename.endswith(".tgz") or filename.endswith(".tar.gz"):
        ok = extract_tgz(download_path, selfplay_dir)
        if not ok:
            return url, False, "extraction failed"
        if not keep_archives:
            try:
                os.remove(download_path)
            except OSError:
                pass
        return url, True, f"downloaded+extracted ({msg})"

    # Move standalone .npz files
    elif filename.endswith(".npz"):
        target = os.path.join(selfplay_dir, filename)
        if not os.path.exists(target):
            shutil.copy2(download_path, target)
            if not keep_archives:
                try:
                    os.remove(download_path)
                except OSError:
                    pass
        return url, True, f"downloaded ({msg})"

    return url, True, f"downloaded ({msg})"


def main():
    parser = argparse.ArgumentParser(
        description="Download KataGo training data from katagoarchive.org",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
The archive (S3 bucket: katago-public) serves index.html pages via
S3 static website hosting. This script parses those pages to find
.tgz files, downloads them, and extracts the .npz selfplay data.

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
        "--output-dir", default="data/selfplay",
        help="Output directory for extracted NPZ files (default: data/selfplay)",
    )
    parser.add_argument(
        "--download-dir", default=None,
        help="Directory for downloaded archives (default: <output-dir>/../downloads)",
    )
    parser.add_argument(
        "--run", default="kata1", choices=["kata1", "g170", "g104", "g65"],
        help="Which training run to download (default: kata1)",
    )
    parser.add_argument(
        "--latest", type=int, default=None,
        help="Only download the latest N files (most recent by listing order)",
    )
    parser.add_argument(
        "--start", type=int, default=None,
        help="Start index for file range to download",
    )
    parser.add_argument(
        "--end", type=int, default=None,
        help="End index (exclusive) for file range to download",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel download workers (default: 4)",
    )
    parser.add_argument(
        "--keep-archives", action="store_true",
        help="Keep .tgz files after extraction (default: delete)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only list files, do not download",
    )
    parser.add_argument(
        "--url", default=None,
        help="Override the archive index URL (must point to an index.html page)",
    )

    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = (args.output_dir if os.path.isabs(args.output_dir)
                  else os.path.join(script_dir, args.output_dir))
    download_dir = (args.download_dir if args.download_dir
                    else os.path.join(os.path.dirname(output_dir), "downloads"))

    # Determine index URL
    if args.url:
        index_url = args.url
        if not index_url.endswith(".html"):
            index_url = index_url.rstrip("/") + "/index.html"
    else:
        index_url = ARCHIVE_URLS[args.run]

    print("=== KataGo Training Data Downloader ===")
    print(f"Run:          {args.run}")
    print(f"Index URL:    {index_url}")
    print(f"Output dir:   {output_dir}")
    print(f"Download dir: {download_dir}")
    print(f"Workers:      {args.workers}")
    print()

    # --- Discover files ---
    print("Discovering available files...")
    try:
        all_files = discover_files(index_url)
    except Exception as e:
        print(f"\nERROR: Could not fetch file listing from {index_url}")
        print(f"  {e}")
        print()
        print("Troubleshooting:")
        print(f"  1. Open in browser: {index_url}")
        print(f"  2. curl -L '{index_url}' | head -100")
        print(f"  3. wget alternative:")
        base_dir = index_url.rsplit("/", 1)[0]
        print(f"     wget -r -np -nd -A '*.tgz' -P downloads/ '{base_dir}/'")
        sys.exit(1)

    # Deduplicate preserving order
    seen = set()
    unique_files = []
    for f in all_files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)
    all_files = unique_files

    if not all_files:
        print("No downloadable files (.tgz/.npz) found on the index page.")
        print()
        print("The page might have a different structure. Try inspecting it:")
        print(f"  curl -L '{index_url}' | head -200")
        sys.exit(1)

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

    # --- Dry run ---
    if args.dry_run:
        print("\nFiles that would be downloaded:")
        for i, url in enumerate(all_files):
            print(f"  [{i:4d}] {url.rsplit('/', 1)[-1]}")
        print(f"\nTotal: {len(all_files)} files")
        print(f"\nFull URLs saved to stdout. Pipe to a file for batch download:")
        print(f"  python {sys.argv[0]} --dry-run 2>/dev/null | grep http > urls.txt")
        return

    # --- Download ---
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(download_dir, exist_ok=True)

    print(f"\nDownloading {len(all_files)} files with {args.workers} workers...")
    print()

    success_count = 0
    fail_count = 0
    skip_count = 0
    total = len(all_files)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_and_extract_one, url, download_dir, output_dir,
                args.keep_archives
            ): url
            for url in all_files
        }

        done = 0
        for future in as_completed(futures):
            done += 1
            url = futures[future]
            filename = url.rsplit("/", 1)[-1]
            try:
                _, success, msg = future.result()
                if success:
                    if "already exists" in msg:
                        skip_count += 1
                        status = "SKIP"
                    else:
                        success_count += 1
                        status = "OK  "
                else:
                    fail_count += 1
                    status = "FAIL"
                print(f"  [{done}/{total}] {status} {filename} ({msg})")
            except Exception as e:
                fail_count += 1
                print(f"  [{done}/{total}] ERR  {filename}: {e}")

    # --- Summary ---
    print()
    print("=== Download Complete ===")
    print(f"  Downloaded: {success_count}")
    print(f"  Skipped:    {skip_count}")
    print(f"  Failed:     {fail_count}")

    npz_count = sum(
        1 for root, _, files in os.walk(output_dir)
        for f in files if f.endswith(".npz")
    )
    print(f"  NPZ files:  {npz_count}")
    print(f"  Output dir: {output_dir}")

    if fail_count > 0:
        print(f"\n  {fail_count} files failed. Re-run the script to retry "
              "(already-downloaded files will be skipped).")

    if npz_count > 0:
        data_dir = os.path.dirname(output_dir)
        print(f"\nNext steps - shuffle data for training:")
        print(f"  cd {os.path.join(script_dir, 'train')}")
        print(f"  bash shuffle.sh {data_dir} {data_dir} {data_dir}/tmp 8 384")


if __name__ == "__main__":
    main()
