#!/bin/bash
set -euo pipefail

# =============================================================================
# Download KataGo training data from katagoarchive.org
#
# The archive (S3 bucket: katago-public) hosts training data as .tgz archives,
# each containing .npz selfplay data files. The site uses S3 static website
# hosting - index.html pages serve as directory listings.
#
# Usage:
#   bash download_training_data.sh [OUTPUT_DIR] [OPTIONS]
#
# Examples:
#   bash download_training_data.sh                          # Default: data/selfplay
#   bash download_training_data.sh /path/to/data/selfplay   # Custom output dir
#   bash download_training_data.sh data/selfplay --keep-tgz # Keep archives
#   bash download_training_data.sh data/selfplay --dry-run   # List only
#
# After downloading, prepare data for training:
#   cd train
#   bash shuffle.sh ../data ../data ../data/tmp 8 384
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults
OUTPUT_DIR="${1:-${SCRIPT_DIR}/data/selfplay}"
shift || true

INDEX_URL="https://katagoarchive.org/kata1/trainingdata/index.html"
DOWNLOAD_DIR="$(dirname "$OUTPUT_DIR")/downloads"
KEEP_TGZ=false
DRY_RUN=false
PARALLEL=4
RETRIES=4

# Parse optional flags
while [[ $# -gt 0 ]]; do
    case $1 in
        --keep-tgz)   KEEP_TGZ=true; shift ;;
        --dry-run)    DRY_RUN=true; shift ;;
        --parallel)   PARALLEL="$2"; shift 2 ;;
        --retries)    RETRIES="$2"; shift 2 ;;
        --url)        INDEX_URL="$2"; shift 2 ;;
        --run)
            case "$2" in
                kata1) INDEX_URL="https://katagoarchive.org/kata1/trainingdata/index.html" ;;
                g170)  INDEX_URL="https://katagoarchive.org/g170/selfplay/index.html" ;;
                g104)  INDEX_URL="https://katagoarchive.org/g104/index.html" ;;
                g65)   INDEX_URL="https://katagoarchive.org/g65/index.html" ;;
                *)     echo "Unknown run: $2"; exit 1 ;;
            esac
            shift 2
            ;;
        --help|-h)
            head -25 "$0" | tail -20
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Ensure index URL ends with index.html
if [[ "$INDEX_URL" != *".html" ]]; then
    INDEX_URL="${INDEX_URL%/}/index.html"
fi

# Base URL for resolving relative links (directory containing index.html)
BASE_URL="${INDEX_URL%/index.html}"

echo "=== KataGo Training Data Downloader ==="
echo "Index URL:     ${INDEX_URL}"
echo "Output dir:    ${OUTPUT_DIR}"
echo "Download dir:  ${DOWNLOAD_DIR}"
echo "Keep archives: ${KEEP_TGZ}"
echo "Parallel:      ${PARALLEL}"
echo ""

# Check for curl or wget
if command -v curl &>/dev/null; then
    FETCHER="curl"
elif command -v wget &>/dev/null; then
    FETCHER="wget"
else
    echo "ERROR: Either curl or wget is required."
    exit 1
fi

fetch_url() {
    local url="$1"
    if [ "$FETCHER" = "curl" ]; then
        curl -sL --retry "$RETRIES" --retry-delay 2 --max-time 180 "$url"
    else
        wget -qO- --tries="$RETRIES" --timeout=180 "$url"
    fi
}

download_file() {
    local url="$1"
    local target="$2"
    if [ "$FETCHER" = "curl" ]; then
        curl -sL --retry "$RETRIES" --retry-delay 2 --max-time 600 \
            -C - -o "$target" "$url"
    else
        wget -q -c --tries="$RETRIES" --timeout=600 -O "$target" "$url"
    fi
}

mkdir -p "$OUTPUT_DIR"
mkdir -p "$DOWNLOAD_DIR"

# ---- Step 1: Discover available files from index.html ----

echo "Step 1: Fetching index page..."

TGZ_LIST_FILE=$(mktemp)
trap "rm -f ${TGZ_LIST_FILE}" EXIT

# Fetch index.html and extract href links to .tgz files
# Uses sed/grep compatible with both macOS and Linux (no grep -P)
INDEX_CONTENT=$(fetch_url "$INDEX_URL" 2>/dev/null || true)

if [ -z "$INDEX_CONTENT" ]; then
    echo "WARNING: Got empty response from ${INDEX_URL}"
    echo ""
    echo "The S3 static website may be temporarily unavailable."
    echo "Try opening in a browser: ${INDEX_URL}"
    echo ""
    echo "Alternative: use the Python script instead:"
    echo "  python download_training_data.py"
    exit 1
fi

# Extract all href="..." values that end in .tgz, .tar.gz, or .npz
# This uses only POSIX-compatible grep/sed (works on macOS and Linux)
echo "$INDEX_CONTENT" \
    | sed 's/href="/\nhref="/g' \
    | grep '^href="' \
    | sed 's/^href="//;s/".*//' \
    | grep -E '\.(tgz|tar\.gz|npz)$' \
    | sort -u > "$TGZ_LIST_FILE"

NUM_FILES=$(wc -l < "$TGZ_LIST_FILE" | tr -d ' ')

echo "  Received $(echo "$INDEX_CONTENT" | wc -c | tr -d ' ') bytes"
echo "  Found ${NUM_FILES} downloadable files"

if [ "$NUM_FILES" -eq 0 ]; then
    echo ""
    echo "No .tgz/.npz files found in the index page."
    echo ""
    echo "Debug: first 500 chars of response:"
    echo "$INDEX_CONTENT" | head -c 500
    echo ""
    echo ""
    echo "Try the Python script instead:"
    echo "  python download_training_data.py"
    exit 1
fi

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "Files that would be downloaded:"
    head -20 "$TGZ_LIST_FILE" | while read -r f; do echo "  $f"; done
    if [ "$NUM_FILES" -gt 20 ]; then
        echo "  ... and $((NUM_FILES - 20)) more"
    fi
    echo ""
    echo "Total: ${NUM_FILES} files"
    exit 0
fi

# ---- Step 2: Download files ----
echo ""
echo "Step 2: Downloading ${NUM_FILES} files (${PARALLEL} parallel)..."
echo ""

SUCCESS=0
FAIL=0
SKIP=0
COUNT=0

download_one() {
    local filename="$1"
    local url

    # Build full URL from relative filename
    if [[ "$filename" == http* ]]; then
        url="$filename"
        filename=$(basename "$filename")
    elif [[ "$filename" == /* ]]; then
        url="https://katagoarchive.org${filename}"
        filename=$(basename "$filename")
    else
        url="${BASE_URL}/${filename}"
        filename=$(basename "$filename")
    fi

    local target="${DOWNLOAD_DIR}/${filename}"

    # Skip if already downloaded
    if [ -f "$target" ]; then
        echo "  SKIP  ${filename}"
        return 0
    fi

    # Download with retries and resume
    local tmp="${target}.part"
    if download_file "$url" "$tmp" 2>/dev/null; then
        mv "$tmp" "$target"
        local size
        size=$(du -h "$target" 2>/dev/null | cut -f1)
        echo "  OK    ${filename} (${size})"
    else
        rm -f "$tmp"
        echo "  FAIL  ${filename}"
        return 1
    fi
}

export -f download_one download_file
export BASE_URL DOWNLOAD_DIR RETRIES FETCHER

# Run downloads in parallel using xargs
cat "$TGZ_LIST_FILE" | xargs -P "$PARALLEL" -I {} bash -c 'download_one "$@"' _ {}

echo ""

# ---- Step 3: Extract archives ----
echo "Step 3: Extracting archives to ${OUTPUT_DIR}..."

EXTRACTED=0
EXTRACT_FAIL=0

for tgz_file in "$DOWNLOAD_DIR"/*.tgz "$DOWNLOAD_DIR"/*.tar.gz; do
    [ -f "$tgz_file" ] || continue

    basename_file=$(basename "$tgz_file")
    echo "  Extracting ${basename_file}..."

    if tar xzf "$tgz_file" -C "$OUTPUT_DIR" 2>/dev/null; then
        EXTRACTED=$((EXTRACTED + 1))
        if [ "$KEEP_TGZ" = false ]; then
            rm -f "$tgz_file"
        fi
    else
        echo "  WARNING: Failed to extract ${basename_file}"
        EXTRACT_FAIL=$((EXTRACT_FAIL + 1))
    fi
done

# Move any standalone .npz files
for npz_file in "$DOWNLOAD_DIR"/*.npz; do
    [ -f "$npz_file" ] || continue
    mv "$npz_file" "$OUTPUT_DIR/" 2>/dev/null || true
done

# ---- Summary ----
NPZ_COUNT=0
if command -v find &>/dev/null; then
    NPZ_COUNT=$(find "$OUTPUT_DIR" -name "*.npz" -type f 2>/dev/null | wc -l | tr -d ' ')
fi

echo ""
echo "=== Download Complete ==="
echo "  Archives extracted: ${EXTRACTED}"
echo "  Extract failures:   ${EXTRACT_FAIL}"
echo "  Total NPZ files:    ${NPZ_COUNT}"
echo "  Output directory:   ${OUTPUT_DIR}"
echo ""

if [ "$NPZ_COUNT" -gt 0 ]; then
    DATA_DIR=$(dirname "$OUTPUT_DIR")
    echo "Next steps - shuffle and train:"
    echo "  cd ${SCRIPT_DIR}/train"
    echo "  bash shuffle.sh ${DATA_DIR} ${DATA_DIR} ${DATA_DIR}/tmp 8 384"
else
    echo "No NPZ files found. Re-run the script to retry failed downloads."
    echo "Or try the Python script: python download_training_data.py"
fi
