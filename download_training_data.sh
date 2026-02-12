#!/bin/bash
set -euo pipefail

# =============================================================================
# Download KataGo training data from katagoarchive.org
#
# The archive hosts training data as .tgz archives, each containing
# multiple .npz selfplay data files. This script downloads and extracts
# them into the directory structure expected by shuffle.sh and train scripts.
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
#   bash train_muon_ki.sh ../data ../data/shuffleddata/current ...
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults
OUTPUT_DIR="${1:-${SCRIPT_DIR}/data/selfplay}"
shift || true

ARCHIVE_URL="https://katagoarchive.org/kata1/trainingdata"
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
        --url)        ARCHIVE_URL="$2"; shift 2 ;;
        --help|-h)
            head -25 "$0" | tail -20
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== KataGo Training Data Downloader ==="
echo "Archive URL:   ${ARCHIVE_URL}"
echo "Output dir:    ${OUTPUT_DIR}"
echo "Download dir:  ${DOWNLOAD_DIR}"
echo "Keep archives: ${KEEP_TGZ}"
echo "Parallel:      ${PARALLEL}"
echo ""

# Check dependencies
for cmd in wget; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: $cmd is required but not found. Install it first."
        echo "  Ubuntu/Debian: sudo apt-get install wget"
        echo "  macOS: brew install wget"
        exit 1
    fi
done

mkdir -p "$OUTPUT_DIR"
mkdir -p "$DOWNLOAD_DIR"

# ---- Step 1: Discover available files ----

echo "Step 1: Discovering files from index page..."
INDEX_URL="${ARCHIVE_URL}/index.html"

# Try to fetch the index and extract all .tgz links
TGZ_LIST_FILE=$(mktemp)
trap "rm -f ${TGZ_LIST_FILE}" EXIT

# Method 1: wget + parse HTML for links
wget -q -O - --tries="$RETRIES" --timeout=30 "$INDEX_URL" 2>/dev/null \
    | grep -oP 'href="[^"]*\.tgz"' \
    | sed 's/href="//;s/"$//' \
    | sort -u > "$TGZ_LIST_FILE" 2>/dev/null || true

# If no results, try the directory listing directly
if [ ! -s "$TGZ_LIST_FILE" ]; then
    wget -q -O - --tries="$RETRIES" --timeout=30 "${ARCHIVE_URL}/" 2>/dev/null \
        | grep -oP 'href="[^"]*\.(tgz|tar\.gz|npz)"' \
        | sed 's/href="//;s/"$//' \
        | sort -u > "$TGZ_LIST_FILE" 2>/dev/null || true
fi

# Count discovered files
NUM_FILES=$(wc -l < "$TGZ_LIST_FILE" | tr -d ' ')

if [ "$NUM_FILES" -eq 0 ]; then
    echo ""
    echo "Could not automatically discover files from the index page."
    echo ""
    echo "Falling back to wget recursive download..."
    echo "This will crawl the archive and download all .tgz files."
    echo ""

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would run:"
        echo "  wget -r -np -nd -A '*.tgz' --tries=${RETRIES} -P '${DOWNLOAD_DIR}' '${ARCHIVE_URL}/'"
        exit 0
    fi

    # Recursive wget download
    wget -r -np -nd -A '*.tgz,*.npz' \
        --tries="$RETRIES" \
        --timeout=60 \
        --wait=1 \
        --random-wait \
        -c \
        -P "$DOWNLOAD_DIR" \
        "${ARCHIVE_URL}/" || {
            echo ""
            echo "wget recursive download completed (some errors may be normal)."
        }

else
    echo "Found ${NUM_FILES} files to download."
    echo ""

    if [ "$DRY_RUN" = true ]; then
        echo "Files that would be downloaded:"
        cat "$TGZ_LIST_FILE" | head -20
        if [ "$NUM_FILES" -gt 20 ]; then
            echo "  ... and $((NUM_FILES - 20)) more"
        fi
        echo ""
        echo "Total: ${NUM_FILES} files"
        exit 0
    fi

    # ---- Step 2: Download files ----
    echo "Step 2: Downloading ${NUM_FILES} files (${PARALLEL} parallel)..."
    echo ""

    # Build full URLs and download with wget
    SUCCESS=0
    FAIL=0
    SKIP=0

    download_one() {
        local filename="$1"
        local url

        # Build full URL
        if [[ "$filename" == http* ]]; then
            url="$filename"
            filename=$(basename "$filename")
        else
            url="${ARCHIVE_URL}/${filename}"
        fi

        local target="${DOWNLOAD_DIR}/${filename}"

        # Skip if already downloaded
        if [ -f "$target" ]; then
            echo "  SKIP  ${filename}"
            return 0
        fi

        # Download with retries and resume
        wget -q -c \
            --tries="$RETRIES" \
            --timeout=60 \
            --waitretry=2 \
            -O "${target}.part" \
            "$url" && \
            mv "${target}.part" "$target" && \
            echo "  OK    ${filename}" || \
            { echo "  FAIL  ${filename}"; return 1; }
    }

    export -f download_one
    export ARCHIVE_URL DOWNLOAD_DIR RETRIES

    # Run downloads in parallel
    cat "$TGZ_LIST_FILE" | xargs -P "$PARALLEL" -I {} bash -c 'download_one "$@"' _ {}

fi

# ---- Step 3: Extract archives ----
echo ""
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
NPZ_COUNT=$(find "$OUTPUT_DIR" -name "*.npz" -type f 2>/dev/null | wc -l | tr -d ' ')

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
    echo "  bash train_muon_ki.sh ${DATA_DIR} ${DATA_DIR}/shuffleddata/current \\"
    echo "    b14c192h6tfrs_1 b14c192h6tfrs-bng-silu 384 extra -multi-gpus 0,1,2,3"
else
    echo "No NPZ files found. The archive structure may differ from expected."
    echo "Please check the archive manually at:"
    echo "  ${ARCHIVE_URL}/index.html"
fi
