#!/bin/bash
# Preprocess KataGo NPZ training data for faster loading.
#
# Usage:
#   ./preprocess.sh <input_base_dir> <output_base_dir> [options...]
#
# Example:
#   ./preprocess.sh ../data/shuffleddata/kata1 ../data/preprocessed/kata1 \
#       --pos-len 19 --symmetry-type xyt --symmetry-mode expand --workers 8

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <input_base_dir> <output_base_dir> [preprocess.py options...]"
    echo ""
    echo "Preprocesses train/ and val/ subdirectories."
    echo "Extra arguments are passed to preprocess.py."
    exit 1
fi

INPUT_BASE="$1"
OUTPUT_BASE="$2"
shift 2

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for subdir in train val; do
    input_dir="${INPUT_BASE}/${subdir}"
    output_dir="${OUTPUT_BASE}/${subdir}"

    if [ ! -d "$input_dir" ]; then
        echo "Skipping ${subdir}/ (not found: ${input_dir})"
        continue
    fi

    echo "=== Preprocessing ${subdir}/ ==="
    python3 "${SCRIPT_DIR}/preprocess.py" \
        --input-dir "$input_dir" \
        --output-dir "$output_dir" \
        "$@"
done

echo "=== All done ==="
