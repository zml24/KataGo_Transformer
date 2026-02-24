#!/usr/bin/env bash
set -euo pipefail

# Example: prepare train/val npz from data/selfplay, then run train/train.py.
# Usage:
#   bash train_trainpy_from_selfplay_example.sh
#
# Optional:
#   CUDA_VISIBLE_DEVICES=0 bash train_trainpy_from_selfplay_example.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

SELFPLAY_DIR="${ROOT_DIR}/data/selfplay"
SHUFFLED_DIR="${ROOT_DIR}/data/shuffleddata/trainpy_current"
TMP_ROOT="${TMPDIR:-/tmp}/katago_trainpy_shuffle"
TRAIN_OUT="${ROOT_DIR}/data/train/trainpy_b14c192h6"

MODEL_KIND="b11c96h3tfrs-bng-silu"
POS_LEN=19
BATCH_SIZE=256
NUM_PROCESSES=8
MAX_TRAINING_SAMPLES=100000000
LR=3e-4

if [ ! -d "${SELFPLAY_DIR}" ]; then
  echo "selfplay directory not found: ${SELFPLAY_DIR}" >&2
  exit 1
fi

# echo "[1/3] Prepare output dirs"
# rm -rf "${SHUFFLED_DIR}"
# mkdir -p "${SHUFFLED_DIR}" "${TMP_ROOT}/train" "${TMP_ROOT}/val"

# echo "[2/3] Shuffle selfplay -> train split"
# python3 train/shuffle.py \
#   "${SELFPLAY_DIR}" \
#   -expand-window-per-row 1.0 \
#   -taper-window-exponent 1.0 \
#   -min-rows 1 \
#   -keep-target-rows 999999999 \
#   -out-dir "${SHUFFLED_DIR}/train" \
#   -out-tmp-dir "${TMP_ROOT}/train" \
#   -approx-rows-per-out-file 70000 \
#   -num-processes "${NUM_PROCESSES}" \
#   -batch-size "${BATCH_SIZE}" \
#   -only-include-md5-path-prop-lbound 0.00 \
#   -only-include-md5-path-prop-ubound 0.95 \
#   -output-npz

# echo "[2/3] Shuffle selfplay -> val split"
# python3 train/shuffle.py \
#   "${SELFPLAY_DIR}" \
#   -expand-window-per-row 1.0 \
#   -taper-window-exponent 1.0 \
#   -min-rows 1 \
#   -keep-target-rows 999999999 \
#   -out-dir "${SHUFFLED_DIR}/val" \
#   -out-tmp-dir "${TMP_ROOT}/val" \
#   -approx-rows-per-out-file 70000 \
#   -num-processes "${NUM_PROCESSES}" \
#   -batch-size "${BATCH_SIZE}" \
#   -only-include-md5-path-prop-lbound 0.95 \
#   -only-include-md5-path-prop-ubound 1.00 \
#   -output-npz

echo "[3/3] Start train/train.py"
python3 -u train/train.py \
  -traindir "${TRAIN_OUT}" \
  -datadir "${SHUFFLED_DIR}" \
  -pos-len "${POS_LEN}" \
  -batch-size "${BATCH_SIZE}" \
  -model-kind "${MODEL_KIND}" \
  -lr "${LR}" \
  -max-training-samples "${MAX_TRAINING_SAMPLES}" \
  -symmetry-type xyt \
  -print-every 50 \
  -save-every-samples 1000000 \
  -val-every-samples 1000000 \
  -warmup-samples 2000000 \
  -enable-history-matrices
