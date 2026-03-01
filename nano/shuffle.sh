#!/bin/bash -eu
set -o pipefail
{
# Shuffles all NPZ data for nano training, splitting into train/val sets.
# Usage: bash shuffle.sh INPUTDIR DATADIR TMPDIR NTHREADS BATCHSIZE [extra shuffle.py args...]

if [[ $# -lt 5 ]]; then
    echo "Usage: $0 INPUTDIR DATADIR TMPDIR NTHREADS BATCHSIZE [extra args...]"
    echo "INPUTDIR   directory containing NPZ files (searched recursively)"
    echo "DATADIR    directory to place shuffleddata/ output"
    echo "TMPDIR     scratch space, ideally on fast local disk"
    echo "NTHREADS   number of parallel processes for shuffling"
    echo "BATCHSIZE  batch size for training examples"
    exit 0
fi
INPUTDIR="$1"
shift
DATADIR="$1"
shift
TMPDIR="$1"
shift
NTHREADS="$1"
shift
BATCHSIZE="$1"
shift

SCRIPTDIR="$(cd "$(dirname "$0")" && pwd)"

#------------------------------------------------------------------------------

OUTDIR=$(date "+%Y%m%d-%H%M%S")

mkdir -p "$DATADIR"/shuffleddata/$OUTDIR
mkdir -p "$TMPDIR"/train
mkdir -p "$TMPDIR"/val

echo "Beginning shuffle at $(date "+%Y-%m-%d %H:%M:%S")"

# Train split (95% of files by md5 hash)
(
    time python3 "$SCRIPTDIR"/shuffle.py \
         "$INPUTDIR" \
         --out-dir "$DATADIR"/shuffleddata/$OUTDIR/train \
         --tmp-dir "$TMPDIR"/train \
         --num-processes "$NTHREADS" \
         --batch-size "$BATCHSIZE" \
         --approx-rows-per-out-file 70000 \
         --md5-lbound 0.00 \
         --md5-ubound 0.95 \
         "$@" \
         2>&1 | tee "$DATADIR"/shuffleddata/$OUTDIR/outtrain.txt &

    wait
)

# Validation split (5% of files by md5 hash)
(
    time python3 "$SCRIPTDIR"/shuffle.py \
         "$INPUTDIR" \
         --out-dir "$DATADIR"/shuffleddata/$OUTDIR/val \
         --tmp-dir "$TMPDIR"/val \
         --num-processes "$NTHREADS" \
         --batch-size "$BATCHSIZE" \
         --approx-rows-per-out-file 70000 \
         --md5-lbound 0.95 \
         --md5-ubound 1.00 \
         "$@" \
         2>&1 | tee "$DATADIR"/shuffleddata/$OUTDIR/outval.txt &

    wait
)

rm -rf "$DATADIR"/shuffleddata/current
mv "$DATADIR"/shuffleddata/$OUTDIR "$DATADIR"/shuffleddata/current

echo "Finished shuffle at $(date "+%Y-%m-%d %H:%M:%S")"
echo "Output: $DATADIR/shuffleddata/current/{train,val}/"
echo ""

exit 0
}
