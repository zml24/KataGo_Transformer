#!/bin/bash -eu
set -o pipefail
{
# Shuffles all NPZ data for nano training, splitting into train/val sets.
# Usage: bash shuffle.sh INPUTDIR OUTPUTDIR TMPDIR NTHREADS BATCHSIZE [extra shuffle.py args...]

if [[ $# -lt 5 ]]; then
    echo "Usage: $0 INPUTDIR OUTPUTDIR TMPDIR NTHREADS BATCHSIZE [extra args...]"
    echo "INPUTDIR   directory containing NPZ files (searched recursively)"
    echo "OUTPUTDIR  output directory, will contain train/ and val/ subdirectories"
    echo "TMPDIR     scratch space, ideally on fast local disk"
    echo "NTHREADS   number of parallel processes for shuffling"
    echo "BATCHSIZE  batch size for training examples"
    exit 0
fi
INPUTDIR="$1"
shift
OUTPUTDIR="$1"
shift
TMPDIR="$1"
shift
NTHREADS="$1"
shift
BATCHSIZE="$1"
shift

SCRIPTDIR="$(cd "$(dirname "$0")" && pwd)"

#------------------------------------------------------------------------------

mkdir -p "$OUTPUTDIR"
mkdir -p "$TMPDIR"/train
mkdir -p "$TMPDIR"/val

echo "Beginning shuffle at $(date "+%Y-%m-%d %H:%M:%S")"

time python3 "$SCRIPTDIR"/shuffle.py \
     "$INPUTDIR" \
     --num-processes "$NTHREADS" \
     --batch-size "$BATCHSIZE" \
     --approx-rows-per-out-file 131072 \
     --split "train:0.00:0.95:$OUTPUTDIR/train:$TMPDIR/train" \
     --split "val:0.95:1.00:$OUTPUTDIR/val:$TMPDIR/val" \
     "$@" \
     2>&1 | tee "$OUTPUTDIR"/output.txt

echo "Finished shuffle at $(date "+%Y-%m-%d %H:%M:%S")"
echo "Output: $OUTPUTDIR/{train,val}/"
echo ""

exit 0
}
