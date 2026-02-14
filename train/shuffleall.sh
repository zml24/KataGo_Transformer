#!/bin/bash -eu
set -o pipefail
{
# Shuffles ALL selfplay npz data (no dynamic window) for supervised/distillation training.
# Usage: bash shuffleall.sh BASEDIR DATADIR TMPDIR NTHREADS BATCHSIZE [extra shuffle.py args...]

if [[ $# -lt 5 ]]
then
    echo "Usage: $0 BASEDIR DATADIR TMPDIR NTHREADS BATCHSIZE [extra args...]"
    echo "BASEDIR  directory containing selfplay/ subdirectory with npz data"
    echo "DATADIR  directory to place shuffleddata/ output, usually same as BASEDIR"
    echo "TMPDIR   scratch space, ideally on fast local disk"
    echo "NTHREADS number of parallel processes for shuffling"
    echo "BATCHSIZE batch size for training examples"
    exit 0
fi
BASEDIR="$1"
shift
DATADIR="$1"
shift
TMPDIR="$1"
shift
NTHREADS="$1"
shift
BATCHSIZE="$1"
shift

#------------------------------------------------------------------------------

OUTDIR=$(date "+%Y%m%d-%H%M%S")
OUTDIRTRAIN=$OUTDIR/train
OUTDIRVAL=$OUTDIR/val

mkdir -p "$DATADIR"/shuffleddata/$OUTDIR
mkdir -p "$TMPDIR"/train
mkdir -p "$TMPDIR"/val

echo "Beginning shuffle-all at" $(date "+%Y-%m-%d %H:%M:%S")

# Train split (95% of files by md5 hash)
(
    time python3 ./shuffle.py \
         "$BASEDIR"/kata1_trainingdata_2601_npz/ \
         -expand-window-per-row 1.0 \
         -taper-window-exponent 1.0 \
         -min-rows 1 \
         -keep-target-rows 9999999999 \
         -out-dir "$DATADIR"/shuffleddata/$OUTDIRTRAIN \
         -out-tmp-dir "$TMPDIR"/train \
         -approx-rows-per-out-file 70000 \
         -num-processes "$NTHREADS" \
         -batch-size "$BATCHSIZE" \
         -only-include-md5-path-prop-lbound 0.00 \
         -only-include-md5-path-prop-ubound 0.95 \
         -output-npz \
         "$@" \
         2>&1 | tee "$DATADIR"/shuffleddata/$OUTDIR/outtrain.txt &

    wait
)

# Validation split (5% of files by md5 hash)
(
    time python3 ./shuffle.py \
         "$BASEDIR"/kata1_trainingdata_2601_npz/ \
         -expand-window-per-row 1.0 \
         -taper-window-exponent 1.0 \
         -min-rows 1 \
         -keep-target-rows 9999999999 \
         -out-dir "$DATADIR"/shuffleddata/$OUTDIRVAL \
         -out-tmp-dir "$TMPDIR"/val \
         -approx-rows-per-out-file 70000 \
         -num-processes "$NTHREADS" \
         -batch-size "$BATCHSIZE" \
         -only-include-md5-path-prop-lbound 0.95 \
         -only-include-md5-path-prop-ubound 1.00 \
         -output-npz \
         "$@" \
         2>&1 | tee "$DATADIR"/shuffleddata/$OUTDIR/outval.txt &

    wait
)

sleep 10

rm -rf "$DATADIR"/shuffleddata/current_all
mv "$DATADIR"/shuffleddata/$OUTDIR "$DATADIR"/shuffleddata/current_all

# Cleanup: among shuffled dirs older than 2 hours, keep only the most recent 5
echo "Cleaning up any old dirs"
find "$DATADIR"/shuffleddata/ -mindepth 1 -maxdepth 1 -type d -mmin +120 | sort | head -n -5 | xargs --no-run-if-empty rm -r

echo "Finished shuffle-all at" $(date "+%Y-%m-%d %H:%M:%S")
echo ""
echo ""

exit 0
}
