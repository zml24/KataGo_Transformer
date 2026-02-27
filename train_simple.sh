# 多卡 DDP 训练：取消下面 MULTI_GPUS 的注释，batch-size 为 per-GPU
# MULTI_GPUS="-multi-gpus 0,1"
MULTI_GPUS=""

python3 -u train/train.py \
  -traindir data/train/base_adam_lr2e-4 \
  -datadir data/shuffleddata/kata1_trainingdata_25q4_2601 \
  -pos-len 19 \
  -batch-size 1024 \
  -model-kind b12c768h12tfrs \
  -lr 2e-4 \
  -max-training-samples 300000000 \
  -symmetry-type xyt \
  -print-every 1 \
  -save-every-samples 1000000 \
  -val-every-samples 1000000 \
  -warmup-samples 2000000 \
  -enable-history-matrices \
  ${MULTI_GPUS}