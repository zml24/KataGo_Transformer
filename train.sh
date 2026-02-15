cd train

bash train_muon_ki.sh \
    ../data ../data/shuffleddata/current_all \
    b14c192h6tfrs_adam_bsz1k_lr5e-3_std0.02 b14c192h6tfrs-bng-silu 1024 extra \
    -multi-gpus 0 \
    -optimizer-type adam -lr-base 5e-3 -init-std 0.02 \
    -export-prob 0.003 -print-every 1

cd ..
