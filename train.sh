cd train

bash train_muon_ki.sh \
    ../data ../data/shuffleddata/current_all \
    b14c192h6tfrs_1 b14c192h6tfrs-bng-silu 256 extra \
    -multi-gpus 0 \
    -gnorm-clip-scale 1.0 -lr-scale-auto-type custom -wd-scale 1.0 \
    -export-prob 0.003 -print-every 1

cd ..
