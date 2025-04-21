# This script extracts TAD features for the Assembly101 dataset using a pretrained model.
#!/usr/bin bash
set -x  # print the commands

export CUDA_VISIBLE_DEVICES=0


python extract_features.py \
    --data_set Assembly101 \
    --data_path data/assembly101/resized \
    --data_list TAD/part_0.txt \
    --save_path workdir/assembly101_tsm_feature \
