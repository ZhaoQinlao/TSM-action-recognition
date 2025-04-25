#!/bin/bash

# Conda 初始化路径（根据你实际安装位置修改）
CONDA_INIT="source /home/fitz_joye/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="protas"  # 替换为你的环境名

for i in {0..7}
do
    GPU_ID=0  # GPU 从 0~3 循环分配
    # GPU_ID=0
    SCREEN_NAME="tad_part_$i"

    screen -S "$SCREEN_NAME" -dm bash -c "
        $CONDA_INIT
        conda activate $CONDA_ENV
        export CUDA_VISIBLE_DEVICES=$GPU_ID

        python extract_features.py \
        --data_set Assembly101 \
        --data_path data/assembly101/resized \
        --data_list TAD/part_$i.txt \
        --save_path workdir/assembly101_tsm_feature_48f
    "
done
