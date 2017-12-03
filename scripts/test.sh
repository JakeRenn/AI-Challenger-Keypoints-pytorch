#!/usr/bin/env bash

IMAGES_DIR="./media/ori_images"
RESULT_PATH="./results/test_results.json"
CHECKPOINT="./checkpoints/model_best.pth.tar"

export CUDA_VISIBLE_DEVICES=0

python tools/test.py \
$IMAGES_DIR \
--test_results=$RESULT_PATH \
--resume=$CHECKPOINT \

