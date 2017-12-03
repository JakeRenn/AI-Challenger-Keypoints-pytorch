#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

IMAGES_DIR="./media/ori_images"
OUTPUT_DIR="./media/pre_images"
RESULT_PATH="./results/test_results.json"

python tools/visualize_test.py \
$RESULT_PATH \
$IMAGES_DIR \
$OUTPUT_DIR \

