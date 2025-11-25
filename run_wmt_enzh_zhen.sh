#!/bin/bash

set -e  # 任意一步出错就终止脚本
export CUDA_VISIBLE_DEVICES=1,2,3,4

echo "=============================================================="
echo "[1/6] Running baseline wmt23_zh-en ..."
echo "=============================================================="
python main.py \
  --xcomet_ckpt /ltstorage/home/4xin/models/XCOMET-XL/checkpoints/model.ckpt \
  --test_files wmt23_zh-en.jsonl \
  --xcomet_gpus 1,2 \
  --qwen_gpus 3,4 \
  --output_file test_baseline_wmt23_zh-en.json


echo "=============================================================="
echo "[2/6] Running baseline wmt24_en-zh ..."
echo "=============================================================="
python main.py \
  --xcomet_ckpt /ltstorage/home/4xin/models/XCOMET-XL/checkpoints/model.ckpt \
  --test_files wmt24_en-zh.jsonl \
  --xcomet_gpus 1,2 \
  --qwen_gpus 3,4 \
  --output_file test_baseline_wmt24_en-zh.json


echo "=============================================================="
echo "[3/6] Running extended wmt23_zh-en ..."
echo "=============================================================="
python main.py \
  --xcomet_ckpt /ltstorage/home/4xin/models/XCOMET-XL/checkpoints/model.ckpt \
  --test_files wmt23_zh-en.jsonl \
  --pipeline_mode extended \
  --xcomet_gpus 1,2 \
  --qwen_gpus 3,4 \
  --output_file test_extended_wmt23_zh-en.json


echo "=============================================================="
echo "[4/6] Running extended wmt24_en-zh ..."
echo "=============================================================="
python main.py \
  --xcomet_ckpt /ltstorage/home/4xin/models/XCOMET-XL/checkpoints/model.ckpt \
  --test_files wmt24_en-zh.jsonl \
  --pipeline_mode extended \
  --xcomet_gpus 1,2 \
  --qwen_gpus 3,4 \
  --output_file test_extended_wmt24_en-zh.json


echo "=============================================================="
echo "[5/6] Running metrics.py ..."
echo "=============================================================="
python /ltstorage/home/4xin/SimultaneousTranslation/data/test/metrics.py


echo "=============================================================="
echo "[6/6] Running merge.py ..."
echo "=============================================================="
python /ltstorage/home/4xin/SimultaneousTranslation/data/test/merge.py


echo "=============================================================="
echo "🎉 All tasks finished successfully!"
echo "=============================================================="