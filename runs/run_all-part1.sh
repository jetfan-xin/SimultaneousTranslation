#!/bin/bash

CKPT=/ltstorage/home/4xin/models/XCOMET-XL/checkpoints/model.ckpt
DATA_DIR=/ltstorage/home/4xin/SimultaneousTranslation/data/test/used-part1
RESULT_DIR=/ltstorage/home/4xin/SimultaneousTranslation/results_Qwen3-4B-part1

CUDA_VISIBLE_DEVICES=0,1 python - <<EOF
import os
from pathlib import Path

ckpt = "$CKPT"
data_dir = Path("$DATA_DIR")
result_dir = Path("$RESULT_DIR")
pipeline_mode = "baseline"

# 遍历所有 jsonl
for jsonl in data_dir.glob("*.jsonl"):
    fname = jsonl.name
    # 输出文件命名规则
    out_name = f"test_{pipeline_mode}_{fname.replace('.jsonl','')}.json"
    out_file = result_dir / out_name

    # 需要运行
    cmd = f"""
CUDA_VISIBLE_DEVICES=0,1 python main.py \\
  --data_dir {data_dir} \
  --xcomet_ckpt {ckpt} \\
  --test_files {jsonl} \\
  --xcomet_gpus 0 \\
  --qwen_gpus 1 \\
  --pipeline_mode {pipeline_mode} \\
  --output_file {out_file}

CUDA_VISIBLE_DEVICES=0,1 python main.py \\
  --data_dir /ltstorage/home/4xin/SimultaneousTranslation/data/test/used \
  --xcomet_ckpt /ltstorage/home/4xin/models/XCOMET-XL/checkpoints/model.ckpt \\
  --test_files /ltstorage/home/4xin/SimultaneousTranslation/data/test/used/wmt24_en-zh.jsonl \\
  --xcomet_gpus 0 \\
  --qwen_gpus 3 \\
  --pipeline_mode baseline \\
  --output_file /ltstorage/home/4xin/SimultaneousTranslation/results_Qwen3-8B/wmt24_en-zh_baseline.json
"""
    print(f"[Run] {cmd}")
    os.system(cmd)

EOF
