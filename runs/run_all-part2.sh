#!/bin/bash

CKPT=/ltstorage/home/4xin/models/XCOMET-XL/checkpoints/model.ckpt
DATA_DIR=/ltstorage/home/4xin/SimultaneousTranslation/data/test/used-part2
RESULT_DIR=/ltstorage/home/4xin/SimultaneousTranslation/results_Qwen3-4B-part2

CUDA_VISIBLE_DEVICES=2,4 python - <<EOF
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
CUDA_VISIBLE_DEVICES=2,4 python main.py \\
  --data_dir {data_dir} \
  --xcomet_ckpt {ckpt} \\
  --test_files {jsonl} \\
  --xcomet_gpus 2 \\
  --qwen_gpus 4 \\
  --pipeline_mode {pipeline_mode} \\
  --output_file {out_file}
"""
    print(f"[Run] {cmd}")
    os.system(cmd)

EOF
