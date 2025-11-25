#!/bin/bash

CKPT=/ltstorage/home/4xin/models/XCOMET-XL/checkpoints/model.ckpt
DATA_DIR=/ltstorage/home/4xin/SimultaneousTranslation/data/test/used
RESULT_DIR=/ltstorage/home/4xin/SimultaneousTranslation/results

CUDA_VISIBLE_DEVICES=1,2,3,4 python - <<EOF
import os
from pathlib import Path

ckpt = "$CKPT"
data_dir = Path("$DATA_DIR")
result_dir = Path("$RESULT_DIR")

# 遍历所有 jsonl
for jsonl in data_dir.glob("*.jsonl"):
    fname = jsonl.name
    # 输出文件命名规则
    out_name = f"test_baseline_{fname.replace('.jsonl','')}.json"
    out_file = result_dir / out_name

    if out_file.exists():
        print(f"[Skip] {out_name} 已存在，跳过。")
        continue

    # 需要运行
    cmd = f"""
CUDA_VISIBLE_DEVICES=1,2,3,4 python main.py \\
  --xcomet_ckpt {ckpt} \\
  --test_files {jsonl} \\
  --xcomet_gpus 1,2 \\
  --qwen_gpus 3,4 \\
  --output_file {out_file}
"""
    print(f"[Run] {cmd}")
    os.system(cmd)

EOF