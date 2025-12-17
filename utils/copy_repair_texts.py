#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从一个结果目录拷贝每条样本的 repair_generated_text 到另一个目录的同名文件。

用法示例：
python copy_repair_texts.py \
  --src_dir results_Qwen3-4B-part2_copy \
  --tgt_dir results_Qwen3-4B-part2 \
  --pattern "test_baseline_*.json"
"""

import argparse
import json
from pathlib import Path
from typing import List

def load_list(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} 内容不是列表")
    return data


def save_list(path: Path, data: List[dict]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def copy_repair(src_dir: Path, tgt_dir: Path, pattern: str):
    src_files = sorted(src_dir.glob(pattern))
    if not src_files:
        print(f"[Info] 源目录无匹配文件: {pattern}")
        return

    for src_file in src_files:
        tgt_file = tgt_dir / src_file.name
        if not tgt_file.exists():
            print(f"[Skip] 目标不存在: {tgt_file}")
            continue

        src_data = load_list(src_file)
        tgt_data = load_list(tgt_file)

        if len(src_data) != len(tgt_data):
            print(f"[Warn] 条目数不一致: {src_file} ({len(src_data)}) vs {tgt_file} ({len(tgt_data)}), 将按较短长度覆盖。")

        n = min(len(src_data), len(tgt_data))
        for i in range(n):
            tgt_data[i]["repair_generated_text"] = src_data[i].get("repair_generated_text")

        save_list(tgt_file, tgt_data)
        print(f"[Done] 覆盖 repair_generated_text -> {tgt_file}")


def main():
    parser = argparse.ArgumentParser(description="拷贝 repair_generated_text 到另一目录的同名文件")
    parser.add_argument("--src_dir", required=True, help="源结果目录")
    parser.add_argument("--tgt_dir", required=True, help="目标结果目录")
    parser.add_argument("--pattern", default="*.json", help="匹配的结果文件名模式，默认 *.json")
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    tgt_dir = Path(args.tgt_dir)
    if not src_dir.exists():
        raise FileNotFoundError(f"源目录不存在: {src_dir}")
    if not tgt_dir.exists():
        raise FileNotFoundError(f"目标目录不存在: {tgt_dir}")

    copy_repair(src_dir, tgt_dir, args.pattern)


if __name__ == "__main__":
    main()
