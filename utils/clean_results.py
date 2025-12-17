#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量清理结果文件：去掉指定键，并将文件名中的 extended 改为 baseline。
"""

import argparse
import json
from pathlib import Path
from typing import List

DROP_KEYS = [
    "draft_segments",
    "draft_segment_results",
    "repair_segment_outputs",
    "final_segments",
    "repair_segment_prompts",
    "repair_segment_format_scores",
]


def clean_and_rename(path: Path, drop_keys: List[str]) -> Path:
    """移除指定键并重命名文件（extended -> baseline）。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print(f"[Skip] {path} 不是列表结构，跳过。")
        return path

    modified = False
    for item in data:
        if not isinstance(item, dict):
            continue
        for k in drop_keys:
            if k in item:
                item.pop(k, None)
                modified = True

    target_name = path.name.replace("extended", "baseline", 1)
    target_path = path.with_name(target_name)
    if target_path.exists() and target_path != path:
        raise FileExistsError(f"目标文件已存在，避免覆盖：{target_path}")

    # 写回清理后的内容，然后重命名
    if modified or path != target_path:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    if target_path != path:
        path.rename(target_path)

    print(f"[Done] {path.name} -> {target_path.name}")
    return target_path


def main():
    parser = argparse.ArgumentParser(description="清理结果文件并重命名")
    parser.add_argument(
        "--dir",
        type=str,
        default="/ltstorage/home/4xin/SimultaneousTranslation/results_Qwen3-4B",
        help="结果文件目录",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="test_extended_*.json",
        help="需要处理的文件通配符",
    )
    args = parser.parse_args()

    root = Path(args.dir)
    if not root.exists():
        raise FileNotFoundError(f"目录不存在: {root}")

    files = sorted(root.glob(args.pattern))
    if not files:
        print(f"[Info] 未找到匹配文件: {args.pattern}")
        return

    print(f"[Info] 将处理 {len(files)} 个文件")
    for fp in files:
        clean_and_rename(fp, DROP_KEYS)


if __name__ == "__main__":
    main()
