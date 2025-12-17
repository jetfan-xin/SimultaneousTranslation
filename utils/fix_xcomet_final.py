#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为已有结果文件补充缺失的 xcomet_final：
- 若某条目 draft_translation 为空/为 null，则写入一个错误占位的 xcomet_final。
- 默认处理 results_Qwen3-4B/test_baseline_*.json，可自定义输入与输出目录。
"""

import argparse
import json
from pathlib import Path
from typing import List


def load_results(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} 内容不是列表")
    return data


def save_results(path: Path, data: List[dict]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="为 draft 为空的样本补充 xcomet_final 占位")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["results_Qwen3-4B/test_baseline_*.json"],
        help="要处理的结果文件（支持glob）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录（默认覆盖原文件）",
    )
    args = parser.parse_args()

    files: List[Path] = []
    for pattern in args.inputs:
        p = Path(pattern)
        if any(ch in pattern for ch in "*?[]"):
            files.extend(Path(p.parent or ".").glob(p.name))
        elif p.exists():
            files.append(p)
    files = sorted(set(f.resolve() for f in files if f.exists()))
    if not files:
        print("[Info] 未找到匹配文件")
        return

    out_dir = Path(args.output_dir).resolve() if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for fp in files:
        data = load_results(fp)
        changed = False

        for item in data:
            draft_translation = item.get("draft_translation")
            if not draft_translation:
                item["xcomet_final"] = {
                    "score": None,
                    "error_spans": [],
                    "error": "Draft translation is empty",
                }
                changed = True

        target_path = out_dir / fp.name if out_dir else fp
        if changed or out_dir:
            save_results(target_path, data)
            print(f"[Done] 写入 {target_path}")
        else:
            print(f"[Skip] {fp} 无需要更新的样本")


if __name__ == "__main__":
    main()
