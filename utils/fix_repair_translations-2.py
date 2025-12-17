#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复历史结果中错误的 <translate> 标记，并重新计算润色后的 XCOMET 评分。

场景：
- 早期 prompt 里把 </translate> 写成了第二个 <translate>，导致润色输出无法正确提取。
- 本脚本从结果 JSON 中读取 repair_generated_text，采用宽松规则提取翻译，再用 XCOMET 重新评分。
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import List

from xcomet_loader import XCOMETLoader


def extract_translation(text: str) -> str:
    """
    宽松提取翻译内容：
    - 先移除 <think>...</think>
    - 优先使用正确的 <translate>...</translate>
    - 兼容错误的 <translate>...<translate>（两次开标签）取中间部分
    - 若仍无，返回null
    """
    if not text or not isinstance(text, str):
        return None

    cleaned = re.sub(r"(?is)<think>.*?</think>", "", text)

    # 正常闭合
    start = cleaned.lower().rfind("<translate>")
    end = cleaned.lower().rfind("</translate>")
    if start != -1 and end != -1 and end > start:
        cand = cleaned[start + len("<translate>"):end].strip()
        if cand:
            return cand

    # 错误的两次开标签
    opens = list(re.finditer(r"(?is)<translate>", cleaned))
    if len(opens) >= 2:
        cand = cleaned[opens[0].end():opens[-1].start()].strip()
        if cand:
            return cand

    # 兜底：直接返回无
    return None


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
    parser = argparse.ArgumentParser(description="修复润色翻译并重跑XCOMET")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["results_Qwen3-4B-part2/test_baseline_*.json"],
        help="要处理的结果文件（支持glob），默认处理 part2 的 baseline 结果",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录（默认覆盖原文件）",
    )
    parser.add_argument(
        "--xcomet_ckpt",
        type=str,
        default=None,
        help="XCOMET checkpoint，默认取环境变量 WORD_QE_CKPT 或 /ltstorage/... 路径",
    )
    parser.add_argument("--xcomet_gpus", type=str, default=None, help="XCOMET GPU，如 '0'")
    parser.add_argument("--xcomet_cpu", action="store_true", help="强制CPU跑XCOMET")
    parser.add_argument("--batch_size", type=int, default=32, help="XCOMET batch size")
    args = parser.parse_args()

    # 汇总待处理文件
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

    # 设定默认 ckpt
    ckpt = (
        args.xcomet_ckpt
        or os.getenv("WORD_QE_CKPT")
        or "/ltstorage/home/4xin/models/XCOMET-XL/checkpoints/model.ckpt"
    )
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"XCOMET checkpoint 不存在: {ckpt}")

    xcomet = XCOMETLoader(
        checkpoint_path=ckpt,
        force_cpu=args.xcomet_cpu,
        gpu_ids=args.xcomet_gpus,
    )

    out_dir = Path(args.output_dir).resolve() if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for fp in files:
        print(f"\n[File] 处理 {fp}")
        data = load_results(fp)

        # 提取翻译并准备评分
        triplets = []
        idx_map = []
        for i, item in enumerate(data):
            repaired = extract_translation(item.get("repair_generated_text"))
            xcomet_draft = item.get("xcomet_draft", {})
            draft_translation = item.get("draft_translation")
            draft_error_spans = xcomet_draft.get("error_spans", [])

            if len(draft_error_spans) == 0:
                # 若 draft 无错误, 没有draft，则直接用 draft 作为 final
                item["final_translation"] = draft_translation
                item["repair_format_score"] = 0
                # 若没有draft
                if not draft_translation:
                    item["xcomet_final"] = {
                            "score": None,
                            "error_spans": [],
                            "error": "Draft translation is empty",
                        }
            else:
                item["final_translation"] = repaired
                item["repair_format_score"] = 1 if repaired else 0

                # 若 final格式错误
                if not item["repair_format_score"]:
                    item["xcomet_final"] = {
                        "score": None,
                        "error_spans": [],
                        "error": "Repaired translation format invalid",
                    }

            if item["final_translation"]:
                src = str(item.get("src_text", "")).strip()
                ref = str(item.get("tgt_text", "")).strip()
                triplets.append({"src": src, "mt": item["final_translation"], "ref": ref})
                idx_map.append(i)

        if not triplets:
            print("[Warning] 无可评分样本，跳过 XCOMET")
            continue

        scores = xcomet.predict(triplets, batch_size=args.batch_size, return_system_score=True)
        for result, di in zip(scores, idx_map):
            data[di]["xcomet_final"] = result

        out_path = out_dir / fp.name if out_dir else fp
        save_results(out_path, data)
        print(f"[Done] 保存至 {out_path}")


if __name__ == "__main__":
    main()
