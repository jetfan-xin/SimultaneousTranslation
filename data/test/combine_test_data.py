#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preprocess all evaluation test sets into unified jsonl format and save to:
    /ltstorage/home/4xin/SimultaneousTranslation/data/test/used/

统一输出格式：
{
    "data_source": "commonmt" / "culturemt" / "drt" / "flores101" / "rtt",
    "lg": "zh-en" / "en-es" / "en-zh" / "en-de" / ...,
    "src_text": "...",
    "tgt_text": "..."
}
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any


# =========================================================
# 1. 各数据集专用 preprocess 函数
# =========================================================

def preprocess_data_commonmt(
    data: List[Dict[str, Any]],
    data_source: str = "commonmt",
    lang_pair: str = "zh-en",
) -> List[Dict[str, Any]]:
    """
    CSV-style common data with columns:
        chinese_source, english_target_correct, english_target_wrong (ignored)
    """
    processed_data = []
    for example in data:
        src_text = (example.get("chinese_source") or "").strip()
        tgt_text = (example.get("english_target_correct") or "").strip()

        processed_example = {
            "data_source": data_source,   # 不再从 example 里拿
            "lg": lang_pair,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

        if processed_example["src_text"] and processed_example["tgt_text"]:
            processed_data.append(processed_example)

    return processed_data


def preprocess_data_culturemt(
    data: List[Dict[str, Any]],
    data_source: str = "culturemt",
    lang_pair: str = "en-es",
) -> List[Dict[str, Any]]:
    """
    CultureMT-style:
        {
            "source": "...",
            "target": "...",
            "labels": [...],   # ignored
            ...
        }
    """
    processed = []

    for example in data:
        src = (example.get("source") or "").strip()
        tgt = (example.get("target") or "").strip()

        processed_example = {
            "data_source": data_source,
            "lg": lang_pair,
            "src_text": src,
            "tgt_text": tgt,
        }

        if src and tgt:
            processed.append(processed_example)

    return processed


def preprocess_data_drt(
    data: List[Dict[str, Any]],
    data_source: str = "drt",
    lang_pair: str = "en-zh",
) -> List[Dict[str, Any]]:
    """
    DRT-style jsonl:
        {
            "text":  <source text>,
            "trans": <target translation>,
            "thought": <reasoning process>  # ignored
        }
    """
    processed = []

    for example in data:
        src = (example.get("text") or "").strip()
        tgt = (example.get("trans") or "").strip()

        processed_example = {
            "data_source": data_source,
            "lg": lang_pair,
            "src_text": src,
            "tgt_text": tgt,
        }

        if src and tgt:
            processed.append(processed_example)

    return processed

def preprocess_data_rtt(
    data: Dict[str, List[str]],
    data_source: str = "rtt",
    language_pair: str = "en-de",
) -> List[Dict[str, Any]]:
    """
    RTT raw test data:
        {
            "src": [...],   # e.g. lines from test.en
            "tgt": [...],   # e.g. lines from test.de
        }
    """
    src_list = data.get("src", [])
    tgt_list = data.get("tgt", [])

    if len(src_list) != len(tgt_list):
        raise ValueError(
            f"RTT src/tgt length mismatch: {len(src_list)} vs {len(tgt_list)}"
        )

    processed = []

    for src, tgt in zip(src_list, tgt_list):
        src = src.strip()
        tgt = tgt.strip()
        if not src or not tgt:
            continue

        processed.append(
            {
                "data_source": data_source,
                "lg": language_pair,
                "src_text": src,
                "tgt_text": tgt,
            }
        )

    return processed


# =========================================================
# 2. 读写工具函数
# =========================================================

def save_jsonl(samples: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ex in samples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"[SAVE] {out_path}  ({len(samples)} samples)")


def read_csv_dicts(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def read_json_list(path: Path) -> List[Dict[str, Any]]:
    """
    既兼容 JSON 数组，也兼容 JSONL（逐行）的 reader。
    """
    with path.open("r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return []
        if content[0] == "[":
            return json.loads(content)
        data = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
        return data


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


# =========================================================
# 3. 主流程：读取所有 raw data → 处理 → 写到 used/
# =========================================================

def main():
    ROOT = Path("/ltstorage/home/4xin/SimultaneousTranslation")
    TEST_ROOT = ROOT / "data/test"
    USED_ROOT = TEST_ROOT / "used"

    # ------------ 1) commonMT: 三个 CSV（zh-en）------------
    commonmt_dir = TEST_ROOT / "commonMT"
    commonmt_files = [
        "contextless syntactic ambiguity.csv",
        "contextual syntactic ambiguity.csv",
        "lexical ambiguity.csv",
    ]

    for fname in commonmt_files:
        csv_path = commonmt_dir / fname
        if not csv_path.exists():
            print(f"[WARN] commonMT file not found: {csv_path}")
            continue

        raw = read_csv_dicts(csv_path)
        processed = preprocess_data_commonmt(
            raw, data_source="commonmt", lang_pair="zh-en"
        )

        # 输出文件名可以带上子集信息，但 data_source 字段仍为 "commonmt"
        stem = fname.replace(" ", "_").replace(".csv", "")
        out_path = USED_ROOT / f"commonmt_{stem}_zh-en.jsonl"
        save_jsonl(processed, out_path)

    # ------------ 2) CultureMT: 多个 en-XX.json ------------
    culture_dir = TEST_ROOT / "CultureMT"
    culture_files = [
        "en-es.json",
        "en-fr.json",
        "en-hi.json",
        "en-ta.json",
        "en-te.json",
        "en-zh.json",
    ]

    for fname in culture_files:
        json_path = culture_dir / fname
        if not json_path.exists():
            print(f"[WARN] CultureMT file not found: {json_path}")
            continue

        raw = read_json_list(json_path)
        lang_pair = fname.replace(".json", "")  # e.g. "en-es"
        processed = preprocess_data_culturemt(
            raw, data_source="culturemt", lang_pair=lang_pair
        )

        out_path = USED_ROOT / f"culturemt_{lang_pair}.jsonl"
        save_jsonl(processed, out_path)

    # ------------ 3) DRT: MetaphorTrans_test.jsonl ------------
    drt_path = TEST_ROOT / "DRT" / "MetaphorTrans_test.jsonl"
    if drt_path.exists():
        raw_drt = read_jsonl(drt_path)
        processed_drt = preprocess_data_drt(
            raw_drt, data_source="drt", lang_pair="en-zh"
        )
        out_path = USED_ROOT / "drt_MetaphorTrans_en-zh.jsonl"
        save_jsonl(processed_drt, out_path)
    else:
        print(f"[WARN] DRT file not found: {drt_path}")

    # ------------ 5) RTT: test.en + test.de ------------
    rtt_dir = TEST_ROOT / "RTT"
    test_en = rtt_dir / "test.en"
    test_de = rtt_dir / "test.de"

    if test_en.exists() and test_de.exists():
        with test_en.open("r", encoding="utf-8") as f_en, \
             test_de.open("r", encoding="utf-8") as f_de:
            src_lines = f_en.read().splitlines()
            tgt_lines = f_de.read().splitlines()

        raw_rtt = {
            "src": src_lines,
            "tgt": tgt_lines,
        }
        processed_rtt = preprocess_data_rtt(
            raw_rtt, data_source="rtt", language_pair="en-de"
        )
        out_path = USED_ROOT / "rtt_en-de.jsonl"
        save_jsonl(processed_rtt, out_path)
    else:
        print(f"[WARN] RTT test.en/test.de not found in {rtt_dir}")


if __name__ == "__main__":
    main()