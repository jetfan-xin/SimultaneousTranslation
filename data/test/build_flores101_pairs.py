#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import Dict, List, Any

# 你给的映射：逻辑语言代码 -> 文件中使用的 stub
LANG2FILE: Dict[str, str] = {
    "en": "eng",
    "zh": "zho_simpl",
    "ja": "jpn",
    "ru": "rus",
    "fr": "fra",
    "de": "deu",
    "th": "tha",
    "nl": "nld",
    "vi": "vie",
    "tr": "tur",
    "cs": "ces",
}


# --------- 小工具函数 --------- #

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def save_jsonl(samples: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ex in samples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"[SAVE] {out_path} ({len(samples)} samples)")


def load_lang_sentences(
    split_dir: Path,
    lang_code: str,
    file_stub: str,
    id_field: str = "id",
    sent_field: str = "sentence",
) -> Dict[int, str]:
    """
    读取某个语言的 devtest 单语 jsonl，返回：
        id -> sentence
    """
    path = split_dir / f"devtest.{file_stub}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing FLORES split file for {lang_code}: {path}")

    data = read_jsonl(path)
    mapping: Dict[int, str] = {}
    for ex in data:
        idx = ex.get(id_field)
        if idx is None:
            continue
        sent = (ex.get(sent_field) or "").strip()
        if not sent:
            continue
        mapping[int(idx)] = sent
    return mapping


def build_pair_samples(
    src_map: Dict[int, str],
    tgt_map: Dict[int, str],
    src_lang: str,
    tgt_lang: str,
    data_source: str = "flores101",
) -> List[Dict[str, Any]]:
    """
    根据两种语言的 id->sentence 映射，构造统一格式样本：
        {
            "data_source": "flores101",
            "lg": f"{src_lang}-{tgt_lang}",
            "src_text": ...,
            "tgt_text": ...
        }
    """
    common_ids = sorted(set(src_map.keys()) & set(tgt_map.keys()))
    samples: List[Dict[str, Any]] = []
    lg = f"{src_lang}-{tgt_lang}"

    for idx in common_ids:
        src = src_map[idx]
        tgt = tgt_map[idx]
        if not src or not tgt:
            continue
        samples.append(
            {
                "data_source": data_source,
                "lg": lg,
                "src_text": src,
                "tgt_text": tgt,
            }
        )
    return samples


def main():
    ROOT = Path("/ltstorage/home/4xin/SimultaneousTranslation")
    FLORES_SPLIT = ROOT / "data/test/flores_101/devtest_split"
    USED_ROOT = ROOT / "data/test/used"

    # 1) 先一次性把所有语言的 id->sentence 映射读进来
    lang2id2sent: Dict[str, Dict[int, str]] = {}
    for lang, stub in LANG2FILE.items():
        mapping = load_lang_sentences(FLORES_SPLIT, lang, stub)
        lang2id2sent[lang] = mapping
        print(f"[LOAD] {lang} ({stub}): {len(mapping)} sentences")

    # 2) 构造所有需要的语言对：
    #    - xx2en, en2xx, xx2zh, zh2xx
    pairs = set()

    langs = list(LANG2FILE.keys())

    for xx in langs:
        if xx == "en" and xx == "zh":
            continue

        if xx != "en":
            pairs.add((xx, "en"))  # xx2en
            pairs.add(("en", xx))  # en2xx

        if xx != "zh":
            pairs.add((xx, "zh"))  # xx2zh
            pairs.add(("zh", xx))  # zh2xx

    # 3) 对每个语言对构建数据集并保存
    for src_lang, tgt_lang in sorted(pairs):
        if src_lang not in lang2id2sent or tgt_lang not in lang2id2sent:
            print(f"[WARN] skip pair {src_lang}-{tgt_lang}: missing lang mapping")
            continue

        src_map = lang2id2sent[src_lang]
        tgt_map = lang2id2sent[tgt_lang]

        samples = build_pair_samples(
            src_map,
            tgt_map,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            data_source="flores101",
        )

        if not samples:
            print(f"[WARN] no common ids for pair {src_lang}-{tgt_lang}, skip")
            continue

        out_path = USED_ROOT / f"flores101_{src_lang}-{tgt_lang}.jsonl"
        save_jsonl(samples, out_path)


if __name__ == "__main__":
    main()