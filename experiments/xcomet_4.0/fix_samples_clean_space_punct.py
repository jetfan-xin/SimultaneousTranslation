#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix mt_segs in samples_clean for space/punctuation-only alignment mismatches.

This script reads segment_alignment_mismatches.json, finds cases labeled as
space_or_punct, and rewrites mt_segs_{good|bad} so that concatenation equals
mt_full_{good|bad} while preserving segment count/order.
"""

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_MISMATCH_PATH = Path(
    "/ltstorage/home/4xin/SimultaneousTranslation/experiments/xcomet_4.0/"
    "xcomet_eval_outputs/segment_alignment_mismatches.json"
)
DEFAULT_SAMPLES_DIR = Path(
    "/ltstorage/home/4xin/SimultaneousTranslation/experiments/xcomet_4.0/"
    "cases/samples_clean"
)
DEFAULT_REPORT_PATH = Path(
    "/ltstorage/home/4xin/SimultaneousTranslation/experiments/xcomet_4.0/"
    "xcomet_eval_outputs/space_or_punct_samples_clean_fixes.json"
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def is_space_or_punct(ch: str) -> bool:
    return ch.isspace() or unicodedata.category(ch).startswith("P")


def is_space_or_punct_only_diff(text_a: str, text_b: str) -> bool:
    def normalize(text: str) -> str:
        return "".join(ch for ch in (text or "") if not is_space_or_punct(ch))
    return normalize(text_a) == normalize(text_b)


def extract_mismatches(mismatch_payload: Any) -> List[Dict[str, Any]]:
    if isinstance(mismatch_payload, dict) and "mismatches" in mismatch_payload:
        data = mismatch_payload.get("mismatches") or []
    elif isinstance(mismatch_payload, list):
        data = mismatch_payload
    else:
        data = []
    return [item for item in data if isinstance(item, dict)]


def build_norm_map(text: str) -> List[int]:
    indices: List[int] = []
    for idx, ch in enumerate(text or ""):
        if not is_space_or_punct(ch):
            indices.append(idx)
    return indices


def count_norm_chars(text: str) -> int:
    return sum(1 for ch in (text or "") if not is_space_or_punct(ch))


def rebuild_segments(mt_text: str, mt_segs: List[str]) -> Tuple[List[str], str]:
    if not mt_segs:
        return [], "no_segments"

    joined = "".join(mt_segs)
    if joined == (mt_text or ""):
        return mt_segs, "already_aligned"

    if not is_space_or_punct_only_diff(joined, mt_text or ""):
        return mt_segs, "not_space_or_punct"

    norm_map = build_norm_map(mt_text or "")
    norm_lens = [count_norm_chars(seg) for seg in mt_segs]
    if sum(norm_lens) != len(norm_map):
        return mt_segs, "norm_length_mismatch"

    start_norms: List[int] = []
    acc = 0
    for length in norm_lens:
        start_norms.append(acc)
        acc += length

    start_indices: List[int] = []
    for idx, start_norm in enumerate(start_norms):
        if idx == 0:
            start_indices.append(0)
            continue
        if start_norm >= len(norm_map):
            start_indices.append(len(mt_text or ""))
        else:
            start_indices.append(norm_map[start_norm])

    # Ensure non-decreasing starts.
    for i in range(1, len(start_indices)):
        if start_indices[i] < start_indices[i - 1]:
            return mt_segs, "non_monotonic_boundaries"

    new_segments: List[str] = []
    for i, start in enumerate(start_indices):
        end = start_indices[i + 1] if i + 1 < len(start_indices) else len(mt_text or "")
        new_segments.append((mt_text or "")[start:end])

    if "".join(new_segments) != (mt_text or ""):
        return mt_segs, "rebuild_failed"

    return new_segments, "updated"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix samples_clean mt_segs for space_or_punct mismatches."
    )
    parser.add_argument("--mismatch-file", default=str(DEFAULT_MISMATCH_PATH))
    parser.add_argument("--samples-dir", default=str(DEFAULT_SAMPLES_DIR))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    args = parser.parse_args()

    mismatch_path = Path(args.mismatch_file)
    samples_dir = Path(args.samples_dir)
    report_path = Path(args.report)

    mismatch_payload = load_json(mismatch_path)
    mismatches = extract_mismatches(mismatch_payload)

    filtered: List[Dict[str, Any]] = []
    for item in mismatches:
        issue = item.get("alignment_issue")
        if issue:
            if issue != "space_or_punct":
                continue
        else:
            mt_text = item.get("mt_text", "")
            mt_joined = item.get("mt_segs_joined", "")
            if not is_space_or_punct_only_diff(mt_text, mt_joined):
                continue
        filtered.append(item)

    samples_cache: Dict[str, Dict[str, Any]] = {}
    updated_files: Dict[str, bool] = {}

    report = {
        "mismatch_file": str(mismatch_path),
        "total_mismatches": len(mismatches),
        "space_or_punct_mismatches": len(filtered),
        "datasets_updated": 0,
        "cases_updated": 0,
        "segments_updated": 0,
        "details": [],
    }

    processed_cases = set()

    for item in filtered:
        dataset = item.get("dataset_name")
        case_id = item.get("case_id")
        scenario = item.get("scenario", "BAD")
        if not dataset or not case_id:
            continue

        key = (dataset, case_id, scenario)
        if key in processed_cases:
            continue
        processed_cases.add(key)

        samples_path = samples_dir / f"{dataset}.json"
        if not samples_path.is_file():
            report["details"].append({
                "dataset": dataset,
                "case_id": case_id,
                "scenario": scenario,
                "status": "missing_samples_file",
                "samples_path": str(samples_path),
            })
            continue

        if dataset not in samples_cache:
            samples_cache[dataset] = load_json(samples_path)

        case = samples_cache[dataset].get(case_id)
        if not isinstance(case, dict):
            report["details"].append({
                "dataset": dataset,
                "case_id": case_id,
                "scenario": scenario,
                "status": "case_not_found",
            })
            continue

        if scenario == "GOOD":
            mt_text = case.get("mt_full_good") or ""
            mt_segs = case.get("mt_segs_good") or []
            seg_key = "mt_segs_good"
        else:
            mt_text = case.get("mt_full_bad") or ""
            mt_segs = case.get("mt_segs_bad") or []
            seg_key = "mt_segs_bad"

        new_segs, status = rebuild_segments(mt_text, mt_segs)
        if status == "updated":
            case[seg_key] = new_segs
            updated_files[dataset] = True
            report["cases_updated"] += 1
            report["segments_updated"] += len(new_segs)
            report["details"].append({
                "dataset": dataset,
                "case_id": case_id,
                "scenario": scenario,
                "status": "updated",
                "old_join_len": len("".join(mt_segs)),
                "new_join_len": len("".join(new_segs)),
                "segment_count": len(new_segs),
            })
        elif status == "already_aligned":
            report["details"].append({
                "dataset": dataset,
                "case_id": case_id,
                "scenario": scenario,
                "status": "already_aligned",
            })
        else:
            report["details"].append({
                "dataset": dataset,
                "case_id": case_id,
                "scenario": scenario,
                "status": status,
            })

    for dataset, should_write in updated_files.items():
        if not should_write:
            continue
        samples_path = samples_dir / f"{dataset}.json"
        save_json(samples_path, samples_cache[dataset])
        report["datasets_updated"] += 1

    save_json(report_path, report)
    print(f"[Done] Updated datasets: {report['datasets_updated']}")
    print(f"[Done] Updated cases: {report['cases_updated']}")
    print(f"[Done] Updated segments: {report['segments_updated']}")
    print(f"[Saved] Report: {report_path}")


if __name__ == "__main__":
    main()
