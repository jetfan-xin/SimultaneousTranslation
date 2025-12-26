#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix error_spans offsets for space/punctuation-only alignment mismatches.

This script reads segment_alignment_mismatches.json, finds entries classified
as space_or_punct, and rewrites the corresponding error_spans_full_bad offsets
by matching the span "text" against mt_full_bad.
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
DEFAULT_ERROR_SPANS_DIR = Path(
    "/ltstorage/home/4xin/SimultaneousTranslation/experiments/xcomet_4.0/"
    "cases/error_spans"
)
DEFAULT_REPORT_PATH = Path(
    "/ltstorage/home/4xin/SimultaneousTranslation/experiments/xcomet_4.0/"
    "xcomet_eval_outputs/space_or_punct_error_span_fixes.json"
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


def find_occurrences(text: str, pattern: str) -> List[int]:
    if not pattern:
        return []
    indices: List[int] = []
    start = 0
    while True:
        idx = text.find(pattern, start)
        if idx < 0:
            break
        indices.append(idx)
        start = idx + 1
    return indices


def choose_best_occurrence(indices: List[int], target_start: int) -> int:
    if not indices:
        return -1
    return min(indices, key=lambda x: abs(x - target_start))


def extract_mismatches(mismatch_payload: Any) -> List[Dict[str, Any]]:
    if isinstance(mismatch_payload, dict) and "mismatches" in mismatch_payload:
        data = mismatch_payload.get("mismatches") or []
    elif isinstance(mismatch_payload, list):
        data = mismatch_payload
    else:
        data = []
    return [item for item in data if isinstance(item, dict)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix error_spans for space_or_punct alignment mismatches."
    )
    parser.add_argument("--mismatch-file", default=str(DEFAULT_MISMATCH_PATH))
    parser.add_argument("--samples-dir", default=str(DEFAULT_SAMPLES_DIR))
    parser.add_argument("--error-spans-dir", default=str(DEFAULT_ERROR_SPANS_DIR))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    args = parser.parse_args()

    mismatch_path = Path(args.mismatch_file)
    samples_dir = Path(args.samples_dir)
    error_spans_dir = Path(args.error_spans_dir)
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
    errors_cache: Dict[str, Dict[str, Any]] = {}
    updated_files: Dict[str, bool] = {}

    report = {
        "mismatch_file": str(mismatch_path),
        "total_mismatches": len(mismatches),
        "space_or_punct_mismatches": len(filtered),
        "datasets_updated": 0,
        "cases_updated": 0,
        "spans_updated": 0,
        "spans_skipped_no_text": 0,
        "spans_unresolved": 0,
        "details": [],
    }

    processed_cases = set()

    for item in filtered:
        dataset = item.get("dataset_name")
        case_id = item.get("case_id")
        if not dataset or not case_id:
            continue

        scenario = item.get("scenario", "BAD")
        key = (dataset, case_id, scenario)
        if key in processed_cases:
            continue
        processed_cases.add(key)

        samples_path = samples_dir / f"{dataset}.json"
        error_path = error_spans_dir / f"{dataset}_error_spans.json"
        if not samples_path.is_file() or not error_path.is_file():
            report["details"].append({
                "dataset": dataset,
                "case_id": case_id,
                "scenario": scenario,
                "status": "missing_files",
                "samples_path": str(samples_path),
                "error_path": str(error_path),
            })
            continue

        if dataset not in samples_cache:
            samples_cache[dataset] = load_json(samples_path)
        if dataset not in errors_cache:
            errors_cache[dataset] = load_json(error_path)

        case = samples_cache[dataset].get(case_id)
        if not isinstance(case, dict):
            report["details"].append({
                "dataset": dataset,
                "case_id": case_id,
                "scenario": scenario,
                "status": "case_not_found",
            })
            continue

        error_case = errors_cache[dataset].get(case_id)
        if not isinstance(error_case, dict):
            report["details"].append({
                "dataset": dataset,
                "case_id": case_id,
                "scenario": scenario,
                "status": "error_spans_not_found",
            })
            continue

        if scenario == "GOOD":
            mt_text = case.get("mt_full_good") or ""
            spans = error_case.get("error_spans_full_good")
            if spans is None:
                spans = error_case.get("gt_error_spans_full_good")
        else:
            mt_text = case.get("mt_full_bad") or ""
            spans = error_case.get("error_spans_full_bad")

        if not isinstance(spans, list):
            report["details"].append({
                "dataset": dataset,
                "case_id": case_id,
                "scenario": scenario,
                "status": "no_span_list",
            })
            continue
        case_changes: List[Dict[str, Any]] = []
        spans_updated = 0

        for idx, span in enumerate(spans):
            if not isinstance(span, dict):
                continue
            span_text = span.get("text")
            if not span_text:
                report["spans_skipped_no_text"] += 1
                continue

            indices = find_occurrences(mt_text, span_text)
            if not indices:
                report["spans_unresolved"] += 1
                case_changes.append({
                    "span_index": idx,
                    "status": "text_not_found",
                    "text": span_text,
                    "start": span.get("start"),
                    "end": span.get("end"),
                })
                continue

            target_start = span.get("start")
            target_start = int(target_start) if isinstance(target_start, int) or isinstance(target_start, float) else 0
            new_start = choose_best_occurrence(indices, target_start)
            new_end = new_start + len(span_text)

            if span.get("start") != new_start or span.get("end") != new_end:
                case_changes.append({
                    "span_index": idx,
                    "status": "updated",
                    "text": span_text,
                    "old_start": span.get("start"),
                    "old_end": span.get("end"),
                    "new_start": new_start,
                    "new_end": new_end,
                })
                span["start"] = new_start
                span["end"] = new_end
                span["text"] = mt_text[new_start:new_end]
                spans_updated += 1

        if spans_updated:
            report["cases_updated"] += 1
            report["spans_updated"] += spans_updated
            report["details"].append({
                "dataset": dataset,
                "case_id": case_id,
                "scenario": scenario,
                "status": "updated",
                "updated_spans": spans_updated,
                "changes": case_changes,
            })
            updated_files[dataset] = True
        elif case_changes:
            report["details"].append({
                "dataset": dataset,
                "case_id": case_id,
                "scenario": scenario,
                "status": "no_updates",
                "changes": case_changes,
            })

    for dataset, should_write in updated_files.items():
        if not should_write:
            continue
        error_path = error_spans_dir / f"{dataset}_error_spans.json"
        save_json(error_path, errors_cache[dataset])
        report["datasets_updated"] += 1

    save_json(report_path, report)
    print(f"[Done] Updated datasets: {report['datasets_updated']}")
    print(f"[Done] Updated cases: {report['cases_updated']}")
    print(f"[Done] Updated spans: {report['spans_updated']}")
    print(f"[Saved] Report: {report_path}")


if __name__ == "__main__":
    main()
