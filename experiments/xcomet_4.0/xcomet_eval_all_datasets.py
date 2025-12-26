#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run XCOMET on all datasets under cases/samples_clean and evaluate span metrics.

Outputs:
  - Per-dataset XCOMET results (raw + filtered spans) in xcomet_results/
  - Per-dataset metrics (accuracy/IoU/F1) in xcomet_metrics/
  - Aggregated summary by dataset type, language pair, and overall
  - Segment alignment mismatch report in segment_alignment_mismatches.json
  - Space/punctuation-only alignment mismatch report in segment_alignment_space_punct_mismatches.json
Note:
  Output directory defaults to XCOMET-XL_eval_outputs or XCOMET-XXL_eval_outputs
  based on --xcomet-model unless --output-dir is provided.
"""

import argparse
import json
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# ================= Path Setup =================

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ================= Config =================

IOU_THRESHOLD = 0.5
ERROR_CONFIDENCE_THRESHOLD = 0.5
ALLOWED_ERROR_SEVERITIES = {"major", "critical"}

DEFAULT_XCOMET_MODEL = "XL"
XCOMET_MODEL_CKPTS = {
    "XL": "/ltstorage/home/4xin/models/XCOMET-XL/checkpoints/model.ckpt",
    "XXL": "/mnt/data1/users/4xin/hf/hub/models--Unbabel--XCOMET-XXL/snapshots/873bac1b1c461e410c4a6e379f6790d3d1c7c214/checkpoints/model.ckpt",
}
XCOMET_MODEL_OUTPUT_DIRS = {
    "XL": CURRENT_DIR / "XCOMET-XL_eval_outputs",
    "XXL": CURRENT_DIR / "XCOMET-XXL_eval_outputs",
}

DEFAULT_SAMPLES_DIR = CURRENT_DIR / "cases" / "samples_clean"
DEFAULT_ERROR_SPANS_DIR = CURRENT_DIR / "cases" / "error_spans"
DEFAULT_OUTPUT_DIR = CURRENT_DIR / "XCOMET-XL_eval_outputs"

STRATEGY_NAMES = {
    "1.1": "Strategy 1.1 (S_full, MT_full, No Ref)",
    "1.2": "Strategy 1.2 (S_full, MT_full, With Ref)",
}

CASE_ID_RE = re.compile(r"^Case(\d+)$")


# ================= Data Loading =================

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def resolve_xcomet_config(model_name: str,
                          ckpt_override: str,
                          output_override: str) -> Tuple[str, str, Path]:
    model_key = (model_name or DEFAULT_XCOMET_MODEL).upper()
    if model_key not in XCOMET_MODEL_CKPTS:
        raise ValueError(f"Unknown XCOMET model: {model_name}")
    ckpt_path = ckpt_override or XCOMET_MODEL_CKPTS[model_key]
    output_dir = Path(output_override) if output_override else XCOMET_MODEL_OUTPUT_DIRS[model_key]
    return model_key, ckpt_path, output_dir


def parse_dataset_info(dataset_name: str) -> Tuple[str, str]:
    parts = dataset_name.split("_")
    if len(parts) >= 2:
        return parts[0], parts[-1]
    return dataset_name, "unknown"


def normalize_dataset_type(dataset_type: str) -> str:
    if dataset_type and dataset_type.startswith("wmt") and dataset_type[3:].isdigit():
        return "wmt"
    return dataset_type


def list_dataset_files(samples_dir: Path) -> List[Path]:
    if not samples_dir.is_dir():
        raise FileNotFoundError(f"Samples dir not found: {samples_dir}")
    files = [p for p in samples_dir.iterdir() if p.suffix == ".json"]
    return sorted(files)


def dataset_name_from_results_file(path: Path) -> str:
    stem = path.stem
    suffix = "_xcomet_results"
    return stem[:-len(suffix)] if stem.endswith(suffix) else stem


def sort_case_ids(case_ids: List[str]) -> List[str]:
    def key(cid: str) -> Tuple[int, Any]:
        match = CASE_ID_RE.match(cid)
        if match:
            return (0, int(match.group(1)))
        return (1, cid)
    return sorted(case_ids, key=key)


def load_error_spans(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Error spans file not found: {path}")
    data = load_json(path)
    if not isinstance(data, dict):
        print(f"[Warning] Error spans file is not a dict: {path}")
        return {}
    spans_by_case: Dict[str, List[Dict[str, Any]]] = {}
    for case_id, payload in data.items():
        spans: List[Dict[str, Any]] = []
        if isinstance(payload, dict):
            spans = payload.get("error_spans_full_bad")
            if spans is None:
                spans = payload.get("gt_error_spans_full_bad")
            if spans is None:
                spans = payload.get("error_spans") or payload.get("gt_error_spans") or []
        elif isinstance(payload, list):
            spans = payload
        spans_by_case[case_id] = spans or []
    return spans_by_case


def derive_diff_spans(mt_full_good: str, mt_full_bad: str) -> List[Dict[str, Any]]:
    matcher = SequenceMatcher(None, mt_full_good or "", mt_full_bad or "", autojunk=False)
    spans: List[Dict[str, Any]] = []
    for tag, _, _, j1, j2 in matcher.get_opcodes():
        if tag == "equal" or j1 == j2:
            continue
        spans.append({
            "start": j1,
            "end": j2,
            "tag": tag,
            "text": (mt_full_bad or "")[j1:j2],
        })
    return spans


def find_first_mismatch_index(text_a: str, text_b: str) -> int:
    min_len = min(len(text_a), len(text_b))
    for idx in range(min_len):
        if text_a[idx] != text_b[idx]:
            return idx
    return min_len


def build_alignment_mismatch_info(mt_text: str, mt_segs: List[str]) -> Dict[str, Any]:
    text = mt_text or ""
    joined = "".join(mt_segs or [])
    if joined == text:
        return {}
    mismatch_idx = find_first_mismatch_index(text, joined)
    radius = 30
    start = max(0, mismatch_idx - radius)
    end = mismatch_idx + radius
    return {
        "mt_text_len": len(text),
        "mt_segs_joined_len": len(joined),
        "first_mismatch_index": mismatch_idx,
        "mt_text_context": text[start:end],
        "mt_segs_joined_context": joined[start:end],
        "mt_text": text,
        "mt_segs_joined": joined,
        "mt_segs": mt_segs or [],
    }


def is_space_or_punct_only_diff(text_a: str, text_b: str) -> bool:
    def normalize(text: str) -> str:
        return "".join(
            ch for ch in (text or "")
            if not (ch.isspace() or unicodedata.category(ch).startswith("P"))
        )
    if (text_a or "") == (text_b or ""):
        return False
    return normalize(text_a) == normalize(text_b)


def classify_alignment_issue(mt_text: str, mt_segs: List[str]) -> str:
    if not mt_segs:
        return "no_segments"
    combined = "".join(mt_segs)
    if combined == (mt_text or ""):
        return "exact"
    if is_space_or_punct_only_diff(combined, mt_text or ""):
        return "space_or_punct"
    return "other_mismatch"


def collect_alignment_mismatches(
    cases: Dict[str, Dict[str, Any]],
    dataset_name: str,
    dataset_type: str,
    language_pair: str,
) -> List[Dict[str, Any]]:
    mismatches: List[Dict[str, Any]] = []
    for case_id in sort_case_ids(list(cases.keys())):
        case = cases[case_id]
        if not isinstance(case, dict):
            print(f"[Warning] Skipping non-dict case entry in {dataset_name}: {case_id}")
            continue
        case_label = case.get("label", case_id)
        source_index = case.get("source_index")

        for scenario, mt_text, mt_segs in (
            ("BAD", case.get("mt_full_bad") or "", case.get("mt_segs_bad") or []),
            ("GOOD", case.get("mt_full_good") or "", case.get("mt_segs_good") or []),
        ):
            alignment_issue = classify_alignment_issue(mt_text, mt_segs)
            if alignment_issue == "exact":
                continue
            info = build_alignment_mismatch_info(mt_text, mt_segs)
            if not info:
                joined = "".join(mt_segs or [])
                info = {
                    "mt_text_len": len(mt_text or ""),
                    "mt_segs_joined_len": len(joined),
                    "first_mismatch_index": None,
                    "mt_text_context": "",
                    "mt_segs_joined_context": "",
                    "mt_text": mt_text or "",
                    "mt_segs_joined": joined,
                    "mt_segs": mt_segs or [],
                }
            mismatches.append({
                "dataset_name": dataset_name,
                "dataset_type": dataset_type,
                "language_pair": language_pair,
                "case_id": case_id,
                "case_label": case_label,
                "source_index": source_index,
                "scenario": scenario,
                "alignment_issue": alignment_issue,
                **info,
            })
    return mismatches


def build_eval_entries(
    cases: Dict[str, Dict[str, Any]],
    error_spans: Dict[str, List[Dict[str, Any]]],
    dataset_name: str,
    dataset_type: str,
    language_pair: str,
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for case_id in sort_case_ids(list(cases.keys())):
        case = cases[case_id]
        if not isinstance(case, dict):
            print(f"[Warning] Skipping non-dict case entry in {dataset_name}: {case_id}")
            continue
        src_full = case.get("src_full") or ""
        ref_full = case.get("ref_full") or ""
        mt_full_good = case.get("mt_full_good") or ""
        mt_full_bad = case.get("mt_full_bad") or ""
        if not src_full or not mt_full_good or not mt_full_bad:
            print(f"[Warning] Missing fields in {dataset_name}:{case_id}, skipping.")
            continue

        mt_segs_good = case.get("mt_segs_good") or []
        mt_segs_bad = case.get("mt_segs_bad") or []

        if case_id not in error_spans:
            raise ValueError(f"Missing error spans for {dataset_name}:{case_id}")

        gt_bad_spans = error_spans[case_id]
        gt_source = "error_spans_file"

        base = {
            "dataset_name": dataset_name,
            "dataset_type": dataset_type,
            "language_pair": language_pair,
            "case_id": case_id,
            "case_label": case.get("label", case_id),
            "source_index": case.get("source_index"),
            "src": src_full,
            "ref": ref_full,
        }

        entries.append({
            **base,
            "entry_index": len(entries),
            "scenario": "BAD",
            "mt_text": mt_full_bad,
            "mt_segs": mt_segs_bad,
            "gt_spans": gt_bad_spans,
            "gt_source": gt_source,
        })
        entries.append({
            **base,
            "entry_index": len(entries),
            "scenario": "GOOD",
            "mt_text": mt_full_good,
            "mt_segs": mt_segs_good,
            "gt_spans": [],
            "gt_source": "clean_example",
        })
    return entries


# ================= Span Utilities =================

def get_char_mask(length: int, spans: List[Dict[str, Any]]) -> List[bool]:
    mask = [False] * length
    for span in spans or []:
        s = span.get("start")
        e = span.get("end")
        if s is None or e is None:
            continue
        try:
            s = int(s)
            e = int(e)
        except (TypeError, ValueError):
            continue
        s = max(0, min(s, length))
        e = max(s, min(e, length))
        for i in range(s, e):
            mask[i] = True
    return mask


def compute_span_metrics(mt_text: str,
                         pred_spans: List[Dict[str, Any]],
                         gt_spans: List[Dict[str, Any]]) -> Dict[str, Any]:
    length = len(mt_text or "")
    mask_pred = get_char_mask(length, pred_spans or [])
    mask_gt = get_char_mask(length, gt_spans or [])

    pred_total = sum(mask_pred)
    gt_total = sum(mask_gt)
    intersection = sum(p and g for p, g in zip(mask_pred, mask_gt))
    union = sum(p or g for p, g in zip(mask_pred, mask_gt))

    precision = intersection / pred_total if pred_total > 0 else (1.0 if gt_total == 0 else 0.0)
    recall = intersection / gt_total if gt_total > 0 else (1.0 if pred_total == 0 else 0.0)
    iou = intersection / union if union > 0 else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else (1.0 if gt_total == 0 and pred_total == 0 else 0.0)

    if gt_total == 0 and pred_total == 0:
        is_accurate = True
    elif gt_total == 0:
        is_accurate = False
    elif pred_total == 0:
        is_accurate = False
    else:
        is_accurate = (iou >= IOU_THRESHOLD)

    return {
        "mt_length": length,
        "pred_coverage": pred_total,
        "gt_coverage": gt_total,
        "intersection": intersection,
        "union": union,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "is_accurate": is_accurate,
    }


def filter_predicted_error_spans(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for span in spans or []:
        if not isinstance(span, dict):
            continue
        try:
            confidence = float(span.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence <= ERROR_CONFIDENCE_THRESHOLD:
            continue

        severity_raw = span.get("severity") or span.get("label") or span.get("type")
        severity = str(severity_raw).lower().strip() if severity_raw is not None else ""
        if severity not in ALLOWED_ERROR_SEVERITIES:
            continue

        start = span.get("start")
        end = span.get("end")
        if start is None or end is None:
            continue
        try:
            start_int = int(start)
            end_int = int(end)
        except (TypeError, ValueError):
            continue
        if end_int <= start_int:
            continue

        clean_span = dict(span)
        clean_span.update({
            "start": start_int,
            "end": end_int,
            "confidence": confidence,
            "severity": severity,
        })
        filtered.append(clean_span)

    filtered.sort(key=lambda s: s["start"])
    return filtered


def build_segments(mt_text: str, mt_segs: List[str]) -> Tuple[List[Dict[str, Any]], bool, str]:
    if not mt_segs:
        return [{
            "index": 0,
            "start": 0,
            "end": len(mt_text or ""),
            "text": mt_text or "",
        }], True, "no_segments"
    combined = "".join(mt_segs)
    if combined != (mt_text or ""):
        if is_space_or_punct_only_diff(combined, mt_text or ""):
            return [{
                "index": 0,
                "start": 0,
                "end": len(mt_text or ""),
                "text": mt_text or "",
            }], False, "space_or_punct"
        return [{
            "index": 0,
            "start": 0,
            "end": len(mt_text or ""),
            "text": mt_text or "",
        }], False, "other_mismatch"
    segments: List[Dict[str, Any]] = []
    offset = 0
    for idx, seg in enumerate(mt_segs):
        seg_len = len(seg)
        segments.append({
            "index": idx,
            "start": offset,
            "end": offset + seg_len,
            "text": seg,
        })
        offset += seg_len
    return segments, True, "exact"


def extract_segment_spans(spans: List[Dict[str, Any]],
                          seg_start: int,
                          seg_end: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rel: List[Dict[str, Any]] = []
    full: List[Dict[str, Any]] = []
    for span in spans or []:
        if not isinstance(span, dict):
            continue
        start = span.get("start")
        end = span.get("end")
        if start is None or end is None:
            continue
        try:
            start_int = int(start)
            end_int = int(end)
        except (TypeError, ValueError):
            continue
        overlap_start = max(start_int, seg_start)
        overlap_end = min(end_int, seg_end)
        if overlap_end <= overlap_start:
            continue
        full_span = dict(span)
        full_span.update({"start": overlap_start, "end": overlap_end})
        rel_span = dict(span)
        rel_span.update({"start": overlap_start - seg_start, "end": overlap_end - seg_start})
        full.append(full_span)
        rel.append(rel_span)
    return rel, full


# ================= Stats Container =================

class MetricBucket:
    def __init__(self):
        self.total = 0
        self.accurate = 0
        self.sum_iou = 0.0
        self.sum_precision = 0.0
        self.sum_recall = 0.0
        self.sum_f1 = 0.0

    def update(self, metrics: Dict[str, Any]):
        self.total += 1
        self.accurate += 1 if metrics.get("is_accurate") else 0
        self.sum_iou += metrics.get("iou", 0.0)
        self.sum_precision += metrics.get("precision", 0.0)
        self.sum_recall += metrics.get("recall", 0.0)
        self.sum_f1 += metrics.get("f1", 0.0)

    def merge(self, other: "MetricBucket") -> None:
        self.total += other.total
        self.accurate += other.accurate
        self.sum_iou += other.sum_iou
        self.sum_precision += other.sum_precision
        self.sum_recall += other.sum_recall
        self.sum_f1 += other.sum_f1

    def accuracy_percent(self) -> float:
        return (self.accurate / self.total * 100) if self.total else 0.0

    def avg(self, field_sum: float) -> float:
        return (field_sum / self.total) if self.total else 0.0

    def to_dict(self):
        return {
            "total": self.total,
            "accurate": self.accurate,
            "accuracy_percent": self.accuracy_percent(),
            "avg_iou": self.avg(self.sum_iou),
            "avg_precision": self.avg(self.sum_precision),
            "avg_recall": self.avg(self.sum_recall),
            "avg_f1": self.avg(self.sum_f1),
        }


class StrategyStats:
    def __init__(self, name: str):
        self.name = name
        self.overall = MetricBucket()
        self.by_scenario: Dict[str, MetricBucket] = {
            "GOOD": MetricBucket(),
            "BAD": MetricBucket(),
        }

    def update(self, scenario: str, metrics: Dict[str, Any]):
        self.overall.update(metrics)
        if scenario not in self.by_scenario:
            self.by_scenario[scenario] = MetricBucket()
        self.by_scenario[scenario].update(metrics)

    def merge(self, other: "StrategyStats") -> None:
        self.overall.merge(other.overall)
        for scenario, bucket in other.by_scenario.items():
            if scenario not in self.by_scenario:
                self.by_scenario[scenario] = MetricBucket()
            self.by_scenario[scenario].merge(bucket)

    def to_dict(self):
        return {
            "name": self.name,
            "overall": self.overall.to_dict(),
            "by_scenario": {k: v.to_dict() for k, v in self.by_scenario.items()},
        }


def create_strategy_stats() -> Dict[str, StrategyStats]:
    return {key: StrategyStats(name) for key, name in STRATEGY_NAMES.items()}


# ================= XCOMET + Metrics =================

def build_quality_record(entry: Dict[str, Any],
                         strategy_id: str,
                         raw_result: Dict[str, Any],
                         record_index: int) -> Dict[str, Any]:
    raw_result = raw_result or {}
    raw_spans = raw_result.get("error_spans") or []
    filtered_spans = filter_predicted_error_spans(raw_spans)
    return {
        "record_index": record_index,
        "entry_index": entry["entry_index"],
        "dataset_name": entry["dataset_name"],
        "dataset_type": entry["dataset_type"],
        "language_pair": entry["language_pair"],
        "case_id": entry["case_id"],
        "case_label": entry["case_label"],
        "source_index": entry.get("source_index"),
        "scenario": entry["scenario"],
        "strategy_id": strategy_id,
        "src_text": entry["src"],
        "ref_text": entry["ref"],
        "mt_text": entry["mt_text"],
        "mt_segs": entry.get("mt_segs") or [],
        "ground_truth_error_spans": entry["gt_spans"],
        "ground_truth_source": entry["gt_source"],
        "xcomet_raw_output": raw_result,
        "xcomet_score": raw_result.get("score"),
        "predicted_error_spans_raw": raw_spans,
        "predicted_error_spans_filtered": filtered_spans,
        "span_filter": {
            "confidence_gt": ERROR_CONFIDENCE_THRESHOLD,
            "allowed_severities": sorted(ALLOWED_ERROR_SEVERITIES),
        },
    }


def run_xcomet_on_entries(xcomet, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    total = len(entries)
    if total == 0:
        return []

    input_1_1 = [{"src": e["src"], "mt": e["mt_text"]} for e in entries]
    input_1_2 = [{"src": e["src"], "mt": e["mt_text"], "ref": e["ref"]} for e in entries]

    results_1_1 = xcomet.predict(input_1_1)
    results_1_2 = xcomet.predict(input_1_2)

    if len(results_1_1) != total:
        print(f"[Warning] Strategy 1.1 returned {len(results_1_1)} results (expected {total})")
    if len(results_1_2) != total:
        print(f"[Warning] Strategy 1.2 returned {len(results_1_2)} results (expected {total})")

    quality_results: List[Dict[str, Any]] = []
    record_index = 0
    for idx, entry in enumerate(entries):
        res_1_1 = results_1_1[idx] if idx < len(results_1_1) else {"score": 0.0, "error_spans": []}
        res_1_2 = results_1_2[idx] if idx < len(results_1_2) else {"score": 0.0, "error_spans": []}
        quality_results.append(build_quality_record(entry, "1.1", res_1_1, record_index))
        record_index += 1
        quality_results.append(build_quality_record(entry, "1.2", res_1_2, record_index))
        record_index += 1
    return quality_results


def compute_accuracy_from_quality_results(
    quality_results: List[Dict[str, Any]]
) -> Tuple[Dict[str, StrategyStats], List[Dict[str, Any]], int, List[Dict[str, Any]]]:
    stats = create_strategy_stats()
    detailed_logs: List[Dict[str, Any]] = []
    total_segments_evaluated = 0
    alignment_mismatches: List[Dict[str, Any]] = []
    seen_alignment_keys = set()

    for record in quality_results or []:
        strategy_id = record.get("strategy_id")
        if strategy_id not in stats:
            continue

        scenario = record.get("scenario", "UNKNOWN")
        filtered_spans_full = record.get("predicted_error_spans_filtered")
        if filtered_spans_full is None:
            filtered_spans_full = filter_predicted_error_spans(record.get("predicted_error_spans_raw", []))

        gt_spans_full = record.get("ground_truth_error_spans", [])
        mt_text = record.get("mt_text", "")
        mt_segs = record.get("mt_segs") or []
        segments, alignment_ok, alignment_issue = build_segments(mt_text, mt_segs)
        if alignment_issue == "space_or_punct":
            entry_key = record.get("entry_index")
            if entry_key is None:
                entry_key = record.get("case_id")
            alignment_key = (entry_key, record.get("scenario"))
            if alignment_key not in seen_alignment_keys:
                info = build_alignment_mismatch_info(mt_text, mt_segs)
                if info:
                    alignment_mismatches.append({
                        "dataset_name": record.get("dataset_name"),
                        "dataset_type": record.get("dataset_type"),
                        "language_pair": record.get("language_pair"),
                        "case_id": record.get("case_id"),
                        "case_label": record.get("case_label"),
                        "source_index": record.get("source_index"),
                        "scenario": record.get("scenario"),
                        "entry_index": record.get("entry_index"),
                        "alignment_issue": alignment_issue,
                        **info,
                    })
                    seen_alignment_keys.add(alignment_key)

        for seg in segments:
            seg_text = seg["text"]
            seg_start = seg["start"]
            seg_end = seg["end"]
            pred_rel, pred_full = extract_segment_spans(filtered_spans_full, seg_start, seg_end)
            gt_rel, gt_full = extract_segment_spans(gt_spans_full, seg_start, seg_end)

            metrics = compute_span_metrics(seg_text, pred_rel, gt_rel)
            stats[strategy_id].update(scenario, metrics)
            total_segments_evaluated += 1

            detailed_logs.append({
                "record_index": record.get("record_index"),
                "entry_index": record.get("entry_index"),
                "dataset_name": record.get("dataset_name"),
                "dataset_type": record.get("dataset_type"),
                "language_pair": record.get("language_pair"),
                "case_id": record.get("case_id"),
                "case_label": record.get("case_label"),
                "source_index": record.get("source_index"),
                "scenario": scenario,
                "strategy_id": strategy_id,
                "segment_index": seg["index"],
                "segment_start": seg_start,
                "segment_end": seg_end,
                "segment_text": seg_text,
                "segment_alignment_ok": alignment_ok,
                "src_text": record.get("src_text"),
                "ref_text": record.get("ref_text"),
                "mt_text": mt_text,
                "predicted_error_spans_raw": record.get("predicted_error_spans_raw", []),
                "predicted_error_spans": pred_rel,
                "predicted_error_spans_full": pred_full,
                "ground_truth_error_spans": gt_rel,
                "ground_truth_error_spans_full": gt_full,
                "ground_truth_source": record.get("ground_truth_source"),
                "metrics": metrics,
                "sentence_score": record.get("xcomet_score"),
                "score": record.get("xcomet_score"),
                "span_filter": record.get("span_filter"),
                "xcomet_raw_output": record.get("xcomet_raw_output"),
            })

    return stats, detailed_logs, total_segments_evaluated, alignment_mismatches


def merge_strategy_stats(target: Dict[str, StrategyStats], src: Dict[str, StrategyStats]) -> None:
    for strategy_id, src_stats in src.items():
        if strategy_id not in target:
            target[strategy_id] = StrategyStats(src_stats.name)
        target[strategy_id].merge(src_stats)


def load_xcomet_loader(checkpoint_path: str):
    try:
        from xcomet_loader import XCOMETLoader
    except ImportError:
        print("Error: Could not import XCOMETLoader. Make sure you are in the project root.")
        sys.exit(1)
    return XCOMETLoader(checkpoint_path=checkpoint_path)


# ================= Main Workflow =================

def run_xcomet_stage(
    samples_dir: Path,
    error_spans_dir: Path,
    output_dir: Path,
    checkpoint_path: str,
    datasets_filter: List[str],
    predict_scope: str,
) -> None:
    results_dir = output_dir / "xcomet_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset_files = list_dataset_files(samples_dir)
    if datasets_filter:
        dataset_files = [p for p in dataset_files if p.stem in datasets_filter]
    if not dataset_files:
        print("[Warning] No datasets found for XCOMET stage.")
        return

    xcomet = load_xcomet_loader(checkpoint_path)

    if predict_scope == "all":
        dataset_meta: Dict[str, Dict[str, Any]] = {}
        all_entries: List[Dict[str, Any]] = []
        iterator = tqdm(dataset_files, desc="XCOMET datasets") if tqdm else dataset_files
        for dataset_path in iterator:
            dataset_name = dataset_path.stem
            dataset_type, language_pair = parse_dataset_info(dataset_name)
            error_spans_path = error_spans_dir / f"{dataset_name}_error_spans.json"

            cases = load_json(dataset_path)
            if not isinstance(cases, dict) or not cases:
                print(f"[Warning] No cases in {dataset_path}")
                continue

            error_spans = load_error_spans(error_spans_path)
            entries = build_eval_entries(cases, error_spans, dataset_name, dataset_type, language_pair)
            print(f"[Stage:XCOMET] {dataset_name} entries: {len(entries)}")

            dataset_meta[dataset_name] = {
                "dataset_name": dataset_name,
                "dataset_type": dataset_type,
                "language_pair": language_pair,
                "num_cases": len(cases),
                "num_entries": len(entries),
            }
            all_entries.extend(entries)

        if not all_entries:
            print("[Warning] No entries collected for XCOMET stage.")
            return

        print(f"[Stage:XCOMET] Running combined prediction for {len(all_entries)} entries...")
        quality_results = run_xcomet_on_entries(xcomet, all_entries)

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for record in quality_results:
            dataset_name = record.get("dataset_name") or "unknown"
            grouped.setdefault(dataset_name, []).append(record)

        for dataset_name, records in grouped.items():
            for idx, record in enumerate(records):
                record["record_index"] = idx
            meta = dataset_meta.get(dataset_name, {})
            payload = {
                "dataset_name": dataset_name,
                "dataset_type": meta.get("dataset_type"),
                "language_pair": meta.get("language_pair"),
                "num_cases": meta.get("num_cases"),
                "num_entries": meta.get("num_entries", len(records) // 2),
                "strategies": sorted(STRATEGY_NAMES.keys()),
                "thresholds": {
                    "iou": IOU_THRESHOLD,
                    "confidence": ERROR_CONFIDENCE_THRESHOLD,
                    "severities": sorted(ALLOWED_ERROR_SEVERITIES),
                },
                "quality_results": records,
            }
            output_path = results_dir / f"{dataset_name}_xcomet_results.json"
            save_json(output_path, payload)
            print(f"[Saved] {output_path}")
    else:
        iterator = tqdm(dataset_files, desc="XCOMET datasets") if tqdm else dataset_files
        for dataset_path in iterator:
            dataset_name = dataset_path.stem
            dataset_type, language_pair = parse_dataset_info(dataset_name)
            error_spans_path = error_spans_dir / f"{dataset_name}_error_spans.json"

            cases = load_json(dataset_path)
            if not isinstance(cases, dict) or not cases:
                print(f"[Warning] No cases in {dataset_path}")
                continue

            error_spans = load_error_spans(error_spans_path)
            entries = build_eval_entries(cases, error_spans, dataset_name, dataset_type, language_pair)
            print(f"[Stage:XCOMET] {dataset_name} entries: {len(entries)}")

            quality_results = run_xcomet_on_entries(xcomet, entries)
            payload = {
                "dataset_name": dataset_name,
                "dataset_type": dataset_type,
                "language_pair": language_pair,
                "num_cases": len(cases),
                "num_entries": len(entries),
                "strategies": sorted(STRATEGY_NAMES.keys()),
                "thresholds": {
                    "iou": IOU_THRESHOLD,
                    "confidence": ERROR_CONFIDENCE_THRESHOLD,
                    "severities": sorted(ALLOWED_ERROR_SEVERITIES),
                },
                "quality_results": quality_results,
            }
            output_path = results_dir / f"{dataset_name}_xcomet_results.json"
            save_json(output_path, payload)
            print(f"[Saved] {output_path}")


def run_alignment_check(
    samples_dir: Path,
    output_dir: Path,
    datasets_filter: List[str],
) -> None:
    output_path = output_dir / "segment_alignment_mismatches.json"
    dataset_files = list_dataset_files(samples_dir)
    if datasets_filter:
        dataset_files = [p for p in dataset_files if p.stem in datasets_filter]
    if not dataset_files:
        print("[Warning] No datasets found for alignment check.")
        return

    all_mismatches: List[Dict[str, Any]] = []
    summary_by_dataset: Dict[str, Any] = {}
    total_cases = 0
    overall_issue_counts = {
        "space_or_punct": 0,
        "other_mismatch": 0,
        "no_segments": 0,
    }

    iterator = tqdm(dataset_files, desc="Alignment check") if tqdm else dataset_files
    for dataset_path in iterator:
        dataset_name = dataset_path.stem
        dataset_type, language_pair = parse_dataset_info(dataset_name)
        cases = load_json(dataset_path)
        if not isinstance(cases, dict) or not cases:
            print(f"[Warning] No cases in {dataset_path}")
            continue
        total_cases += len(cases)
        mismatches = collect_alignment_mismatches(cases, dataset_name, dataset_type, language_pair)
        if mismatches:
            scenario_counts = {"GOOD": 0, "BAD": 0}
            issue_counts = {
                "space_or_punct": 0,
                "other_mismatch": 0,
                "no_segments": 0,
            }
            for item in mismatches:
                scenario_counts[item["scenario"]] += 1
                issue = item.get("alignment_issue")
                if issue in issue_counts:
                    issue_counts[issue] += 1
            summary_by_dataset[dataset_name] = {
                "dataset_type": dataset_type,
                "language_pair": language_pair,
                "total_cases": len(cases),
                "mismatch_entries": len(mismatches),
                "mismatch_cases": len({m["case_id"] for m in mismatches}),
                "by_scenario": scenario_counts,
                "by_issue": issue_counts,
            }
            for issue, count in issue_counts.items():
                overall_issue_counts[issue] += count
        all_mismatches.extend(mismatches)

    summary = {
        "num_datasets": len(dataset_files),
        "num_cases": total_cases,
        "num_mismatch_entries": len(all_mismatches),
        "datasets_with_mismatches": len(summary_by_dataset),
        "by_issue": overall_issue_counts,
        "by_dataset": summary_by_dataset,
    }

    payload = {
        "summary": summary,
        "mismatches": all_mismatches,
    }
    save_json(output_path, payload)
    print(f"[Alignment] Mismatch entries: {len(all_mismatches)}")
    print(f"[Saved] {output_path}")


def run_metrics_stage(
    output_dir: Path,
    datasets_filter: List[str],
) -> None:
    results_dir = output_dir / "xcomet_results"
    metrics_dir = output_dir / "xcomet_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.is_dir():
        print(f"[Warning] Results dir not found: {results_dir}")
        return

    result_files = sorted([p for p in results_dir.iterdir() if p.suffix == ".json"])
    if datasets_filter:
        result_files = [p for p in result_files if dataset_name_from_results_file(p) in datasets_filter]
    if not result_files:
        print("[Warning] No result files found for metrics stage.")
        return

    overall_stats = create_strategy_stats()
    by_type: Dict[str, Dict[str, StrategyStats]] = {}
    by_lang: Dict[str, Dict[str, StrategyStats]] = {}
    dataset_summaries: Dict[str, Any] = {}
    space_punct_mismatches: List[Dict[str, Any]] = []
    space_punct_summary_by_dataset: Dict[str, Any] = {}

    iterator = tqdm(result_files, desc="Metrics datasets") if tqdm else result_files
    for result_path in iterator:
        payload = load_json(result_path)
        if isinstance(payload, dict) and "quality_results" in payload:
            quality_results = payload.get("quality_results") or []
            dataset_name = payload.get("dataset_name") or dataset_name_from_results_file(result_path)
            dataset_type = payload.get("dataset_type")
            language_pair = payload.get("language_pair")
            num_cases = payload.get("num_cases")
        else:
            quality_results = payload if isinstance(payload, list) else []
            dataset_name = dataset_name_from_results_file(result_path)
            dataset_type, language_pair = parse_dataset_info(dataset_name)
            num_cases = None

        if not dataset_type or not language_pair:
            dataset_type, language_pair = parse_dataset_info(dataset_name)

        dataset_type = normalize_dataset_type(dataset_type)

        stats, detailed_logs, total_segments, alignment_mismatches = compute_accuracy_from_quality_results(quality_results)
        if alignment_mismatches:
            scenario_counts = {"GOOD": 0, "BAD": 0}
            for item in alignment_mismatches:
                scenario_counts[item["scenario"]] += 1
            space_punct_summary_by_dataset[dataset_name] = {
                "dataset_type": dataset_type,
                "language_pair": language_pair,
                "num_records": len(quality_results),
                "mismatch_entries": len(alignment_mismatches),
                "mismatch_cases": len({m["case_id"] for m in alignment_mismatches}),
                "by_scenario": scenario_counts,
            }
            space_punct_mismatches.extend(alignment_mismatches)

        metrics_payload = {
            "dataset_name": dataset_name,
            "dataset_type": dataset_type,
            "language_pair": language_pair,
            "num_cases": num_cases,
            "num_records": len(quality_results),
            "num_segments_evaluated": total_segments,
            "evaluation_unit": "segment",
            "thresholds": {
                "iou": IOU_THRESHOLD,
                "confidence": ERROR_CONFIDENCE_THRESHOLD,
                "severities": sorted(ALLOWED_ERROR_SEVERITIES),
            },
            "strategies": {k: v.to_dict() for k, v in stats.items()},
            "detailed_logs": detailed_logs,
        }

        metrics_path = metrics_dir / f"{dataset_name}_metrics.json"
        save_json(metrics_path, metrics_payload)
        print(f"[Saved] {metrics_path}")

        dataset_summaries[dataset_name] = {
            "dataset_type": dataset_type,
            "language_pair": language_pair,
            "num_cases": num_cases,
            "num_records": len(quality_results),
            "num_segments_evaluated": total_segments,
            "metrics_file": str(metrics_path),
            "overall": {k: v.overall.to_dict() for k, v in stats.items()},
            "overall_by_scenario": {k: {s: b.to_dict() for s, b in v.by_scenario.items()} for k, v in stats.items()},
            "strategies": {k: v.to_dict() for k, v in stats.items()},
        }

        merge_strategy_stats(overall_stats, stats)

        if dataset_type not in by_type:
            by_type[dataset_type] = create_strategy_stats()
        merge_strategy_stats(by_type[dataset_type], stats)

        if language_pair not in by_lang:
            by_lang[language_pair] = create_strategy_stats()
        merge_strategy_stats(by_lang[language_pair], stats)

    summary_payload = {
        "evaluation_unit": "segment",
        "thresholds": {
            "iou": IOU_THRESHOLD,
            "confidence": ERROR_CONFIDENCE_THRESHOLD,
            "severities": sorted(ALLOWED_ERROR_SEVERITIES),
        },
        "overall": {k: v.to_dict() for k, v in overall_stats.items()},
        "by_dataset_type": {k: {sid: s.to_dict() for sid, s in v.items()} for k, v in by_type.items()},
        "by_language_pair": {k: {sid: s.to_dict() for sid, s in v.items()} for k, v in by_lang.items()},
        "datasets": dataset_summaries,
    }

    summary_path = metrics_dir / "xcomet_metrics_summary.json"
    save_json(summary_path, summary_payload)
    print(f"[Saved] {summary_path}")

    space_punct_payload = {
        "summary": {
            "num_datasets": len(result_files),
            "num_mismatch_entries": len(space_punct_mismatches),
            "datasets_with_mismatches": len(space_punct_summary_by_dataset),
            "by_dataset": space_punct_summary_by_dataset,
        },
        "mismatches": space_punct_mismatches,
    }
    space_punct_path = output_dir / "segment_alignment_space_punct_mismatches.json"
    save_json(space_punct_path, space_punct_payload)
    print(f"[Saved] {space_punct_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="XCOMET evaluation across all datasets.")
    parser.add_argument("--samples-dir", default=str(DEFAULT_SAMPLES_DIR), help="Directory with samples_clean JSONs.")
    parser.add_argument("--error-spans-dir", default=str(DEFAULT_ERROR_SPANS_DIR), help="Directory with error spans JSONs.")
    parser.add_argument("--xcomet-model", choices=sorted(XCOMET_MODEL_CKPTS.keys()), default=DEFAULT_XCOMET_MODEL, help="XCOMET model size (sets default ckpt/output dir).")
    parser.add_argument("--output-dir", default=None, help="Output directory for results/metrics (default depends on --xcomet-model).")
    parser.add_argument("--xcomet-ckpt", default=None, help="XCOMET checkpoint path (override default for --xcomet-model).")
    parser.add_argument("--mode", choices=["all", "check", "xcomet", "metrics"], default="all", help="Run stage(s).")
    parser.add_argument("--xcomet-scope", choices=["dataset", "all"], default="dataset", help="Predict per dataset or all at once.")
    parser.add_argument("--dataset", action="append", default=[], help="Dataset name (stem) to process; can repeat.")
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir)
    error_spans_dir = Path(args.error_spans_dir)
    model_name, ckpt_path, output_dir = resolve_xcomet_config(args.xcomet_model, args.xcomet_ckpt, args.output_dir)
    datasets_filter = args.dataset or []

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("all", "check"):
        run_alignment_check(samples_dir, output_dir, datasets_filter)

    if args.mode in ("all", "xcomet"):
        run_xcomet_stage(samples_dir, error_spans_dir, output_dir, ckpt_path, datasets_filter, args.xcomet_scope)

    if args.mode in ("all", "metrics"):
        run_metrics_stage(output_dir, datasets_filter)


if __name__ == "__main__":
    main()
