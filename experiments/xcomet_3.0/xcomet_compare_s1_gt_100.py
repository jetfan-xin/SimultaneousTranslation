# -*- coding: utf-8 -*-
"""
Compare XCOMET strategy 1 (Baselines 1.1 & 1.2) on 100 constructed cases with Accuracy evaluation.
Saves structured output for further analysis.

Ground Truth: 100 constructed cases with known good & bad translations.
    - Good examples: No errors
    - Bad examples: Known error segments. But maybe just parts of the segments (spans) are wrong.
                    Need char-level IoU evaluation or other more appropriate fine-grained evaluation.

Baselines:
    - For NO-REF sub-strategies: Baseline 1.1 (S_full, MT_full)
    - For WITH-REF sub-strategies: Baseline 1.2 (S_full, MT_full, Ref_full)

Evaluation:
    How well do strategy 1 (Baselines 1.1 & 1.2) match the ground truth error spans in good & wrong examples?

Output:
    - Console report
    - xcomet_strategy_1_details.json (Detailed per-sentence logs)
    - xcomet_strategy_1_summary.json (Final accuracy stats)

Usage:
    Ensure 'xcomet_build_cases.py' is in the same directory or PYTHONPATH.
    python experiments/xcomet_2.0/xcomet_compare_s1_gt_100.py
"""

import os
import sys
import json
from difflib import SequenceMatcher
from typing import List, Dict, Any, Tuple

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# 添加当前脚本所在目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 添加项目根目录到路径
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from xcomet_loader import XCOMETLoader
except ImportError:
    print("Error: Could not import XCOMETLoader. Make sure you are in the project root or environment is set.")
    sys.exit(1)

# 导入你的数据构建模块
try:
    from xcomet_build_cases import build_wmt24_100_cases
except ImportError:
    print("Error: Could not import build_wmt24_100_cases from xcomet_build_cases.py")
    print("Please ensure xcomet_build_cases.py is in the same directory as this script.")
    sys.exit(1)

# ================= Config =================

# Char-level thresholds used to decide whether a predicted span matches GT.
IOU_THRESHOLD = 0.5

# Only count XCOMET spans as errors when they pass both filters.
ERROR_CONFIDENCE_THRESHOLD = 0.5
ALLOWED_ERROR_SEVERITIES = {"major", "critical"}

# Default XCOMET checkpoint path (override env-based loading to avoid setting WORD_QE_CKPT).
DEFAULT_XCOMET_CKPT = "/ltstorage/home/4xin/models/XCOMET-XL/checkpoints/model.ckpt"

# Ground-truth error spans for BAD translations.
GT_BAD_SPANS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xcomet_gt_error_spans_bad")

# Human-friendly names for each evaluated strategy.
STRATEGY_NAMES = {
    "1.1": "Strategy 1.1 (S_full, MT_full, No Ref)",
    "1.2": "Strategy 1.2 (S_full, MT_full, With Ref)",
}


# ================= Utils & Metrics =================

def pretty(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def get_char_mask(length: int, spans: List[Dict[str, Any]]) -> List[bool]:
    mask = [False] * length
    for span in spans or []:
        s = span.get("start")
        e = span.get("end")
        if s is None or e is None:
            continue
        s = max(0, min(int(s), length))
        e = max(s, min(int(e), length))
        for i in range(s, e):
            mask[i] = True
    return mask


def map_segment_spans_to_full(segment_spans: List[List[Dict[str, Any]]], mt_segments: List[str]) -> List[Dict[str, Any]]:
    """Map per-segment spans to offsets on the concatenated full MT text."""
    if not segment_spans:
        return []
    mapped: List[Dict[str, Any]] = []
    offset = 0
    for idx, seg in enumerate(mt_segments):
        spans_in_seg = segment_spans[idx] if idx < len(segment_spans) else []
        for span in spans_in_seg or []:
            if span is None:
                continue
            start = max(0, min(int(span.get("start", 0) or 0), len(seg)))
            end = max(start, min(int(span.get("end", 0) or 0), len(seg)))
            mapped.append({
                "start": offset + start,
                "end": offset + end,
                "source_seg": idx,
                "text": seg[start:end],
                "label": span.get("label", span.get("type"))
            })
        offset += len(seg)
    return mapped


def derive_diff_spans(mt_full_good: str, mt_full_bad: str) -> List[Dict[str, Any]]:
    """Fallback GT: char-level diffs between good MT and bad MT."""
    matcher = SequenceMatcher(None, mt_full_good or "", mt_full_bad or "", autojunk=False)
    spans: List[Dict[str, Any]] = []
    for tag, _, _, j1, j2 in matcher.get_opcodes():
        if tag == "equal" or j1 == j2:
            continue
        spans.append({
            "start": j1,
            "end": j2,
            "tag": tag,
            "text": (mt_full_bad or "")[j1:j2]
        })
    return spans


def load_external_bad_gt_spans(spans_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    spans_map: Dict[str, List[Dict[str, Any]]] = {}
    if not spans_dir or not os.path.isdir(spans_dir):
        print(f"[Warning] GT spans directory not found: {spans_dir}")
        return spans_map

    for filename in sorted(os.listdir(spans_dir)):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(spans_dir, filename)
        try:
            if os.path.getsize(path) == 0:
                print(f"[Warning] GT spans file is empty: {path}")
                continue
            raw = open(path, "r", encoding="utf-8").read().strip()
            if not raw:
                print(f"[Warning] GT spans file is empty: {path}")
                continue
            data = json.loads(raw)
        except Exception as exc:
            print(f"[Warning] Failed to load GT spans from {path}: {exc}")
            continue

        if not isinstance(data, list):
            print(f"[Warning] GT spans file is not a list: {path}")
            continue

        for item in data:
            if not isinstance(item, dict):
                continue
            case_id = item.get("case_id")
            if not case_id:
                continue
            spans = item.get("gt_error_spans_full_bad")
            if spans is None:
                spans = []
            spans_map[case_id] = spans

    return spans_map


def resolve_ground_truth_spans(case: Dict[str, Any],
                               scenario_label: str,
                               case_id: str = "",
                               external_bad_spans: Dict[str, List[Dict[str, Any]]] = None) -> Tuple[List[Dict[str, Any]], str]:
    """
    Pick GT spans for the given scenario.
    Priority:
      0) External GT spans for BAD translations (xcomet_gt_error_spans_bad).
      1) Explicit full-text GT (gt_error_spans_full_bad / gt_error_spans_full_good).
      2) Explicit per-segment GT (gt_error_spans_bad_by_seg).
      3) Fallback: char-level diff between good MT and bad MT.
    """
    if scenario_label == "GOOD":
        manual_good = case.get("gt_error_spans_full_good")
        if manual_good is not None:
            return manual_good, "manual_full_good"
        return [], "clean_example"

    if external_bad_spans and case_id in external_bad_spans:
        return external_bad_spans[case_id], "external_bad_spans"

    manual_full = (
        case.get("gt_error_spans_full_bad")
        or case.get("gt_error_spans_bad_full")
        or case.get("gt_error_spans_bad")
    )
    if manual_full is not None:
        return manual_full, "manual_full_bad"

    manual_by_seg = (
        case.get("gt_error_spans_bad_by_seg")
        or case.get("gt_error_spans_by_seg")
        or case.get("bad_error_spans_by_seg")
    )
    if manual_by_seg and case.get("mt_segs_bad"):
        return map_segment_spans_to_full(manual_by_seg, case["mt_segs_bad"]), "manual_segments_bad"

    # Fallback: derive diff between good/bad MT
    return derive_diff_spans(case.get("mt_full_good", ""), case.get("mt_full_bad", "")), "auto_diff_vs_good"


def compute_span_metrics(mt_text: str,
                         pred_spans: List[Dict[str, Any]],
                         gt_spans: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Char-level IoU / precision / recall / F1 with a boolean accuracy flag."""
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
        "is_accurate": is_accurate
    }


def filter_predicted_error_spans(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep only spans that meet the confidence/severity constraints so that only
    high-confidence major/critical errors are counted.
    """
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
            "severity": severity
        })
        filtered.append(clean_span)

    filtered.sort(key=lambda s: s["start"])
    return filtered


def build_segments(mt_text: str, mt_segs: List[str]) -> Tuple[List[Dict[str, Any]], bool]:
    if not mt_segs:
        return [{
            "index": 0,
            "start": 0,
            "end": len(mt_text or ""),
            "text": mt_text or ""
        }], True
    combined = "".join(mt_segs)
    if combined != (mt_text or ""):
        return [{
            "index": 0,
            "start": 0,
            "end": len(mt_text or ""),
            "text": mt_text or ""
        }], False
    segments: List[Dict[str, Any]] = []
    offset = 0
    for idx, seg in enumerate(mt_segs):
        seg_len = len(seg)
        segments.append({
            "index": idx,
            "start": offset,
            "end": offset + seg_len,
            "text": seg
        })
        offset += seg_len
    return segments, True


def extract_segment_spans(spans: List[Dict[str, Any]],
                          seg_start: int,
                          seg_end: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (relative_spans, full_spans) that overlap the segment."""
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
            "avg_f1": self.avg(self.sum_f1)
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

    def to_dict(self):
        return {
            "name": self.name,
            "overall": self.overall.to_dict(),
            "by_scenario": {k: v.to_dict() for k, v in self.by_scenario.items()}
        }


def create_strategy_stats() -> Dict[str, StrategyStats]:
    return {key: StrategyStats(name) for key, name in STRATEGY_NAMES.items()}


def build_quality_result(entry: Dict[str, Any],
                         strategy_id: str,
                         raw_result: Dict[str, Any]) -> Dict[str, Any]:
    raw_result = raw_result or {}
    raw_spans = raw_result.get("error_spans") or []
    filtered_spans = filter_predicted_error_spans(raw_spans)
    return {
        "case_id": entry["case_id"],
        "case_label": entry["case_label"],
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
            "allowed_severities": sorted(ALLOWED_ERROR_SEVERITIES)
        }
    }


def compute_accuracy_from_quality_results(quality_results: List[Dict[str, Any]]) -> Tuple[Dict[str, StrategyStats], List[Dict[str, Any]], int]:
    stats = create_strategy_stats()
    detailed_logs: List[Dict[str, Any]] = []
    total_segments_evaluated = 0

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
        segments, alignment_ok = build_segments(mt_text, mt_segs)

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
                "case_id": record.get("case_id"),
                "case_label": record.get("case_label"),
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
                "xcomet_raw_output": record.get("xcomet_raw_output")
            })

    return stats, detailed_logs, total_segments_evaluated


# ================= Comparison Logic =================

def run_evaluation(xcomet: XCOMETLoader, cases: Dict[str, Dict[str, Any]]):

    external_bad_spans = load_external_bad_gt_spans(GT_BAD_SPANS_DIR)
    if external_bad_spans:
        print(f"[Main] Loaded external BAD GT spans for {len(external_bad_spans)} cases from {GT_BAD_SPANS_DIR}")
    else:
        print(f"[Warning] No external BAD GT spans loaded from {GT_BAD_SPANS_DIR}")

    # Collect all GOOD/BAD translations first to do two batched XCOMET calls (1.1, 1.2)
    print(f"Starting evaluation on {len(cases)} cases (GOOD + BAD)...")
    eval_entries = []
    for key, case in cases.items():
        src_full = case["src_full"]
        ref_full = case.get("ref_full", "")
        scenarios = [
            ("BAD", case["mt_full_bad"], case.get("mt_segs_bad") or []),
            ("GOOD", case["mt_full_good"], case.get("mt_segs_good") or [])
        ]
        for scenario_label, mt_full, mt_segs in scenarios:
            gt_spans, gt_source = resolve_ground_truth_spans(
                case,
                scenario_label,
                case_id=key,
                external_bad_spans=external_bad_spans
            )
            eval_entries.append({
                "case_id": key,
                "case_label": case.get("label", key),
                "scenario": scenario_label,
                "src": src_full,
                "ref": ref_full,
                "mt_text": mt_full,
                "mt_segs": mt_segs,
                "gt_spans": gt_spans,
                "gt_source": gt_source
            })

    total_translations = len(eval_entries)
    if tqdm:
        pbar = tqdm(total=2, desc="XCOMET batch eval", unit="strategy")
    else:
        pbar = None

    # --- Strategy 1.1 (no ref) in one batch ---
    print(f"[Stage] Running Strategy 1.1 on {total_translations} translations ...")
    input_1_1 = [{"src": e["src"], "mt": e["mt_text"]} for e in eval_entries]
    results_1_1 = xcomet.predict(input_1_1)
    if pbar:
        pbar.update(1)

    # --- Strategy 1.2 (with ref) in one batch ---
    print(f"[Stage] Running Strategy 1.2 on {total_translations} translations ...")
    input_1_2 = [{"src": e["src"], "mt": e["mt_text"], "ref": e["ref"]} for e in eval_entries]
    results_1_2 = xcomet.predict(input_1_2)
    if pbar:
        pbar.update(1)
        pbar.close()

    if len(results_1_1) != total_translations:
        print(f"[Warning] Strategy 1.1 returned {len(results_1_1)} results (expected {total_translations})")
    if len(results_1_2) != total_translations:
        print(f"[Warning] Strategy 1.2 returned {len(results_1_2)} results (expected {total_translations})")

    # Map results back to each case/scenario in order, with span filtering & structured logging
    quality_results = []
    for idx, entry in enumerate(eval_entries):
        res_1_1 = results_1_1[idx] if idx < len(results_1_1) else {"score": 0.0, "error_spans": []}
        res_1_2 = results_1_2[idx] if idx < len(results_1_2) else {"score": 0.0, "error_spans": []}
        quality_results.append(build_quality_result(entry, "1.1", res_1_1))
        quality_results.append(build_quality_result(entry, "1.2", res_1_2))

    # Save all XCOMET translation quality outputs (raw + filtered spans) for reuse
    quality_results_file = "xcomet_strategy_1_quality_results.json"
    with open(quality_results_file, 'w', encoding='utf-8') as f:
        json.dump(quality_results, f, ensure_ascii=False, indent=2)
    print(f"[Saved] XCOMET quality outputs saved to {quality_results_file}")

    # Use the saved quality file to compute accuracy so that the metric logic can
    # be easily updated later without rerunning XCOMET.
    with open(quality_results_file, 'r', encoding='utf-8') as f:
        stored_quality_results = json.load(f)

    stats, detailed_logs, total_segments_evaluated = compute_accuracy_from_quality_results(stored_quality_results)

    # ================= Print Final Stats =================
    total_translations_evaluated = len(stored_quality_results)

    pretty("Final Accuracy Report (Strategy 1 vs Ground Truth)")
    print(f"Total segments evaluated: {total_segments_evaluated} (GOOD + BAD)")
    print(f"Total translations evaluated: {total_translations_evaluated} (for reference)")
    print(f"Decision thresholds -> IoU: {IOU_THRESHOLD}, confidence: >{ERROR_CONFIDENCE_THRESHOLD}, severities: {sorted(ALLOWED_ERROR_SEVERITIES)}")
    print("=" * 100)
    print(f"{'Strategy':<45} | {'Accurate':<10} | {'Total':<10} | {'Accuracy %':<10} | {'Avg IoU':<8} | {'Avg F1'}")
    print("-" * 100)
    for key in ("1.1", "1.2"):
        bucket = stats[key].overall
        print(f"{stats[key].name:<45} | {bucket.accurate:<10} | {bucket.total:<10} | {bucket.accuracy_percent():.2f}% | {bucket.avg(bucket.sum_iou):.3f} | {bucket.avg(bucket.sum_f1):.3f}")

    # ================= Save Structured Results =================

    summary_data = {
        "num_cases": len(cases),
        "num_translations_evaluated": total_translations_evaluated,
        "num_segments_evaluated": total_segments_evaluated,
        "evaluation_unit": "segment",
        "quality_results_file": quality_results_file,
        "ground_truth_bad_spans_dir": GT_BAD_SPANS_DIR,
        "thresholds": {
            "iou": IOU_THRESHOLD,
            "confidence": ERROR_CONFIDENCE_THRESHOLD,
            "severities": sorted(ALLOWED_ERROR_SEVERITIES)
        },
        "strategies": {k: v.to_dict() for k, v in stats.items()}
    }

    summary_file = "xcomet_strategy_1_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    print(f"\n[Saved] Summary stats saved to {summary_file}")

    details_file = "xcomet_strategy_1_details.json"
    with open(details_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_logs, f, ensure_ascii=False, indent=2)
    print(f"[Saved] Detailed logs saved to {details_file}")


# ================= Main =================

def main():
    print("[Main] Loading model...")
    xcomet = XCOMETLoader(checkpoint_path=DEFAULT_XCOMET_CKPT)

    print("[Main] Loading cases from xcomet_build_cases.py ...")
    cases = build_wmt24_100_cases()
    print(f"[Main] Total Cases Loaded: {len(cases)}")

    if len(cases) == 0:
        print("Warning: No cases loaded. Please check xcomet_build_cases.py.")
        return

    run_evaluation(xcomet, cases)


if __name__ == "__main__":
    main()
