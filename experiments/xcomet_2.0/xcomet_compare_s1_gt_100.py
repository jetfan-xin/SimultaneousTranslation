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

# Default XCOMET checkpoint path (override env-based loading to avoid setting WORD_QE_CKPT).
DEFAULT_XCOMET_CKPT = "/ltstorage/home/4xin/models/XCOMET-XL/checkpoints/model.ckpt"


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


def resolve_ground_truth_spans(case: Dict[str, Any], scenario_label: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    Pick GT spans for the given scenario.
    Priority:
      1) Explicit full-text GT (gt_error_spans_full_bad / gt_error_spans_full_good).
      2) Explicit per-segment GT (gt_error_spans_bad_by_seg).
      3) Fallback: char-level diff between good MT and bad MT.
    """
    if scenario_label == "GOOD":
        manual_good = case.get("gt_error_spans_full_good")
        if manual_good is not None:
            return manual_good, "manual_full_good"
        return [], "clean_example"

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


# ================= Comparison Logic =================

def run_evaluation(xcomet: XCOMETLoader, cases: Dict[str, Dict[str, Any]]):

    stats = {
        "1.1": StrategyStats("Strategy 1.1 (S_full, MT_full, No Ref)"),
        "1.2": StrategyStats("Strategy 1.2 (S_full, MT_full, With Ref)"),
    }

    # Store detailed logs for JSON output
    detailed_logs = []

    # Collect all GOOD/BAD translations first to do two batched XCOMET calls (1.1, 1.2)
    print(f"Starting evaluation on {len(cases)} cases (GOOD + BAD)...")
    eval_entries = []
    for key, case in cases.items():
        src_full = case["src_full"]
        ref_full = case.get("ref_full", "")
        scenarios = [
            ("BAD", case["mt_full_bad"]),
            ("GOOD", case["mt_full_good"])
        ]
        for scenario_label, mt_full in scenarios:
            gt_spans, gt_source = resolve_ground_truth_spans(case, scenario_label)
            eval_entries.append({
                "case_id": key,
                "case_label": case.get("label", key),
                "scenario": scenario_label,
                "src": src_full,
                "ref": ref_full,
                "mt_text": mt_full,
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

    # Map results back to each case/scenario in order
    for idx, entry in enumerate(eval_entries):
        res_1_1 = results_1_1[idx] if idx < len(results_1_1) else {"score": 0.0, "error_spans": []}
        pred_spans_1_1 = res_1_1.get("error_spans") or []
        metrics_1_1 = compute_span_metrics(entry["mt_text"], pred_spans_1_1, entry["gt_spans"])
        score_1_1 = res_1_1.get("score")
        stats["1.1"].update(entry["scenario"], metrics_1_1)
        detailed_logs.append({
            "case_id": entry["case_id"],
            "case_label": entry["case_label"],
            "scenario": entry["scenario"],
            "strategy_id": "1.1",
            "src_text": entry["src"],
            "ref_text": entry["ref"],
            "mt_text": entry["mt_text"],
            "predicted_error_spans": pred_spans_1_1,
            "ground_truth_error_spans": entry["gt_spans"],
            "ground_truth_source": entry["gt_source"],
            "metrics": metrics_1_1,
            "sentence_score": score_1_1,
            "score": score_1_1
        })

        res_1_2 = results_1_2[idx] if idx < len(results_1_2) else {"score": 0.0, "error_spans": []}
        pred_spans_1_2 = res_1_2.get("error_spans") or []
        metrics_1_2 = compute_span_metrics(entry["mt_text"], pred_spans_1_2, entry["gt_spans"])
        score_1_2 = res_1_2.get("score")
        stats["1.2"].update(entry["scenario"], metrics_1_2)
        detailed_logs.append({
            "case_id": entry["case_id"],
            "case_label": entry["case_label"],
            "scenario": entry["scenario"],
            "strategy_id": "1.2",
            "src_text": entry["src"],
            "ref_text": entry["ref"],
            "mt_text": entry["mt_text"],
            "predicted_error_spans": pred_spans_1_2,
            "ground_truth_error_spans": entry["gt_spans"],
            "ground_truth_source": entry["gt_source"],
            "metrics": metrics_1_2,
            "sentence_score": score_1_2,
            "score": score_1_2
        })

    # ================= Print Final Stats =================
    pretty("Final Accuracy Report (Strategy 1 vs Ground Truth)")
    print(f"Total translations evaluated: {len(cases) * 2} (GOOD + BAD)")
    print(f"Decision thresholds -> IoU: {IOU_THRESHOLD}")
    print("=" * 100)
    print(f"{'Strategy':<45} | {'Accurate':<10} | {'Total':<10} | {'Accuracy %':<10} | {'Avg IoU':<8} | {'Avg F1'}")
    print("-" * 100)
    for key in ("1.1", "1.2"):
        bucket = stats[key].overall
        print(f"{stats[key].name:<45} | {bucket.accurate:<10} | {bucket.total:<10} | {bucket.accuracy_percent():.2f}% | {bucket.avg(bucket.sum_iou):.3f} | {bucket.avg(bucket.sum_f1):.3f}")

    # ================= Save Structured Results =================

    summary_data = {
        "num_cases": len(cases),
        "num_translations_evaluated": len(cases) * 2,
        "thresholds": {
            "iou": IOU_THRESHOLD
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
