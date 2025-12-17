# -*- coding: utf-8 -*-
"""
Compare XCOMET strategies on 100 constructed cases with Accuracy evaluation.
Saves structured output for further analysis.

Baselines:
    - For NO-REF sub-strategies: Baseline 1.1 (S_full, MT_full)
    - For WITH-REF sub-strategies: Baseline 1.2 (S_full, MT_full, Ref_full)

Evaluation:
    How well do sub-strategies (Segment-level) match the error spans found by their
    respective Baselines? (Using Character-level IoU).

Output:
    - Console report
    - xcomet_strategy_details.json (Detailed per-segment logs)
    - xcomet_strategy_summary.json (Final accuracy stats)

Usage:
    Ensure 'xcomet_build_cases.py' is in the same directory or PYTHONPATH.
    python experiments/xcomet_compare_strategies.py
"""

import os
import sys
import json
from typing import List, Dict, Any

# 添加当前脚本所在目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 添加项目根目录到路径
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
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


# ================= Utils & Metrics =================

def pretty(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def get_char_mask(length: int, spans: List[Dict[str, Any]]) -> List[bool]:
    mask = [False] * length
    for span in spans:
        s = span.get('start')
        e = span.get('end')
        if s is not None and e is not None:
            s = max(0, s)
            e = min(e, length)
            for i in range(s, e):
                mask[i] = True
    return mask

def check_accuracy_on_segment(seg_text: str, 
                              baseline_spans_in_seg: List[Dict[str, Any]], 
                              strategy_spans: List[Dict[str, Any]]) -> bool:
    """
    Definition of Accurate:
    1. Both find NO errors.
    2. Both find errors, and the overlap is significant (>= 75% of the union).
    """
    seg_len = len(seg_text)
    if seg_len == 0:
        return True

    # 1. True Negative
    if not baseline_spans_in_seg and not strategy_spans:
        return True
    
    # 2. Mismatch (False Positive or False Negative)
    if bool(baseline_spans_in_seg) != bool(strategy_spans):
        return False

    # 3. Check Overlap (True Positive quality)
    mask_base = get_char_mask(seg_len, baseline_spans_in_seg)
    mask_strat = get_char_mask(seg_len, strategy_spans)

    intersection = sum(b and s for b, s in zip(mask_base, mask_strat))
    union = sum(b or s for b, s in zip(mask_base, mask_strat))
    
    if union == 0: return True
    
    iou = intersection / union
    return iou >= 0.75

def map_baseline_to_segments(mt_segs: List[str], baseline_result: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    """
    Maps global error spans from Baseline (MT_full) to each segment (MT_seg).
    """
    segment_errors = []
    current_offset = 0
    global_spans = baseline_result.get('error_spans', []) or []

    for seg in mt_segs:
        seg_len = len(seg)
        seg_start = current_offset
        seg_end = current_offset + seg_len
        
        local_spans = []
        for span in global_spans:
            g_start = span.get('start', -1)
            g_end = span.get('end', -1)
            
            # Check overlap
            overlap_start = max(g_start, seg_start)
            overlap_end = min(g_end, seg_end)
            
            if overlap_end > overlap_start:
                l_start = overlap_start - seg_start
                l_end = overlap_end - seg_start
                local_spans.append({
                    'start': l_start,
                    'end': l_end,
                    'text': seg[l_start:l_end],
                    'severity': span.get('severity', 'major')
                })
        
        segment_errors.append(local_spans)
        current_offset += seg_len
    
    return segment_errors

# ================= Stats Container =================

class StrategyStats:
    def __init__(self, name):
        self.name = name
        self.total_segments = 0
        self.accurate_segments = 0
    
    def update(self, is_accurate: bool):
        self.total_segments += 1
        if is_accurate:
            self.accurate_segments += 1
            
    def get_accuracy(self):
        return (self.accurate_segments / self.total_segments) * 100 if self.total_segments > 0 else 0.0
    
    def to_dict(self):
        return {
            "name": self.name,
            "total_segments": self.total_segments,
            "accurate_segments": self.accurate_segments,
            "accuracy_percent": self.get_accuracy()
        }

# ================= Comparison Logic =================

def run_evaluation(xcomet: XCOMETLoader, cases: Dict[str, Dict[str, Any]]):
    
    stats = {
        "2.1": StrategyStats("Strategy 2.1 (S_seg, MT_seg, No Ref)"),
        "2.2": StrategyStats("Strategy 2.2 (S_seg, MT_seg, With Ref)"),
        "3.1": StrategyStats("Strategy 3.1 (S_full, MT_seg, No Ref)"),
        "3.2": StrategyStats("Strategy 3.2 (S_full, MT_seg, With Ref)"),
    }

    # Store detailed logs for JSON output
    detailed_logs = []

    print(f"Starting evaluation on {len(cases)} cases...")

    for key, case in cases.items():
        # Scenarios to test: BAD translation (has errors), GOOD translation (no errors)
        scenarios = [
            ("BAD", case["mt_full_bad"], case["mt_segs_bad"]),
            ("GOOD", case["mt_full_good"], case["mt_segs_good"])
        ]
        
        src_full = case["src_full"]
        ref_full = case["ref_full"]
        src_segs = case["src_segs"]
        ref_segs = case["ref_segs"]

        for scenario_label, mt_full, mt_segs in scenarios:
            
            # --- 1. Run Baselines (Ground Truths) ---
            
            # Baseline 1.1: NO Ref
            baseline_input_noref = [{"src": src_full, "mt": mt_full}]
            baseline_output_noref = xcomet.predict(baseline_input_noref)[0]
            expected_errors_noref = map_baseline_to_segments(mt_segs, baseline_output_noref)

            # Baseline 1.2: WITH Ref
            baseline_input_ref = [{"src": src_full, "mt": mt_full, "ref": ref_full}]
            baseline_output_ref = xcomet.predict(baseline_input_ref)[0]
            expected_errors_ref = map_baseline_to_segments(mt_segs, baseline_output_ref)

            # Helper to record results
            def record_segment_result(strat_id, seg_idx, mt_text, pred_spans, expected_spans, is_acc):
                stats[strat_id].update(is_acc)
                detailed_logs.append({
                    "case_id": key,
                    "case_label": case["label"],
                    "scenario": scenario_label,
                    "strategy_id": strat_id,
                    "segment_index": seg_idx,
                    "segment_text": mt_text,
                    "predicted_errors": pred_spans,
                    "expected_errors_from_baseline": expected_spans,
                    "is_accurate": is_acc
                })

            # --- 2. Run Strategy 2 (Source-side Segmentation) ---
            
            # [2.1] Without Ref -> Compare vs Baseline 1.1
            inputs_2_1 = [{"src": s, "mt": m} for s, m in zip(src_segs, mt_segs)]
            results_2_1 = xcomet.predict(inputs_2_1)
            for i, res in enumerate(results_2_1):
                pred = res.get('error_spans', [])
                is_acc = check_accuracy_on_segment(mt_segs[i], expected_errors_noref[i], pred)
                record_segment_result("2.1", i, mt_segs[i], pred, expected_errors_noref[i], is_acc)

            # [2.2] With Ref -> Compare vs Baseline 1.2
            inputs_2_2 = [{"src": s, "mt": m, "ref": r} for s, m, r in zip(src_segs, mt_segs, ref_segs)]
            results_2_2 = xcomet.predict(inputs_2_2)
            for i, res in enumerate(results_2_2):
                pred = res.get('error_spans', [])
                is_acc = check_accuracy_on_segment(mt_segs[i], expected_errors_ref[i], pred)
                record_segment_result("2.2", i, mt_segs[i], pred, expected_errors_ref[i], is_acc)

            # --- 3. Run Strategy 3 (Full Source, Segmented Output) ---
            
            # [3.1] Without Ref -> Compare vs Baseline 1.1
            inputs_3_1 = [{"src": src_full, "mt": m} for m in mt_segs]
            results_3_1 = xcomet.predict(inputs_3_1)
            for i, res in enumerate(results_3_1):
                pred = res.get('error_spans', [])
                is_acc = check_accuracy_on_segment(mt_segs[i], expected_errors_noref[i], pred)
                record_segment_result("3.1", i, mt_segs[i], pred, expected_errors_noref[i], is_acc)

            # [3.2] With Ref -> Compare vs Baseline 1.2
            inputs_3_2 = [{"src": src_full, "mt": m, "ref": ref_full} for m in mt_segs]
            results_3_2 = xcomet.predict(inputs_3_2)
            for i, res in enumerate(results_3_2):
                pred = res.get('error_spans', [])
                is_acc = check_accuracy_on_segment(mt_segs[i], expected_errors_ref[i], pred)
                record_segment_result("3.2", i, mt_segs[i], pred, expected_errors_ref[i], is_acc)

    # ================= Print Final Stats =================
    pretty("Final Accuracy Report (vs Corresponding Baselines)")
    print("Baseline Mapping:")
    print("  - Strategies 2.1 & 3.1 (No Ref) are compared against Strategy 1.1 (S_full, MT_full)")
    print("  - Strategies 2.2 & 3.2 (With Ref) are compared against Strategy 1.2 (S_full, MT_full, Ref_full)")
    print("=" * 85)
    print(f"Total Cases processed: {len(cases)} (Both GOOD and BAD scenarios)")
    print(f"{'Strategy':<45} | {'Accurate':<10} | {'Total Segs':<10} | {'Accuracy %'}")
    print("-" * 85)
    for key in sorted(stats.keys()):
        s = stats[key]
        print(f"{s.name:<45} | {s.accurate_segments:<10} | {s.total_segments:<10} | {s.get_accuracy():.2f}%")

    # ================= Save Structured Results =================
    
    # Save Summary
    summary_data = {k: v.to_dict() for k, v in stats.items()}
    summary_file = "xcomet_strategy_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    print(f"\n[Saved] Summary stats saved to {summary_file}")

    # Save Detailed Logs
    details_file = "xcomet_strategy_details.json"
    with open(details_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_logs, f, ensure_ascii=False, indent=2)
    print(f"[Saved] Detailed logs saved to {details_file}")

# ================= Main =================

def main():
    print("[Main] Loading model...")
    xcomet = XCOMETLoader()
    
    print("[Main] Loading cases from xcomet_build_cases.py ...")
    cases = build_wmt24_100_cases()
    print(f"[Main] Total Cases Loaded: {len(cases)}")
    
    if len(cases) == 0:
        print("Warning: No cases loaded. Please check xcomet_build_cases.py.")
        return

    run_evaluation(xcomet, cases)

if __name__ == "__main__":
    main()