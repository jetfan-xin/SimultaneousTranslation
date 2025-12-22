#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate batched prompt files from xcomet_build_cases.py.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List


DEFAULT_TEMPLATE = """You are a translation error annotator. You will receive a list of cases
(English source, Chinese reference, good MT, bad MT, and optional segments). Your task is to
mark error spans in the bad translation.

Input format (JSON):
[
  {
    "case_id": "Case1",
    "label": "...",
    "src_full": "...",
    "ref_full": "...",
    "mt_full_good": "...",
    "mt_full_bad": "...",
    "src_segs": [...],
    "ref_segs": [...],
    "mt_segs_good": [...],
    "mt_segs_bad": [...]
  },
  ...
]

Rules:
1) Only mark errors in mt_full_bad. Use src_full/ref_full/mt_full_good to judge errors.
2) Do not do character-level diff. Use syntactic units (word/phrase/clause) as the minimum span.
3) Spans must be contiguous substrings of mt_full_bad. Provide 0-based start/end (end exclusive).
4) Spans must not overlap. Output in ascending start order.
5) For omissions, mark the smallest clause/phrase in mt_full_bad where the missing info should be.
6) Include a "text" field that exactly matches mt_full_bad[start:end] for validation.
7) Optional field: error_type (semantic/factual/polarity/hallucination/omission/grammar).

Output format (JSON only, no explanation):
[
  {
    "case_id": "Case1",
    "gt_error_spans_full_bad": [
      {"start": 0, "end": 12, "text": "...", "error_type": "semantic"}
    ]
  },
  ...
]

Now annotate these cases:
<CASES_JSON>
"""


def load_cases() -> Dict[str, Dict[str, Any]]:
    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    from xcomet_build_cases import build_wmt24_100_cases
    return build_wmt24_100_cases()


def order_cases(cases: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    def sort_key(item):
        case_id = item[0]
        if case_id.startswith("Case"):
            try:
                return int(case_id[4:])
            except ValueError:
                pass
        return case_id

    ordered = []
    for case_id, case in sorted(cases.items(), key=sort_key):
        ordered.append({
            "case_id": case_id,
            "label": case.get("label"),
            "src_full": case.get("src_full"),
            "ref_full": case.get("ref_full"),
            "mt_full_good": case.get("mt_full_good"),
            "mt_full_bad": case.get("mt_full_bad"),
            "src_segs": case.get("src_segs"),
            "ref_segs": case.get("ref_segs"),
            "mt_segs_good": case.get("mt_segs_good"),
            "mt_segs_bad": case.get("mt_segs_bad"),
        })
    return ordered


def load_template(template_path: str) -> str:
    if not template_path:
        return DEFAULT_TEMPLATE
    path = Path(template_path)
    if not path.exists():
        print(f"[Warning] Template not found: {path}. Falling back to default template.")
        return DEFAULT_TEMPLATE
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        print(f"[Warning] Template file is empty: {path}. Falling back to default template.")
        return DEFAULT_TEMPLATE
    return text


def render_prompt(template: str, batch: List[Dict[str, Any]]) -> str:
    cases_json = json.dumps(batch, ensure_ascii=False, indent=2)
    return template.replace("<CASES_JSON>", cases_json)


def chunk_cases(cases: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    if batch_size <= 0:
        return [cases]
    return [cases[i:i + batch_size] for i in range(0, len(cases), batch_size)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate batched error-span prompts from cases.")
    parser.add_argument("--batch-size", type=int, default=10, help="Cases per prompt file (<=0 for single file).")
    parser.add_argument("--output-dir", default="/ltstorage/home/4xin/SimultaneousTranslation/experiments/xcomet_3.0/xcomet_error_span_prompts", help="Directory to store prompt files.")
    parser.add_argument("--template", default="/ltstorage/home/4xin/SimultaneousTranslation/experiments/xcomet_3.0/prompt.txt", help="Path to a custom prompt template file.")
    args = parser.parse_args()

    cases = load_cases()
    ordered = order_cases(cases)
    template = load_template(args.template)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.batch_size <= 0:
        prompt_text = render_prompt(template, ordered)
        output_path = output_dir / "prompt_all.txt"
        output_path.write_text(prompt_text, encoding="utf-8")
        print(f"[Saved] Single prompt file to {output_path}")
        return

    batches = chunk_cases(ordered, args.batch_size)

    total = len(batches)
    width = len(str(total))
    for idx, batch in enumerate(batches, start=1):
        prompt_text = render_prompt(template, batch)
        filename = f"prompt_batch_{idx:0{width}d}_of_{total:0{width}d}.txt"
        (output_dir / filename).write_text(prompt_text, encoding="utf-8")

    print(f"[Saved] {total} prompt file(s) in {output_dir}")


if __name__ == "__main__":
    main()
