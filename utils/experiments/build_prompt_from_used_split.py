#!/usr/bin/env python3
import argparse
from pathlib import Path
'''
python /ltstorage/home/4xin/SimultaneousTranslation/utils/experiments/build_prompt_from_used_split.py \
  --input-file /ltstorage/home/4xin/SimultaneousTranslation/experiments/xcomet_3.0/cases/used_split_json/wmt24_en-zh.part001.json \
  --case-start 1
'''

LANG_NAME = {
    "en": "English",
    "zh": "Chinese",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "ja": "Japanese",
}


PROMPT_TEMPLATE = """Role:
You are an expert data engineer and linguist specializing in Machine Translation evaluation. Your task is to process a dataset and output the result directly in **valid JSON format**.

Input Data Source:
- File: `{file_name}`
- Structure: Each line is a JSON object. `src_text` is the Source ({src_lang}), and `tgt_text` is the Reference ({tgt_lang}).

Core Task:
Process EVERY SINGLE ENTRY contained in the provided file content.

Sequential Processing: Iterate through the dataset from the first line to the last line available to you.
One-to-One Mapping: For every source line, generate exactly one corresponding entry in the final JSON object.
Segmentation Logic: Do **NOT** target a specific number of segments. Instead, perform **natural segmentation** based on the sentence structure, length, and punctuation (e.g., split by commas, semicolons, or logical clauses).
Segment Count: There is **NO upper or lower limit** on the number of segments. If a sentence is short or indivisible, it counts as 1 segment. If it is a long paragraph, it might have 10+ segments.

Strict Constraints:
1. Data Consistency (Non-negotiable): For every case, `ref_full` MUST be **bit-for-bit identical** to `"".join(mt_segs_good)`. Do not change a single character of the original reference when constructing the "good" segments.
2. UNo Skipping: Do not skip any hard/long sentences. Process everything.
3. Entirety: Provide **FULL text content**. No ellipsis (`...`).
4. Error Injection Strategy (CRITICAL - READ CAREFULLY):
\t- Target: `mt_segs_bad`
\t- Scope: Inject errors into a subset of the segments (roughly **50% of the segments**). The remaining segments must be perfect (identical to reference).
\t- Random Positioning: You must **randomize the location** of the errors. Do not always put errors at the beginning or end.
\t- Error Span Logic: The injected errors must be designed such that a human annotator could identify them as **contiguous Error Spans** (syntactic units like words, phrases, or clauses).
\t- Nature of Errors:
\t\t- Syntactic Units: Errors must affect complete words, phrases, or clauses. Do **NOT** do character-level or signle-word-level changes.
\t\t- Detectable Types: Create errors that fall into categories like **Semantic Shift** (meaning changed), **Factual Error**, **Complicated Semantic Reversal**, **Hallucination** (confused info), or **Broken Grammar** (that affects meaning).
\t\t- Context: The error must fit the flow of the bad sentence (it represents a bad translation, not a random string).
\t- Examples of Valid Error Injection:
\t\t- Ref Segment: "...after the lengthy meeting ended."
\t\t- Bad Segment: "...**before** the long-time gathering **started**."
\t\t- Ref Segment: "The output is stable."
\t\t- Bad Segment: "The otcome **fluctuates wildly**."

Output Format:
Generate a valid JSON object. The keys should be "Case1", "Case2", etc.

JSON Structure Template:
```json
{{
  "Case1": {{
    "label": "Case 1 (Topic Summary)",
    "source_index": 9, # index of the record in the data file
    "src_full": "Original {src_lang} text",
    "ref_full": "Original {tgt_lang} Reference",
    "mt_segs_good": ["Seg 1 ({tgt_lang})", "Seg 2 ({tgt_lang})", "Seg 3 ({tgt_lang})"], 
    "mt_segs_bad": ["Seg 1 ({tgt_lang})", "Error Seg 2", "Seg 3 ({tgt_lang})"],
    "mt_full_good": "Identical to ref_full",
    "mt_full_bad": "Joined string of mt_segs_bad"
  }},
  "Case2": {{
    ...
  }}
}}
```

Action:
Give out the JSON script now, constructing cases from {file_name} following these rules. Start numbering from Case {case_start}.
"""


def infer_lang_pair(name: str):
    parts = name.replace(".", "_").split("_")
    for part in parts:
        if "-" in part:
            items = part.split("-")
            if len(items) == 2:
                return items[0], items[1]
    return None, None


def resolve_language(code: str) -> str:
    if not code:
        return "Source Language"
    return LANG_NAME.get(code.lower(), code)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a prompt for building XCOMET cases from a used_split file."
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="File name or path under used_split (e.g., wmt23_zh-en.part020.json).",
    )
    parser.add_argument(
        "--lang-pair",
        default=None,
        help="Language pair like zh-en. If omitted, inferred from file name.",
    )
    parser.add_argument(
        "--case-start",
        type=int,
        required=True,
        help="Starting Case number.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the prompt text.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    file_name = args.input_file.name
    if args.lang_pair:
        src_code, tgt_code = args.lang_pair.split("-", 1)
    else:
        src_code, tgt_code = infer_lang_pair(file_name)

    src_lang = resolve_language(src_code)
    tgt_lang = resolve_language(tgt_code)

    prompt = PROMPT_TEMPLATE.format(
        file_name=file_name,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        case_start=args.case_start,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(prompt, encoding="utf-8")
    else:
        print(prompt)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
