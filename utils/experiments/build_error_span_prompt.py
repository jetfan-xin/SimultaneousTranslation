#!/usr/bin/env python3
import argparse
from pathlib import Path

'''
python /ltstorage/home/4xin/SimultaneousTranslation/utils/experiments/build_error_span_prompt.py \
  --input-file /ltstorage/home/4xin/SimultaneousTranslation/experiments/xcomet_3.0/cases/samples_clean/culturemt_en-hi.json
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

EXAMPLE_TEXTS = {
    "en": ("Entertainment news", "is hardly noticeable"),
    "zh": ("娱乐新闻", "几乎看不出来"),
    "de": ("Unterhaltungsnachrichten", "ist kaum bemerkbar"),
    "fr": ("actualités du divertissement", "est à peine perceptible"),
    "es": ("noticias de entretenimiento", "apenas se nota"),
    "hi": ("मनोरंजन समाचार", "लगभग ध्यान देने योग्य नहीं"),
    "ta": ("பொழுதுபோக்கு செய்திகள்", "கிட்டத்தட்ட தெரியாது"),
    "te": ("వినోద వార్తలు", "దాదాపు గమనించబడదు"),
    "ja": ("エンターテインメントニュース", "ほとんど目立たない"),
}


PROMPT_TEMPLATE = """Role: You are a professional translation error annotator and linguist. You will receive a dataset of Machine Translation (MT) cases. Each case includes the original {src_lang} source, the {tgt_lang} reference, a "Good" MT (identical to the reference), and a "Bad" MT (containing injected errors).

Task: Your task is to identify and annotate all the specific error spans within the mt_full_bad text for each case. You must compare mt_full_bad against ref_full (mt_full_good) carefully to pinpoint exactly where the translation failed (e.g., semantic errors, hallucinations, reversals).

Input Format: The input is a JSON file `{file_name}` of case objects with the format:
```json
{{
  "Case1": {{
    "label": "Case 1 (Topic Summary)",
    "source_index": 685, # index of the record in the data file
    "src_full": "Original {src_lang} text",
    "ref_full": "Original {tgt_lang} Reference",
    "mt_segs_good": [
        "Seg 1 ({tgt_lang})", 
        "Seg 2 ({tgt_lang})", 
        "Seg 3 ({tgt_lang})"
    ],
    "mt_segs_bad": [
        "Seg 1 ({tgt_lang})", 
        "Error Seg 2", 
        "Seg 3 ({tgt_lang})"
    ],
    "mt_full_good": "Identical to ref_full",
    "mt_full_bad": "Joined string of mt_segs_bad"
  }},
  ...
}}
```

Annotation Rules (Strict Adherence):
1. Target: ONLY mark all the errors found in mt_full_bad. Use ref_full (mt_full_good) as the gold standard for correctness.
2. Span Granularity: Do not mark character-level differences (like distinct punctuation styles unless they change meaning). Use syntactic units (words, phrases, or clauses) as the minimum span. Mark all differences found in mt_full_bad compared to mt_full_good
\t- Bad: {{\"text\": \"not\"}} (if the error is the whole phrase "did not go")
\t- Good: {{\"text\": \"did not go\"}}
3. Contiguous Spans: Spans must be exact, contiguous substrings of mt_full_bad.
4. Indices: Provide 0-based start index and end index (end is exclusive, standard Python slicing). mt_full_bad[start:end] must exactly equal the text field.
5. No Overlap: Error spans within a single case must not overlap.
6. Ordering: Output spans in ascending order of their start index.
7. Omissions: If text is missing, mark the smallest clause or phrase in mt_full_bad where the missing information should have been or which is grammatically affected. If the whole sentence is missing context, mark the relevant segment that failed to convey the meaning.
8. Error Types: Assign one of the following types to each span:
\t- semantic: Meaning is changed, wrong word choice, or logic error.
\t- reversal: Specific semantic error where meaning is inverted (e.g., "accepted" vs "rejected").
\t- hallucination: Information that are confusing compared to the source.
\t- omission: Key information missing.
\t- factual: Numerical or entity error (e.g., "1999" vs "2022").
\t- grammar: Severe grammatical error that hinders understanding (not just stylistic).
\t- other: other situations that are not listed here.

Output Format: Output ONLY a valid JSON list containing the annotated error spans. Do not include markdown formatting or explanations.
```json
{{
  "Case1": {{
    "error_spans_full_bad": [
      {{
        "start": {ex1_start},
        "end": {ex1_end},
        "text": "{ex1_text}",
        "error_type": "semantic"
      }},
      {{
        "start": {ex2_start},
        "end": {ex2_end},
        "text": "{ex2_text}",
        "error_type": "reversal"
      }},
      ...
    ]
  }},
  "Case2": {{
  ...
  }},
  ...
}}
```

Action:
Give out the JSON list now, identifying error spans from {file_name} following these rules. Start numbering from Case1.
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


def build_example_spans(tgt_code: str):
    text1, text2 = EXAMPLE_TEXTS.get(tgt_code, EXAMPLE_TEXTS["en"])
    start1 = 10
    gap = 9
    end1 = start1 + len(text1)
    start2 = end1 + gap
    end2 = start2 + len(text2)
    return {
        "ex1_start": start1,
        "ex1_end": end1,
        "ex1_text": text1,
        "ex2_start": start2,
        "ex2_end": end2,
        "ex2_text": text2,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an error span annotation prompt for a samples_clean file."
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="File name or path under samples_clean (e.g., wmt23_zh-en.json).",
    )
    parser.add_argument(
        "--lang-pair",
        default=None,
        help="Language pair like zh-en. If omitted, inferred from file name.",
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
    spans = build_example_spans((tgt_code or "").lower())

    prompt = PROMPT_TEMPLATE.format(
        file_name=file_name,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        **spans,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(prompt, encoding="utf-8")
    else:
        print(prompt)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
