#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

'''
python /ltstorage/home/4xin/SimultaneousTranslation/utils/experiments/jsonl_to_json.py --datasets wmt24_en-zh.part001.jsonl,wmt24_en-zh.part002.jsonl,wmt24_en-zh.part003.jsonl,wmt24_en-zh.part004.jsonl,wmt24_en-zh.part005.jsonl,wmt24_en-zh.part006.jsonl,wmt24_en-zh.part007.jsonl,wmt24_en-zh.part008.jsonl,wmt24_en-zh.part009.jsonl,wmt24_en-zh.part010.jsonl
'''

def normalize_dataset_names(raw_names):
    names = []
    for raw in raw_names:
        for part in raw.split(","):
            name = part.strip()
            if not name:
                continue
            if not name.endswith(".jsonl"):
                name = f"{name}.jsonl"
            names.append(name)
    return names


def load_jsonl(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} line {line_no}: {exc}") from exc
    return records


def write_json(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert JSONL files into valid JSON arrays."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "/ltstorage/home/4xin/SimultaneousTranslation/experiments/xcomet_3.0/cases/used_split"
        ),
        help="Directory containing .jsonl files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/ltstorage/home/4xin/SimultaneousTranslation/experiments/xcomet_3.0/cases/used_split_json"
        ),
        help="Directory to write .json files.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help=(
            "JSONL file names to process (with or without .jsonl). "
            "If omitted, process all .jsonl files. "
            "Supports comma-separated lists."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input_dir.exists() or not args.input_dir.is_dir():
        print(f"Input directory not found: {args.input_dir}", file=sys.stderr)
        return 1

    jsonl_files = sorted(p for p in args.input_dir.iterdir() if p.suffix == ".jsonl")
    if not jsonl_files:
        print(f"No .jsonl files found in {args.input_dir}", file=sys.stderr)
        return 1

    if args.datasets:
        wanted = set(normalize_dataset_names(args.datasets))
        if not wanted:
            print("No valid dataset names provided.", file=sys.stderr)
            return 1
        jsonl_files = [p for p in jsonl_files if p.name in wanted]
        missing = wanted - {p.name for p in jsonl_files}
        if missing:
            missing_list = ", ".join(sorted(missing))
            print(f"Dataset(s) not found: {missing_list}", file=sys.stderr)
            return 1

    for path in jsonl_files:
        records = load_jsonl(path)
        out_path = args.output_dir / f"{path.stem}.json"
        write_json(out_path, records)
        print(f"Wrote {len(records)} records to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
