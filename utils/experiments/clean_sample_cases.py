#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

'''
python /ltstorage/home/4xin/SimultaneousTranslation/utils/experiments/clean_sample_cases.py --datasets culturemt_en-hi.json
'''
RECORD_KEYS = (
    "label",
    "source_index",
    "src_full",
    "ref_full",
    "mt_segs_good",
    "mt_segs_bad",
    "mt_full_good",
    "mt_full_bad",
)


def normalize_dataset_names(raw_names):
    names = []
    for raw in raw_names:
        for part in raw.split(","):
            name = part.strip()
            if not name:
                continue
            if not name.endswith(".json"):
                name = f"{name}.json"
            names.append(name)
    return names


def load_index_list(dataset_path: Path):
    indexes = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{dataset_path} line {line_no}: {exc}") from exc
            if "index" not in obj:
                raise KeyError(f"{dataset_path} line {line_no}: missing key 'index'")
            idx_value = obj["index"]
            try:
                idx_value = int(idx_value)
            except (TypeError, ValueError):
                pass
            indexes.append(idx_value)
    return indexes


def clean_case(case, assigned_index, dataset_name: str):
    for key in RECORD_KEYS:
        if key not in case:
            raise KeyError(f"{dataset_name}: case missing key '{key}'")

    cleaned = {key: case[key] for key in RECORD_KEYS}
    cleaned["source_index"] = assigned_index
    return cleaned


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Align sample case source_index with dataset index and keep only required fields."
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=Path(
            "/ltstorage/home/4xin/SimultaneousTranslation/experiments/xcomet_3.0/cases/samples"
        ),
        help="Directory containing sample JSON files.",
    )
    parser.add_argument(
        "--used-dir",
        type=Path,
        default=Path(
            "/ltstorage/home/4xin/SimultaneousTranslation/experiments/xcomet_3.0/cases/used"
        ),
        help="Directory containing source datasets in JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/ltstorage/home/4xin/SimultaneousTranslation/experiments/xcomet_3.0/cases/samples_clean"
        ),
        help="Directory to write cleaned sample JSON files.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help=(
            "Sample JSON file names to process (with or without .json). "
            "If omitted, process all .json files. "
            "Supports comma-separated lists."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.samples_dir.exists() or not args.samples_dir.is_dir():
        print(f"Samples directory not found: {args.samples_dir}", file=sys.stderr)
        return 1
    if not args.used_dir.exists() or not args.used_dir.is_dir():
        print(f"Used directory not found: {args.used_dir}", file=sys.stderr)
        return 1

    sample_files = sorted(p for p in args.samples_dir.iterdir() if p.suffix == ".json")
    if not sample_files:
        print(f"No .json files found in {args.samples_dir}", file=sys.stderr)
        return 1

    if args.datasets:
        wanted = set(normalize_dataset_names(args.datasets))
        if not wanted:
            print("No valid dataset names provided.", file=sys.stderr)
            return 1
        sample_files = [p for p in sample_files if p.name in wanted]
        missing = wanted - {p.name for p in sample_files}
        if missing:
            missing_list = ", ".join(sorted(missing))
            print(f"Dataset(s) not found: {missing_list}", file=sys.stderr)
            return 1

    for sample_path in sample_files:
        dataset_path = args.used_dir / f"{sample_path.stem}.jsonl"
        if not dataset_path.exists():
            print(
                f"Dataset not found for {sample_path.name}: {dataset_path}",
                file=sys.stderr,
            )
            return 1
        index_list = load_index_list(dataset_path)
        data = load_json(sample_path)
        if not isinstance(data, dict):
            print(f"{sample_path.name}: expected a JSON object at top level.", file=sys.stderr)
            return 1

        case_items = [(key, value) for key, value in data.items() if isinstance(value, dict) and "src_full" in value]
        if len(index_list) != len(case_items):
            print(
                f"{sample_path.name}: case count {len(case_items)} does not match "
                f"index count {len(index_list)} in {dataset_path.name}.",
                file=sys.stderr,
            )
            return 1

        cleaned = {}
        index_iter = iter(index_list)
        for key, value in data.items():
            if isinstance(value, dict) and "src_full" in value:
                assigned_index = next(index_iter)
                cleaned[key] = clean_case(value, assigned_index, sample_path.name)
            else:
                cleaned[key] = value

        out_path = args.output_dir / sample_path.name
        write_json(out_path, cleaned)
        print(f"Wrote cleaned file to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
