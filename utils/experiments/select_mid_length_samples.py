#!/usr/bin/env python3
import argparse
import json
import random
import sys
from pathlib import Path

# python /ltstorage/home/4xin/SimultaneousTranslation/utils/experiments/select_mid_length_samples.py --num 103 --datasets wmt24_en-zh


def load_jsonl(path: Path, src_key: str):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} line {line_no}: {exc}") from exc
            if src_key not in obj:
                raise KeyError(f"{path} line {line_no}: missing key '{src_key}'")
            text = obj[src_key]
            if not isinstance(text, str):
                text = str(text)
            records.append((len(text), obj))
    return records


def select_records(records, min_percent: float, max_percent: float, num: int, rng, pair_mode: bool):
    if not records:
        return []
    records.sort(key=lambda item: item[0])
    total = len(records)
    start = int(total * min_percent)
    end = int(total * max_percent)
    if end <= start:
        start = 0
        end = total

    eligible = records[start:end]
    if not pair_mode:
        if num > len(eligible):
            raise ValueError(
                f"Requested {num} samples, but only {len(eligible)} records "
                f"available in the {min_percent:.0%}-{max_percent:.0%} range."
            )

        indices = sorted(rng.sample(range(len(eligible)), k=num))
        return [eligible[i][1] for i in indices]

    if num % 2 != 0:
        raise ValueError(
            "For commonmt datasets, --num must be even to keep adjacent index pairs."
        )

    eligible_by_index = {}
    for _, obj in eligible:
        if "index" not in obj:
            raise KeyError("Missing key 'index' required for commonmt pairing.")
        idx_value = obj["index"]
        try:
            idx_int = int(idx_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid index value {idx_value!r} for commonmt pairing."
            ) from exc
        eligible_by_index[idx_int] = obj

    pairs = []
    for idx in sorted(eligible_by_index):
        if idx % 2 == 0 and (idx + 1) in eligible_by_index:
            pairs.append((idx, eligible_by_index[idx], eligible_by_index[idx + 1]))

    num_pairs = num // 2
    if num_pairs > len(pairs):
        raise ValueError(
            f"Requested {num} samples ({num_pairs} pairs), but only {len(pairs)} pairs "
            f"available in the {min_percent:.0%}-{max_percent:.0%} range."
        )

    selected_pairs = rng.sample(pairs, k=num_pairs)
    selected_pairs.sort(key=lambda item: item[0])
    selected = []
    for _, even_obj, odd_obj in selected_pairs:
        selected.append(even_obj)
        selected.append(odd_obj)
    return selected


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for obj in records:
            handle.write(json.dumps(obj, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sort by source length, sample from the 10%-90% range, "
            "and write one file per dataset."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/ltstorage/home/4xin/SimultaneousTranslation/data/test/used"),
        help="Directory containing .jsonl datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/ltstorage/home/4xin/SimultaneousTranslation/experiments/xcomet_3.0/cases/used"
        ),
        help="Directory to write sampled datasets.",
    )
    parser.add_argument(
        "--num",
        type=int,
        required=True,
        help=(
            "Number of samples to draw per dataset (commonmt requires an even number; "
            "odd values are rounded down for commonmt datasets)."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help=(
            "Dataset file names to process (with or without .jsonl). "
            "If omitted, process all .jsonl files. "
            "Supports comma-separated lists."
        ),
    )
    parser.add_argument(
        "--src-key",
        default="src_text",
        help="JSON key holding the source text.",
    )
    parser.add_argument(
        "--min-percent",
        type=float,
        default=0.05,
        help="Lower percentile bound (0.0-1.0).",
    )
    parser.add_argument(
        "--max-percent",
        type=float,
        default=0.95,
        help="Upper percentile bound (0.0-1.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling.",
    )
    return parser.parse_args()


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


def adjust_num_for_dataset(num: int, pair_mode: bool, path: Path) -> int:
    if not pair_mode or num % 2 == 0:
        return num
    adjusted = num - 1
    if adjusted <= 0:
        raise ValueError(
            f"{path.name}: --num must be >= 2 for commonmt datasets to keep pairs."
        )
    print(
        f"{path.name}: commonmt requires even --num; using {adjusted} instead of {num}.",
        file=sys.stderr,
    )
    return adjusted


def main() -> int:
    args = parse_args()
    if not args.input_dir.exists() or not args.input_dir.is_dir():
        print(f"Input directory not found: {args.input_dir}", file=sys.stderr)
        return 1
    if not (0.0 <= args.min_percent < args.max_percent <= 1.0):
        print("Percent bounds must satisfy 0 <= min < max <= 1.", file=sys.stderr)
        return 1
    if args.num <= 0:
        print("--num must be positive.", file=sys.stderr)
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

    rng = random.Random(args.seed)
    for path in jsonl_files:
        records = load_jsonl(path, args.src_key)
        pair_mode = path.name.startswith("commonmt")
        effective_num = adjust_num_for_dataset(args.num, pair_mode, path)
        selected = select_records(
            records, args.min_percent, args.max_percent, effective_num, rng, pair_mode
        )
        out_path = args.output_dir / path.name
        write_jsonl(out_path, selected)
        if pair_mode:
            print(f"Wrote {len(selected)} records ({len(selected)//2} pairs) to {out_path}")
        else:
            print(f"Wrote {len(selected)} records to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
