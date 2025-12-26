#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

'''
python /ltstorage/home/4xin/SimultaneousTranslation/utils/experiments/split_sampled_jsonl.py --chunk-size 10 --datasets wmt24_en-zh
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


def iter_nonempty_lines(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield line


def split_jsonl(path: Path, output_dir: Path, chunk_size: int) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    part_idx = 0
    record_idx = 0
    out_handle = None

    def open_next():
        nonlocal part_idx, out_handle
        if out_handle:
            out_handle.close()
        part_idx += 1
        out_path = output_dir / f"{path.stem}.part{part_idx:03d}{path.suffix}"
        out_handle = out_path.open("w", encoding="utf-8")
        return out_path

    out_path = None
    for line in iter_nonempty_lines(path):
        if record_idx % chunk_size == 0:
            out_path = open_next()
        out_handle.write(line if line.endswith("\n") else f"{line}\n")
        record_idx += 1

    if out_handle:
        out_handle.close()

    if record_idx == 0:
        if out_path and out_path.exists():
            out_path.unlink()
        return 0

    return part_idx


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split sampled JSONL datasets into fixed-size chunks."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "/ltstorage/home/4xin/SimultaneousTranslation/experiments/xcomet_3.0/cases/used"
        ),
        help="Directory containing sampled .jsonl datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/ltstorage/home/4xin/SimultaneousTranslation/experiments/xcomet_3.0/cases/used_split"
        ),
        help="Directory to write split .jsonl files.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        required=True,
        help="Number of records per split file.",
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input_dir.exists() or not args.input_dir.is_dir():
        print(f"Input directory not found: {args.input_dir}", file=sys.stderr)
        return 1
    if args.chunk_size <= 0:
        print("--chunk-size must be positive.", file=sys.stderr)
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
        parts = split_jsonl(path, args.output_dir, args.chunk_size)
        if parts == 0:
            print(f"{path.name}: no records to split")
        else:
            print(f"{path.name}: wrote {parts} file(s) to {args.output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
