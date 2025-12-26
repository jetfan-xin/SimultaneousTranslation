#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path


def add_index_to_file(path: Path, start: int, dry_run: bool, keep_existing: bool) -> int:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    count = 0

    with path.open("r", encoding="utf-8") as src, tmp_path.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, start=1):
            if not line.strip():
                dst.write(line)
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} line {line_no}: {exc}") from exc

            if not (keep_existing and "index" in obj):
                obj["index"] = start + count

            dst.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1

    if dry_run:
        tmp_path.unlink(missing_ok=True)
    else:
        os.replace(tmp_path, path)

    return count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add an index field to each JSON object in .jsonl datasets."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/ltstorage/home/4xin/SimultaneousTranslation/data/test/used"),
        help="Directory containing .jsonl datasets.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting index value for each file.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not overwrite an existing index field.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process files but do not modify them.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir.exists() or not data_dir.is_dir():
        print(f"Directory not found: {data_dir}", file=sys.stderr)
        return 1

    jsonl_files = sorted(p for p in data_dir.iterdir() if p.suffix == ".jsonl")
    if not jsonl_files:
        print(f"No .jsonl files found in {data_dir}", file=sys.stderr)
        return 1

    for path in jsonl_files:
        count = add_index_to_file(path, args.start, args.dry_run, args.keep_existing)
        action = "would update" if args.dry_run else "updated"
        print(f"{action} {path.name}: {count} records")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
