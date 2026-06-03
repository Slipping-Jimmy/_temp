#!/usr/bin/env python3
"""Merge multiple SFT content CSVs into one input for expansion or training."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import read_csv, require_columns, write_csv


BUILD_DIR = Path("training_data/build/20260518")
DEFAULT_OUTPUT = BUILD_DIR / "gemma3_sft_key_qa_plus_hard_dependents.csv"


def merge_sft_csvs(input_paths: list[Path], output_path: Path, dedupe_content: bool) -> None:
    if not input_paths:
        raise ValueError("At least one --input path is required")

    fieldnames: list[str] = []
    merged_rows: list[dict[str, str]] = []
    seen_content: set[str] = set()
    seen_ids: set[str] = set()

    for input_path in input_paths:
        source_fieldnames, rows = read_csv(input_path)
        require_columns(input_path, source_fieldnames, ["id", "content"])

        for name in source_fieldnames:
            if name not in fieldnames:
                fieldnames.append(name)

        for row in rows:
            content = row.get("content", "")
            if dedupe_content and content in seen_content:
                continue
            seen_content.add(content)

            row = dict(row)
            row_id = row.get("id", "").strip()
            if row_id in seen_ids:
                row["id"] = f"{input_path.stem}-{len(merged_rows) + 1}"
            seen_ids.add(row["id"])
            merged_rows.append(row)

    write_csv(output_path, fieldnames, merged_rows, extrasaction="ignore")
    print(f"output: {output_path}")
    print(f"inputs: {len(input_paths)}")
    print(f"rows: {len(merged_rows)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge SFT content CSV files.")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-dedupe-content", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_sft_csvs(
        input_paths=args.input,
        output_path=args.output,
        dedupe_content=not args.no_dedupe_content,
    )


if __name__ == "__main__":
    main()
