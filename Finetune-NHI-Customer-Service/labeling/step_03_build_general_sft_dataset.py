#!/usr/bin/env python3
"""Build general-question SFT CSV from prompt, reference, user input, and answer columns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import read_csv, read_prompt, require_columns


BUILD_DIR = Path("training_data/build/20260518")
CONFIG_DIR = Path("training_data/config")
FINAL_DIR = Path("training_data/20260518")
DEFAULT_INPUT = BUILD_DIR / "general_data_260518.csv"
DEFAULT_PROMPT = CONFIG_DIR / "prompt.txt"
DEFAULT_OUTPUT = FINAL_DIR / "gemma3_sft_general_260518.csv"


def build_system_prompt(prompt_template: str, reference: str) -> str:
    full_prompt = prompt_template.replace("{model_contexts}", reference or "")
    system_part, _ = full_prompt.split("[USER_INPUT]", 1)
    system_part = system_part.replace("[SYSTEM_PROMPT]", "", 1).rstrip("=\n ")
    return system_part.strip()


def build_content(
    prompt_template: str,
    user_input: str,
    reference: str,
    answer: str,
) -> str:
    conversation = [
        {
            "role": "system",
            "content": build_system_prompt(prompt_template, reference),
        },
        {
            "role": "user",
            "content": user_input.strip(),
        },
        {
            "role": "assistant",
            "content": answer.strip(),
        },
    ]
    return json.dumps(conversation, ensure_ascii=False)


def build_dataset(
    input_path: Path,
    prompt_path: Path,
    output_path: Path,
    user_column: str,
    reference_column: str,
    answer_column: str,
    id_prefix: str,
    limit: int,
    keep_source_columns: bool,
    allow_empty_answer: bool,
) -> None:
    prompt_template = read_prompt(prompt_path)

    source_fieldnames, rows = read_csv(input_path)
    require_columns(input_path, source_fieldnames, [user_column, reference_column, answer_column])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "content"]
    if keep_source_columns:
        for column in (user_column, reference_column, answer_column):
            if column not in fieldnames:
                fieldnames.append(column)

    written = 0
    skipped_empty_answer = 0
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row_index, row in enumerate(rows, start=1):
            if limit and written >= limit:
                break

            user_input = row.get(user_column, "") or ""
            reference = row.get(reference_column, "") or ""
            answer = row.get(answer_column, "") or ""

            if not answer.strip() and not allow_empty_answer:
                skipped_empty_answer += 1
                continue

            output_row = {
                "id": f"{id_prefix}-{row_index}",
                "content": build_content(
                    prompt_template=prompt_template,
                    user_input=user_input,
                    reference=reference,
                    answer=answer,
                ),
            }
            if keep_source_columns:
                output_row[user_column] = user_input
                output_row[reference_column] = reference
                output_row[answer_column] = answer

            writer.writerow(output_row)
            written += 1

    print(f"input: {input_path}")
    print(f"output: {output_path}")
    print(f"written: {written}")
    print(f"skipped_empty_answer: {skipped_empty_answer}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert generated general QA data into SFT content CSV."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--user-column", default="user_input")
    parser.add_argument("--reference-column", default="reference")
    parser.add_argument("--answer-column", default="gpt-5.5")
    parser.add_argument("--id-prefix", default="general-260518")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--keep-source-columns", action="store_true")
    parser.add_argument("--allow-empty-answer", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(
        input_path=args.input,
        prompt_path=args.prompt,
        output_path=args.output,
        user_column=args.user_column,
        reference_column=args.reference_column,
        answer_column=args.answer_column,
        id_prefix=args.id_prefix,
        limit=args.limit,
        keep_source_columns=args.keep_source_columns,
        allow_empty_answer=args.allow_empty_answer,
    )


if __name__ == "__main__":
    main()
