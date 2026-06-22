#!/usr/bin/env python3
"""Add NHI smart chat reference contexts to a CSV."""

from __future__ import annotations

import argparse
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from common import load_env, read_csv, require_columns, write_csv
from nhi_reference_api import fetch_chat, format_information


BUILD_DIR = Path("training_data/build/20260518")
SOURCE_DIR = Path("training_data/sources/20260518")
CONFIG_DIR = Path("training_data/config")
DEFAULT_ENV = CONFIG_DIR / ".env"

PRESETS = {
    "general": {
        "input": BUILD_DIR / "general_data_260518.csv",
        "output": BUILD_DIR / "general_data_260518.csv",
        "answer_column": None,
        "workers": 6,
        "checkpoint_every": 25,
        "sleep_seconds": 0.0,
    },
    "key-qa": {
        "input": SOURCE_DIR / "key_qa_raw.csv",
        "output": BUILD_DIR / "key_qa_with_reference.csv",
        "answer_column": "model_output",
        "workers": 1,
        "checkpoint_every": 1,
        "sleep_seconds": 0.2,
    },
}


def merge_existing(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    output_path: Path,
    key_column: str,
) -> list[str]:
    if not output_path.exists():
        return fieldnames

    output_fieldnames, existing_rows = read_csv(output_path)
    if key_column not in output_fieldnames:
        return fieldnames

    for name in output_fieldnames:
        if name not in fieldnames:
            fieldnames.append(name)

    existing_by_key = {
        (row.get(key_column, "") or "").strip(): row
        for row in existing_rows
        if (row.get(key_column, "") or "").strip()
    }
    for row in rows:
        key = (row.get(key_column, "") or "").strip()
        existing = existing_by_key.get(key)
        if not existing:
            continue
        for name in output_fieldnames:
            if existing.get(name, "") and not row.get(name, ""):
                row[name] = existing.get(name, "")

    return fieldnames


def build_reference_cache(
    rows: list[dict[str, str]],
    user_column: str,
    reference_column: str,
) -> dict[str, str]:
    cache: dict[str, str] = {}
    for row in rows:
        question = (row.get(user_column, "") or "").strip()
        reference = row.get(reference_column, "")
        if question and reference:
            cache[question] = reference
    return cache


def fetch_reference(question: str, retries: int, retry_sleep: float) -> str:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            _, _, _, contexts = fetch_chat(question)
            return format_information(contexts).strip()
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(retry_sleep * attempt)
    raise RuntimeError(f"failed after {retries} attempts: {last_error}")


def add_references(
    input_path: Path,
    output_path: Path,
    user_column: str,
    reference_column: str,
    answer_column: str | None,
    checkpoint_every: int,
    sleep_seconds: float,
    retries: int,
    retry_sleep: float,
    workers: int,
    max_fetch: int,
    overwrite: bool,
    env_path: Path,
) -> None:
    load_env(env_path)

    fieldnames, rows = read_csv(input_path)
    required = [user_column]
    if answer_column:
        required.append(answer_column)
    require_columns(input_path, fieldnames, required)

    if reference_column not in fieldnames:
        fieldnames.append(reference_column)
        for row in rows:
            row[reference_column] = ""

    fieldnames = merge_existing(rows, fieldnames, output_path, user_column)
    cache = build_reference_cache(rows, user_column, reference_column)

    total = len(rows)
    print(f"Input rows: {total}")
    print(f"Cached references: {len(cache)}")

    for row in rows:
        question = (row.get(user_column, "") or "").strip()
        if question in cache and (overwrite or not row.get(reference_column, "").strip()):
            row[reference_column] = cache[question]

    pending: list[tuple[int, str]] = []
    scheduled_questions: set[str] = set()
    for idx, row in enumerate(rows):
        question = (row.get(user_column, "") or "").strip()
        if not question or question in scheduled_questions:
            continue
        if row.get(reference_column, "").strip() and not overwrite:
            continue
        scheduled_questions.add(question)
        pending.append((idx, question))
        if max_fetch and len(pending) >= max_fetch:
            break

    completed = sum(1 for row in rows if row.get(reference_column, "").strip())
    print(f"Completed rows before fetch: {completed}/{total}")
    print(f"Unique questions to fetch: {len(pending)}")

    since_checkpoint = 0
    next_pending = 0
    in_flight = {}
    failed: list[tuple[int, str, str]] = []

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        while next_pending < len(pending) or in_flight:
            while next_pending < len(pending) and len(in_flight) < max(1, workers):
                idx, question = pending[next_pending]
                next_pending += 1
                print(f"[{idx + 1}/{total}] Fetching reference: {question[:80]}", flush=True)
                future = executor.submit(fetch_reference, question, retries, retry_sleep)
                in_flight[future] = (idx, question)
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)

            done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                idx, question = in_flight.pop(future)
                try:
                    reference = future.result()
                except Exception as exc:
                    failed.append((idx + 1, question, str(exc)))
                    print(f"[{idx + 1}/{total}] FAILED: {exc}", flush=True)
                    continue

                cache[question] = reference
                for row in rows:
                    if (row.get(user_column, "") or "").strip() == question:
                        row[reference_column] = reference
                        completed += 1
                since_checkpoint += 1

                if since_checkpoint >= checkpoint_every:
                    write_csv(output_path, fieldnames, rows, extrasaction="ignore")
                    print(f"Checkpoint saved: {completed}/{total}", flush=True)
                    since_checkpoint = 0

    write_csv(output_path, fieldnames, rows, extrasaction="ignore")
    print(f"input: {input_path}")
    print(f"output: {output_path}")
    print(f"completed rows: {sum(1 for row in rows if row.get(reference_column, '').strip())}/{total}")
    if failed:
        print(f"failed unique questions: {len(failed)}")
        for line_number, question, error in failed[:10]:
            print(f"  [{line_number}] {question[:60]} | {error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch NHI references for a CSV.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="general")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--user-column", default="user_input")
    parser.add_argument("--reference-column", default="reference")
    parser.add_argument("--answer-column", default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=None)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--max-fetch", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset = PRESETS[args.preset]
    add_references(
        input_path=args.input or preset["input"],
        output_path=args.output or preset["output"],
        user_column=args.user_column,
        reference_column=args.reference_column,
        answer_column=args.answer_column if args.answer_column is not None else preset["answer_column"],
        checkpoint_every=args.checkpoint_every or preset["checkpoint_every"],
        sleep_seconds=args.sleep_seconds if args.sleep_seconds is not None else preset["sleep_seconds"],
        retries=args.retries,
        retry_sleep=args.retry_sleep,
        workers=args.workers or preset["workers"],
        max_fetch=args.max_fetch,
        overwrite=args.overwrite,
        env_path=args.env,
    )


if __name__ == "__main__":
    main()
