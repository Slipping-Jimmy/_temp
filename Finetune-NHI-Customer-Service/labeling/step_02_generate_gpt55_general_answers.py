#!/usr/bin/env python3
"""Generate GPT-5.5 answers for general_data_260518.csv."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from openai import OpenAI, OpenAIError

from common import load_env, read_prompt, response_text, write_csv

BUILD_DIR = Path("training_data/build/20260518")
CONFIG_DIR = Path("training_data/config")
DEFAULT_DATA = BUILD_DIR / "general_data_260518.csv"
DEFAULT_PROMPT = CONFIG_DIR / "prompt.txt"
DEFAULT_ENV = CONFIG_DIR / ".env"
DEFAULT_BATCH_JSONL = BUILD_DIR / "gpt55_batch_requests.jsonl"
DEFAULT_BATCH_META = BUILD_DIR / "gpt55_batch_meta.json"
DEFAULT_BATCH_OUTPUT = BUILD_DIR / "gpt55_batch_output.jsonl"
DEFAULT_BATCH_ERRORS = BUILD_DIR / "gpt55_batch_errors.jsonl"
OUTPUT_COLUMN = "gpt-5.5"


def build_messages(prompt_template: str, user_input: str, reference: str) -> list[dict[str, str]]:
    full_prompt = prompt_template.replace("{model_contexts}", reference or "")
    system_part, _ = full_prompt.split("[USER_INPUT]", 1)
    system_part = system_part.replace("[SYSTEM_PROMPT]", "", 1).rstrip("=\n ")
    return [
        {"role": "system", "content": system_part.strip()},
        {"role": "user", "content": user_input.strip()},
    ]


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    write_csv(path, fieldnames, rows)


def load_rows(data_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with data_path.open("r", newline="", encoding="utf-8-sig") as f:
        import csv

        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No CSV header found: {data_path}")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    for required in ("user_input", "reference"):
        if required not in fieldnames:
            raise ValueError(f"{data_path} must contain {required}")

    if OUTPUT_COLUMN not in fieldnames:
        fieldnames.append(OUTPUT_COLUMN)
        for row in rows:
            row[OUTPUT_COLUMN] = ""
        write_rows(data_path, fieldnames, rows)

    return fieldnames, rows


def pending_row_indexes(rows: list[dict[str, str]], start: int, limit: int) -> list[int]:
    indexes: list[int] = []
    for row_index, row in enumerate(rows, start=1):
        if row_index < start:
            continue
        if row.get(OUTPUT_COLUMN, "").strip():
            continue
        indexes.append(row_index)
        if limit and len(indexes) >= limit:
            break
    return indexes


def response_body_text(body: dict[str, object]) -> str:
    output_text = body.get("output_text")
    if output_text:
        return str(output_text).strip()

    chunks: list[str] = []
    for item in body.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                chunks.append(str(content["text"]))
    if chunks:
        return "".join(chunks).strip()

    choices = body.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict) and message.get("content"):
                return str(message["content"]).strip()

    return json.dumps(body, ensure_ascii=False)


def read_batch_id(meta_path: Path, batch_id: str | None) -> str:
    if batch_id:
        return batch_id
    if not meta_path.exists():
        raise RuntimeError(f"Batch id is not set and {meta_path} does not exist")
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    loaded = data.get("batch_id")
    if not loaded:
        raise RuntimeError(f"{meta_path} does not contain batch_id")
    return str(loaded)


def binary_response_bytes(response: object) -> bytes:
    if hasattr(response, "read"):
        data = response.read()
        if isinstance(data, str):
            return data.encode("utf-8")
        return bytes(data)
    if hasattr(response, "content"):
        content = response.content
        if isinstance(content, str):
            return content.encode("utf-8")
        return bytes(content)
    if hasattr(response, "text"):
        return str(response.text).encode("utf-8")
    return bytes(response)


def generate(
    data_path: Path,
    prompt_path: Path,
    env_path: Path,
    model: str,
    limit: int,
    start: int,
    overwrite: bool,
    checkpoint_every: int,
    sleep_seconds: float,
    retries: int,
    retry_sleep: float,
) -> None:
    load_env(env_path)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(f"OPENAI_API_KEY is not set. Check {env_path}")

    prompt_template = read_prompt(prompt_path)
    client = OpenAI()

    fieldnames, rows = load_rows(data_path)

    generated = 0
    checked = 0
    for row_index, row in enumerate(rows, start=1):
        if row_index < start:
            continue
        if limit and checked >= limit:
            break
        if row.get(OUTPUT_COLUMN, "").strip() and not overwrite:
            continue

        checked += 1
        user_input = row.get("user_input", "")
        reference = row.get("reference", "")
        messages = build_messages(prompt_template, user_input, reference)
        print(f"[{row_index}] generating: {user_input[:80]}", flush=True)

        response = None
        for attempt in range(1, retries + 1):
            try:
                response = client.responses.create(
                    model=model,
                    input=messages,
                )
                break
            except OpenAIError as exc:
                if attempt >= retries:
                    write_rows(data_path, fieldnames, rows)
                    print(f"OpenAI API error at row {row_index}: {exc}", flush=True)
                    return
                print(
                    f"OpenAI API error at row {row_index}, "
                    f"retry {attempt}/{retries}: {exc}",
                    flush=True,
                )
                time.sleep(retry_sleep * attempt)

        if response is None:
            write_rows(data_path, fieldnames, rows)
            print(f"No response returned at row {row_index}", flush=True)
            return

        row[OUTPUT_COLUMN] = response_text(response)
        generated += 1

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        if generated % checkpoint_every == 0:
            write_rows(data_path, fieldnames, rows)
            print(f"checkpoint saved: {generated}", flush=True)

    write_rows(data_path, fieldnames, rows)
    print(f"saved: {data_path}")
    print(f"generated: {generated}")


def submit_batch(
    data_path: Path,
    prompt_path: Path,
    env_path: Path,
    model: str,
    limit: int,
    start: int,
    batch_jsonl_path: Path,
    batch_meta_path: Path,
    max_output_tokens: int | None,
) -> None:
    load_env(env_path)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(f"OPENAI_API_KEY is not set. Check {env_path}")

    prompt_template = read_prompt(prompt_path)
    _, rows = load_rows(data_path)
    indexes = pending_row_indexes(rows, start=start, limit=limit)
    if not indexes:
        print("no pending rows")
        return

    batch_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with batch_jsonl_path.open("w", encoding="utf-8") as f:
        for row_index in indexes:
            row = rows[row_index - 1]
            body: dict[str, object] = {
                "model": model,
                "input": build_messages(
                    prompt_template,
                    row.get("user_input", ""),
                    row.get("reference", ""),
                ),
            }
            if max_output_tokens:
                body["max_output_tokens"] = max_output_tokens
            request = {
                "custom_id": f"row-{row_index}",
                "method": "POST",
                "url": "/v1/responses",
                "body": body,
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")

    client = OpenAI()
    with batch_jsonl_path.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={
            "data_path": str(data_path),
            "output_column": OUTPUT_COLUMN,
            "model": model,
        },
    )

    meta = {
        "batch_id": batch.id,
        "input_file_id": uploaded.id,
        "data_path": str(data_path),
        "prompt_path": str(prompt_path),
        "model": model,
        "output_column": OUTPUT_COLUMN,
        "request_count": len(indexes),
        "first_row": indexes[0],
        "last_row": indexes[-1],
        "batch_jsonl_path": str(batch_jsonl_path),
        "status": batch.status,
    }
    batch_meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"batch_id: {batch.id}")
    print(f"input_file_id: {uploaded.id}")
    print(f"status: {batch.status}")
    print(f"requests: {len(indexes)} rows {indexes[0]}-{indexes[-1]}")
    print(f"meta saved: {batch_meta_path}")


def status_batch(env_path: Path, batch_meta_path: Path, batch_id: str | None) -> None:
    load_env(env_path)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(f"OPENAI_API_KEY is not set. Check {env_path}")
    client = OpenAI()
    resolved_batch_id = read_batch_id(batch_meta_path, batch_id)
    batch = client.batches.retrieve(resolved_batch_id)
    dumped = batch.model_dump() if hasattr(batch, "model_dump") else dict(batch)
    print(json.dumps(dumped, ensure_ascii=False, indent=2, default=str))


def collect_batch(
    data_path: Path,
    env_path: Path,
    batch_meta_path: Path,
    batch_output_path: Path,
    batch_errors_path: Path,
    batch_id: str | None,
) -> None:
    load_env(env_path)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(f"OPENAI_API_KEY is not set. Check {env_path}")

    client = OpenAI()
    resolved_batch_id = read_batch_id(batch_meta_path, batch_id)
    batch = client.batches.retrieve(resolved_batch_id)
    if batch.status != "completed":
        print(f"batch status: {batch.status}")
        print("batch is not completed yet")
        return

    output_file_id = getattr(batch, "output_file_id", None)
    if not output_file_id:
        raise RuntimeError("completed batch has no output_file_id")

    output_response = client.files.content(output_file_id)
    output_bytes = binary_response_bytes(output_response)
    batch_output_path.write_bytes(output_bytes)

    error_file_id = getattr(batch, "error_file_id", None)
    if error_file_id:
        error_response = client.files.content(error_file_id)
        batch_errors_path.write_bytes(binary_response_bytes(error_response))

    fieldnames, rows = load_rows(data_path)
    applied = 0
    failed = 0
    for raw_line in output_bytes.decode("utf-8").splitlines():
        if not raw_line.strip():
            continue
        item = json.loads(raw_line)
        custom_id = str(item.get("custom_id", ""))
        if not custom_id.startswith("row-"):
            failed += 1
            continue
        row_index = int(custom_id.removeprefix("row-"))
        row = rows[row_index - 1]
        response = item.get("response") or {}
        error = item.get("error")
        if error or response.get("status_code") != 200:
            failed += 1
            continue
        body = response.get("body") or {}
        if not isinstance(body, dict):
            failed += 1
            continue
        text = response_body_text(body)
        if text:
            row[OUTPUT_COLUMN] = text
            applied += 1
        else:
            failed += 1

    write_rows(data_path, fieldnames, rows)
    print(f"output saved: {batch_output_path}")
    if error_file_id:
        print(f"errors saved: {batch_errors_path}")
    print(f"applied: {applied}")
    print(f"failed_or_skipped: {failed}")
    print(f"saved: {data_path}")


def watch_batch(
    data_path: Path,
    env_path: Path,
    batch_meta_path: Path,
    batch_output_path: Path,
    batch_errors_path: Path,
    batch_id: str | None,
    poll_seconds: float,
) -> None:
    load_env(env_path)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(f"OPENAI_API_KEY is not set. Check {env_path}")

    client = OpenAI()
    resolved_batch_id = read_batch_id(batch_meta_path, batch_id)
    terminal_statuses = {"completed", "failed", "cancelled", "expired"}
    while True:
        batch = client.batches.retrieve(resolved_batch_id)
        counts = getattr(batch, "request_counts", None)
        counts_text = ""
        if counts is not None:
            counts_dict = counts.model_dump() if hasattr(counts, "model_dump") else counts
            counts_text = f" counts={counts_dict}"
        print(f"batch {resolved_batch_id} status={batch.status}{counts_text}", flush=True)

        if batch.status == "completed":
            collect_batch(
                data_path=data_path,
                env_path=env_path,
                batch_meta_path=batch_meta_path,
                batch_output_path=batch_output_path,
                batch_errors_path=batch_errors_path,
                batch_id=resolved_batch_id,
            )
            return
        if batch.status in terminal_statuses:
            print(f"batch ended without completion: {batch.status}", flush=True)
            return
        time.sleep(poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill the gpt-5.5 column in general_data_260518.csv."
    )
    parser.add_argument(
        "--mode",
        choices=("sync", "submit-batch", "status-batch", "collect-batch", "watch-batch"),
        default="sync",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--limit", type=int, default=3, help="Number of rows to generate.")
    parser.add_argument("--start", type=int, default=1, help="1-based row number in the data.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=10.0)
    parser.add_argument("--batch-jsonl", type=Path, default=DEFAULT_BATCH_JSONL)
    parser.add_argument("--batch-meta", type=Path, default=DEFAULT_BATCH_META)
    parser.add_argument("--batch-output", type=Path, default=DEFAULT_BATCH_OUTPUT)
    parser.add_argument("--batch-errors", type=Path, default=DEFAULT_BATCH_ERRORS)
    parser.add_argument("--batch-id")
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "submit-batch":
        submit_batch(
            data_path=args.data,
            prompt_path=args.prompt,
            env_path=args.env,
            model=args.model,
            limit=args.limit,
            start=args.start,
            batch_jsonl_path=args.batch_jsonl,
            batch_meta_path=args.batch_meta,
            max_output_tokens=args.max_output_tokens,
        )
        return
    if args.mode == "status-batch":
        status_batch(
            env_path=args.env,
            batch_meta_path=args.batch_meta,
            batch_id=args.batch_id,
        )
        return
    if args.mode == "collect-batch":
        collect_batch(
            data_path=args.data,
            env_path=args.env,
            batch_meta_path=args.batch_meta,
            batch_output_path=args.batch_output,
            batch_errors_path=args.batch_errors,
            batch_id=args.batch_id,
        )
        return
    if args.mode == "watch-batch":
        watch_batch(
            data_path=args.data,
            env_path=args.env,
            batch_meta_path=args.batch_meta,
            batch_output_path=args.batch_output,
            batch_errors_path=args.batch_errors,
            batch_id=args.batch_id,
            poll_seconds=args.poll_seconds,
        )
        return

    generate(
        data_path=args.data,
        prompt_path=args.prompt,
        env_path=args.env,
        model=args.model,
        limit=args.limit,
        start=args.start,
        overwrite=args.overwrite,
        checkpoint_every=args.checkpoint_every,
        sleep_seconds=args.sleep_seconds,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
    )


if __name__ == "__main__":
    main()
