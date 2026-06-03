#!/usr/bin/env python3
"""Generate OpenAI paraphrases for key QA and build Gemma-3 SFT expansion CSV."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from openai import OpenAI, OpenAIError

from common import load_env, read_csv, response_text, write_csv


BUILD_DIR = Path("training_data/build/20260518")
CONFIG_DIR = Path("training_data/config")
FINAL_DIR = Path("training_data/20260518")
DEFAULT_ENV = CONFIG_DIR / ".env"
DEFAULT_INPUT_CANDIDATES = [
    BUILD_DIR / "gemma3_sft_key_qa.csv",
]
DEFAULT_PARAPHRASE_CACHE = BUILD_DIR / "key_qa_openai_paraphrases.csv"
DEFAULT_OUTPUT = FINAL_DIR / "gemma3_sft_key_qa_expansion.csv"


def default_input_path() -> Path:
    for path in DEFAULT_INPUT_CANDIDATES:
        if path.exists():
            return path
    return DEFAULT_INPUT_CANDIDATES[0]


def parse_content(content: str) -> list[dict[str, Any]]:
    conv = json.loads(content)
    if not isinstance(conv, list):
        raise ValueError("content must be a JSON conversation list")
    return conv


def role_name(msg: dict[str, Any]) -> str:
    role = str(msg.get("role", "")).strip().lower()
    if role == "model":
        return "assistant"
    return role


def first_user_index(conv: list[dict[str, Any]]) -> int:
    for index, msg in enumerate(conv):
        if isinstance(msg, dict) and role_name(msg) == "user":
            return index
    raise ValueError("content conversation has no user message")


def source_items(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    items = []
    for row_index, row in enumerate(rows, start=1):
        conv = parse_content(row["content"])
        user_index = first_user_index(conv)
        question = str(conv[user_index].get("content", "")).strip()
        if not question:
            raise ValueError(f"Empty user question at source row {row_index}")
        source_id = str(row.get("id") or f"key-qa-{row_index}")
        items.append(
            {
                "source_id": source_id,
                "source_row": row,
                "conversation": conv,
                "user_index": user_index,
                "question": question,
            }
        )
    return items


def extract_json_array(text: str) -> list[Any]:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.S)
    if fenced:
        stripped = fenced.group(1).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("[")
        end = stripped.rfind("]")
        if start < 0 or end <= start:
            raise
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, list):
        raise ValueError("OpenAI response is not a JSON array")
    return value


def normalize_variant(text: str) -> str:
    text = re.sub(r"^\s*[-*\d.、)）]+", "", str(text)).strip()
    return re.sub(r"\s+", " ", text)


def validate_variants(raw_variants: list[Any], original_question: str, count: int) -> list[str]:
    variants = []
    seen = {original_question.strip()}
    for raw_variant in raw_variants:
        variant = normalize_variant(raw_variant)
        if not variant or variant in seen:
            continue
        variants.append(variant)
        seen.add(variant)
        if len(variants) >= count:
            break
    if len(variants) < count:
        raise ValueError(f"Need {count} variants, got {len(variants)}")
    return variants


def build_prompt(question: str, count: int) -> list[dict[str, str]]:
    system = (
        "你是台灣繁體中文客服訓練資料增強助手。"
        "你的任務是改寫使用者問題，不能新增事實、條件、人物、金額、日期或答案。"
        "每個改寫都必須和原問題語意等價，且仍然是在詢問同一件健保業務。"
        "只輸出 JSON array of strings，不要 Markdown，不要解釋。"
    )
    user = (
        f"請將下列問題改寫成 {count} 個不同問法。\n"
        "規則：\n"
        "1. 使用繁體中文。\n"
        "2. 可改變語氣、口語程度、詞序、客服詢問方式。\n"
        "3. 不可以改變使用者身分、辦理事項、條件或答案範圍。\n"
        "4. 不要包含答案。\n"
        "5. 不要和原句完全相同。\n\n"
        f"原問題：{question}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def load_paraphrase_cache(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    _fieldnames, rows = read_csv(path)
    cache: dict[str, list[str]] = {}
    for row in rows:
        source_id = row.get("source_id", "").strip()
        paraphrase = row.get("paraphrase", "").strip()
        if source_id and paraphrase:
            cache.setdefault(source_id, []).append(paraphrase)
    return cache


def write_paraphrase_cache(path: Path, cache: dict[str, list[str]]) -> None:
    rows = []
    for source_id in sorted(cache):
        for index, paraphrase in enumerate(cache[source_id], start=1):
            rows.append(
                {
                    "source_id": source_id,
                    "variant_index": str(index),
                    "paraphrase": paraphrase,
                }
            )
    write_csv(path, ["source_id", "variant_index", "paraphrase"], rows)


def generate_variants(
    client: OpenAI,
    model: str,
    question: str,
    count: int,
    retries: int,
    retry_sleep: float,
) -> list[str]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = client.responses.create(
                model=model,
                input=build_prompt(question, count),
            )
            raw_variants = extract_json_array(response_text(response))
            return validate_variants(raw_variants, question, count)
        except (OpenAIError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            print(f"  retry {attempt}/{retries}: {exc}", flush=True)
            time.sleep(retry_sleep * attempt)
    raise RuntimeError(f"OpenAI paraphrase generation failed: {last_error}")


def build_expansion_rows(
    items: list[dict[str, Any]],
    cache: dict[str, list[str]],
    repeat: int,
) -> list[dict[str, str]]:
    rows = []
    for item in items:
        source_id = item["source_id"]
        variants = cache[source_id]
        for variant_index, variant in enumerate(variants, start=1):
            for repeat_index in range(1, repeat + 1):
                conv = [dict(msg) for msg in item["conversation"]]
                conv[item["user_index"]]["content"] = variant

                row = dict(item["source_row"])
                row["id"] = f"{source_id}-para-{variant_index:02d}-rep-{repeat_index:02d}"
                row["content"] = json.dumps(conv, ensure_ascii=False)
                rows.append(row)
    return rows


def build_key_qa_expansion(
    input_path: Path,
    output_path: Path,
    cache_path: Path,
    env_path: Path,
    model: str,
    paraphrases_per_qa: int,
    repeat: int,
    retries: int,
    retry_sleep: float,
    sleep_seconds: float,
) -> None:
    load_env(env_path)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(f"OPENAI_API_KEY is not set; checked env and {env_path}")

    fieldnames, source_rows = read_csv(input_path)
    if "content" not in fieldnames:
        raise ValueError(f"{input_path} must contain a content column")
    if "id" not in fieldnames:
        fieldnames = ["id"] + fieldnames

    items = source_items(source_rows)
    cache = load_paraphrase_cache(cache_path)
    client = OpenAI()

    for item_index, item in enumerate(items, start=1):
        source_id = item["source_id"]
        cached = cache.get(source_id, [])
        if len(cached) >= paraphrases_per_qa:
            cache[source_id] = cached[:paraphrases_per_qa]
            print(f"[{item_index}/{len(items)}] cached: {source_id}", flush=True)
            continue

        print(
            f"[{item_index}/{len(items)}] generating {paraphrases_per_qa}: "
            f"{item['question'][:80]}",
            flush=True,
        )
        cache[source_id] = generate_variants(
            client=client,
            model=model,
            question=item["question"],
            count=paraphrases_per_qa,
            retries=retries,
            retry_sleep=retry_sleep,
        )
        write_paraphrase_cache(cache_path, cache)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    expansion_rows = build_expansion_rows(items, cache, repeat)
    write_csv(output_path, fieldnames, expansion_rows)

    print(f"input: {input_path}")
    print(f"paraphrase_cache: {cache_path}")
    print(f"output: {output_path}")
    print(f"source rows: {len(items)}")
    print(f"paraphrases per QA: {paraphrases_per_qa}")
    print(f"repeat per paraphrase: {repeat}")
    print(f"expanded rows: {len(expansion_rows)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate OpenAI key_qa paraphrases and write expanded Gemma-3 SFT CSV."
    )
    parser.add_argument("--input", type=Path, default=default_input_path())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_PARAPHRASE_CACHE)
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--paraphrases-per-qa", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=3.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_key_qa_expansion(
        input_path=args.input,
        output_path=args.output,
        cache_path=args.cache,
        env_path=args.env,
        model=args.model,
        paraphrases_per_qa=args.paraphrases_per_qa,
        repeat=args.repeat,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
        sleep_seconds=args.sleep_seconds,
    )


if __name__ == "__main__":
    main()
