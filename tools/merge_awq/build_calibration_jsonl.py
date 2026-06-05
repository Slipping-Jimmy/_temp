#!/usr/bin/env python3
import argparse
import ast
import csv
import json
import os
import random

from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build Gemma 3 chat-template calibration JSONL from SFT CSV files."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input CSV path. Can be passed multiple times.",
    )
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    parser.add_argument("--tokenizer", default="./models/google--gemma-3-27b-it")
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--max-chars",
        type=int,
        default=30000,
        help="Skip samples with rendered text longer than this many characters.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def safe_parse(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        return None


def normalize_role(role):
    role = str(role).strip().lower()
    if role == "model":
        return "assistant"
    return role


def normalize_conversation(conv):
    if not isinstance(conv, list) or len(conv) == 0:
        return None

    normalized = []
    system_text = ""
    for msg in conv:
        if not isinstance(msg, dict):
            continue
        role = normalize_role(msg.get("role", ""))
        content = msg.get("content")
        if role not in {"system", "user", "assistant"}:
            continue
        if content is None or str(content).strip() == "":
            continue

        content = str(content).strip()
        if role == "system":
            system_text = content if not system_text else system_text + "\n\n" + content
            continue

        if role == "user" and system_text:
            content = "[SYSTEM INSTRUCTION]\n" + system_text + "\n\n[USER]\n" + content
            system_text = ""

        if normalized and normalized[-1]["role"] == role:
            normalized[-1]["content"] += "\n\n" + content
        else:
            normalized.append({"role": role, "content": content})

    if system_text:
        normalized.insert(0, {"role": "user", "content": "[SYSTEM INSTRUCTION]\n" + system_text})

    while normalized and normalized[0]["role"] != "user":
        normalized = normalized[1:]

    if len(normalized) < 2 or not any(msg["role"] == "assistant" for msg in normalized):
        return None
    return normalized


def read_conversations(paths):
    conversations = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "content" not in reader.fieldnames:
                raise ValueError(f"{path} must contain a content column")
            for row in reader:
                conv = normalize_conversation(safe_parse(row.get("content")))
                if conv is not None:
                    conversations.append(conv)
    return conversations


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=args.trust_remote_code,
    )

    conversations = read_conversations(args.input)
    random.Random(args.seed).shuffle(conversations)
    if args.limit > 0:
        conversations = conversations[: args.limit]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    written = 0
    skipped = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for conv in conversations:
            try:
                text = tokenizer.apply_chat_template(
                    conv,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception:
                skipped += 1
                continue
            if not text or len(text) > args.max_chars:
                skipped += 1
                continue
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            written += 1

    print(f"written={written} skipped={skipped} out={args.out}")


if __name__ == "__main__":
    main()

