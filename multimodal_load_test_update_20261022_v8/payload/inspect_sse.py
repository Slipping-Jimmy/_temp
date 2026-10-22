#!/usr/bin/env python3
"""Inspect API-received SSE without the CSV preview's truncation/escaping."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any

from loadtest_lib import parse_sse_data


def data_value(line: str) -> str | None:
    if not line.startswith("data:"):
        return None
    # SSE removes at most one optional space, not all leading/trailing spaces.
    value = line[5:]
    return value[1:] if value.startswith(" ") else value


def content_value(data: str) -> str:
    """Preserve plain-text numeric tokens instead of treating them as JSON numbers."""
    if data.strip() in {"[END]", "[DONE]"} or data.strip().upper().startswith("[ERROR]"):
        return ""
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return data
    if isinstance(payload, dict):
        return str(parse_sse_data(data)["text"])
    # The endpoint streams unquoted text fragments, including valid JSON scalars.
    return data


def read_samples(root: Path) -> list[dict[str, Any]]:
    samples = []
    paths = [root] if root.is_file() else sorted(root.rglob("sse_raw.jsonl"))
    for path in paths:
        metadata = {}
        csv_path = path.with_name("transactions.csv")
        if csv_path.exists():
            with csv_path.open(encoding="utf-8", newline="") as handle:
                metadata = {row["chat_id"]: row for row in csv.DictReader(handle)}
        grouped: dict[str, dict[str, Any]] = {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                event = json.loads(line)
                chat_id = event["chat_id"]
                sample = grouped.setdefault(chat_id, {
                    "chat_id": chat_id, "stage": path.parent.name,
                    "metadata": metadata.get(chat_id, {}), "events": [], "parts": [],
                })
                sample["events"].append(event)
                data = data_value(event["line"])
                if data is not None:
                    sample["parts"].append(content_value(data))
        for sample in grouped.values():
            sample["content"] = "".join(sample.pop("parts"))
            samples.append(sample)
    return samples


def render_report(samples: list[dict[str, Any]]) -> str:
    esc = lambda value: html.escape(str(value))
    cards = []
    for sample in samples:
        meta = sample["metadata"]
        timeline = "\n".join(
            f'{event["elapsed_ms"]:10.3f} ms  {event["line"]}' for event in sample["events"]
        )
        cards.append(
            f'<article><h2>{esc(sample["stage"])} · round {esc(meta.get("round", ""))}'
            f' · sequence {esc(meta.get("sequence", ""))}</h2>'
            f'<p>chat_id: <code>{esc(sample["chat_id"])}</code><br>'
            f'conversation_id: <code>{esc(meta.get("conversation_id", ""))}</code><br>'
            f'image: {esc(meta.get("image_path", ""))}<br>'
            f'success={esc(meta.get("success", ""))}; completion_marker={esc(meta.get("completion_marker", ""))}; '
            f'error={esc(meta.get("error_type", ""))}</p>'
            '<h3>API data 內容串接（排除完成標記及錯誤事件）</h3>'
            f'<pre>{esc(sample["content"])}</pre>'
            '<details><summary>相同內容的 JSON 表示法（辨識真正換行與字面符號）</summary>'
            f'<pre>{esc(json.dumps(sample["content"], ensure_ascii=False))}</pre></details>'
            '<details><summary>原始 SSE 非空行與時間（含 heartbeat、完成標記及錯誤）</summary>'
            f'<pre>{esc(timeline)}</pre></details></article>'
        )
    return ('<!doctype html><html lang="zh-Hant"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>API SSE 回覆明細</title><style>'
            'body{font-family:system-ui;max-width:1100px;margin:30px auto;padding:0 18px}'
            'article{border:1px solid #ccd5df;border-radius:8px;padding:18px;margin:20px 0}'
            'h2{font-size:18px}h3{font-size:15px}pre{white-space:pre-wrap;overflow-wrap:anywhere;'
            'background:#f3f5f7;padding:12px}code{overflow-wrap:anywhere}summary{cursor:pointer}'
            '</style><h1>API SSE 回覆明細</h1>'
            '<p>這是測試端收到的 API 資料，不能單憑此報告區分模型產生或後端加入。'
            '不套用 CSV safe_preview，不截短內容、不刪除數字，不將 ↵／␣ 自動轉換成空白。'
            '原始行仍以 sse_raw.jsonl 為準；[answer] 等前綴保留以便比對。</p>'
            f'<p>共 {len(samples)} 筆</p>' + ''.join(cards) + '</html>')


def write_answer_report(root: Path) -> Path | None:
    samples = read_samples(root)
    if not samples:
        return None
    output = root / "answers.html"
    output.write_text(render_report(samples), encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="Result directory or sse_raw.jsonl")
    parser.add_argument("--chat-id", help="Inspect one chat ID")
    parser.add_argument("--limit", type=int, default=5, help="Console sample count; 0 means all")
    parser.add_argument("--html", type=Path, help="Write all matching samples to an HTML report")
    args = parser.parse_args()
    if not args.result.exists():
        parser.error(f"Not found: {args.result}")
    samples = read_samples(args.result)
    if args.chat_id:
        samples = [sample for sample in samples if sample["chat_id"] == args.chat_id]
    if not samples:
        parser.error("No matching SSE records; check CAPTURE_RESPONSE=1 and the result path")
    for sample in samples[:args.limit or None]:
        print(f'\n{sample["stage"]} chat_id={sample["chat_id"]}')
        print(sample["content"])
        print("JSON:", json.dumps(sample["content"], ensure_ascii=False))
    if args.html:
        args.html.write_text(render_report(samples), encoding="utf-8")
        print(f"\nHTML: {args.html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
