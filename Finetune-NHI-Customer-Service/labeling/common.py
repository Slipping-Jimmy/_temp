"""Shared helpers for labeling data builders."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No CSV header found: {path}")
        return list(reader.fieldnames), list(reader)


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
    *,
    extrasaction: str = "raise",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction=extrasaction)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def read_prompt(path: Path) -> str:
    prompt = path.read_text(encoding="utf-8")
    if "{model_contexts}" not in prompt:
        raise ValueError(f"{path} must contain {{model_contexts}}")
    if "[USER_INPUT]" not in prompt:
        raise ValueError(f"{path} must contain [USER_INPUT]")
    return prompt


def response_text(response: object) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return str(text).strip()
    if hasattr(response, "model_dump"):
        dumped = response.model_dump()
        chunks: list[str] = []
        for item in dumped.get("output", []) or []:
            for content in item.get("content", []) or []:
                if content.get("type") in {"output_text", "text"} and content.get("text"):
                    chunks.append(content["text"])
        if chunks:
            return "".join(chunks).strip()
    return str(response).strip()


def require_columns(path: Path, fieldnames: list[str], columns: list[str]) -> None:
    missing = [column for column in columns if column not in fieldnames]
    if missing:
        raise ValueError(f"{path} must contain columns: {', '.join(missing)}")
