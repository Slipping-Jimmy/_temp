"""Dependency-free helpers shared by the load test and offline tools."""

from __future__ import annotations

import json
import math
import os
from typing import Any, Iterable


ID_KEYS = ("id", "_id", "conversation_id", "attachment_id")
ANSWER_KEYS = ("content", "answer", "text", "token", "response")
CONTROL_WORDS = {"ping", "connected", "start", "done", "success", "heartbeat"}


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def extract_api_id(payload: Any, preferred_keys: Iterable[str] = ID_KEYS) -> str | None:
    """Extract an ID from APIs that may return a JSON string or nested object."""

    if isinstance(payload, str):
        value = payload.strip()
        return value or None
    if isinstance(payload, (int, float)) and not isinstance(payload, bool):
        return str(payload)
    if isinstance(payload, dict):
        for key in preferred_keys:
            value = payload.get(key)
            if isinstance(value, (str, int)) and str(value).strip():
                return str(value).strip()
        for key in ("data", "result", "conversation", "attachment"):
            if key in payload:
                found = extract_api_id(payload[key], preferred_keys)
                if found:
                    return found
    if isinstance(payload, list) and len(payload) == 1:
        return extract_api_id(payload[0], preferred_keys)
    return None


def extract_answer_text(payload: Any) -> str:
    """Return model text from common SSE JSON schemas, excluding status events."""

    if not isinstance(payload, dict):
        return ""

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            for container_key in ("delta", "message"):
                container = choice.get(container_key)
                if isinstance(container, dict):
                    content = container.get("content")
                    if isinstance(content, str) and content:
                        return content

    message = payload.get("message")
    if isinstance(message, dict):
        for key in ("content", "text"):
            value = message.get(key)
            if isinstance(value, str) and value.strip():
                return value

    for key in ANSWER_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value

    data = payload.get("data")
    if isinstance(data, dict):
        return extract_answer_text(data)
    if isinstance(data, str) and data.strip() and str(payload.get("type", "")).lower() in {
        "token",
        "content",
        "delta",
        "answer",
    }:
        return data
    return ""


def parse_sse_data(data: str, event_name: str = "") -> dict[str, Any]:
    """Parse one SSE data field into a normalized event description."""

    stripped = data.strip()
    result: dict[str, Any] = {
        "text": "",
        "done": event_name.lower() in {"done", "complete", "completed", "end"},
        "error": "",
        "json": None,
        "finish_reason": "",
    }
    if not stripped:
        if event_name.lower() == "error":
            result["error"] = "SSE error event"
        elif data:
            result["text"] = data
        return result
    if stripped in {"[DONE]", "[END]"}:
        result["done"] = True
        return result
    if stripped.upper().startswith("[ERROR]"):
        result["error"] = stripped
        return result

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        if stripped.lower() not in CONTROL_WORDS and not stripped.lower().startswith(
            ("status:", "stage:", "event:")
        ):
            if event_name.lower() == "error":
                result["error"] = data
            else:
                result["text"] = data
        return result

    result["json"] = payload
    if not isinstance(payload, dict):
        result["text"] = payload if isinstance(payload, str) else data
        if event_name.lower() == "error":
            result["error"], result["text"] = str(result["text"]), ""
        return result

    error = payload.get("error")
    if error:
        if isinstance(error, str):
            result["error"] = error
        else:
            result["error"] = json.dumps(error, ensure_ascii=False)

    choices = payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        if choices[0].get("finish_reason") is not None:
            result["done"] = True
            result["finish_reason"] = str(choices[0]["finish_reason"])

    event_type = str(payload.get("type", payload.get("event", event_name))).lower()
    status = str(payload.get("status", "")).lower()
    if event_type == "error" and not result["error"]:
        result["error"] = str(payload.get("message", "SSE error event"))
    if event_type in {"done", "complete", "completed", "end"} or status in {
        "done",
        "complete",
        "completed",
    } or payload.get("done") is True:
        result["done"] = True

    result["text"] = extract_answer_text(payload)
    # This endpoint also sends unquoted text fragments. A complete model JSON
    # object is content unless it has a recognized transport-envelope shape.
    envelope_keys = {"choices", "error", "type", "event", "status", "done", "usage", "id", "object"}
    if not result["text"] and not (set(payload) & envelope_keys):
        result["text"] = data
    return result


def protocol_whitespace(text: str) -> str:
    """Decode API display whitespace for interpretation; retain raw data separately."""
    return text.replace("↵", "\n").replace("␣", " ")


def percentile(values: Iterable[float], percent: float) -> float | None:
    """Nearest-rank percentile with linear interpolation."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def safe_preview(value: str, limit: int) -> str:
    return value.replace("\x00", "").replace("\r", " ").replace("\n", " ")[:limit]
