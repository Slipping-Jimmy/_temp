#!/usr/bin/env python3
"""Open-loop and burst load generator for one authenticated playground vendor.

Unlike Locust's default closed-loop users, this tool can schedule requests without
waiting for earlier requests to finish. It is intended to find the point where a
single vendor's concurrent requests hit the application's 60-second cutoff.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import mimetypes
import random
import shlex
import shutil
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable

import requests

from loadtest_lib import extract_api_id, parse_sse_data, percentile, safe_preview, protocol_whitespace
from inspect_sse import write_answer_report
from sse_stream import AbsoluteSSETimeout, DeadlineSSE


CONVERSATIONS_ENDPOINT = "/api/playground/conversations"
SCRIPT_VERSION = "20261027-v10"
ATTACHMENT_ENDPOINT = "/api/playground/conversations/{conversation_id}/chat/{chat_id}/attachment"
SSE_ENDPOINT = "/api/playground/conversations/{conversation_id}/chat/sse"
TRACE_HEADER_NAMES = ("x-request-id", "x-correlation-id", "traceparent", "request-id")

RECORD_FIELDS = [
    "run_id",
    "stage",
    "mode",
    "target_value",
    "round",
    "sequence",
    "scheduled_at_utc",
    "prepared_at_utc",
    "burst_released_at_utc",
    "started_at_utc",
    "ended_at_utc",
    "scheduled_epoch",
    "start_epoch",
    "end_epoch",
    "inflight_at_dispatch",
    "release_skew_ms",
    "request_dispatch_skew_ms",
    "preparation_success",
    "round_valid",
    "conversation_id",
    "chat_id",
    "attachment_id",
    "conversation_strategy",
    "conversation_status",
    "upload_status",
    "sse_status",
    "conversation_trace_id",
    "upload_trace_id",
    "sse_trace_id",
    "image_path",
    "image_category",
    "image_width",
    "image_height",
    "image_bytes",
    "question_id",
    "prompt_profile",
    "conversation_ms",
    "upload_ms",
    "sse_headers_ms",
    "ttft_ms",
    "first_token_within_60s",
    "max_sse_event_gap_ms",
    "sse_total_ms",
    "transaction_ms",
    "sse_event_count",
    "sse_nonempty_line_count",
    "sse_data_line_count",
    "sse_heartbeat_line_count",
    "finish_reason",
    "content_event_count",
    "response_chars",
    "completion_marker",
    "answer_valid",
    "normalized_answer",
    "success",
    "timeout_basis",
    "error_type",
    "failure_phase",
    "error_message",
    "sse_debug_preview",
    "response_preview",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def parse_duration(value: str) -> float:
    raw = value.strip().lower()
    if raw.endswith("ms"):
        return float(raw[:-2]) / 1000.0
    if raw.endswith("s"):
        return float(raw[:-1])
    if raw.endswith("m"):
        return float(raw[:-1]) * 60.0
    if raw.endswith("h"):
        return float(raw[:-1]) * 3600.0
    return float(raw)


def parse_number_list(value: str, *, integer: bool) -> list[float | int]:
    parsed: list[float | int] = []
    for item in value.replace(",", " ").split():
        number = int(item) if integer else float(item)
        if number <= 0:
            raise ValueError(f"Values must be positive: {item!r}")
        parsed.append(number)
    if not parsed:
        raise ValueError("Provide at least one load value")
    return parsed


def parse_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        elif value:
            try:
                words = shlex.split(value, comments=True, posix=True)
                value = words[0] if words else ""
            except ValueError:
                pass
        values[key] = value
    return values


def config_path(value: str, config_file: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config_file.parent / path).resolve()


def trace_id(headers: requests.structures.CaseInsensitiveDict[str]) -> str:
    for name in TRACE_HEADER_NAMES:
        value = headers.get(name)
        if value:
            return str(value)
    return ""


def short_error(response: requests.Response) -> str:
    try:
        body = response.text
    except Exception:  # pragma: no cover - response body failure is uncommon
        body = ""
    return safe_preview(body or "", 500)


def classify_sse_error(message: str) -> str:
    if "語言模型通訊異常" in message or "無法解析" in message:
        return "sse_model_communication_error"
    if "無法儲存紀錄" in message:
        return "sse_record_save_error"
    return "sse_error_event"


def record_duration_ms(record: dict[str, Any]) -> float:
    """Return the best available end-to-end duration for cutoff detection."""

    for field in ("sse_total_ms", "transaction_ms"):
        try:
            return float(record.get(field, ""))
        except (TypeError, ValueError):
            continue
    return 0.0


def normalize_binary_answer(response_text: str) -> str:
    """Return the required binary answer, or an empty string for malformed output."""

    compact = "".join(protocol_whitespace(response_text).split())
    for marker in ("[answer]", "[ANSWER]", "[end]", "[END]", "[done]", "[DONE]"):
        compact = compact.replace(marker, "")
    return compact if compact in {"符合", "不符合"} else ""


def numeric_value(record: dict[str, Any], field: str) -> float:
    try:
        return float(record.get(field, ""))
    except (TypeError, ValueError):
        return 0.0


def classify_60s_cutoff(record: dict[str, Any]) -> str:
    """Identify the confirmed product limit: no first model token within 60s.

    Late first tokens count even if the stream eventually succeeds. Failures
    before first content near 60s remain candidates pending server-log checks.
    Post-token errors never count as this product timeout.
    """
    failure_phase = str(record.get("failure_phase", ""))
    if failure_phase in {"conversation", "upload", "preparation"}:
        return ""

    ttft_ms = numeric_value(record, "ttft_ms")
    if ttft_ms > 0:
        return "first_token" if ttft_ms > 60_000 else ""

    error_type = str(record.get("error_type", "")).lower()
    message = str(record.get("error_message", "")).lower()
    semantic_markers = (
        "內容已達語言模型上限",
        "上下文長度",
        "context length",
        "context window",
        "maximum context",
        "token limit",
    )
    non_timeout_error_types = {"client_sse_absolute_timeout"}
    if error_type in non_timeout_error_types or any(marker in message for marker in semantic_markers):
        return ""

    if str(record.get("success", "0")) == "1":
        return ""
    return "first_token" if 55_000 <= record_duration_ms(record) <= 75_000 else ""


def has_model_content(text: str) -> bool:
    """Protocol-only prefixes must not start the model TTFT clock."""
    for marker in ("[answer]", "[thought]", "[done]", "[end]"):
        text = text.replace(marker, "").replace(marker.upper(), "")
    return bool(protocol_whitespace(text).strip())


def is_60s_cutoff_candidate(record: dict[str, Any]) -> bool:
    return bool(classify_60s_cutoff(record))


@dataclass(frozen=True)
class Settings:
    base_url: str
    token: str
    dataset_dir: Path
    manifest_path: Path
    question_path: Path
    prompt_profile: str
    timeout_seconds: float
    sse_max_total_seconds: float
    require_done: bool
    allow_raw_stream: bool
    capture_response: bool
    preview_chars: int
    conversation_strategy: str
    fixed_conversation_id: str
    llm_argument: dict[str, Any] = field(default_factory=dict)


DEFAULT_LLM_ARGUMENT = {
    "max_tokens": 1024,
    "temperature": 1.0,
    "top_p": 0.95,
    "presence_penalty": 0.2,
    "enable_thinking": False,
}


def parse_llm_argument(raw: str | None) -> dict[str, Any]:
    """Validate requested API arguments before sending any workload."""
    overrides = json.loads(raw) if raw is not None else {}
    if not isinstance(overrides, dict):
        raise ValueError("LLM_ARGUMENT_JSON must be a JSON object")
    unknown = set(overrides) - set(DEFAULT_LLM_ARGUMENT)
    if unknown:
        raise ValueError(f"Unknown LLM_ARGUMENT_JSON keys: {sorted(unknown)}")
    args = {**DEFAULT_LLM_ARGUMENT, **overrides}
    for name in ("max_tokens",):
        if type(args[name]) is not int or args[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(args["enable_thinking"]) is not bool:
        raise ValueError("enable_thinking must be a JSON boolean")
    for name, low, high, exclusive in (
        ("temperature", 0, 1, False), ("top_p", 0, 1, True),
        ("presence_penalty", 0, 2, False),
    ):
        value = args[name]
        if (type(value) not in (int, float) or not math.isfinite(value)
                or value > high or value < low or (exclusive and value == low)):
            raise ValueError(f"Invalid {name}: {value!r}")
    return args


def build_sse_payload(conversation_id: str, chat_id: str, prompt: str,
                      attachment_id: str, llm_argument: dict[str, Any]) -> dict[str, Any]:
    return {
        "parent_id": conversation_id,
        "chat_id": chat_id,
        "prompt": prompt,
        "message": "",
        "attachments": [attachment_id],
        "llm_argument": dict(llm_argument),
    }


@dataclass
class PreparedRequest:
    """An uploaded attachment that is ready for synchronized SSE release."""

    record: dict[str, Any]
    payload: dict[str, Any]


class BurstRelease:
    """Barrier that releases only after every SSE worker is ready."""

    def __init__(self, expected: int):
        self.expected = expected
        self.ready = 0
        self.epoch = 0.0
        self.condition = threading.Condition()

    def wait(self) -> float:
        with self.condition:
            self.ready += 1
            self.condition.notify_all()
            while not self.epoch:
                self.condition.wait()
            return self.epoch

    def release_when_ready(self) -> float:
        with self.condition:
            while self.ready < self.expected:
                self.condition.wait()
            self.epoch = time.time()
            self.condition.notify_all()
            return self.epoch


def load_settings(config_file: Path, strategy_override: str | None) -> Settings:
    values = parse_dotenv(config_file)
    base_url = values.get("PLAYGROUND_BASE_URL", "").rstrip("/")
    token = values.get("PLAYGROUND_TOKEN", "")
    if not base_url or not token or token.startswith("replace-"):
        raise ValueError("Set PLAYGROUND_BASE_URL and a valid PLAYGROUND_TOKEN in the config file")
    strategy = (strategy_override or values.get("UNTHROTTLED_CONVERSATION_STRATEGY", "shared")).lower()
    if strategy not in {"shared", "pool", "per_request"}:
        raise ValueError("Conversation strategy must be shared, pool, or per_request")
    dataset_dir = config_path(values.get("DATASET_DIR", "./data/tono"), config_file)
    manifest_path = config_path(values.get("IMAGE_MANIFEST", "./images_manifest.csv"), config_file)
    question_path = config_path(values.get("QUESTION_FILE", "./questions.json"), config_file)
    for label, path in (("DATASET_DIR", dataset_dir), ("IMAGE_MANIFEST", manifest_path), ("QUESTION_FILE", question_path)):
        if (label == "DATASET_DIR" and not path.is_dir()) or (label != "DATASET_DIR" and not path.is_file()):
            raise ValueError(f"{label} does not exist: {path}")
    return Settings(
        base_url=base_url,
        token=token,
        dataset_dir=dataset_dir,
        manifest_path=manifest_path,
        question_path=question_path,
        prompt_profile=values.get("PROMPT_PROFILE", "short"),
        timeout_seconds=float(values.get("REQUEST_TIMEOUT_SECONDS", "120")),
        sse_max_total_seconds=float(values.get("SSE_MAX_TOTAL_SECONDS", "180")),
        require_done=values.get("REQUIRE_DONE_MARKER", "1").strip().lower() in {"1", "true", "yes", "on"},
        allow_raw_stream=values.get("ALLOW_RAW_STREAM", "0").strip().lower() in {"1", "true", "yes", "on"},
        capture_response=values.get("CAPTURE_RESPONSE", "1").strip().lower() in {"1", "true", "yes", "on"},
        preview_chars=int(values.get("RESPONSE_PREVIEW_CHARS", "500")),
        conversation_strategy=strategy,
        fixed_conversation_id=values.get("FIXED_CONVERSATION_ID", "").strip(),
        llm_argument=parse_llm_argument(values.get("LLM_ARGUMENT_JSON")),
    )


def load_images(settings: Settings) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with settings.manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("valid", "1")).strip().lower() not in {"1", "true", "yes"}:
                continue
            path = settings.dataset_dir / row["path"]
            if path.is_file():
                row["absolute_path"] = str(path)
                rows.append(row)
    if not rows:
        raise ValueError("No valid images found in the manifest")
    return rows


def load_questions(settings: Settings) -> tuple[list[dict[str, Any]], list[float]]:
    questions = json.loads(settings.question_path.read_text(encoding="utf-8"))
    selected = [item for item in questions if settings.prompt_profile in item.get("profiles", [])]
    if not selected:
        raise ValueError(f"No questions for PROMPT_PROFILE={settings.prompt_profile!r}")
    return selected, [float(item.get("weight", 1)) for item in selected]


class CsvWriter:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=RECORD_FIELDS, extrasaction="ignore")
        self.writer.writeheader()
        self.sse_handle = None

    def write_sse_line(self, record: dict[str, Any], elapsed_ms: float, line: str) -> None:
        with self.lock:
            if self.sse_handle is None:
                self.sse_handle = self.path.with_name("sse_raw.jsonl").open("w", encoding="utf-8")
            self.sse_handle.write(json.dumps({
                "round": record["round"], "sequence": record["sequence"],
                "chat_id": record["chat_id"], "elapsed_ms": round(elapsed_ms, 3),
                "line": line,
            }, ensure_ascii=False) + "\n")

    def write(self, record: dict[str, Any]) -> None:
        ttft = numeric_value(record, "ttft_ms")
        record["first_token_within_60s"] = int(0 < ttft <= 60_000)
        with self.lock:
            self.writer.writerow(record)
            self.handle.flush()

    def close(self) -> None:
        with self.lock:
            self.handle.close()
            if self.sse_handle is not None:
                self.sse_handle.close()


class TransactionClient:
    def __init__(
        self,
        settings: Settings,
        images: list[dict[str, str]],
        questions: list[dict[str, Any]],
        weights: list[float],
        writer: CsvWriter,
        run_id: str,
    ):
        self.settings = settings
        self.images = images
        self.questions = questions
        self.weights = weights
        self.writer = writer
        self.run_id = run_id
        self.local = threading.local()

    def session(self) -> requests.Session:
        if not hasattr(self.local, "session"):
            session = requests.Session()
            session.headers.update({"Authorization": f"Bearer {self.settings.token}"})
            self.local.session = session
        return self.local.session

    def capture_sse_line(self, record: dict[str, Any], line: str, elapsed_ms: float) -> None:
        if self.settings.capture_response:
            self.writer.write_sse_line(record, elapsed_ms, line)

    @staticmethod
    def stream_diagnostics(record: dict[str, Any], stream: DeadlineSSE | None) -> None:
        if stream is None:
            return
        record.update({
            "sse_nonempty_line_count": stream.nonempty_lines,
            "sse_data_line_count": stream.data_lines,
            "sse_heartbeat_line_count": stream.heartbeat_lines,
            "max_sse_event_gap_ms": round(max(stream.max_wire_gap_ms,
                                             stream.elapsed_ms() - stream.last_received_ms), 3),
        })
        if stream.request_started_epoch and record.get("scheduled_epoch"):
            record["request_dispatch_skew_ms"] = round(
                (stream.request_started_epoch - float(record["scheduled_epoch"])) * 1000, 3)

    def create_conversation(self) -> str:
        response = self.session().post(
            f"{self.settings.base_url}{CONVERSATIONS_ENDPOINT}",
            timeout=self.settings.timeout_seconds,
        )
        if not response.ok:
            raise RuntimeError(f"Conversation setup failed: HTTP {response.status_code}: {short_error(response)}")
        try:
            conversation_id = extract_api_id(response.json(), ("conversation_id", "id", "_id"))
        except ValueError as exc:
            raise RuntimeError(f"Conversation setup returned invalid JSON: {exc}") from exc
        if not conversation_id:
            raise RuntimeError("Conversation setup returned no ID")
        return conversation_id

    def pick_workload(self, sequence: int) -> tuple[dict[str, str], dict[str, Any]]:
        rng = random.Random(f"{self.run_id}:{sequence}")
        image = self.images[rng.randrange(len(self.images))]
        question = rng.choices(self.questions, weights=self.weights, k=1)[0]
        return image, question

    def run(
        self,
        *,
        stage: str,
        mode: str,
        target_value: str,
        sequence: int,
        conversation_id: str | None,
        scheduled_epoch: float,
        inflight_at_dispatch: int,
        round_number: int,
        start_gate: threading.Event | BurstRelease | None = None,
    ) -> dict[str, Any]:
        release_epoch: float | None = None
        if isinstance(start_gate, BurstRelease):
            release_epoch = start_gate.wait()
            scheduled_epoch = release_epoch
        elif start_gate is not None:
            start_gate.wait()
        image, question = self.pick_workload(sequence)
        started_at_utc = utc_now()
        start_epoch = time.time()
        started = time.perf_counter()
        chat_id = str(uuid.uuid4())
        response_text = ""
        sse_debug: list[str] = []
        event_count = 0
        content_event_count = 0
        completion_marker = False
        error_type = ""
        error_message = ""
        attachment_id = ""

        record: dict[str, Any] = {
            "run_id": self.run_id,
            "stage": stage,
            "mode": mode,
            "target_value": target_value,
            "round": round_number,
            "sequence": sequence,
            "scheduled_at_utc": datetime.fromtimestamp(scheduled_epoch, timezone.utc).isoformat(timespec="milliseconds"),
            "prepared_at_utc": "",
            "burst_released_at_utc": (
                datetime.fromtimestamp(release_epoch, timezone.utc).isoformat(timespec="milliseconds")
                if release_epoch is not None
                else ""
            ),
            "started_at_utc": started_at_utc,
            "scheduled_epoch": f"{scheduled_epoch:.6f}",
            "start_epoch": f"{start_epoch:.6f}",
            "inflight_at_dispatch": inflight_at_dispatch,
            "release_skew_ms": (
                round((start_epoch - release_epoch) * 1000, 3) if release_epoch is not None else ""
            ),
            "preparation_success": "",
            "round_valid": 1,
            "conversation_id": conversation_id or "",
            "chat_id": chat_id,
            "attachment_id": "",
            "conversation_strategy": self.settings.conversation_strategy,
            "conversation_status": "reused" if conversation_id else "",
            "upload_status": "",
            "sse_status": "",
            "conversation_trace_id": "",
            "upload_trace_id": "",
            "sse_trace_id": "",
            "image_path": image.get("path", ""),
            "image_category": image.get("category", ""),
            "image_width": image.get("width", ""),
            "image_height": image.get("height", ""),
            "image_bytes": image.get("bytes", ""),
            "question_id": question["id"],
            "prompt_profile": self.settings.prompt_profile,
            "conversation_ms": 0 if conversation_id else "",
            "upload_ms": "",
            "sse_headers_ms": "",
            "ttft_ms": "",
            "max_sse_event_gap_ms": "",
            "sse_total_ms": "",
            "transaction_ms": "",
            "sse_event_count": 0,
            "content_event_count": 0,
            "response_chars": 0,
            "completion_marker": 0,
            "answer_valid": 0,
            "normalized_answer": "",
            "success": 0,
            "timeout_basis": "",
            "error_type": "",
            "failure_phase": "",
            "error_message": "",
            "sse_debug_preview": "",
            "response_preview": "",
        }
        try:
            session = self.session()
            if not conversation_id:
                step = time.perf_counter()
                response = session.post(
                    f"{self.settings.base_url}{CONVERSATIONS_ENDPOINT}",
                    timeout=self.settings.timeout_seconds,
                )
                record["conversation_ms"] = round((time.perf_counter() - step) * 1000, 3)
                record["conversation_status"] = response.status_code
                record["conversation_trace_id"] = trace_id(response.headers)
                if not response.ok:
                    error_type = "conversation_http_error"
                    record["failure_phase"] = "conversation"
                    error_message = f"HTTP {response.status_code}: {short_error(response)}"
                    return record
                conversation_id = extract_api_id(response.json(), ("conversation_id", "id", "_id"))
                if not conversation_id:
                    error_type = "conversation_missing_id"
                    record["failure_phase"] = "conversation"
                    error_message = short_error(response)
                    return record
                record["conversation_id"] = conversation_id

            attachment_url = self.settings.base_url + ATTACHMENT_ENDPOINT.format(
                conversation_id=conversation_id, chat_id=chat_id
            )
            image_path = Path(image["absolute_path"])
            mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
            step = time.perf_counter()
            with image_path.open("rb") as handle:
                response = session.post(
                    attachment_url,
                    files={"attachment": (image_path.name, handle, mime_type)},
                    timeout=self.settings.timeout_seconds,
                )
            record["upload_ms"] = round((time.perf_counter() - step) * 1000, 3)
            record["upload_status"] = response.status_code
            record["upload_trace_id"] = trace_id(response.headers)
            if not response.ok:
                error_type = "upload_http_error"
                record["failure_phase"] = "upload"
                error_message = f"HTTP {response.status_code}: {short_error(response)}"
                return record
            attachment_id = extract_api_id(response.json(), ("attachment_id", "id", "_id")) or ""
            if not attachment_id:
                error_type = "upload_missing_id"
                record["failure_phase"] = "upload"
                error_message = short_error(response)
                return record
            record["attachment_id"] = attachment_id

            payload = build_sse_payload(
                conversation_id, chat_id, question["text"], attachment_id, self.settings.llm_argument
            )
            step = time.perf_counter()
            sse_started = step
            last_sse_event = step
            max_sse_event_gap_ms = 0.0
            sse_stream = DeadlineSSE(
                session, self.settings.base_url + SSE_ENDPOINT.format(conversation_id=conversation_id),
                headers={"Content-Type": "application/json"}, json=payload,
                total_seconds=self.settings.sse_max_total_seconds, timeout=self.settings.timeout_seconds,
            )
            with sse_stream as response:
                record["sse_headers_ms"] = round((time.perf_counter() - step) * 1000, 3)
                record["sse_status"] = response.status_code
                record["sse_trace_id"] = trace_id(response.headers)
                if not response.ok:
                    error_type = "sse_http_error"
                    record["failure_phase"] = "sse"
                    error_message = f"HTTP {response.status_code}"
                    return record
                for data, event_name, received_ms in response.events(
                    on_line=lambda line, ms: self.capture_sse_line(record, line, ms),
                    allow_raw=self.settings.allow_raw_stream,
                ):
                    event_now = time.perf_counter()
                    max_sse_event_gap_ms = max(
                        max_sse_event_gap_ms, (event_now - last_sse_event) * 1000
                    )
                    last_sse_event = event_now
                    event_count += 1
                    if self.settings.capture_response and len(sse_debug) < 5:
                        sse_debug.append(safe_preview(data, 200))
                    parsed = parse_sse_data(data, event_name)
                    if parsed["error"]:
                        error_type = classify_sse_error(parsed["error"])
                        record["failure_phase"] = "sse"
                        error_message = safe_preview(parsed["error"], 500)
                        return record
                    if parsed["text"]:
                        if not record["ttft_ms"] and has_model_content(response_text + parsed["text"]):
                            record["ttft_ms"] = round(received_ms - response.request_started_ms, 3)
                        response_text += parsed["text"]
                        content_event_count += 1
                    if parsed["done"]:
                        record["finish_reason"] = parsed["finish_reason"]
                        completion_marker = True
                        if parsed["finish_reason"] in {"length", "max_tokens"}:
                            error_type = "sse_output_length_limit"
                            record["failure_phase"] = "sse"
                            error_message = "Model output reached its token limit"
                            return record
                        break
                record["sse_total_ms"] = round((time.perf_counter() - step) * 1000, 3)
                if not has_model_content(response_text):
                    error_type = "sse_no_content"
                    record["failure_phase"] = "sse"
                    error_message = "Stream ended without model content"
                    return record
                if self.settings.require_done and not completion_marker:
                    error_type = "sse_missing_done_marker"
                    record["failure_phase"] = "sse"
                    error_message = "Stream ended without a recognized completion marker"
                    return record
                record["success"] = 1
        except AbsoluteSSETimeout as exc:
            error_type = "client_sse_absolute_timeout"
            record["failure_phase"] = "sse"
            error_message = str(exc)
        except requests.RequestException as exc:
            error_type = f"client_{type(exc).__name__}"
            record["failure_phase"] = "full_flow"
            error_message = str(exc)
        except Exception as exc:  # Includes invalid API JSON and local I/O failures.
            error_type = f"client_{type(exc).__name__}"
            error_message = str(exc)
        finally:
            end_epoch = time.time()
            if "sse_started" in locals():
                finished = time.perf_counter()
                record["sse_total_ms"] = round((finished - sse_started) * 1000, 3)
                max_sse_event_gap_ms = max(
                    max_sse_event_gap_ms, (finished - last_sse_event) * 1000
                )
                record["max_sse_event_gap_ms"] = round(max_sse_event_gap_ms, 3)
            self.stream_diagnostics(record, locals().get("sse_stream"))
            normalized_answer = normalize_binary_answer(response_text)
            record.update(
                {
                    "ended_at_utc": utc_now(),
                    "end_epoch": f"{end_epoch:.6f}",
                    "transaction_ms": round((time.perf_counter() - started) * 1000, 3),
                    "sse_event_count": event_count,
                    "content_event_count": content_event_count,
                    "response_chars": len(response_text),
                    "completion_marker": int(completion_marker),
                    "answer_valid": int(bool(normalized_answer)),
                    "normalized_answer": normalized_answer,
                    "error_type": error_type,
                    "error_message": safe_preview(error_message, 500),
                    "sse_debug_preview": safe_preview(" | ".join(sse_debug), 1000)
                    if self.settings.capture_response
                    else "",
                    "response_preview": safe_preview(response_text, self.settings.preview_chars)
                    if self.settings.capture_response
                    else "",
                }
            )
            record["timeout_basis"] = classify_60s_cutoff(record)
            self.writer.write(record)
        return record

    def prepare_sse_request(
        self,
        *,
        stage: str,
        target_value: str,
        sequence: int,
        conversation_id: str | None,
        round_number: int,
    ) -> PreparedRequest | dict[str, Any]:
        """Create/upload everything needed for an SSE request without starting SSE."""

        image, question = self.pick_workload(sequence)
        prepared_started = time.perf_counter()
        chat_id = str(uuid.uuid4())
        record: dict[str, Any] = {
            "run_id": self.run_id,
            "stage": stage,
            "mode": "sse_burst",
            "target_value": target_value,
            "round": round_number,
            "sequence": sequence,
            "scheduled_at_utc": "",
            "prepared_at_utc": "",
            "burst_released_at_utc": "",
            "started_at_utc": "",
            "scheduled_epoch": "",
            "start_epoch": "",
            "end_epoch": "",
            "inflight_at_dispatch": "",
            "release_skew_ms": "",
            "preparation_success": 0,
            "round_valid": 1,
            "conversation_id": conversation_id or "",
            "chat_id": chat_id,
            "attachment_id": "",
            "conversation_strategy": self.settings.conversation_strategy,
            "conversation_status": "reused" if conversation_id else "",
            "upload_status": "",
            "sse_status": "",
            "conversation_trace_id": "",
            "upload_trace_id": "",
            "sse_trace_id": "",
            "image_path": image.get("path", ""),
            "image_category": image.get("category", ""),
            "image_width": image.get("width", ""),
            "image_height": image.get("height", ""),
            "image_bytes": image.get("bytes", ""),
            "question_id": question["id"],
            "prompt_profile": self.settings.prompt_profile,
            "conversation_ms": 0 if conversation_id else "",
            "upload_ms": "",
            "sse_headers_ms": "",
            "ttft_ms": "",
            "max_sse_event_gap_ms": "",
            "sse_total_ms": "",
            "transaction_ms": "",
            "sse_event_count": 0,
            "content_event_count": 0,
            "response_chars": 0,
            "completion_marker": 0,
            "answer_valid": 0,
            "normalized_answer": "",
            "success": 0,
            "timeout_basis": "",
            "error_type": "",
            "failure_phase": "",
            "error_message": "",
            "sse_debug_preview": "",
            "response_preview": "",
        }
        error_type = ""
        error_message = ""
        try:
            session = self.session()
            if not conversation_id:
                step = time.perf_counter()
                response = session.post(
                    f"{self.settings.base_url}{CONVERSATIONS_ENDPOINT}",
                    timeout=self.settings.timeout_seconds,
                )
                record["conversation_ms"] = round((time.perf_counter() - step) * 1000, 3)
                record["conversation_status"] = response.status_code
                record["conversation_trace_id"] = trace_id(response.headers)
                if not response.ok:
                    error_type = "conversation_http_error"
                    error_message = f"HTTP {response.status_code}: {short_error(response)}"
                    record["failure_phase"] = "conversation"
                    raise RuntimeError(error_message)
                conversation_id = extract_api_id(response.json(), ("conversation_id", "id", "_id"))
                if not conversation_id:
                    error_type = "conversation_missing_id"
                    error_message = short_error(response)
                    record["failure_phase"] = "conversation"
                    raise RuntimeError(error_message)
                record["conversation_id"] = conversation_id

            image_path = Path(image["absolute_path"])
            mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
            step = time.perf_counter()
            with image_path.open("rb") as handle:
                response = session.post(
                    self.settings.base_url
                    + ATTACHMENT_ENDPOINT.format(conversation_id=conversation_id, chat_id=chat_id),
                    files={"attachment": (image_path.name, handle, mime_type)},
                    timeout=self.settings.timeout_seconds,
                )
            record["upload_ms"] = round((time.perf_counter() - step) * 1000, 3)
            record["upload_status"] = response.status_code
            record["upload_trace_id"] = trace_id(response.headers)
            if not response.ok:
                error_type = "upload_http_error"
                error_message = f"HTTP {response.status_code}: {short_error(response)}"
                record["failure_phase"] = "upload"
                raise RuntimeError(error_message)
            attachment_id = extract_api_id(response.json(), ("attachment_id", "id", "_id")) or ""
            if not attachment_id:
                error_type = "upload_missing_id"
                error_message = short_error(response)
                record["failure_phase"] = "upload"
                raise RuntimeError(error_message)
            record.update(
                {
                    "attachment_id": attachment_id,
                    "prepared_at_utc": utc_now(),
                    "preparation_success": 1,
                }
            )
            return PreparedRequest(
                record=record,
                payload=build_sse_payload(
                    conversation_id, chat_id, question["text"], attachment_id, self.settings.llm_argument
                ),
            )
        except requests.RequestException as exc:
            if not error_type:
                error_type = f"preparation_{type(exc).__name__}"
                error_message = str(exc)
                record["failure_phase"] = "preparation"
        except Exception as exc:
            if not error_type:
                error_type = f"preparation_{type(exc).__name__}"
                error_message = str(exc)
                record["failure_phase"] = "preparation"
        record.update(
            {
                "round_valid": 0,
                "ended_at_utc": utc_now(),
                "end_epoch": f"{time.time():.6f}",
                "transaction_ms": round((time.perf_counter() - prepared_started) * 1000, 3),
                "error_type": error_type,
                "error_message": safe_preview(error_message, 500),
            }
        )
        self.writer.write(record)
        return record

    def abort_prepared(self, prepared: PreparedRequest, message: str) -> dict[str, Any]:
        record = prepared.record
        record.update(
            {
                "round_valid": 0,
                "ended_at_utc": utc_now(),
                "end_epoch": f"{time.time():.6f}",
                "success": 0,
                "error_type": "round_aborted_preparation_failure",
                "failure_phase": "preparation",
                "error_message": safe_preview(message, 500),
            }
        )
        self.writer.write(record)
        return record

    def run_prepared_sse(
        self,
        prepared: PreparedRequest,
        *,
        release: BurstRelease,
        inflight_at_dispatch: int,
    ) -> dict[str, Any]:
        """Wait at the barrier, then measure only the SSE/model response path."""

        release_epoch = release.wait()
        record = prepared.record
        start_epoch = time.time()
        started = time.perf_counter()
        response_text = ""
        sse_debug: list[str] = []
        event_count = 0
        content_event_count = 0
        completion_marker = False
        error_type = ""
        error_message = ""
        sse_started: float | None = None
        last_sse_event: float | None = None
        max_sse_event_gap_ms = 0.0
        record.update(
            {
                "scheduled_at_utc": datetime.fromtimestamp(release_epoch, timezone.utc).isoformat(timespec="milliseconds"),
                "burst_released_at_utc": datetime.fromtimestamp(release_epoch, timezone.utc).isoformat(timespec="milliseconds"),
                "started_at_utc": utc_now(),
                "scheduled_epoch": f"{release_epoch:.6f}",
                "start_epoch": f"{start_epoch:.6f}",
                "inflight_at_dispatch": inflight_at_dispatch,
                "release_skew_ms": round((start_epoch - release_epoch) * 1000, 3),
            }
        )
        try:
            step = time.perf_counter()
            sse_started = step
            last_sse_event = step
            conversation_id = record["conversation_id"]
            sse_stream = DeadlineSSE(
                self.session(), self.settings.base_url + SSE_ENDPOINT.format(conversation_id=conversation_id),
                headers={"Content-Type": "application/json"}, json=prepared.payload,
                total_seconds=self.settings.sse_max_total_seconds, timeout=self.settings.timeout_seconds,
            )
            with sse_stream as response:
                record["sse_headers_ms"] = round((time.perf_counter() - step) * 1000, 3)
                record["sse_status"] = response.status_code
                record["sse_trace_id"] = trace_id(response.headers)
                if not response.ok:
                    error_type = "sse_http_error"
                    record["failure_phase"] = "sse"
                    error_message = f"HTTP {response.status_code}"
                    return record
                for data, event_name, received_ms in response.events(
                    on_line=lambda line, ms: self.capture_sse_line(record, line, ms),
                    allow_raw=self.settings.allow_raw_stream,
                ):
                    event_now = time.perf_counter()
                    if last_sse_event is not None:
                        max_sse_event_gap_ms = max(
                            max_sse_event_gap_ms, (event_now - last_sse_event) * 1000
                        )
                    last_sse_event = event_now
                    event_count += 1
                    if self.settings.capture_response and len(sse_debug) < 5:
                        sse_debug.append(safe_preview(data, 200))
                    parsed = parse_sse_data(data, event_name)
                    if parsed["error"]:
                        error_type = classify_sse_error(parsed["error"])
                        error_message = safe_preview(parsed["error"], 500)
                        return record
                    if parsed["text"]:
                        if not record["ttft_ms"] and has_model_content(response_text + parsed["text"]):
                            record["ttft_ms"] = round(received_ms - response.request_started_ms, 3)
                        response_text += parsed["text"]
                        content_event_count += 1
                    if parsed["done"]:
                        record["finish_reason"] = parsed["finish_reason"]
                        completion_marker = True
                        if parsed["finish_reason"] in {"length", "max_tokens"}:
                            error_type = "sse_output_length_limit"
                            error_message = "Model output reached its token limit"
                            return record
                        break
                record["sse_total_ms"] = round((time.perf_counter() - step) * 1000, 3)
                if not has_model_content(response_text):
                    error_type = "sse_no_content"
                    error_message = "Stream ended without model content"
                    return record
                if self.settings.require_done and not completion_marker:
                    error_type = "sse_missing_done_marker"
                    error_message = "Stream ended without a recognized completion marker"
                    return record
                record["success"] = 1
        except AbsoluteSSETimeout as exc:
            error_type = "client_sse_absolute_timeout"
            error_message = str(exc)
        except requests.RequestException as exc:
            error_type = f"client_{type(exc).__name__}"
            error_message = str(exc)
        except Exception as exc:
            error_type = f"client_{type(exc).__name__}"
            error_message = str(exc)
        finally:
            finished = time.perf_counter()
            if sse_started is not None:
                record["sse_total_ms"] = round((finished - sse_started) * 1000, 3)
                if last_sse_event is not None:
                    max_sse_event_gap_ms = max(
                        max_sse_event_gap_ms, (finished - last_sse_event) * 1000
                    )
                record["max_sse_event_gap_ms"] = round(max_sse_event_gap_ms, 3)
            self.stream_diagnostics(record, locals().get("sse_stream"))
            normalized_answer = normalize_binary_answer(response_text)
            record.update(
                {
                    "ended_at_utc": utc_now(),
                    "end_epoch": f"{time.time():.6f}",
                    "transaction_ms": round((time.perf_counter() - started) * 1000, 3),
                    "sse_event_count": event_count,
                    "content_event_count": content_event_count,
                    "response_chars": len(response_text),
                    "completion_marker": int(completion_marker),
                    "answer_valid": int(bool(normalized_answer)),
                    "normalized_answer": normalized_answer,
                    "error_type": error_type,
                    "failure_phase": "sse" if error_type else "",
                    "error_message": safe_preview(error_message, 500),
                    "sse_debug_preview": safe_preview(" | ".join(sse_debug), 1000)
                    if self.settings.capture_response
                    else "",
                    "response_preview": safe_preview(response_text, self.settings.preview_chars)
                    if self.settings.capture_response
                    else "",
                }
            )
            record["timeout_basis"] = classify_60s_cutoff(record)
            self.writer.write(record)
        return record


class InflightCounter:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.current = 0
        self.maximum = 0

    def add(self) -> int:
        with self.lock:
            self.current += 1
            self.maximum = max(self.maximum, self.current)
            return self.current

    def remove(self) -> None:
        with self.lock:
            self.current -= 1

    def value(self) -> int:
        with self.lock:
            return self.current


def create_conversation_pool(client: TransactionClient, count: int, settings: Settings) -> list[str | None]:
    if settings.fixed_conversation_id:
        return [settings.fixed_conversation_id] * count
    if settings.conversation_strategy == "per_request":
        return [None] * count
    if settings.conversation_strategy == "shared":
        return [client.create_conversation()] * count
    return [client.create_conversation() for _ in range(count)]


def numeric_values(records: Iterable[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for record in records:
        try:
            value = float(record.get(field, ""))
        except (TypeError, ValueError):
            continue
        values.append(value)
    return values


def stage_summary(
    *,
    stage: str,
    mode: str,
    target_value: str,
    records: list[dict[str, Any]],
    scheduled: int,
    admission_limited: int,
    run_window_seconds: float,
    drain_seconds: float,
    max_inflight: int,
) -> dict[str, Any]:
    successes = [record for record in records if record.get("success") == 1]
    valid_answers = [record for record in successes if record.get("answer_valid") == 1]
    errors = Counter(record.get("error_type") or "unknown" for record in records if record.get("success") != 1)
    cutoff_bases = Counter(
        basis for record in records if (basis := classify_60s_cutoff(record))
    )
    result: dict[str, Any] = {
        "stage": stage,
        "mode": mode,
        "target_value": target_value,
        "scheduled": scheduled,
        "admitted": len(records),
        "admission_limited": admission_limited,
        "successes": len(successes),
        "valid_answers": len(valid_answers),
        "answer_valid_rate_percent": len(valid_answers) / len(successes) * 100 if successes else 0.0,
        "failures": len(records) - len(successes),
        "cutoff_60s_candidates": sum(cutoff_bases.values()),
        "first_token_timeout_candidates": cutoff_bases["first_token"],
        "total_request_timeout_candidates": cutoff_bases["total_request"],
        "idle_stream_timeout_candidates": cutoff_bases["idle_stream"],
        "explicit_timeout_candidates": cutoff_bases["explicit_timeout"],
        "timeout_basis_counts": json.dumps(cutoff_bases, ensure_ascii=False, sort_keys=True),
        "preparation_failures": sum(
            1 for record in records if record.get("failure_phase") in {"conversation", "upload", "preparation"}
        ),
        "error_rate_percent": (len(records) - len(successes)) / len(records) * 100 if records else 100.0,
        "scheduled_rps": scheduled / run_window_seconds if run_window_seconds else 0.0,
        "admitted_rps": len(records) / run_window_seconds if run_window_seconds else 0.0,
        "successful_rps": len(successes) / run_window_seconds if run_window_seconds else 0.0,
        "run_window_seconds": run_window_seconds,
        "drain_seconds": drain_seconds,
        "max_observed_inflight": max_inflight,
        "error_counts": json.dumps(errors, ensure_ascii=False, sort_keys=True),
    }
    for field in ("ttft_ms", "max_sse_event_gap_ms", "sse_total_ms", "transaction_ms"):
        values = numeric_values(successes, field)
        result[f"mean_{field}"] = sum(values) / len(values) if values else ""
        result[f"p50_{field}"] = percentile(values, 50)
        result[f"p95_{field}"] = percentile(values, 95)
        result[f"p99_{field}"] = percentile(values, 99)
        result[f"max_{field}"] = max(values) if values else ""
    release_skews = numeric_values(records, "release_skew_ms")
    result["max_release_skew_ms"] = max(release_skews) if release_skews else ""
    dispatch_skews = numeric_values(records, "request_dispatch_skew_ms")
    result["max_request_dispatch_skew_ms"] = max(dispatch_skews) if dispatch_skews else ""
    all_ttfts = numeric_values(records, "ttft_ms")
    result["first_token_received_count"] = len(all_ttfts)
    result["no_first_token_count"] = len(records) - len(all_ttfts)
    result["late_first_token_count"] = sum(value > 60_000 for value in all_ttfts)
    result["mean_all_ttft_ms"] = mean(all_ttfts) if all_ttfts else ""
    result["p95_all_ttft_ms"] = percentile(all_ttfts, 95)
    result["max_all_ttft_ms"] = max(all_ttfts) if all_ttfts else ""
    return result


def write_summary(result_root: Path, summaries: list[dict[str, Any]], config: dict[str, str]) -> None:
    summary_path = result_root / "summary.csv"
    fields = list(summaries[0]) if summaries else []
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)

    report = [f"# Unthrottled load test: {config['RUN_ID']}", "", "## Stages", ""]
    report.append(
        "| Stage | Mode | Target | Scheduled | Admitted | Success | Error % | "
        "Valid answers | Valid % | 60s candidates | Prep failures | Mean E2E ms | p95 E2E ms | Max E2E ms | Max release skew ms |"
    )
    report.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summaries:
        report.append(
            f"| {row['stage']} | {row['mode']} | {row['target_value']} | {row['scheduled']} | "
            f"{row['admitted']} | {row['successes']} | {row['error_rate_percent']:.2f} | "
            f"{row['valid_answers']} | {row['answer_valid_rate_percent']:.2f} | "
            f"{row['cutoff_60s_candidates']} | {row['preparation_failures']} | "
            f"{format_metric(row.get('mean_transaction_ms'))} | {format_metric(row.get('p95_transaction_ms'))} | "
            f"{format_metric(row.get('max_transaction_ms'))} | {format_metric(row.get('max_release_skew_ms'))} |"
        )
        if row["error_counts"] != "{}":
            report.append(f"  - errors: `{row['error_counts']}`")
        if row["cutoff_60s_candidates"]:
            report.append(f"  - 60-second cutoff candidates: **{row['cutoff_60s_candidates']}**")
            report.append(f"  - timeout basis: `{row['timeout_basis_counts']}`")
    first_cutoff = next((row for row in summaries if row["cutoff_60s_candidates"] > 0), None)
    if first_cutoff:
        report.extend(
            [
                "",
                "## 60-second cutoff boundary",
                "",
                f"The first cutoff candidate occurred at **{first_cutoff['stage']}** "
                f"(target={first_cutoff['target_value']}, "
                f"count={first_cutoff['cutoff_60s_candidates']}).",
            ]
        )
    else:
        report.extend(
            [
                "",
                "## 60-second cutoff boundary",
                "",
                "No 60-second cutoff candidate was observed in these stages; increase the burst size "
                "or arrival rate to continue finding the boundary.",
            ]
        )
    report.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `burst` sends the target number of jobs together, then waits for the round to finish.",
            "- `sse_burst` uploads all attachments first and then releases all SSE/model requests together; its E2E values measure SSE release through complete model answer.",
            "- `open_loop` schedules at its target requests/s even while earlier jobs are still running.",
            "- `admission_limited > 0` means the client safety cap was reached; increase `--max-inflight` before treating that stage as a server capacity result.",
            "- `cutoff_60s_candidates` uses the confirmed first-token limit: TTFT > 60000 ms, or failure near 60s before any model content. Post-token failures remain errors but do not count as this cutoff.",
            "- `timeout_basis_counts` reports `first_token`. Confirm candidates with API/vLLM logs. E2E measures completed answers and is not the timeout criterion.",
            "- With CAPTURE_RESPONSE=1, `sse_raw.jsonl` stores received SSE lines with elapsed times and chat IDs, including blank event separators, event names and completion markers. Wire-line counters distinguish empty responses from heartbeat-only streams.",
            "- `answer_valid_rate_percent` checks whether a successful short-profile response is exactly `符合` or `不符合` after removing protocol markers.",
        ]
    )
    report.extend([
        "", "## All-request first-token timing", "",
        "Includes failed requests that received content. Requests without a first token are counted separately, not assigned zero latency.",
        "", "| Target | First token received | No first token | TTFT >60s | Mean ms | p95 ms | Max ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in summaries:
        report.append(f"| {row['target_value']} | {row['first_token_received_count']} | "
                      f"{row['no_first_token_count']} | {row['late_first_token_count']} | "
                      f"{format_metric(row['mean_all_ttft_ms'])} | {format_metric(row['p95_all_ttft_ms'])} | "
                      f"{format_metric(row['max_all_ttft_ms'])} |")
    (result_root / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (result_root / "report.html").write_text(build_html_report(summaries, config["RUN_ID"]), encoding="utf-8")
    write_answer_report(result_root)


def build_html_report(summaries: list[dict[str, Any]], run_id: str) -> str:
    """Build a dependency-free capacity chart that can be opened off-network."""

    rows = [row for row in summaries if row.get("mode") in {"burst", "sse_burst"}]
    width, height = 960, 520
    left, right, top, bottom = 85, 30, 45, 75
    plot_w, plot_h = width - left - right, height - top - bottom
    x_values = [float(row["target_value"]) for row in rows] or [0.0]
    def seconds_or_none(value: Any) -> float | None:
        try:
            return float(value) / 1000
        except (TypeError, ValueError):
            return None

    series = {
        "mean": [seconds_or_none(row.get("mean_transaction_ms")) for row in rows],
        "p95": [seconds_or_none(row.get("p95_transaction_ms")) for row in rows],
        "max": [seconds_or_none(row.get("max_transaction_ms")) for row in rows],
    }
    x_min, x_max = min(x_values), max(x_values)
    if x_min == x_max:
        x_min, x_max = max(0, x_min - 1), x_max + 1
    observed_max = max(
        (value for values in series.values() for value in values if value is not None), default=0
    )
    y_max = max(1.0, math.ceil(observed_max))

    def x_pos(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def y_pos(value: float) -> float:
        return top + plot_h - value / y_max * plot_h

    svg: list[str] = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Capacity curve">']
    svg.append(f'<rect width="{width}" height="{height}" fill="#fff"/>')
    for tick in range(0, 6):
        value = y_max * tick / 5
        y = y_pos(value)
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        svg.append(f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end">{value:g}</text>')
    colors = {"mean": "#2563eb", "p95": "#f59e0b", "max": "#7c3aed"}
    labels = {"mean": "平均完整回答", "p95": "p95", "max": "最大值"}
    for key, values in series.items():
        points = " ".join(
            f"{x_pos(x):.1f},{y_pos(y):.1f}" for x, y in zip(x_values, values) if y is not None
        )
        if points:
            svg.append(f'<polyline points="{points}" fill="none" stroke="{colors[key]}" stroke-width="3"/>')
        for row, x, y in zip(rows, x_values, values):
            if y is None:
                continue
            failed = int(row.get("failures", 0)) > 0
            svg.append(
                f'<circle cx="{x_pos(x):.1f}" cy="{y_pos(y):.1f}" r="5" fill="{colors[key]}" '
                f'stroke="{("#dc2626" if failed else "#fff")}" stroke-width="{("3" if failed else "1")}"><title>'
                f'{labels[key]}: {y:.3f}s; failures={row.get("failures", 0)}; timeout={row.get("cutoff_60s_candidates", 0)}'
                f'</title></circle>'
            )
    for row, x in zip(rows, x_values):
        if int(row.get("cutoff_60s_candidates", 0)):
            marker_x, marker_y = x_pos(x), y_pos(60)
            svg.append(
                f'<path d="M {marker_x-7:.1f} {marker_y-7:.1f} L {marker_x+7:.1f} {marker_y+7:.1f} '
                f'M {marker_x+7:.1f} {marker_y-7:.1f} L {marker_x-7:.1f} {marker_y+7:.1f}" '
                f'stroke="#dc2626" stroke-width="4"><title>60 秒 cutoff 候選：{row["cutoff_60s_candidates"]}</title></path>'
            )
    for x in x_values:
        svg.append(f'<text x="{x_pos(x):.1f}" y="{height-bottom+28}" text-anchor="middle">{x:g}</text>')
    svg.append(f'<text x="{left+plot_w/2:.1f}" y="{height-18}" text-anchor="middle">同時放行的請求數</text>')
    svg.append(f'<text transform="translate(22 {top+plot_h/2:.1f}) rotate(-90)" text-anchor="middle">完整回答時間（秒）</text>')
    legend_x = left
    for key in ("mean", "p95", "max"):
        svg.append(f'<circle cx="{legend_x}" cy="22" r="5" fill="{colors[key]}"/><text x="{legend_x+10}" y="27">{labels[key]}</text>')
        legend_x += 155
    svg.append("</svg>")

    table_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['target_value']))}</td><td>{row['successes']}/{row['admitted']}</td>"
        f"<td>{format_metric(row.get('mean_transaction_ms'))}</td><td>{format_metric(row.get('p95_transaction_ms'))}</td>"
        f"<td>{format_metric(row.get('max_transaction_ms'))}</td><td>{row['cutoff_60s_candidates']}</td>"
        f"<td>{html.escape(str(row.get('timeout_basis_counts', '{}')))}</td>"
        f"<td>{row['preparation_failures']}</td><td>{row.get('valid_answers', 0)}/{row['successes']}</td>"
        f"<td>{format_metric(row.get('max_release_skew_ms'))}</td></tr>"
        for row in rows
    )
    ttft_table = '<table><tr><th>並發</th><th>收到首 token</th><th>未收到</th><th>TTFT &gt;60s</th><th>p95 ms</th><th>最大 ms</th></tr>' + ''.join(
        f"<tr><td>{html.escape(str(row['target_value']))}</td><td>{row.get('first_token_received_count', 0)}</td>"
        f"<td>{row.get('no_first_token_count', 0)}</td><td>{row.get('late_first_token_count', 0)}</td>"
        f"<td>{format_metric(row.get('p95_all_ttft_ms'))}</td><td>{format_metric(row.get('max_all_ttft_ms'))}</td></tr>"
        for row in rows) + '</table>'
    return f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(run_id)} 負載報告</title><style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;color:#172033}}h1{{font-size:25px}}.card{{border:1px solid #dbe3ee;border-radius:12px;padding:18px;margin:18px 0;box-shadow:0 2px 10px #0000000d}}svg text{{font-size:13px;fill:#334155}}table{{border-collapse:collapse;width:100%}}th,td{{padding:9px;border-bottom:1px solid #e5e7eb;text-align:right}}th:first-child,td:first-child{{text-align:left}}.note{{color:#475569}}</style></head><body><h1>同步批次容量報告</h1><p class="note">Run: {html.escape(run_id)}。60 秒 timeout 僅依首個模型 token 判定；完整回答時間另計。平均、p95 與最大值僅包含成功完成的回答；失敗數請同時參考下表。</p><div class="card">{''.join(svg)}</div><div class="card"><table><thead><tr><th>並發數</th><th>成功/送出</th><th>平均 ms</th><th>p95 ms</th><th>最大 ms</th><th>60s timeout</th><th>判定依據</th><th>準備失敗</th><th>有效答案/成功</th><th>最大同步誤差 ms</th></tr></thead><tbody>{table_rows}</tbody></table></div><div class="card"><h2>首 token 統計（含失敗請求）</h2><p>未收到 token 的請求另列，不當作 0 ms。上方完整回答曲線只含成功回答。</p>{ttft_table}</div></body></html>"""


def format_metric(value: Any) -> str:
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return "N/A"


def append_result(
    future: Future[dict[str, Any]],
    records: list[dict[str, Any]],
    records_lock: threading.Lock,
    inflight: InflightCounter,
) -> None:
    try:
        record = future.result()
    except Exception as exc:  # The transaction itself should already record errors.
        record = {"success": 0, "error_type": f"future_{type(exc).__name__}", "error_message": str(exc)}
    with records_lock:
        records.append(record)
    inflight.remove()


def run_burst_stage(
    client: TransactionClient,
    *,
    stage: str,
    concurrency: int,
    rounds: int,
    between_rounds: float,
) -> tuple[list[dict[str, Any]], int, int, float, float, int]:
    records: list[dict[str, Any]] = []
    records_lock = threading.Lock()
    inflight = InflightCounter()
    sequence = 0
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="burst-request") as executor:
        for round_number in range(1, rounds + 1):
            conversations = create_conversation_pool(client, concurrency, client.settings)
            gate = BurstRelease(expected=concurrency)
            futures: list[Future[dict[str, Any]]] = []
            for slot in range(concurrency):
                sequence += 1
                current = inflight.add()
                future = executor.submit(
                    client.run,
                    stage=stage,
                    mode="burst",
                    target_value=str(concurrency),
                    sequence=sequence,
                    conversation_id=conversations[slot],
                    scheduled_epoch=0.0,
                    inflight_at_dispatch=current,
                    round_number=round_number,
                    start_gate=gate,
                )
                future.add_done_callback(
                    lambda done, recs=records, lock=records_lock, counter=inflight: append_result(
                        done, recs, lock, counter
                    )
                )
                futures.append(future)
            gate.release_when_ready()
            wait(futures)
            if round_number < rounds and between_rounds:
                time.sleep(between_rounds)
    run_window = time.monotonic() - started
    return records, sequence, 0, run_window, 0.0, inflight.maximum


def run_sse_burst_stage(
    client: TransactionClient,
    *,
    stage: str,
    concurrency: int,
    rounds: int,
    between_rounds: float,
    prepare_workers: int,
) -> tuple[list[dict[str, Any]], int, int, float, float, int]:
    """Pre-upload N inputs and release exactly N SSE requests through one gate."""

    records: list[dict[str, Any]] = []
    records_lock = threading.Lock()
    inflight = InflightCounter()
    sequence = 0
    started = time.monotonic()
    for round_number in range(1, rounds + 1):
        # A fresh conversation per round avoids prior turns affecting later samples.
        conversations = create_conversation_pool(client, concurrency, client.settings)
        preparation_jobs: list[tuple[int, Future[PreparedRequest | dict[str, Any]]]] = []
        with ThreadPoolExecutor(
            max_workers=min(prepare_workers, concurrency), thread_name_prefix="sse-burst-prepare"
        ) as executor:
            for slot in range(concurrency):
                sequence += 1
                future = executor.submit(
                    client.prepare_sse_request,
                    stage=stage,
                    target_value=str(concurrency),
                    sequence=sequence,
                    conversation_id=conversations[slot],
                    round_number=round_number,
                )
                preparation_jobs.append((sequence, future))

        prepared_requests: list[PreparedRequest] = []
        failed_records: list[dict[str, Any]] = []
        for _, future in preparation_jobs:
            result = future.result()
            if isinstance(result, PreparedRequest):
                prepared_requests.append(result)
            else:
                result["round_valid"] = 0
                failed_records.append(result)
        records.extend(failed_records)

        if failed_records:
            reason = f"Round aborted: {len(failed_records)} of {concurrency} preparations failed"
            records.extend(client.abort_prepared(item, reason) for item in prepared_requests)
        else:
            release = BurstRelease(expected=concurrency)
            futures: list[Future[dict[str, Any]]] = []
            with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="sse-burst-request") as executor:
                for prepared in prepared_requests:
                    current = inflight.add()
                    future = executor.submit(
                        client.run_prepared_sse,
                        prepared,
                        release=release,
                        inflight_at_dispatch=current,
                    )
                    future.add_done_callback(
                        lambda done, recs=records, lock=records_lock, counter=inflight: append_result(
                            done, recs, lock, counter
                        )
                    )
                    futures.append(future)
                release.release_when_ready()
                wait(futures)
        if round_number < rounds and between_rounds:
            time.sleep(between_rounds)
    run_window = time.monotonic() - started
    return records, sequence, 0, run_window, 0.0, inflight.maximum


def run_open_loop_stage(
    client: TransactionClient,
    *,
    stage: str,
    arrival_rate: float,
    duration_seconds: float,
    max_inflight: int,
    conversation_pool_size: int,
    drain_timeout: float,
) -> tuple[list[dict[str, Any]], int, int, float, float, int]:
    conversations = create_conversation_pool(client, conversation_pool_size, client.settings)
    records: list[dict[str, Any]] = []
    records_lock = threading.Lock()
    inflight = InflightCounter()
    futures: set[Future[dict[str, Any]]] = set()
    sequence = 0
    scheduled = 0
    admission_limited = 0
    started = time.monotonic()
    deadline = started + duration_seconds
    next_dispatch = started
    interval = 1.0 / arrival_rate

    with ThreadPoolExecutor(max_workers=max_inflight, thread_name_prefix="open-loop-request") as executor:
        while True:
            now = time.monotonic()
            if now >= deadline:
                break
            if now < next_dispatch:
                time.sleep(min(next_dispatch - now, 0.02))
                continue
            while next_dispatch <= now and next_dispatch < deadline:
                scheduled += 1
                scheduled_epoch = time.time() - max(0.0, now - next_dispatch)
                if inflight.value() >= max_inflight:
                    admission_limited += 1
                else:
                    sequence += 1
                    current = inflight.add()
                    conversation_id = conversations[(sequence - 1) % len(conversations)]
                    future = executor.submit(
                        client.run,
                        stage=stage,
                        mode="open_loop",
                        target_value=f"{arrival_rate:g}",
                        sequence=sequence,
                        conversation_id=conversation_id,
                        scheduled_epoch=scheduled_epoch,
                        inflight_at_dispatch=current,
                        round_number=0,
                    )
                    future.add_done_callback(
                        lambda done, recs=records, lock=records_lock, counter=inflight: append_result(
                            done, recs, lock, counter
                        )
                    )
                    futures.add(future)
                next_dispatch += interval
            futures = {future for future in futures if not future.done()}

        run_window = time.monotonic() - started
        drain_started = time.monotonic()
        remaining = max(0.0, drain_timeout)
        while futures and remaining > 0:
            _, unfinished = wait(futures, timeout=min(remaining, 0.5))
            futures = set(unfinished)
            remaining = drain_timeout - (time.monotonic() - drain_started)
        drain_seconds = time.monotonic() - drain_started
        if futures:
            # Running request threads cannot be safely killed. Mark this explicitly
            # and wait for their configured HTTP timeout before the process exits.
            raise RuntimeError(
                f"{len(futures)} requests did not drain within {drain_timeout:g}s; "
                "increase --drain-timeout or inspect the server connection timeout"
            )
    return records, scheduled, admission_limited, run_window, drain_seconds, inflight.maximum


def safe_run_id() -> str:
    return datetime.now(timezone.utc).strftime("unthrottled-%Y%m%dT%H%M%SZ")


def write_config(path: Path, values: dict[str, str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for key, value in values.items():
            if "TOKEN" not in key:
                handle.write(f"{key}={value}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(".env"))
    parser.add_argument("--output-root", type=Path, default=Path("./results"))
    parser.add_argument("--run-id", default=safe_run_id())
    parser.add_argument("--conversation-strategy", choices=("shared", "pool", "per_request"))
    parser.add_argument("--capture-response", choices=("0", "1"))
    subparsers = parser.add_subparsers(dest="mode", required=True)

    burst = subparsers.add_parser("burst", help="Start N full upload+SSE flows at once.")
    burst.add_argument("--concurrency", required=True, help='For example: "8 16 24 32"')
    burst.add_argument("--rounds", type=int, default=3)
    burst.add_argument("--between-rounds", default="10s")

    sse_burst = subparsers.add_parser(
        "sse-burst", help="Pre-upload N attachments, then release N SSE requests simultaneously."
    )
    sse_burst.add_argument("--concurrency", required=True, help='For example: "1 5 10 15 20"')
    sse_burst.add_argument("--rounds", type=int, default=5)
    sse_burst.add_argument("--between-rounds", default="20s")
    sse_burst.add_argument("--prepare-workers", type=int, default=8)
    sse_burst.add_argument(
        "--allow-fixed-conversation",
        action="store_true",
        help="Allow one fixed conversation across all stages (not recommended; can hit context limits).",
    )
    sse_burst.add_argument(
        "--continue-after-failure",
        action="store_true",
        help="Continue to higher concurrency even after a stage has any failure.",
    )

    open_loop = subparsers.add_parser("open-loop", help="Send at a fixed request rate without waiting.")
    open_loop.add_argument("--arrival-rates", required=True, help='For example: "2 4 6"')
    open_loop.add_argument("--duration", default="5m")
    open_loop.add_argument("--max-inflight", type=int, default=128)
    open_loop.add_argument("--conversation-pool-size", type=int, default=1)
    open_loop.add_argument("--drain-timeout", default="150s")

    args = parser.parse_args()
    if args.mode in {"burst", "sse-burst"} and args.rounds < 1:
        parser.error("--rounds must be at least 1")
    if args.mode == "sse-burst" and args.prepare_workers < 1:
        parser.error("--prepare-workers must be positive")
    if args.mode == "open-loop" and (args.max_inflight < 1 or args.conversation_pool_size < 1):
        parser.error("--max-inflight and --conversation-pool-size must be positive")

    config_file = args.config.expanduser().resolve()
    if not config_file.is_file():
        parser.error(f"Configuration not found: {config_file}")
    settings = load_settings(config_file, args.conversation_strategy)
    if settings.sse_max_total_seconds <= 0:
        parser.error("SSE_MAX_TOTAL_SECONDS must be positive")
    if (
        args.mode == "sse-burst"
        and settings.fixed_conversation_id
        and not args.allow_fixed_conversation
    ):
        parser.error(
            "sse-burst requires a fresh conversation per round. Clear FIXED_CONVERSATION_ID "
            "in .env, or use --allow-fixed-conversation only for deliberate context-limit testing."
        )
    if args.capture_response is not None:
        settings = Settings(**{**settings.__dict__, "capture_response": args.capture_response == "1"})
    images = load_images(settings)
    questions, weights = load_questions(settings)

    result_root = (args.output_root.expanduser().resolve() / args.run_id)
    if result_root.exists():
        parser.error(f"Result directory already exists: {result_root}")
    result_root.mkdir(parents=True)
    write_config(
        result_root / "run_config.txt",
        {
            "RUN_ID": args.run_id,
            "TEST_SCRIPT_VERSION": SCRIPT_VERSION,
            "PLAYGROUND_BASE_URL": settings.base_url,
            "PROMPT_PROFILE": settings.prompt_profile,
            "LLM_ARGUMENT_JSON_REQUESTED": json.dumps(settings.llm_argument, ensure_ascii=False, sort_keys=True),
            "REQUEST_TIMEOUT_SECONDS": str(settings.timeout_seconds),
            "SSE_MAX_TOTAL_SECONDS": str(settings.sse_max_total_seconds),
            "REQUIRE_DONE_MARKER": str(int(settings.require_done)),
            "CONVERSATION_STRATEGY": settings.conversation_strategy,
            "FIXED_CONVERSATION_ID_SET": str(int(bool(settings.fixed_conversation_id))),
            "MODE": args.mode,
            "TARGETS": args.concurrency if args.mode in {"burst", "sse-burst"} else args.arrival_rates,
            "ROUNDS": str(args.rounds) if args.mode in {"burst", "sse-burst"} else "",
            "BETWEEN_ROUNDS": str(args.between_rounds) if args.mode in {"burst", "sse-burst"} else "",
            "PREPARE_WORKERS": str(args.prepare_workers) if args.mode == "sse-burst" else "",
            "DURATION": str(args.duration) if args.mode == "open-loop" else "",
            "MAX_INFLIGHT": str(args.max_inflight) if args.mode == "open-loop" else "",
            "CONVERSATION_POOL_SIZE": str(args.conversation_pool_size) if args.mode == "open-loop" else "",
        },
    )
    shutil.copy2(settings.manifest_path, result_root / "images_manifest.csv")
    shutil.copy2(settings.question_path, result_root / "questions.json")
    print(f"Test script version: {SCRIPT_VERSION}")
    print("Requested llm_argument: " + json.dumps(settings.llm_argument, ensure_ascii=False, sort_keys=True))

    summaries: list[dict[str, Any]] = []
    if args.mode in {"burst", "sse-burst"}:
        values = parse_number_list(args.concurrency, integer=True)
        between_rounds = parse_duration(args.between_rounds)
        for index, raw_value in enumerate(values, 1):
            concurrency = int(raw_value)
            mode_label = "sse_burst" if args.mode == "sse-burst" else "burst"
            stage = f"stage_{index:03d}_{mode_label}_{concurrency}c"
            stage_dir = result_root / stage
            writer = CsvWriter(stage_dir / "transactions.csv")
            client = TransactionClient(settings, images, questions, weights, writer, args.run_id)
            description = "synchronized SSE bursts" if args.mode == "sse-burst" else "full-flow bursts"
            print(f"[{utc_now()}] Starting {stage}: {args.rounds} {description} of {concurrency} requests")
            try:
                if args.mode == "sse-burst":
                    records, scheduled, limited, window, drain, max_seen = run_sse_burst_stage(
                        client,
                        stage=stage,
                        concurrency=concurrency,
                        rounds=args.rounds,
                        between_rounds=between_rounds,
                        prepare_workers=args.prepare_workers,
                    )
                else:
                    records, scheduled, limited, window, drain, max_seen = run_burst_stage(
                        client,
                        stage=stage,
                        concurrency=concurrency,
                        rounds=args.rounds,
                        between_rounds=between_rounds,
                    )
            finally:
                writer.close()
            summary = stage_summary(
                    stage=stage,
                    mode=mode_label,
                    target_value=str(concurrency),
                    records=records,
                    scheduled=scheduled,
                    admission_limited=limited,
                    run_window_seconds=window,
                    drain_seconds=drain,
                    max_inflight=max_seen,
                )
            summaries.append(summary)
            if (args.mode == "sse-burst" and
                    (summary["failures"] or summary["cutoff_60s_candidates"]) and
                    not args.continue_after_failure):
                print(
                    f"[{utc_now()}] Stopping after {stage}: {summary['failures']} failures, "
                    f"{summary['cutoff_60s_candidates']} first-token cutoff candidates "
                    "(use --continue-after-failure to override)."
                )
                break
    else:
        values = parse_number_list(args.arrival_rates, integer=False)
        duration_seconds = parse_duration(args.duration)
        drain_timeout = parse_duration(args.drain_timeout)
        for index, raw_value in enumerate(values, 1):
            arrival_rate = float(raw_value)
            stage = f"stage_{index:03d}_open_{arrival_rate:g}rps"
            stage_dir = result_root / stage
            writer = CsvWriter(stage_dir / "transactions.csv")
            client = TransactionClient(settings, images, questions, weights, writer, args.run_id)
            print(f"[{utc_now()}] Starting {stage}: {arrival_rate:g} requests/s for {duration_seconds:g}s")
            try:
                records, scheduled, limited, window, drain, max_seen = run_open_loop_stage(
                    client,
                    stage=stage,
                    arrival_rate=arrival_rate,
                    duration_seconds=duration_seconds,
                    max_inflight=args.max_inflight,
                    conversation_pool_size=args.conversation_pool_size,
                    drain_timeout=drain_timeout,
                )
            finally:
                writer.close()
            summaries.append(
                stage_summary(
                    stage=stage,
                    mode="open_loop",
                    target_value=f"{arrival_rate:g}",
                    records=records,
                    scheduled=scheduled,
                    admission_limited=limited,
                    run_window_seconds=window,
                    drain_seconds=drain,
                    max_inflight=max_seen,
                )
            )

    write_summary(result_root, summaries, {"RUN_ID": args.run_id})
    print(f"Results are ready: {result_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
