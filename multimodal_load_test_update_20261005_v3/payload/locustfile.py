"""Locust end-to-end load test for the multimodal playground API."""

from __future__ import annotations

import csv
import json
import mimetypes
import os
import queue
import random
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from locust import HttpUser, between, constant, events, task

from loadtest_lib import env_bool, extract_api_id, parse_sse_data, safe_preview


BASE_URL = os.getenv("PLAYGROUND_BASE_URL", "").rstrip("/")
TOKEN = os.getenv("PLAYGROUND_TOKEN", "")
DATASET_DIR = Path(os.getenv("DATASET_DIR", "./data/tono")).expanduser().resolve()
MANIFEST_PATH = Path(os.getenv("IMAGE_MANIFEST", "./images_manifest.csv")).expanduser().resolve()
QUESTION_PATH = Path(os.getenv("QUESTION_FILE", "./questions.json")).expanduser().resolve()
RESULT_DIR = Path(os.getenv("RESULT_DIR", "./results/manual")).expanduser().resolve()
RUN_ID = os.getenv("RUN_ID", "manual")
TARGET_USERS = os.getenv("TARGET_USERS", "")
PROMPT_PROFILE = os.getenv("PROMPT_PROFILE", "short")
TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "120"))
WAIT_SECONDS = float(os.getenv("BETWEEN_REQUESTS_SECONDS", "0"))
REQUIRE_DONE = env_bool("REQUIRE_DONE_MARKER", True)
ALLOW_RAW_STREAM = env_bool("ALLOW_RAW_STREAM", False)
CAPTURE_RESPONSE = env_bool("CAPTURE_RESPONSE", True)
PREVIEW_CHARS = int(os.getenv("RESPONSE_PREVIEW_CHARS", "500"))
BALANCE_CATEGORIES = env_bool("BALANCE_CATEGORIES", True)
CONVERSATION_MODE = os.getenv("CONVERSATION_MODE", "per_user").strip().lower()
FIXED_CONVERSATION_ID = os.getenv("FIXED_CONVERSATION_ID", "").strip()

CONVERSATIONS_ENDPOINT = "/api/playground/conversations"
ATTACHMENT_ENDPOINT = "/api/playground/conversations/{conversation_id}/chat/{chat_id}/attachment"
SSE_ENDPOINT = "/api/playground/conversations/{conversation_id}/chat/sse"

RECORD_FIELDS = [
    "run_id",
    "target_users",
    "started_at_utc",
    "ended_at_utc",
    "start_epoch",
    "end_epoch",
    "user_id",
    "sequence",
    "conversation_id",
    "chat_id",
    "attachment_id",
    "conversation_trace_id",
    "upload_trace_id",
    "sse_trace_id",
    "image_path",
    "image_category",
    "image_set_type",
    "image_width",
    "image_height",
    "image_bytes",
    "question_id",
    "prompt_profile",
    "conversation_reused",
    "conversation_status",
    "upload_status",
    "sse_status",
    "conversation_ms",
    "upload_ms",
    "sse_headers_ms",
    "ttft_ms",
    "sse_total_ms",
    "transaction_ms",
    "sse_event_count",
    "content_event_count",
    "response_chars",
    "completion_marker",
    "success",
    "error_type",
    "error_message",
    "sse_debug_preview",
    "response_preview",
]


def _validate_configuration() -> None:
    errors = []
    if not BASE_URL:
        errors.append("PLAYGROUND_BASE_URL is required")
    if not TOKEN:
        errors.append("PLAYGROUND_TOKEN is required")
    if not DATASET_DIR.is_dir():
        errors.append(f"DATASET_DIR does not exist: {DATASET_DIR}")
    if not MANIFEST_PATH.is_file():
        errors.append(f"IMAGE_MANIFEST does not exist: {MANIFEST_PATH}")
    if not QUESTION_PATH.is_file():
        errors.append(f"QUESTION_FILE does not exist: {QUESTION_PATH}")
    if CONVERSATION_MODE not in {"per_user", "per_request"}:
        errors.append("CONVERSATION_MODE must be per_user or per_request")
    if errors:
        raise RuntimeError("; ".join(errors))


def _load_images() -> tuple[list[dict[str, str]], dict[str, list[dict[str, str]]]]:
    rows: list[dict[str, str]] = []
    by_category: dict[str, list[dict[str, str]]] = defaultdict(list)
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("valid", "1")).strip().lower() not in {"1", "true", "yes"}:
                continue
            image_path = DATASET_DIR / row["path"]
            if not image_path.is_file():
                continue
            row["absolute_path"] = str(image_path)
            rows.append(row)
            by_category[row.get("category", "unknown")].append(row)
    if not rows:
        raise RuntimeError("Manifest contains no valid images present under DATASET_DIR")
    return rows, dict(by_category)


def _load_questions() -> list[dict[str, Any]]:
    with QUESTION_PATH.open(encoding="utf-8") as handle:
        questions = json.load(handle)
    selected = [item for item in questions if PROMPT_PROFILE in item.get("profiles", [])]
    if not selected:
        raise RuntimeError(f"No questions found for PROMPT_PROFILE={PROMPT_PROFILE!r}")
    for item in selected:
        if not item.get("id") or not item.get("text") or float(item.get("weight", 0)) <= 0:
            raise RuntimeError(f"Invalid question entry: {item!r}")
    return selected


_validate_configuration()
IMAGES, IMAGES_BY_CATEGORY = _load_images()
CATEGORIES = sorted(IMAGES_BY_CATEGORY)
QUESTIONS = _load_questions()
QUESTION_WEIGHTS = [float(item["weight"]) for item in QUESTIONS]


class CsvRecordWriter:
    def __init__(self, path: Path):
        self.path = path
        self.queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(target=self._run, name="transaction-csv-writer", daemon=True)
        self.thread.start()

    def write(self, record: dict[str, Any]) -> None:
        self.queue.put(record)

    def close(self) -> None:
        if not self.thread:
            return
        self.queue.put(None)
        self.thread.join(timeout=30)

    def _run(self) -> None:
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=RECORD_FIELDS, extrasaction="ignore")
            writer.writeheader()
            pending = 0
            while True:
                item = self.queue.get()
                if item is None:
                    handle.flush()
                    return
                writer.writerow(item)
                pending += 1
                if pending >= 20:
                    handle.flush()
                    pending = 0


WRITER = CsvRecordWriter(RESULT_DIR / f"transactions_{os.getpid()}.csv")


@events.test_start.add_listener
def _on_test_start(environment, **kwargs):
    WRITER.start()


@events.test_stop.add_listener
def _on_test_stop(environment, **kwargs):
    WRITER.close()


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _trace_id(headers: Any) -> str:
    """Return the first common request/correlation identifier from response headers."""

    for name in ("x-request-id", "x-correlation-id", "traceparent", "request-id"):
        value = headers.get(name)
        if value:
            return str(value)
    return ""


class MultimodalPlaygroundUser(HttpUser):
    host = BASE_URL
    wait_time = constant(WAIT_SECONDS) if WAIT_SECONDS <= 0 else between(WAIT_SECONDS, WAIT_SECONDS)

    def on_start(self) -> None:
        self.local_user_id = str(uuid.uuid4())
        self.sequence = 0
        self.rng = random.Random(f"{RUN_ID}:{self.local_user_id}")
        self.conversation_id = FIXED_CONVERSATION_ID or None

    def _choose_image(self) -> dict[str, str]:
        if BALANCE_CATEGORIES:
            category = self.rng.choice(CATEGORIES)
            return self.rng.choice(IMAGES_BY_CATEGORY[category])
        return self.rng.choice(IMAGES)

    def _choose_question(self) -> dict[str, Any]:
        return self.rng.choices(QUESTIONS, weights=QUESTION_WEIGHTS, k=1)[0]

    @task
    def classify_image(self) -> None:
        self.sequence += 1
        image = self._choose_image()
        question = self._choose_question()
        started_at_utc = _now_utc()
        start_epoch = time.time()
        transaction_start = time.perf_counter()
        response_text = ""
        sse_debug_parts: list[str] = []
        event_count = 0
        content_event_count = 0
        completion_marker = False
        error_type = ""
        error_message = ""
        conversation_id = None
        attachment_id = None
        chat_id = str(uuid.uuid4())

        record: dict[str, Any] = {
            "run_id": RUN_ID,
            "target_users": TARGET_USERS,
            "started_at_utc": started_at_utc,
            "user_id": self.local_user_id,
            "sequence": self.sequence,
            "conversation_id": "",
            "chat_id": chat_id,
            "attachment_id": "",
            "conversation_trace_id": "",
            "upload_trace_id": "",
            "sse_trace_id": "",
            "image_path": image.get("path", ""),
            "image_category": image.get("category", ""),
            "image_set_type": image.get("set_type", ""),
            "image_width": image.get("width", ""),
            "image_height": image.get("height", ""),
            "image_bytes": image.get("bytes", ""),
            "question_id": question["id"],
            "prompt_profile": PROMPT_PROFILE,
            "conversation_reused": 0,
            "conversation_status": "",
            "upload_status": "",
            "sse_status": "",
            "conversation_ms": "",
            "upload_ms": "",
            "sse_headers_ms": "",
            "ttft_ms": "",
            "sse_total_ms": "",
            "transaction_ms": "",
            "sse_event_count": 0,
            "content_event_count": 0,
            "response_chars": 0,
            "completion_marker": 0,
            "success": 0,
            "error_type": "",
            "error_message": "",
            "sse_debug_preview": "",
            "response_preview": "",
        }

        try:
            headers = {"Authorization": f"Bearer {TOKEN}"}
            if self.conversation_id:
                conversation_id = self.conversation_id
                record["conversation_reused"] = 1
                record["conversation_status"] = "reused"
                record["conversation_ms"] = 0
                record["conversation_id"] = conversation_id
            else:
                step_start = time.perf_counter()
                with self.client.post(
                    CONVERSATIONS_ENDPOINT,
                    headers=headers,
                    timeout=TIMEOUT_SECONDS,
                    catch_response=True,
                    name="/api/playground/conversations",
                ) as response:
                    record["conversation_ms"] = round(
                        (time.perf_counter() - step_start) * 1000, 3
                    )
                    record["conversation_status"] = response.status_code
                    record["conversation_trace_id"] = _trace_id(response.headers)
                    if response.status_code < 200 or response.status_code >= 300:
                        error_type = "conversation_http_error"
                        response_detail = str(
                            response.text or getattr(response, "error", "") or ""
                        )
                        error_message = (
                            f"HTTP {response.status_code}: {safe_preview(response_detail, 300)}"
                        )
                        response.failure(error_message)
                        return
                    try:
                        conversation_id = extract_api_id(
                            response.json(), ("conversation_id", "id", "_id")
                        )
                    except ValueError as exc:
                        error_type = "conversation_invalid_json"
                        error_message = str(exc)
                        response.failure(error_type)
                        return
                    if not conversation_id:
                        error_type = "conversation_missing_id"
                        error_message = safe_preview(str(response.text or ""), 300)
                        response.failure(error_type)
                        return
                    response.success()
                record["conversation_id"] = conversation_id
                if CONVERSATION_MODE == "per_user":
                    self.conversation_id = conversation_id

            attachment_path = ATTACHMENT_ENDPOINT.format(
                conversation_id=conversation_id, chat_id=chat_id
            )
            image_path = Path(image["absolute_path"])
            mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
            step_start = time.perf_counter()
            with image_path.open("rb") as image_handle:
                with self.client.post(
                    attachment_path,
                    headers=headers,
                    files={"attachment": (image_path.name, image_handle, mime_type)},
                    timeout=TIMEOUT_SECONDS,
                    catch_response=True,
                    name="/api/playground/conversations/{conversation_id}/chat/{chat_id}/attachment",
                ) as response:
                    record["upload_ms"] = round((time.perf_counter() - step_start) * 1000, 3)
                    record["upload_status"] = response.status_code
                    record["upload_trace_id"] = _trace_id(response.headers)
                    if response.status_code < 200 or response.status_code >= 300:
                        error_type = "upload_http_error"
                        error_message = (
                            f"HTTP {response.status_code}: "
                            f"{safe_preview(str(response.text or ''), 300)}"
                        )
                        response.failure(error_message)
                        return
                    try:
                        attachment_id = extract_api_id(
                            response.json(), ("attachment_id", "id", "_id")
                        )
                    except ValueError as exc:
                        error_type = "upload_invalid_json"
                        error_message = str(exc)
                        response.failure(error_type)
                        return
                    if not attachment_id:
                        error_type = "upload_missing_id"
                        error_message = safe_preview(str(response.text or ""), 300)
                        response.failure(error_type)
                        return
                    response.success()
                    record["attachment_id"] = attachment_id

            payload = {
                "parent_id": conversation_id,
                "chat_id": chat_id,
                "prompt": question["text"],
                "message": "",
                "attachments": [attachment_id],
            }
            sse_path = SSE_ENDPOINT.format(conversation_id=conversation_id)
            sse_start = time.perf_counter()
            with self.client.post(
                sse_path,
                headers={**headers, "Content-Type": "application/json"},
                json=payload,
                stream=True,
                timeout=TIMEOUT_SECONDS,
                catch_response=True,
                name="/api/playground/conversations/{conversation_id}/chat/sse",
            ) as response:
                record["sse_headers_ms"] = round((time.perf_counter() - sse_start) * 1000, 3)
                record["sse_status"] = response.status_code
                record["sse_trace_id"] = _trace_id(response.headers)
                if response.status_code < 200 or response.status_code >= 300:
                    error_type = "sse_http_error"
                    error_message = (
                        f"HTTP {response.status_code}: {safe_preview(str(response.text or ''), 300)}"
                    )
                    response.failure(error_message)
                    return

                for raw_line in response.iter_lines(decode_unicode=True, chunk_size=1):
                    if not raw_line:
                        continue
                    if isinstance(raw_line, bytes):
                        raw_line = raw_line.decode("utf-8", errors="replace")
                    line = raw_line.strip()
                    if line.startswith("data:"):
                        data = line[5:].strip()
                    elif ALLOW_RAW_STREAM and not line.startswith(("event:", "id:", "retry:", ":")):
                        data = line
                    else:
                        continue

                    event_count += 1
                    if CAPTURE_RESPONSE and len(sse_debug_parts) < 5:
                        sse_debug_parts.append(safe_preview(data, 200))
                    parsed = parse_sse_data(data)
                    if parsed["error"]:
                        error_type = "sse_error_event"
                        error_message = safe_preview(parsed["error"], 300)
                        response.failure(error_message)
                        return
                    text = parsed["text"]
                    if text:
                        if not response_text:
                            record["ttft_ms"] = round(
                                (time.perf_counter() - sse_start) * 1000, 3
                            )
                        content_event_count += 1
                        response_text += text
                    if parsed["done"]:
                        completion_marker = True
                        break

                record["sse_total_ms"] = round((time.perf_counter() - sse_start) * 1000, 3)
                if not response_text:
                    error_type = "sse_no_content"
                    error_message = "Stream ended without model content"
                    response.failure(error_message)
                    return
                if REQUIRE_DONE and not completion_marker:
                    error_type = "sse_missing_done_marker"
                    error_message = "Stream ended without a recognized completion marker"
                    response.failure(error_message)
                    return
                response.success()

            record["success"] = 1
        except Exception as exc:
            error_type = error_type or f"client_{type(exc).__name__}"
            error_message = error_message or str(exc)
        except BaseException as exc:
            # Locust/gevent stops active users with GreenletExit, which does not
            # inherit from Exception. Record the interrupted request as failed.
            error_type = error_type or f"client_interrupted_{type(exc).__name__}"
            error_message = error_message or "Request interrupted while Locust stopped the user"
            raise
        finally:
            transaction_ms = (time.perf_counter() - transaction_start) * 1000
            end_epoch = time.time()
            record.update(
                {
                    "ended_at_utc": _now_utc(),
                    "end_epoch": f"{end_epoch:.6f}",
                    "start_epoch": f"{start_epoch:.6f}",
                    "transaction_ms": round(transaction_ms, 3),
                    "sse_event_count": event_count,
                    "content_event_count": content_event_count,
                    "response_chars": len(response_text),
                    "completion_marker": int(completion_marker),
                    "error_type": error_type,
                    "error_message": safe_preview(error_message, 500),
                    "sse_debug_preview": safe_preview(" | ".join(sse_debug_parts), 1000)
                    if CAPTURE_RESPONSE
                    else "",
                    "response_preview": safe_preview(response_text, PREVIEW_CHARS)
                    if CAPTURE_RESPONSE
                    else "",
                }
            )
            if error_type:
                record["success"] = 0
            WRITER.write(record)
            transaction_exception = RuntimeError(error_type) if error_type else None
            events.request.fire(
                request_type="TXN",
                name="multimodal_e2e",
                response_time=transaction_ms,
                response_length=len(response_text),
                exception=transaction_exception,
                context={},
            )
