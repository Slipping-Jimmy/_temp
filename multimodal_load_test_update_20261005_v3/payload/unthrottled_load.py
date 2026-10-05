#!/usr/bin/env python3
"""Open-loop and burst load generator for one authenticated playground vendor.

Unlike Locust's default closed-loop users, this tool can schedule requests without
waiting for earlier requests to finish. It is intended to find the point where a
single vendor's concurrent requests hit the application's 60-second cutoff.
"""

from __future__ import annotations

import argparse
import csv
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable

import requests

from loadtest_lib import extract_api_id, parse_sse_data, percentile, safe_preview


CONVERSATIONS_ENDPOINT = "/api/playground/conversations"
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
    "started_at_utc",
    "ended_at_utc",
    "scheduled_epoch",
    "start_epoch",
    "end_epoch",
    "inflight_at_dispatch",
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


def is_60s_cutoff_candidate(record: dict[str, Any]) -> bool:
    """Identify records consistent with the product's 60-second response cutoff.

    This is deliberately a *candidate* rather than asserting root cause: a reverse
    proxy or browser gateway can close a stream at nearly the same point.  The raw
    error and exact timing remain in transactions.csv for server-side correlation.
    """

    message = str(record.get("error_message", "")).lower()
    error_type = str(record.get("error_type", ""))
    duration_ms = record_duration_ms(record)
    timeout_words = ("timeout", "time out", "逾時", "超時", "60 秒", "60秒")
    return bool(
        (record.get("success") != 1 and duration_ms >= 55_000)
        or (record.get("success") != 1 and any(word in message for word in timeout_words))
        or (error_type == "sse_model_communication_error" and duration_ms >= 55_000)
    )


@dataclass(frozen=True)
class Settings:
    base_url: str
    token: str
    dataset_dir: Path
    manifest_path: Path
    question_path: Path
    prompt_profile: str
    timeout_seconds: float
    require_done: bool
    allow_raw_stream: bool
    capture_response: bool
    preview_chars: int
    conversation_strategy: str
    fixed_conversation_id: str


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
        require_done=values.get("REQUIRE_DONE_MARKER", "1").strip().lower() in {"1", "true", "yes", "on"},
        allow_raw_stream=values.get("ALLOW_RAW_STREAM", "0").strip().lower() in {"1", "true", "yes", "on"},
        capture_response=values.get("CAPTURE_RESPONSE", "1").strip().lower() in {"1", "true", "yes", "on"},
        preview_chars=int(values.get("RESPONSE_PREVIEW_CHARS", "500")),
        conversation_strategy=strategy,
        fixed_conversation_id=values.get("FIXED_CONVERSATION_ID", "").strip(),
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

    def write(self, record: dict[str, Any]) -> None:
        with self.lock:
            self.writer.writerow(record)
            self.handle.flush()

    def close(self) -> None:
        with self.lock:
            self.handle.close()


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
        start_gate: threading.Event | None = None,
    ) -> dict[str, Any]:
        if start_gate is not None:
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
            "started_at_utc": started_at_utc,
            "scheduled_epoch": f"{scheduled_epoch:.6f}",
            "start_epoch": f"{start_epoch:.6f}",
            "inflight_at_dispatch": inflight_at_dispatch,
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
                    return record
                conversation_id = extract_api_id(response.json(), ("conversation_id", "id", "_id"))
                if not conversation_id:
                    error_type = "conversation_missing_id"
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
                error_message = f"HTTP {response.status_code}: {short_error(response)}"
                return record
            attachment_id = extract_api_id(response.json(), ("attachment_id", "id", "_id")) or ""
            if not attachment_id:
                error_type = "upload_missing_id"
                error_message = short_error(response)
                return record
            record["attachment_id"] = attachment_id

            payload = {
                "parent_id": conversation_id,
                "chat_id": chat_id,
                "prompt": question["text"],
                "message": "",
                "attachments": [attachment_id],
            }
            step = time.perf_counter()
            with session.post(
                self.settings.base_url + SSE_ENDPOINT.format(conversation_id=conversation_id),
                headers={"Content-Type": "application/json"},
                json=payload,
                stream=True,
                timeout=self.settings.timeout_seconds,
            ) as response:
                record["sse_headers_ms"] = round((time.perf_counter() - step) * 1000, 3)
                record["sse_status"] = response.status_code
                record["sse_trace_id"] = trace_id(response.headers)
                if not response.ok:
                    error_type = "sse_http_error"
                    error_message = f"HTTP {response.status_code}: {short_error(response)}"
                    return record
                for raw_line in response.iter_lines(decode_unicode=True, chunk_size=1):
                    if not raw_line:
                        continue
                    if isinstance(raw_line, bytes):
                        raw_line = raw_line.decode("utf-8", errors="replace")
                    line = raw_line.strip()
                    if line.startswith("data:"):
                        data = line[5:].strip()
                    elif self.settings.allow_raw_stream and not line.startswith(("event:", "id:", "retry:", ":")):
                        data = line
                    else:
                        continue
                    event_count += 1
                    if self.settings.capture_response and len(sse_debug) < 5:
                        sse_debug.append(safe_preview(data, 200))
                    parsed = parse_sse_data(data)
                    if parsed["error"]:
                        error_type = classify_sse_error(parsed["error"])
                        error_message = safe_preview(parsed["error"], 500)
                        return record
                    if parsed["text"]:
                        if not response_text:
                            record["ttft_ms"] = round((time.perf_counter() - step) * 1000, 3)
                        response_text += parsed["text"]
                        content_event_count += 1
                    if parsed["done"]:
                        completion_marker = True
                        break
                record["sse_total_ms"] = round((time.perf_counter() - step) * 1000, 3)
                if not response_text:
                    error_type = "sse_no_content"
                    error_message = "Stream ended without model content"
                    return record
                if self.settings.require_done and not completion_marker:
                    error_type = "sse_missing_done_marker"
                    error_message = "Stream ended without a recognized completion marker"
                    return record
                record["success"] = 1
        except requests.RequestException as exc:
            error_type = f"client_{type(exc).__name__}"
            error_message = str(exc)
        except Exception as exc:  # Includes invalid API JSON and local I/O failures.
            error_type = f"client_{type(exc).__name__}"
            error_message = str(exc)
        finally:
            end_epoch = time.time()
            record.update(
                {
                    "ended_at_utc": utc_now(),
                    "end_epoch": f"{end_epoch:.6f}",
                    "transaction_ms": round((time.perf_counter() - started) * 1000, 3),
                    "sse_event_count": event_count,
                    "content_event_count": content_event_count,
                    "response_chars": len(response_text),
                    "completion_marker": int(completion_marker),
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
    errors = Counter(record.get("error_type") or "unknown" for record in records if record.get("success") != 1)
    cutoff_candidates = [record for record in records if is_60s_cutoff_candidate(record)]
    result: dict[str, Any] = {
        "stage": stage,
        "mode": mode,
        "target_value": target_value,
        "scheduled": scheduled,
        "admitted": len(records),
        "admission_limited": admission_limited,
        "successes": len(successes),
        "failures": len(records) - len(successes),
        "cutoff_60s_candidates": len(cutoff_candidates),
        "error_rate_percent": (len(records) - len(successes)) / len(records) * 100 if records else 100.0,
        "scheduled_rps": scheduled / run_window_seconds if run_window_seconds else 0.0,
        "admitted_rps": len(records) / run_window_seconds if run_window_seconds else 0.0,
        "successful_rps": len(successes) / run_window_seconds if run_window_seconds else 0.0,
        "run_window_seconds": run_window_seconds,
        "drain_seconds": drain_seconds,
        "max_observed_inflight": max_inflight,
        "error_counts": json.dumps(errors, ensure_ascii=False, sort_keys=True),
    }
    for field in ("ttft_ms", "sse_total_ms", "transaction_ms"):
        values = numeric_values(successes, field)
        result[f"p50_{field}"] = percentile(values, 50)
        result[f"p95_{field}"] = percentile(values, 95)
        result[f"p99_{field}"] = percentile(values, 99)
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
        "60s candidates | Success RPS | Max in-flight | p95 TTFT ms | p95 E2E ms |"
    )
    report.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summaries:
        report.append(
            f"| {row['stage']} | {row['mode']} | {row['target_value']} | {row['scheduled']} | "
            f"{row['admitted']} | {row['successes']} | {row['error_rate_percent']:.2f} | "
            f"{row['cutoff_60s_candidates']} | {row['successful_rps']:.3f} | {row['max_observed_inflight']} | "
            f"{format_metric(row.get('p95_ttft_ms'))} | {format_metric(row.get('p95_transaction_ms'))} |"
        )
        if row["error_counts"] != "{}":
            report.append(f"  - errors: `{row['error_counts']}`")
        if row["cutoff_60s_candidates"]:
            report.append(f"  - 60-second cutoff candidates: **{row['cutoff_60s_candidates']}**")
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
            "- `open_loop` schedules at its target requests/s even while earlier jobs are still running.",
            "- `admission_limited > 0` means the client safety cap was reached; increase `--max-inflight` before treating that stage as a server capacity result.",
            "- A `cutoff_60s_candidates` value is a request that failed at roughly 60 seconds or reported a timeout. Confirm its cause with the API/vLLM logs using the recorded trace IDs.",
        ]
    )
    (result_root / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


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
    conversations = create_conversation_pool(client, concurrency, client.settings)
    records: list[dict[str, Any]] = []
    records_lock = threading.Lock()
    inflight = InflightCounter()
    sequence = 0
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="burst-request") as executor:
        for round_number in range(1, rounds + 1):
            gate = threading.Event()
            futures: list[Future[dict[str, Any]]] = []
            scheduled_epoch = time.time()
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
                    scheduled_epoch=scheduled_epoch,
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
            gate.set()
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

    burst = subparsers.add_parser("burst", help="Send N requests at once, then wait for the round.")
    burst.add_argument("--concurrency", required=True, help='For example: "8 16 24 32"')
    burst.add_argument("--rounds", type=int, default=3)
    burst.add_argument("--between-rounds", default="10s")

    open_loop = subparsers.add_parser("open-loop", help="Send at a fixed request rate without waiting.")
    open_loop.add_argument("--arrival-rates", required=True, help='For example: "2 4 6"')
    open_loop.add_argument("--duration", default="5m")
    open_loop.add_argument("--max-inflight", type=int, default=128)
    open_loop.add_argument("--conversation-pool-size", type=int, default=1)
    open_loop.add_argument("--drain-timeout", default="150s")

    args = parser.parse_args()
    if args.mode == "burst" and args.rounds < 1:
        parser.error("--rounds must be at least 1")
    if args.mode == "open-loop" and (args.max_inflight < 1 or args.conversation_pool_size < 1):
        parser.error("--max-inflight and --conversation-pool-size must be positive")

    config_file = args.config.expanduser().resolve()
    if not config_file.is_file():
        parser.error(f"Configuration not found: {config_file}")
    settings = load_settings(config_file, args.conversation_strategy)
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
            "PLAYGROUND_BASE_URL": settings.base_url,
            "PROMPT_PROFILE": settings.prompt_profile,
            "REQUEST_TIMEOUT_SECONDS": str(settings.timeout_seconds),
            "REQUIRE_DONE_MARKER": str(int(settings.require_done)),
            "CONVERSATION_STRATEGY": settings.conversation_strategy,
            "FIXED_CONVERSATION_ID_SET": str(int(bool(settings.fixed_conversation_id))),
            "MODE": args.mode,
            "TARGETS": args.concurrency if args.mode == "burst" else args.arrival_rates,
            "ROUNDS": str(args.rounds) if args.mode == "burst" else "",
            "BETWEEN_ROUNDS": str(args.between_rounds) if args.mode == "burst" else "",
            "DURATION": str(args.duration) if args.mode == "open-loop" else "",
            "MAX_INFLIGHT": str(args.max_inflight) if args.mode == "open-loop" else "",
            "CONVERSATION_POOL_SIZE": str(args.conversation_pool_size) if args.mode == "open-loop" else "",
        },
    )
    shutil.copy2(settings.manifest_path, result_root / "images_manifest.csv")
    shutil.copy2(settings.question_path, result_root / "questions.json")

    summaries: list[dict[str, Any]] = []
    if args.mode == "burst":
        values = parse_number_list(args.concurrency, integer=True)
        between_rounds = parse_duration(args.between_rounds)
        for index, raw_value in enumerate(values, 1):
            concurrency = int(raw_value)
            stage = f"stage_{index:03d}_burst_{concurrency}c"
            stage_dir = result_root / stage
            writer = CsvWriter(stage_dir / "transactions.csv")
            client = TransactionClient(settings, images, questions, weights, writer, args.run_id)
            print(f"[{utc_now()}] Starting {stage}: {args.rounds} bursts of {concurrency} requests")
            try:
                records, scheduled, limited, window, drain, max_seen = run_burst_stage(
                    client,
                    stage=stage,
                    concurrency=concurrency,
                    rounds=args.rounds,
                    between_rounds=between_rounds,
                )
            finally:
                writer.close()
            summaries.append(
                stage_summary(
                    stage=stage,
                    mode="burst",
                    target_value=str(concurrency),
                    records=records,
                    scheduled=scheduled,
                    admission_limited=limited,
                    run_window_seconds=window,
                    drain_seconds=drain,
                    max_inflight=max_seen,
                )
            )
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
