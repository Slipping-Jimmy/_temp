"""Deadline-bounded requests streaming with SSE event framing.

The reader owns the response, while the caller waits on a bounded queue. This
keeps a stalled HTTP read (including response headers) from blocking the caller
past the wall-clock limit. Each request has one additional daemon reader.
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Any, Callable

import requests


class AbsoluteSSETimeout(TimeoutError):
    pass


class DeadlineSSE:
    def __init__(self, session: requests.Session, url: str, *, total_seconds: float,
                 timeout: float, **kwargs: Any):
        if total_seconds <= 0 or timeout <= 0:
            raise ValueError("SSE deadlines must be positive")
        self.session, self.url, self.kwargs = session, url, kwargs
        self.total_seconds, self.timeout = total_seconds, timeout
        self.queue: queue.Queue = queue.Queue(maxsize=32)
        self.cancelled = threading.Event()
        self.started = 0.0
        self.response = None
        self.nonempty_lines = self.data_lines = self.heartbeat_lines = 0
        self.last_received_ms = 0.0
        self.max_wire_gap_ms = 0.0
        self.request_started_ms = 0.0
        self.request_started_epoch = 0.0

    def _put(self, item: tuple) -> None:
        while not self.cancelled.is_set():
            try:
                self.queue.put(item, timeout=0.05)
                return
            except queue.Full:
                continue

    def _read(self) -> None:
        try:
            self.request_started_ms = self.elapsed_ms()
            self.request_started_epoch = time.time()
            # Bound lingering reads too; the caller's queue deadline is stricter
            # than requests' separate connect/read timeouts.
            with self.session.post(self.url, stream=True,
                                   timeout=min(self.timeout, self.total_seconds),
                                   **self.kwargs) as response:
                self.response = response
                if self.cancelled.is_set():
                    return
                self._put(("headers", response, self.elapsed_ms()))
                if not response.ok:
                    self._put(("http_error", f"HTTP {response.status_code}"))
                    return
                for raw in response.iter_lines(chunk_size=1, decode_unicode=False):
                    if self.cancelled.is_set():
                        break
                    received = self.elapsed_ms()
                    line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
                    if line:
                        self.nonempty_lines += 1
                        self.max_wire_gap_ms = max(self.max_wire_gap_ms, received - self.last_received_ms)
                        self.last_received_ms = received
                        self.data_lines += int(line == "data" or line.startswith("data:"))
                        self.heartbeat_lines += int(line.startswith(":"))
                    self._put(("line", line, received))
                self._put(("eof", None, self.elapsed_ms()))
        except Exception as exc:
            self._put(("error", exc))

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started) * 1000

    def _next(self) -> tuple:
        remaining = self.total_seconds - (time.perf_counter() - self.started)
        if remaining <= 0:
            raise AbsoluteSSETimeout(f"SSE exceeded absolute limit of {self.total_seconds:g}s")
        try:
            item = self.queue.get(timeout=remaining)
        except queue.Empty:
            raise AbsoluteSSETimeout(f"SSE exceeded absolute limit of {self.total_seconds:g}s") from None
        if time.perf_counter() - self.started > self.total_seconds:
            raise AbsoluteSSETimeout(f"SSE exceeded absolute limit of {self.total_seconds:g}s")
        if item[0] == "error":
            raise item[1]
        return item

    def __enter__(self):
        self.started = time.perf_counter()
        threading.Thread(target=self._read, daemon=True, name="sse-reader").start()
        try:
            item = self._next()
        except Exception:
            self.cancel()
            raise
        if item[0] != "headers":
            raise RuntimeError("Missing HTTP response headers")
        response = item[1]
        self.status_code, self.headers, self.ok = response.status_code, response.headers, response.ok
        return self

    def __exit__(self, *args):
        self.cancel()

    def cancel(self) -> None:
        self.cancelled.set()
        # Unblock a pending socket read without waiting on Response.close's read
        # lock. The reader's with-block remains responsible for closing it.
        try:
            import socket
            self.response.raw._fp.fp.raw._sock.shutdown(socket.SHUT_RDWR)
        except (AttributeError, OSError):
            pass

    def events(self, *, on_line: Callable[[str, float], None], allow_raw: bool = False):
        data: list[str] = []
        event_name = ""
        received = 0.0
        try:
            while True:
                item = self._next()
                if item[0] == "eof":
                    if data or event_name:
                        yield "\n".join(data), event_name, received
                    return
                if item[0] != "line":
                    continue
                line, timestamp = item[1], item[2]
                on_line(line, timestamp)
                if not line:
                    if data or event_name:
                        yield "\n".join(data), event_name, received
                    data, event_name = [], ""
                    continue
                if line.startswith(":"):
                    continue
                field, separator, value = line.partition(":")
                if value.startswith(" "):
                    value = value[1:]
                if field == "data":
                    data.append(value)
                    received = timestamp
                elif field == "event":
                    event_name = value
                    received = timestamp
                elif allow_raw and field not in {"id", "retry"}:
                    yield line, "", timestamp
        finally:
            self.cancel()
