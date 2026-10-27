from __future__ import annotations

import json
import sys
import threading
import time
import unittest
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sse_stream import AbsoluteSSETimeout, DeadlineSSE
from unthrottled_load import PreparedRequest, TransactionClient, stage_summary
from loadtest_lib import parse_sse_data


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        request_body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        if self.path.endswith("/chat/sse") and request_body:
            case = json.loads(request_body).get("test_case")
            if case:
                self.path = "/" + case
        if self.path == "/api/playground/conversations" or self.path.endswith("/attachment"):
            body = json.dumps("a" if self.path.endswith("/attachment") else "c").encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/stall-headers":
            time.sleep(0.3)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        if self.path == "/stall-body":
            time.sleep(0.3)
            return
        if self.path == "/framing":
            body = ('data: [answer]\n\ndata: ↵␣\n\n: ping\n\n'
                    'data: 123\n\nevent: message\ndata: {"choices": [\n'
                    'data: {"delta":{"content":"符合"}}]}\n\nevent: completed\n\n')
        elif self.path == "/empty":
            body = ": ping\n\n"
        elif self.path == "/api/playground/conversations/c/chat/sse":
            body = ('data: [answer]\n\ndata: ↵␣\n\ndata: 不\n\ndata: 符合\n\n'
                    'event: completed\n\n')
        elif self.path == "/length":
            body = 'data: {"choices":[{"delta":{"content":"符合"},"finish_reason":"length"}]}\n\n'
        else:
            body = ""
        try:
            self.wfile.write(body.encode("utf-8"))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


class MemoryWriter:
    def __init__(self):
        self.lines = []

    def write_sse_line(self, record, ms, line):
        self.lines.append(line)

    def write(self, record):
        self.record = dict(record)


class SseStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.server.daemon_threads = True
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=1)

    def test_framing_numbers_utf8_and_event_only_completion(self):
        captured = []
        with requests.Session() as session:
            with DeadlineSSE(session, self.base + "/framing", total_seconds=2, timeout=2) as stream:
                events = [parse_sse_data(data, name) for data, name, _ in
                          stream.events(on_line=lambda line, ms: captured.append(line))]
                self.assertEqual(stream.heartbeat_lines, 1)
        self.assertEqual("".join(e["text"] for e in events), "[answer]↵␣123符合")
        self.assertTrue(events[-1]["done"])
        self.assertIn("event: completed", captured)

    def test_absolute_deadline_covers_headers_and_idle_body(self):
        for endpoint in ("/stall-headers", "/stall-body"):
            with self.subTest(endpoint=endpoint), requests.Session() as session:
                started = time.perf_counter()
                with self.assertRaises(AbsoluteSSETimeout):
                    with DeadlineSSE(session, self.base + endpoint, total_seconds=.06, timeout=2) as stream:
                        list(stream.events(on_line=lambda *args: None))
                self.assertLess(time.perf_counter() - started, .25)

    def test_prepared_request_uses_real_http_and_named_completion(self):
        settings = SimpleNamespace(base_url=self.base, timeout_seconds=2, sse_max_total_seconds=2,
                                   capture_response=True, require_done=True, allow_raw_stream=False,
                                   preview_chars=500, token="test")
        writer = MemoryWriter()
        client = TransactionClient(settings, [], [], [], writer, "test")
        prepared = PreparedRequest({"conversation_id": "c", "chat_id": "h", "round": 1,
                                    "sequence": 1, "ttft_ms": "", "success": 0}, {})
        record = client.run_prepared_sse(prepared, release=SimpleNamespace(wait=time.time), inflight_at_dispatch=1)
        self.assertEqual(record["success"], 1)
        self.assertEqual(record["normalized_answer"], "不符合")
        self.assertEqual(record["completion_marker"], 1)
        self.assertEqual(record["sse_data_line_count"], 4)
        self.assertIn("event: completed", writer.lines)
        self.assertIn("", writer.lines)
        client.session().close()

    def test_all_request_ttft_includes_failed_requests(self):
        result = stage_summary(stage="s", mode="sse_burst", target_value="3", records=[
            {"success": 1, "ttft_ms": 100, "transaction_ms": 200},
            {"success": 0, "ttft_ms": 70000, "error_type": "sse_error_event"},
            {"success": 0, "ttft_ms": "", "error_type": "sse_no_content"}],
            scheduled=3, admission_limited=0, run_window_seconds=1, drain_seconds=0, max_inflight=3)
        self.assertEqual(result["max_ttft_ms"], 100)
        self.assertEqual(result["max_all_ttft_ms"], 70000)
        self.assertEqual(result["no_first_token_count"], 1)
        self.assertEqual(result["late_first_token_count"], 1)

    def test_empty_response_and_length_limit_are_distinct_failures(self):
        for case, expected in (("empty", "sse_no_content"), ("length", "sse_output_length_limit")):
            with self.subTest(case=case):
                settings = SimpleNamespace(base_url=self.base, timeout_seconds=2, sse_max_total_seconds=2,
                                           capture_response=True, require_done=True, allow_raw_stream=False,
                                           preview_chars=500, token="test")
                client = TransactionClient(settings, [], [], [], MemoryWriter(), "test")
                record = client.run_prepared_sse(PreparedRequest({
                    "conversation_id": "c", "chat_id": "h", "round": 1, "sequence": 1,
                    "ttft_ms": "", "success": 0}, {"test_case": case}),
                    release=SimpleNamespace(wait=time.time), inflight_at_dispatch=1)
                self.assertEqual(record["success"], 0)
                self.assertEqual(record["error_type"], expected)
                if case == "empty":
                    self.assertEqual(record["sse_data_line_count"], 0)
                    self.assertEqual(record["sse_heartbeat_line_count"], 1)
                    self.assertEqual(record["ttft_ms"], "")
                else:
                    self.assertEqual(record["finish_reason"], "length")
                    self.assertEqual(record["completion_marker"], 1)
                client.session().close()

    def test_full_upload_flow_preserves_conversation_and_completes(self):
        settings = SimpleNamespace(base_url=self.base, timeout_seconds=2, sse_max_total_seconds=2,
                                   capture_response=True, require_done=True, allow_raw_stream=False,
                                   preview_chars=500, token="test", conversation_strategy="per_request",
                                   prompt_profile="short", llm_argument={"presence_penalty": 0.2})
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "test.png"
            image.write_bytes(b"fixture")
            client = TransactionClient(settings, [{"absolute_path": str(image)}],
                                       [{"id": "q", "text": "test"}], [1], MemoryWriter(), "test")
            record = client.run(stage="s", mode="burst", target_value="1", sequence=1,
                                conversation_id=None, scheduled_epoch=time.time(),
                                inflight_at_dispatch=1, round_number=1)
            self.assertEqual(record["conversation_id"], "c")
            self.assertEqual(record["attachment_id"], "a")
            self.assertEqual(record["success"], 1)
            self.assertEqual(record["normalized_answer"], "不符合")
            client.session().close()
