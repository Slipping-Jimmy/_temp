import json
import tempfile
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inspect_sse import content_value, data_value, read_samples, render_report


class InspectSseTests(unittest.TestCase):
    def test_text_tokens_keep_numbers_and_literal_symbols(self):
        parts = ["[answer]", "↵", "S", "64", "04", "40", "9.png", "␣不符合", "[END]"]
        self.assertEqual("".join(map(content_value, parts)), "[answer]↵S6404409.png␣不符合")
        self.assertEqual(data_value("data:  leading "), " leading ")
        self.assertIsNone(data_value(": ping"))
        self.assertEqual(content_value('[ERROR]error'), "")

    def test_structured_content_and_completion(self):
        self.assertEqual(content_value('{"choices":[{"delta":{"content":"符合"}}]}'), "符合")
        self.assertEqual(content_value('{"done":true}'), "")

    def test_report_keeps_wire_lines_and_escapes_html(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            events = [
                {"chat_id": "c", "elapsed_ms": 1, "line": "data: <script>"},
                {"chat_id": "c", "elapsed_ms": 2, "line": "data: 123"},
                {"chat_id": "c", "elapsed_ms": 3, "line": "data: [END]"},
            ]
            (root / "sse_raw.jsonl").write_text("\n".join(map(json.dumps, events)), encoding="utf-8")
            samples = read_samples(root)
            self.assertEqual(samples[0]["content"], "<script>123")
            report = render_report(samples)
            self.assertIn("&lt;script&gt;123", report)
            self.assertNotIn("<script>", report)
            self.assertIn("data: [END]", report)
