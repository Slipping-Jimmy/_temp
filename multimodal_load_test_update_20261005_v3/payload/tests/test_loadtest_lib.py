from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loadtest_lib import extract_api_id, parse_sse_data, percentile


class LoadTestLibTests(unittest.TestCase):
    def test_extract_api_id_from_string_and_nested_dict(self):
        self.assertEqual(extract_api_id("abc"), "abc")
        self.assertEqual(extract_api_id({"data": {"conversation_id": "c-1"}}), "c-1")
        self.assertIsNone(extract_api_id({"data": {"status": "ok"}}))

    def test_parse_openai_sse(self):
        event = parse_sse_data('{"choices":[{"delta":{"content":"符合"}}]}')
        self.assertEqual(event["text"], "符合")
        self.assertFalse(event["done"])

    def test_parse_nested_message_and_token_data(self):
        event = parse_sse_data('{"message":{"content":"合格"}}')
        self.assertEqual(event["text"], "合格")
        event = parse_sse_data('{"type":"token","data":"不合格"}')
        self.assertEqual(event["text"], "不合格")

    def test_boolean_done(self):
        self.assertTrue(parse_sse_data('{"done":true}')["done"])

    def test_parse_done_and_error(self):
        self.assertTrue(parse_sse_data("[DONE]")["done"])
        event = parse_sse_data('{"error":{"message":"overloaded"}}')
        self.assertIn("overloaded", event["error"])

    def test_parse_raw_application_error(self):
        event = parse_sse_data("[ERROR]\n語言模型通訊異常或回傳內容無法解析，請稍後再試！")
        self.assertIn("語言模型通訊異常", event["error"])
        self.assertEqual(event["text"], "")

    def test_control_text_is_not_answer(self):
        self.assertEqual(parse_sse_data("ping")["text"], "")
        self.assertEqual(parse_sse_data("一般文字")["text"], "一般文字")

    def test_percentile(self):
        self.assertEqual(percentile([], 95), None)
        self.assertEqual(percentile([1], 95), 1)
        self.assertAlmostEqual(percentile([1, 2, 3, 4], 50), 2.5)


if __name__ == "__main__":
    unittest.main()
