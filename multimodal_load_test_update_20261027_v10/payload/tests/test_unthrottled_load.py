from __future__ import annotations

import sys
import json
import csv
import tempfile
import threading
import time
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unthrottled_load import (  # noqa: E402
    BurstRelease,
    CsvWriter,
    build_html_report,
    classify_60s_cutoff,
    has_model_content,
    is_60s_cutoff_candidate,
    normalize_binary_answer,
    parse_dotenv,
    parse_duration,
    parse_number_list,
    stage_summary,
)


class UnthrottledLoadTests(unittest.TestCase):
    def test_duration_and_lists(self):
        self.assertEqual(parse_duration("1500ms"), 1.5)
        self.assertEqual(parse_duration("2m"), 120)
        self.assertEqual(parse_number_list("1, 2 3", integer=True), [1, 2, 3])
        self.assertEqual(parse_number_list("0.5 1.25", integer=False), [0.5, 1.25])
        with self.assertRaises(ValueError):
            parse_number_list("0", integer=True)

    def test_dotenv_ignores_comments_and_keeps_quoted_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / ".env"
            path.write_text('TOKEN="abc def"\nEMPTY=\n# ignored\nRATE=2 # comment\n', encoding="utf-8")
            self.assertEqual(
                parse_dotenv(path),
                {"TOKEN": "abc def", "EMPTY": "", "RATE": "2"},
            )

    def test_cutoff_candidate_requires_timeout_evidence(self):
        self.assertTrue(
            is_60s_cutoff_candidate(
                {"success": 0, "transaction_ms": 60_012, "error_type": "sse_error_event"}
            )
        )

    def test_cutoff_uses_confirmed_first_token_limit(self):
        self.assertEqual(
            classify_60s_cutoff(
                {
                    "success": 0,
                    "sse_total_ms": 60_100,
                    "ttft_ms": "",
                    "max_sse_event_gap_ms": 60_000,
                    "failure_phase": "sse",
                }
            ),
            "first_token",
        )
        self.assertEqual(
            classify_60s_cutoff(
                {
                    "success": 0,
                    "sse_total_ms": 60_100,
                    "ttft_ms": 1_500,
                    "max_sse_event_gap_ms": 2_000,
                    "failure_phase": "sse",
                }
            ),
            "",
        )
        self.assertEqual(
            classify_60s_cutoff(
                {
                    "success": 0,
                    "sse_total_ms": 70_000,
                    "ttft_ms": 1_500,
                    "max_sse_event_gap_ms": 60_200,
                    "failure_phase": "sse",
                }
            ),
            "",
        )
        self.assertEqual(
            classify_60s_cutoff(
                {
                    "success": 0,
                    "sse_total_ms": 180_100,
                    "ttft_ms": 2_000,
                    "max_sse_event_gap_ms": 500,
                    "error_type": "client_sse_absolute_timeout",
                    "failure_phase": "sse",
                }
            ),
            "",
        )
        self.assertEqual(
            classify_60s_cutoff(
                {
                    "success": 0,
                    "sse_total_ms": 1_100,
                    "ttft_ms": 400,
                    "max_sse_event_gap_ms": 500,
                    "error_type": "sse_record_save_error",
                    "error_message": "無法儲存紀錄",
                    "failure_phase": "sse",
                }
            ),
            "",
        )
        self.assertEqual(
            classify_60s_cutoff(
                {
                    "success": 0,
                    "sse_total_ms": 60_100,
                    "ttft_ms": 1_000,
                    "max_sse_event_gap_ms": 59_000,
                    "error_type": "sse_record_save_error",
                    "error_message": "無法儲存紀錄",
                    "failure_phase": "sse",
                }
            ),
            "",
        )
        self.assertFalse(
            is_60s_cutoff_candidate(
                {"success": 0, "transaction_ms": 100, "error_message": "upstream timeout"}
            )
        )
        self.assertFalse(is_60s_cutoff_candidate({"success": 0, "transaction_ms": 300}))
        self.assertFalse(
            is_60s_cutoff_candidate(
                {
                    "success": 0,
                    "transaction_ms": 60_000,
                    "error_type": "upload_http_error",
                    "failure_phase": "upload",
                }
            )
        )
        self.assertFalse(
            is_60s_cutoff_candidate(
                {
                    "success": 0,
                    "ttft_ms": 2_000,
                    "transaction_ms": 3_053_000,
                    "response_chars": 100_000,
                    "error_type": "sse_error_event",
                    "failure_phase": "sse",
                    "error_message": "內容已達語言模型上限，請開啟新對話！",
                }
            )
        )
        self.assertTrue(
            is_60s_cutoff_candidate(
                {
                    "success": 0,
                    "ttft_ms": 60_100,
                    "transaction_ms": 60_200,
                    "failure_phase": "sse",
                }
            )
        )

    def test_binary_answer_validation_is_strict(self):
        self.assertEqual(normalize_binary_answer("[answer]\n符合"), "符合")
        self.assertEqual(normalize_binary_answer(" [ANSWER] 不符合 [END] "), "不符合")
        self.assertEqual(normalize_binary_answer("L不符合"), "")
        self.assertEqual(normalize_binary_answer("符合，因為照片清楚"), "")

    def test_protocol_markers_are_not_model_tokens(self):
        self.assertFalse(has_model_content("[answer] [THOUGHT]"))
        self.assertFalse(has_model_content("[answer]↵␣"))
        self.assertTrue(has_model_content("[answer]符合"))
        self.assertEqual(normalize_binary_answer("[answer]↵␣符合"), "符合")

    def test_raw_sse_capture_keeps_event_names_and_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "transactions.csv"
            writer = CsvWriter(path)
            record = {"round": 1, "sequence": 2, "chat_id": "test-chat", "ttft_ms": 458}
            writer.write_sse_line(record, 460, "event: custom_completed")
            writer.write_sse_line(record, 461, "data: [DONE]")
            writer.write(record)
            writer.close()
            events = [json.loads(line) for line in path.with_name("sse_raw.jsonl").read_text().splitlines()]
            self.assertEqual([event["line"] for event in events],
                             ["event: custom_completed", "data: [DONE]"])
            self.assertEqual(events[0]["chat_id"], "test-chat")
            with path.open() as handle:
                self.assertEqual(next(csv.DictReader(handle))["first_token_within_60s"], "1")

    def test_late_token_counts_even_if_answer_eventually_completes(self):
        self.assertEqual(classify_60s_cutoff({"success": 1, "ttft_ms": 60001}), "first_token")
        self.assertEqual(classify_60s_cutoff({"success": 0, "ttft_ms": 59999,
                                           "sse_total_ms": 60100}), "")

    def test_stage_summary_keeps_cutoff_count(self):
        summary = stage_summary(
            stage="stage_001",
            mode="burst",
            target_value="8",
            records=[
                {
                    "success": 1,
                    "answer_valid": 1,
                    "ttft_ms": 100,
                    "sse_total_ms": 200,
                    "transaction_ms": 220,
                },
                {"success": 0, "transaction_ms": 60_000, "error_type": "sse_error_event"},
            ],
            scheduled=2,
            admission_limited=0,
            run_window_seconds=2,
            drain_seconds=0,
            max_inflight=2,
        )
        self.assertEqual(summary["cutoff_60s_candidates"], 1)
        self.assertEqual(summary["first_token_timeout_candidates"], 1)
        self.assertEqual(summary["timeout_basis_counts"], '{"first_token": 1}')
        self.assertEqual(summary["successes"], 1)
        self.assertEqual(summary["valid_answers"], 1)
        self.assertEqual(summary["answer_valid_rate_percent"], 100)
        self.assertEqual(summary["mean_transaction_ms"], 220)

    def test_burst_release_waits_until_all_workers_are_ready(self):
        release = BurstRelease(expected=3)
        observed: list[float] = []
        workers = [threading.Thread(target=lambda: observed.append(release.wait())) for _ in range(3)]
        for worker in workers:
            worker.start()
        deadline = time.monotonic() + 1
        while release.ready < 3 and time.monotonic() < deadline:
            time.sleep(0.001)
        epoch = release.release_when_ready()
        for worker in workers:
            worker.join(timeout=1)
        self.assertEqual(observed, [epoch, epoch, epoch])

    def test_html_report_contains_capacity_chart(self):
        summary = stage_summary(
            stage="stage_001",
            mode="sse_burst",
            target_value="5",
            records=[
                {
                    "success": 1,
                    "ttft_ms": 100,
                    "sse_total_ms": 200,
                    "transaction_ms": 220,
                    "release_skew_ms": 1.5,
                }
            ],
            scheduled=1,
            admission_limited=0,
            run_window_seconds=1,
            drain_seconds=0,
            max_inflight=1,
        )
        report = build_html_report([summary], "test-run")
        self.assertIn("同步批次容量報告", report)
        self.assertIn("60 秒 timeout", report)
        self.assertIn("平均完整回答", report)


if __name__ == "__main__":
    unittest.main()
