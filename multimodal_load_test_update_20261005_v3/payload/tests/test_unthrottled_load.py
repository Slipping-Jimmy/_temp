from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unthrottled_load import (  # noqa: E402
    is_60s_cutoff_candidate,
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
        self.assertTrue(
            is_60s_cutoff_candidate(
                {"success": 0, "transaction_ms": 100, "error_message": "upstream timeout"}
            )
        )
        self.assertFalse(is_60s_cutoff_candidate({"success": 0, "transaction_ms": 300}))

    def test_stage_summary_keeps_cutoff_count(self):
        summary = stage_summary(
            stage="stage_001",
            mode="burst",
            target_value="8",
            records=[
                {"success": 1, "ttft_ms": 100, "sse_total_ms": 200, "transaction_ms": 220},
                {"success": 0, "transaction_ms": 60_000, "error_type": "sse_error_event"},
            ],
            scheduled=2,
            admission_limited=0,
            run_window_seconds=2,
            drain_seconds=0,
            max_inflight=2,
        )
        self.assertEqual(summary["cutoff_60s_candidates"], 1)
        self.assertEqual(summary["successes"], 1)


if __name__ == "__main__":
    unittest.main()
