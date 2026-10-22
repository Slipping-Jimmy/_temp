import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unthrottled_load import build_sse_payload, load_settings, parse_llm_argument


class LlmArgumentTests(unittest.TestCase):
    def test_gemma4_defaults_and_boolean(self):
        args = parse_llm_argument(None)
        self.assertEqual(args["temperature"], 1.0)
        self.assertEqual(args["top_p"], 0.95)
        self.assertEqual(args["frequency_penalty"], 0.2)
        self.assertEqual(set(args), {"max_tokens", "temperature", "top_p", "frequency_penalty",
                                     "presence_penalty", "enable_thinking"})
        self.assertEqual(args["max_tokens"], 1024)
        self.assertIs(args["enable_thinking"], False)

    def test_override_penalty_for_control_experiment(self):
        args = parse_llm_argument('{"frequency_penalty":0.0}')
        self.assertEqual(args["frequency_penalty"], 0.0)
        self.assertEqual(args["max_tokens"], 1024)

    def test_bad_values_are_rejected_before_network_requests(self):
        for raw in ('[]', '{"max_tokens":0}', '{"max_tokens":true}',
                    '{"temperature":2}', '{"top_p":0}', '{"top_k":0}',
                    '{"repetition_penalty":0}', '{"enable_thinking":"false"}',
                    '{"frequency_penalty":3}', '{"frequency_penalty":-0.1}',
                    '{"repeat_penalty":1.1}', '{"repetition_penalty":1.05}', '{"top_k":64}',
                    '{"temperature":NaN}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_llm_argument(raw)

    def test_payload_preserves_ids_and_uses_nested_arguments(self):
        args = parse_llm_argument(None)
        payload = build_sse_payload("conversation-A", "chat-B", "只回答符合或不符合", "attachment-C", args)
        self.assertEqual(payload["parent_id"], "conversation-A")
        self.assertEqual(payload["chat_id"], "chat-B")
        self.assertEqual(payload["attachments"], ["attachment-C"])
        self.assertEqual(payload["llm_argument"], args)
        self.assertEqual(json.loads(json.dumps(payload))["llm_argument"]["enable_thinking"], False)
        payload["llm_argument"]["temperature"] = 0.3
        self.assertEqual(args["temperature"], 1.0)

    def test_existing_env_gets_defaults_and_can_override_without_shell_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data/tono").mkdir(parents=True)
            (root / "images_manifest.csv").touch()
            (root / "questions.json").write_text("[]", encoding="utf-8")
            config = root / ".env"
            initial = "PLAYGROUND_BASE_URL=http://example.invalid\nPLAYGROUND_TOKEN=test-only\n"
            config.write_text(initial, encoding="utf-8")
            self.assertEqual(load_settings(config, None).llm_argument["max_tokens"], 1024)
            config.write_text(initial + 'LLM_ARGUMENT_JSON=\'{"frequency_penalty":0.3}\'\n', encoding="utf-8")
            self.assertEqual(load_settings(config, None).llm_argument["frequency_penalty"], 0.3)
