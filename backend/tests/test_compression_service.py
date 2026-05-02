import unittest
import importlib.util
from pathlib import Path

SERVICE_PATH = Path(__file__).resolve().parents[1] / "services" / "compression_service.py"
SPEC = importlib.util.spec_from_file_location("compression_service", SERVICE_PATH)
COMPRESSION_SERVICE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(COMPRESSION_SERVICE)
compress_prompt = COMPRESSION_SERVICE.compress_prompt


class TestCompressionService(unittest.TestCase):
    def test_prompt_1_filler_removal(self):
        before = (
            "I hope you are doing well. I wanted to ask you if you don't mind, "
            "could you please explain what Docker volumes are?"
        )
        result = compress_prompt(before)
        after = result["compressed_text"]
        print(f"\n[1] BEFORE: {before}\n[1] AFTER:  {after}")

        self.assertIn("explain what docker volumes are", after.lower())
        self.assertNotIn("i hope you are doing well", after.lower())
        self.assertGreaterEqual(result["tokens_saved"], 1)

    def test_prompt_2_redundant_sentence_removal(self):
        before = (
            "Please summarize this report for me. "
            "Can you summarize this report for me? "
            "Focus on key points."
        )
        result = compress_prompt(before)
        after = result["compressed_text"]
        print(f"\n[2] BEFORE: {before}\n[2] AFTER:  {after}")

        self.assertIn("focus on key points", after.lower())
        self.assertEqual(after.lower().count("summarize this report for me"), 1)

    def test_prompt_3_whitespace_normalization(self):
        before = "   Could you please    optimize this SQL query?\n\n\nIt is slow on joins.   "
        result = compress_prompt(before)
        after = result["compressed_text"]
        print(f"\n[3] BEFORE: {before}\n[3] AFTER:  {after}")

        self.assertFalse(after.startswith(" "))
        self.assertFalse(after.endswith(" "))
        self.assertNotIn("  ", after)
        self.assertNotIn("\n\n", after)

    def test_prompt_4_guardrail_keeps_meaning(self):
        before = "I would like to know Kubernetes architecture and scheduler internals in detail."
        result = compress_prompt(before)
        after = result["compressed_text"]
        print(f"\n[4] BEFORE: {before}\n[4] AFTER:  {after}")

        self.assertIn("kubernetes", after.lower())
        self.assertIn("scheduler", after.lower())
        self.assertTrue(result["compression_applied"])

    def test_prompt_5_response_shape(self):
        before = "Thank you so much, would you be able to list Python async best practices?"
        result = compress_prompt(before)
        after = result["compressed_text"]
        print(f"\n[5] BEFORE: {before}\n[5] AFTER:  {after}")

        expected_keys = {
            "original_text",
            "compressed_text",
            "original_tokens",
            "compressed_tokens",
            "tokens_saved",
            "compression_ratio",
            "compression_applied",
        }
        self.assertEqual(set(result.keys()), expected_keys)
        self.assertIsInstance(result["compression_applied"], bool)
        self.assertGreaterEqual(result["original_tokens"], result["compressed_tokens"])


if __name__ == "__main__":
    unittest.main()
