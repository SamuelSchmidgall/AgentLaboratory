import unittest

import inference
from tokenization import encoding_name_for_model, normalize_model_name


class TokenizationTests(unittest.TestCase):
    def test_normalize_model_name_handles_repo_aliases(self):
        self.assertEqual(normalize_model_name("gpt4o"), "gpt-4o")
        self.assertEqual(normalize_model_name("gpt4omini"), "gpt-4o-mini")
        self.assertEqual(normalize_model_name("claude-3.5-sonnet"), "claude-3-5-sonnet")

    def test_encoding_name_for_model_uses_expected_fallbacks(self):
        self.assertEqual(encoding_name_for_model("deepseek-chat"), "cl100k_base")
        self.assertEqual(encoding_name_for_model("o3-mini"), "o200k_base")
        self.assertEqual(encoding_name_for_model("cl100k_base"), "cl100k_base")


class CostEstimationTests(unittest.TestCase):
    def setUp(self):
        inference.TOKENS_IN.clear()
        inference.TOKENS_OUT.clear()

    def test_curr_cost_est_ignores_unknown_models(self):
        inference.TOKENS_IN["unknown-model"] = 1000
        inference.TOKENS_OUT["unknown-model"] = 500

        self.assertEqual(inference.curr_cost_est(), 0.0)


if __name__ == "__main__":
    unittest.main()
