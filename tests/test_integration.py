import unittest
from inference import query_model

class TestDeepSeekV3Integration(unittest.TestCase):

    def setUp(self):
        self.model_str = "deepseek-chat"
        self.prompt = "Test prompt for DeepSeek V3"
        self.system_prompt = "System prompt for DeepSeek V3"
        self.api_key = "test-api-key"

    def test_query_model_with_deepseek_v3(self):
        try:
            response = query_model(
                model_str=self.model_str,
                prompt=self.prompt,
                system_prompt=self.system_prompt,
                openai_api_key=self.api_key
            )
            self.assertIsNotNone(response)
            self.assertIsInstance(response, str)
        except Exception as e:
            self.fail(f"query_model raised an exception: {e}")

    def test_tokenization_with_deepseek_v3(self):
        try:
            response = query_model(
                model_str=self.model_str,
                prompt=self.prompt,
                system_prompt=self.system_prompt,
                openai_api_key=self.api_key
            )
            self.assertIsNotNone(response)
            self.assertIsInstance(response, str)
        except Exception as e:
            self.fail(f"query_model raised an exception: {e}")

    def test_error_handling_with_deepseek_v3(self):
        invalid_model_str = "invalid-deepseek-chat"
        with self.assertRaises(Exception):
            query_model(
                model_str=invalid_model_str,
                prompt=self.prompt,
                system_prompt=self.system_prompt,
                openai_api_key=self.api_key
            )

if __name__ == "__main__":
    unittest.main()
