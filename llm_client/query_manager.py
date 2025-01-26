from typing import Optional, Dict
from .models import TokenCounter
from .factory import LLMStrategyFactory
import time

class LLMQueryManager:
    def __init__(
            self,
            api_keys: Optional[Dict[str, str]] = None,
            token_counter: Optional[TokenCounter] = None
    ):
        self.token_counter = token_counter or TokenCounter()
        self.factory = LLMStrategyFactory(api_keys)

    def query_model(
            self,
            model_name: str,
            prompt: str,
            system_prompt: str,
            api_key: Optional[str] = None,
            max_retries: int = 5,
            timeout: float = 5.0,
            print_cost: bool = True,
            temperature: Optional[float] = None,
    ) -> str:
        strategy = self.factory.create_strategy(model_name)
        for attempt in range(max_retries):
            try:
                answer = strategy.query(prompt, system_prompt, api_key, temperature)

                input_tokens = strategy.count_tokens(system_prompt + prompt)
                output_tokens = strategy.count_tokens(answer)
                self.token_counter.update_counts(model_name, input_tokens, output_tokens)
                if print_cost:
                    cost = self.token_counter.calculate_cost(
                        {model_name: strategy.config}
                    )
                    print(f"Current experiment cost = ${cost:.6f}, ** Approximate values, may not reflect true cost")

                return answer
            except ValueError as e:
                if 'No API key provided for' in str(e):
                    raise ValueError("No API key provided. Please provide an API key.")
                raise e

            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Max retries reached: {str(e)}")
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                time.sleep(timeout)

        raise Exception("Max retries: timeout")