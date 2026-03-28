import os
import tiktoken
import time

from config import OLLAMA_API_BASE_URL
from model_registry import ModelRegistry
from provider import get_provider_response
from utils import remove_thinking_process

# Global registry instance
registry = ModelRegistry()

encoding = tiktoken.get_encoding("cl100k_base")


def curr_cost_est():
    return registry.curr_cost_est()


def query_model(model_str, prompt, system_prompt,
                openai_api_key=None, anthropic_api_key=None,
                tries=5, timeout=5.0,
                temp=None, print_cost=True, version="1.5"):
    # Override the API keys if provided in the function call
    if openai_api_key is not None:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    if anthropic_api_key is not None:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key

    preloaded_openai_api = os.getenv('OPENAI_API_KEY')
    preload_anthropic_api = os.getenv('ANTHROPIC_API_KEY')
    preload_google_api = os.getenv('GOOGLE_API_KEY')
    preload_deepseek_api = os.getenv('DEEPSEEK_API_KEY')

    if (preloaded_openai_api is None and
        preload_anthropic_api is None and
        preload_google_api is None and
        preload_deepseek_api is None):
        raise Exception("No API key provided in query_model function")

    # Handle Ollama passthrough
    if preloaded_openai_api == "ollama":
        return _query_ollama(model_str, prompt, system_prompt, tries, timeout, temp)

    for _ in range(tries):
        try:
            # Resolve model via registry
            canonical = registry.get_canonical_for_cost(model_str)
            provider = registry.get_provider(model_str)
            api_model_name = registry.get_api_model_name(model_str)
            base_url = registry.get_base_url(model_str)

            # Determine API key based on provider
            api_key_map = {
                "openai": os.getenv('OPENAI_API_KEY'),
                "anthropic": os.getenv('ANTHROPIC_API_KEY'),
                "google": os.getenv('GOOGLE_API_KEY'),
                "deepseek": os.getenv('DEEPSEEK_API_KEY'),
            }
            api_key = api_key_map.get(provider)
            if api_key is None:
                raise Exception(f"No API key set for provider '{provider}'")

            answer = get_provider_response(
                provider=provider,
                api_key=api_key,
                model_name=api_model_name,
                user_prompt=prompt,
                system_prompt=system_prompt,
                temperature=temp,
                base_url=base_url,
            )

            answer = remove_thinking_process(answer)

            # Cost estimation
            try:
                try:
                    model_encoding = tiktoken.encoding_for_model(canonical)
                except KeyError:
                    model_encoding = tiktoken.encoding_for_model("gpt-4o")
                if canonical not in registry.tokens_in:
                    registry.tokens_in[canonical] = 0
                    registry.tokens_out[canonical] = 0
                registry.tokens_in[canonical] += len(model_encoding.encode(system_prompt + prompt))
                registry.tokens_out[canonical] += len(model_encoding.encode(answer))
                if print_cost:
                    print(f"Current experiment cost = ${curr_cost_est()}, ** Approximate values, may not reflect true cost")
            except Exception as e:
                if print_cost:
                    print(f"Cost approximation has an error? {e}")

            return answer
        except Exception as e:
            print("Inference Exception:", e)
            time.sleep(timeout)
            continue
    raise Exception("Max retries: timeout")


def _query_ollama(model_str, prompt, system_prompt, tries, timeout, temp):
    """Handle Ollama models — bypass registry, pass model string directly."""
    from provider import OpenaiProvider
    for _ in range(tries):
        try:
            answer = OpenaiProvider.get_response(
                api_key="ollama",
                model_name=model_str,
                user_prompt=prompt,
                system_prompt=system_prompt,
                temperature=temp,
                base_url=OLLAMA_API_BASE_URL,
            )
            return remove_thinking_process(answer)
        except Exception as e:
            print("Inference Exception:", e)
            time.sleep(timeout)
            continue
    raise Exception("Max retries: timeout")
