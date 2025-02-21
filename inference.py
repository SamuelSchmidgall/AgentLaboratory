import os
import tiktoken
import time

from config import GOOGLE_GENERATIVE_API_BASE_URL, DEEPSEEK_API_BASE_URL, OLLAMA_API_BASE_URL
from provider import AnthropicProvider, OpenaiProvider
from utils import remove_thinking_process

TOKENS_IN = dict()
TOKENS_OUT = dict()

encoding = tiktoken.get_encoding("cl100k_base")


def curr_cost_est():
    costmap_in = {
        "gpt-4o": 2.50 / 1000000,
        "gpt-4o-mini": 0.150 / 1000000,
        "o1-preview": 15.00 / 1000000,
        "o1-mini": 3.00 / 1000000,
        "claude-3-5-sonnet-latest": 3.00 / 1000000,
        "claude-3-5-haiku-latest": 0.8 / 1000000,
        "deepseek-chat": 1.00 / 1000000,
        "o1": 15.00 / 1000000,
        "gemini-2.0-flash": 0.1 / 1000000,
        "gemini-2.0-flash-lite": 0.075 / 1000000,
    }
    costmap_out = {
        "gpt-4o": 10.00 / 1000000,
        "gpt-4o-mini": 0.6 / 1000000,
        "o1-preview": 60.00 / 1000000,
        "o1-mini": 12.00 / 1000000,
        "claude-3-5-sonnet-latest": 12.00 / 1000000,
        "claude-3-5-haiku-latest": 1.4 / 1000000,
        "deepseek-chat": 5.00 / 1000000,
        "o1": 60.00 / 1000000,
        "gemini-2.0-flash": 0.4 / 1000000,
        "gemini-2.0-flash-lite": 0.3 / 1000000,
    }
    return sum([costmap_in[_] * TOKENS_IN[_] for _ in TOKENS_IN]) + sum(
        [costmap_out[_] * TOKENS_OUT[_] for _ in TOKENS_OUT])


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

    # If no API key is provided, raise an exception
    if (preloaded_openai_api is None and
        preload_anthropic_api is None and
        preload_google_api is None and
        preload_deepseek_api is None):
        raise Exception("No API key provided in query_model function")

    for _ in range(tries):
        try:
            if model_str == "gpt-4o-mini" or model_str == "gpt4omini" or model_str == "gpt-4omini" or model_str == "gpt4o-mini":
                model_str = "gpt-4o-mini"
                answer = OpenaiProvider.get_response(
                    api_key=os.getenv('OPENAI_API_KEY'),
                    model_name="gpt-4o-mini" if version == "0.28" else "gpt-4o-mini-2024-07-18",
                    user_prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temp,
                )
            elif model_str == "gpt4o" or model_str == "gpt-4o":
                model_str = "gpt-4o"
                answer = OpenaiProvider.get_response(
                    api_key=os.getenv('OPENAI_API_KEY'),
                    model_name="gpt-4o" if version == "0.28" else "gpt-4o-2024-08-06",
                    user_prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temp,
                )
            elif model_str == "o1-mini":
                model_str = "o1-mini"
                answer = OpenaiProvider.get_response(
                    api_key=os.getenv('OPENAI_API_KEY'),
                    model_name="o1-mini" if version == "0.28" else "o1-mini-2024-09-12",
                    user_prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temp,
                )
            elif model_str == "o1":
                model_str = "o1"
                answer = OpenaiProvider.get_response(
                    api_key=os.getenv('OPENAI_API_KEY'),
                    model_name="o1" if version == "0.28" else "o1-2024-12-17",
                    user_prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temp,
                )
            elif model_str == "o1-preview":
                model_str = "o1-preview"
                answer = OpenaiProvider.get_response(
                    api_key=os.getenv('OPENAI_API_KEY'),
                    model_name="o1-preview" if version == "0.28" else "o1-preview-2024-12-17",
                    user_prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temp,
                )
            elif model_str == "claude-3-5-sonnet-latest" or model_str == "claude-3-5-haiku-latest":
                answer = AnthropicProvider.get_response(
                    api_key=os.environ["ANTHROPIC_API_KEY"],
                    model_name=model_str,
                    user_prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temp,
                )
            elif model_str == "deepseek-chat":
                model_str = "deepseek-chat"
                answer = OpenaiProvider.get_response(
                    api_key=os.getenv('DEEPSEEK_API_KEY'),
                    model_name="deepseek-chat",
                    user_prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temp,
                    base_url=DEEPSEEK_API_BASE_URL,
                )
            elif model_str == "gemini-2.0-flash":
                model_str = "gemini-2.0-flash"
                answer = OpenaiProvider.get_response(
                    api_key=os.getenv('GOOGLE_API_KEY'),
                    model_name=model_str,
                    user_prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temp,
                    base_url=GOOGLE_GENERATIVE_API_BASE_URL,
                )
            elif model_str == "gemini-2.0-flash-lite":
                model_str = "gemini-2.0-flash-lite"
                answer = OpenaiProvider.get_response(
                    api_key=os.getenv('GOOGLE_API_KEY'),
                    model_name="gemini-2.0-flash-lite-preview",
                    user_prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temp,
                    base_url=GOOGLE_GENERATIVE_API_BASE_URL,
                )
            elif preloaded_openai_api == "ollama":
                answer = OpenaiProvider.get_response(
                    api_key="ollama",
                    model_name=model_str,
                    user_prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temp,
                    base_url=OLLAMA_API_BASE_URL,
                )
            else:
                raise Exception(f"Model {model_str} not found")

            # Remove the thinking process from the answer
            answer = remove_thinking_process(answer)

            # Cost estimation when not using Ollama
            if preloaded_openai_api != "ollama":
                try:
                    if model_str in [
                        "o1-preview", "o1-mini", "o1",
                        "claude-3-5-sonnet-latest", "claude-3-5-haiku-latest",
                        "gemini-2.0-flash", "gemini-2.0-flash-lite"
                    ]:
                        model_encoding = tiktoken.encoding_for_model("gpt-4o")
                    elif model_str in ["deepseek-chat"]:
                        model_encoding = tiktoken.get_encoding("cl100k_base")
                    else:
                        model_encoding = tiktoken.encoding_for_model(model_str)
                    if model_str not in TOKENS_IN:
                        TOKENS_IN[model_str] = 0
                        TOKENS_OUT[model_str] = 0
                    TOKENS_IN[model_str] += len(model_encoding.encode(system_prompt + prompt))
                    TOKENS_OUT[model_str] += len(model_encoding.encode(answer))
                    if print_cost:
                        print(
                            f"Current experiment cost = ${curr_cost_est()}, ** Approximate values, may not reflect true cost")
                except Exception as e:
                    if print_cost:
                        print(f"Cost approximation has an error? {e}")
            return answer
        except Exception as e:
            print("Inference Exception:", e)
            time.sleep(timeout)
            continue
    raise Exception("Max retries: timeout")

# print(query_model(model_str="o1-mini", prompt="hi", system_prompt="hey"))
