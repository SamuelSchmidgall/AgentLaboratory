import json
import os
import time

import anthropic
import openai
from openai import OpenAI

from tokenization import count_text_tokens, normalize_model_name

TOKENS_IN = dict()
TOKENS_OUT = dict()

OPENAI_MODELS = {"gpt-4o", "gpt-4o-mini", "o1", "o1-mini", "o1-preview", "o3-mini"}
ANTHROPIC_MODELS = {"claude-3-5-sonnet"}
GEMINI_MODELS = {"gemini-1.5-pro", "gemini-2.0-pro"}
DEEPSEEK_MODELS = {"deepseek-chat"}


def curr_cost_est():
    costmap_in = {
        "gpt-4o": 2.50 / 1000000,
        "gpt-4o-mini": 0.150 / 1000000,
        "o1-preview": 15.00 / 1000000,
        "o1-mini": 3.00 / 1000000,
        "claude-3-5-sonnet": 3.00 / 1000000,
        "deepseek-chat": 1.00 / 1000000,
        "o1": 15.00 / 1000000,
        "o3-mini": 1.10 / 1000000,
    }
    costmap_out = {
        "gpt-4o": 10.00 / 1000000,
        "gpt-4o-mini": 0.6 / 1000000,
        "o1-preview": 60.00 / 1000000,
        "o1-mini": 12.00 / 1000000,
        "claude-3-5-sonnet": 12.00 / 1000000,
        "deepseek-chat": 5.00 / 1000000,
        "o1": 60.00 / 1000000,
        "o3-mini": 4.40 / 1000000,
    }

    total_cost = 0.0
    for model_name, token_count in TOKENS_IN.items():
        if model_name in costmap_in:
            total_cost += costmap_in[model_name] * token_count
    for model_name, token_count in TOKENS_OUT.items():
        if model_name in costmap_out:
            total_cost += costmap_out[model_name] * token_count
    return total_cost


def _resolve_api_key(explicit_key, env_var_name):
    return explicit_key or os.getenv(env_var_name)


def _load_genai():
    import google.generativeai as genai

    return genai


def _validate_credentials(model_name, openai_api_key, gemini_api_key, anthropic_api_key, deepseek_api_key):
    if model_name in OPENAI_MODELS and openai_api_key is None:
        raise Exception("OPENAI_API_KEY must be provided for OpenAI-backed models")
    if model_name in GEMINI_MODELS and gemini_api_key is None:
        raise Exception("GEMINI_API_KEY must be provided for Gemini-backed models")
    if model_name in ANTHROPIC_MODELS and anthropic_api_key is None:
        raise Exception("ANTHROPIC_API_KEY must be provided for Anthropic-backed models")
    if model_name in DEEPSEEK_MODELS and deepseek_api_key is None:
        raise Exception("DEEPSEEK_API_KEY must be provided for DeepSeek-backed models")


def _record_token_usage(model_name, system_prompt, prompt, answer, print_cost):
    try:
        TOKENS_IN.setdefault(model_name, 0)
        TOKENS_OUT.setdefault(model_name, 0)
        TOKENS_IN[model_name] += count_text_tokens(system_prompt + prompt, model_name)
        TOKENS_OUT[model_name] += count_text_tokens(answer, model_name)
        if print_cost:
            print(f"Current experiment cost = ${curr_cost_est()}, ** Approximate values, may not reflect true cost")
    except Exception as e:
        if print_cost:
            print(f"Cost approximation has an error? {e}")


def query_model(
    model_str,
    prompt,
    system_prompt,
    openai_api_key=None,
    gemini_api_key=None,
    anthropic_api_key=None,
    tries=5,
    timeout=5.0,
    temp=None,
    print_cost=True,
    version="1.5",
):
    model_str = normalize_model_name(model_str)
    openai_api_key = _resolve_api_key(openai_api_key, "OPENAI_API_KEY")
    gemini_api_key = _resolve_api_key(gemini_api_key, "GEMINI_API_KEY")
    anthropic_api_key = _resolve_api_key(anthropic_api_key, "ANTHROPIC_API_KEY")
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

    _validate_credentials(model_str, openai_api_key, gemini_api_key, anthropic_api_key, deepseek_api_key)

    if openai_api_key is not None:
        openai.api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
    if anthropic_api_key is not None:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
    if gemini_api_key is not None:
        os.environ["GEMINI_API_KEY"] = gemini_api_key

    for _ in range(tries):
        try:
            if model_str == "gpt-4o-mini":
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
                if version == "0.28":
                    request_kwargs = {"model": model_str, "messages": messages}
                    if temp is not None:
                        request_kwargs["temperature"] = temp
                    completion = openai.ChatCompletion.create(**request_kwargs)
                else:
                    client = OpenAI(api_key=openai_api_key)
                    request_kwargs = {"model": "gpt-4o-mini-2024-07-18", "messages": messages}
                    if temp is not None:
                        request_kwargs["temperature"] = temp
                    completion = client.chat.completions.create(**request_kwargs)
                answer = completion.choices[0].message.content

            elif model_str == "gemini-2.0-pro":
                genai = _load_genai()
                genai.configure(api_key=gemini_api_key)
                model = genai.GenerativeModel(
                    model_name="gemini-2.0-pro-exp-02-05",
                    system_instruction=system_prompt,
                )
                answer = model.generate_content(prompt).text

            elif model_str == "gemini-1.5-pro":
                genai = _load_genai()
                genai.configure(api_key=gemini_api_key)
                model = genai.GenerativeModel(
                    model_name="gemini-1.5-pro",
                    system_instruction=system_prompt,
                )
                answer = model.generate_content(prompt).text

            elif model_str == "o3-mini":
                messages = [{"role": "user", "content": system_prompt + prompt}]
                if version == "0.28":
                    completion = openai.ChatCompletion.create(model=model_str, messages=messages)
                else:
                    client = OpenAI(api_key=openai_api_key)
                    completion = client.chat.completions.create(
                        model="o3-mini-2025-01-31",
                        messages=messages,
                    )
                answer = completion.choices[0].message.content

            elif model_str == "claude-3-5-sonnet":
                client = anthropic.Anthropic(api_key=anthropic_api_key)
                message = client.messages.create(
                    model="claude-3-5-sonnet-latest",
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                )
                answer = json.loads(message.to_json())["content"][0]["text"]

            elif model_str == "gpt-4o":
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
                if version == "0.28":
                    request_kwargs = {"model": model_str, "messages": messages}
                    if temp is not None:
                        request_kwargs["temperature"] = temp
                    completion = openai.ChatCompletion.create(**request_kwargs)
                else:
                    client = OpenAI(api_key=openai_api_key)
                    request_kwargs = {"model": "gpt-4o-2024-08-06", "messages": messages}
                    if temp is not None:
                        request_kwargs["temperature"] = temp
                    completion = client.chat.completions.create(**request_kwargs)
                answer = completion.choices[0].message.content

            elif model_str == "deepseek-chat":
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
                if version == "0.28":
                    raise Exception("Please upgrade your OpenAI version to use DeepSeek client")
                deepseek_client = OpenAI(
                    api_key=deepseek_api_key,
                    base_url="https://api.deepseek.com/v1",
                )
                request_kwargs = {"model": "deepseek-chat", "messages": messages}
                if temp is not None:
                    request_kwargs["temperature"] = temp
                completion = deepseek_client.chat.completions.create(**request_kwargs)
                answer = completion.choices[0].message.content

            elif model_str == "o1-mini":
                messages = [{"role": "user", "content": system_prompt + prompt}]
                if version == "0.28":
                    completion = openai.ChatCompletion.create(model=model_str, messages=messages)
                else:
                    client = OpenAI(api_key=openai_api_key)
                    completion = client.chat.completions.create(
                        model="o1-mini-2024-09-12",
                        messages=messages,
                    )
                answer = completion.choices[0].message.content

            elif model_str == "o1":
                messages = [{"role": "user", "content": system_prompt + prompt}]
                if version == "0.28":
                    completion = openai.ChatCompletion.create(
                        model="o1-2024-12-17",
                        messages=messages,
                    )
                else:
                    client = OpenAI(api_key=openai_api_key)
                    completion = client.chat.completions.create(
                        model="o1-2024-12-17",
                        messages=messages,
                    )
                answer = completion.choices[0].message.content

            elif model_str == "o1-preview":
                messages = [{"role": "user", "content": system_prompt + prompt}]
                if version == "0.28":
                    completion = openai.ChatCompletion.create(model=model_str, messages=messages)
                else:
                    client = OpenAI(api_key=openai_api_key)
                    completion = client.chat.completions.create(
                        model="o1-preview",
                        messages=messages,
                    )
                answer = completion.choices[0].message.content

            else:
                raise ValueError(f"Unsupported model '{model_str}'")

            _record_token_usage(model_str, system_prompt, prompt, answer, print_cost)
            return answer
        except Exception as e:
            print("Inference Exception:", e)
            time.sleep(timeout)
            continue
    raise Exception("Max retries: timeout")


# print(query_model(model_str="o1-mini", prompt="hi", system_prompt="hey"))
