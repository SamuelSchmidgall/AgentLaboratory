import time
import tiktoken
from openai import OpenAI
import openai
import os
import anthropic
import json

##############################################################################
# Global Token Counters & Encoding
##############################################################################
TOKENS_IN = dict()
TOKENS_OUT = dict()

# Use the TikToken encoding for the "cl100k_base" by default
encoding = tiktoken.get_encoding("cl100k_base")

##############################################################################
# Cost Estimation
##############################################################################
def curr_cost_est():
    """
    Estimates total cost of the current experiment based on tokens used for each
    model and the pricing maps below. Cost is approximate.
    """
    costmap_in = {
        "gpt-4o": 2.50 / 1000000,
        "gpt-4o-mini": 0.150 / 1000000,
        "o1-preview": 15.00 / 1000000,
        "o1-mini": 3.00 / 1000000,
        "claude-3-5-sonnet": 3.00 / 1000000,
        "deepseek-chat": 1.00 / 1000000,
        "o1": 15.00 / 1000000,
    }
    costmap_out = {
        "gpt-4o": 10.00/1000000,
        "gpt-4o-mini": 0.6/1000000,
        "o1-preview": 60.00/1000000,
        "o1-mini": 12.00/1000000,
        "claude-3-5-sonnet": 12.00/1000000,
        "deepseek-chat": 5.00/1000000,
        "o1": 60.00/1000000,
    }

    # Sum over all models for input tokens and output tokens
    total_cost = sum([costmap_in[m] * TOKENS_IN[m] for m in TOKENS_IN]) \
                 + sum([costmap_out[m] * TOKENS_OUT[m] for m in TOKENS_OUT])
    return total_cost

##############################################################################
# Model Normalization
##############################################################################
def normalize_model_str(model_str):
    """
    Maps certain string variants to canonical model names.
    This replaces the repeated if-conditions that reassign model_str.
    """
    # For GPT-4o-mini variants
    gpt4o_mini_variants = ["gpt-4o-mini", "gpt4omini", "gpt-4omini", "gpt4o-mini"]
    # For GPT-4o variants
    gpt4o_variants = ["gpt4o", "gpt-4o"]
    # Return canonical name if found, otherwise just return the same string.
    if model_str in gpt4o_mini_variants:
        return "gpt-4o-mini"
    elif model_str in gpt4o_variants:
        return "gpt-4o"
    else:
        return model_str

##############################################################################
# Handlers for Each Model
##############################################################################
def handle_gpt4o_mini(model_str, prompt, system_prompt, temp, version):
    """
    Handles queries to GPT-4o-mini (Azure-based or otherwise).
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    if version == "0.28":
        # Use openai.ChatCompletion with or without temperature
        if temp is None:
            completion = openai.ChatCompletion.create(
                model=model_str,
                messages=messages
            )
        else:
            completion = openai.ChatCompletion.create(
                model=model_str,
                messages=messages,
                temperature=temp
            )
    else:
        # Use the python-openai direct client
        client = OpenAI()
        if temp is None:
            completion = client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=messages
            )
        else:
            completion = client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=messages,
                temperature=temp
            )
    return completion.choices[0].message.content


def handle_claude_35_sonnet(model_str, prompt, system_prompt, temp, version):
    """
    Handles queries to Claude 3.5 Sonnet.
    (Note that temperature is not used in original code for Claude.)
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model="claude-3-5-sonnet-latest",
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}]
    )
    # "content" in the JSON is a list of tokens; we pick [0]["text"] as in original
    return json.loads(message.to_json())["content"][0]["text"]


def handle_gpt4o(model_str, prompt, system_prompt, temp, version):
    """
    Handles queries to GPT-4o.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    if version == "0.28":
        # Use openai.ChatCompletion with or without temperature
        if temp is None:
            completion = openai.ChatCompletion.create(
                model=model_str,
                messages=messages
            )
        else:
            completion = openai.ChatCompletion.create(
                model=model_str,
                messages=messages,
                temperature=temp
            )
    else:
        # Use the python-openai direct client
        client = OpenAI()
        if temp is None:
            completion = client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=messages,
            )
        else:
            completion = client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=messages,
                temperature=temp
            )
    return completion.choices[0].message.content


def handle_deepseek_chat(model_str, prompt, system_prompt, temp, version):
    """
    Handles queries to DeepSeek Chat.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    if version == "0.28":
        raise Exception("Please upgrade your OpenAI version to use DeepSeek client")
    else:
        # Use the python-openai client with a different base_url
        deepseek_client = OpenAI(
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com/v1"
        )
        if temp is None:
            completion = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=messages
            )
        else:
            completion = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=temp
            )
    return completion.choices[0].message.content


def handle_o1_mini(model_str, prompt, system_prompt, temp, version):
    """
    Handles queries to o1-mini.
    """
    messages = [
        {"role": "user", "content": system_prompt + prompt}
    ]
    if version == "0.28":
        completion = openai.ChatCompletion.create(
            model=model_str,
            messages=messages
        )
    else:
        client = OpenAI()
        completion = client.chat.completions.create(
            model="o1-mini-2024-09-12",
            messages=messages
        )
    return completion.choices[0].message.content


def handle_o1(model_str, prompt, system_prompt, temp, version):
    """
    Handles queries to o1.
    """
    messages = [
        {"role": "user", "content": system_prompt + prompt}
    ]
    if version == "0.28":
        completion = openai.ChatCompletion.create(
            model="o1-2024-12-17",
            messages=messages
        )
    else:
        client = OpenAI()
        completion = client.chat.completions.create(
            model="o1-2024-12-17",
            messages=messages
        )
    return completion.choices[0].message.content


def handle_o1_preview(model_str, prompt, system_prompt, temp, version):
    """
    Handles queries to o1-preview.
    """
    messages = [
        {"role": "user", "content": system_prompt + prompt}
    ]
    if version == "0.28":
        completion = openai.ChatCompletion.create(
            model=model_str,
            messages=messages
        )
    else:
        client = OpenAI()
        completion = client.chat.completions.create(
            model="o1-preview",
            messages=messages
        )
    return completion.choices[0].message.content

##############################################################################
# Main Dispatch
##############################################################################
def query_model(
    model_str,
    prompt,
    system_prompt,
    openai_api_key=None,
    anthropic_api_key=None,
    tries=5,
    timeout=5.0,
    temp=None,
    print_cost=True,
    version="1.5"
):
    """
    Main function to query one of several models (GPT-4o-mini, GPT-4o, etc.).
    Retries multiple times in case of exceptions.

    :param model_str:          Model name string
    :param prompt:             User prompt
    :param system_prompt:      System instruction/prompt
    :param openai_api_key:     OpenAI key (defaults to env var if None)
    :param anthropic_api_key:  Anthropic key (defaults to env var if None)
    :param tries:              Retry attempts
    :param timeout:            Seconds to sleep between retries
    :param temp:               Temperature for generation
    :param print_cost:         Whether to print cost after each request
    :param version:            Some version parameter used in specific logic
    :return:                   Model's response text
    """

    # Retrieve preloaded API keys if not provided
    preloaded_api = os.getenv('OPENAI_API_KEY')
    if openai_api_key is None and preloaded_api is not None:
        openai_api_key = preloaded_api
    
    if openai_api_key is None and anthropic_api_key is None:
        raise Exception("No API key provided in query_model function")

    # Set environment
    if openai_api_key is not None:
        openai.api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
    if anthropic_api_key is not None:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key

    # Dictionary that dispatches to the appropriate model handler function
    model_dispatch = {
        "gpt-4o-mini": handle_gpt4o_mini,
        "claude-3.5-sonnet": handle_claude_35_sonnet,
        "gpt-4o": handle_gpt4o,
        "deepseek-chat": handle_deepseek_chat,
        "o1-mini": handle_o1_mini,
        "o1": handle_o1,
        "o1-preview": handle_o1_preview
    }

    # Attempt multiple times
    for _ in range(tries):
        try:
            # Normalize model_str if needed (this captures the old if-block logic)
            model_str = normalize_model_str(model_str)

            # Now query the correct model handler
            if model_str not in model_dispatch:
                # If no direct dispatch match, it means there's a code path not covered
                # by the dictionary. Raise an exception as in the original code (or you
                # could handle it differently).
                raise Exception(f"Unknown model: {model_str}")

            # Call the model-specific function to get the answer
            answer = model_dispatch[model_str](
                model_str, prompt, system_prompt, temp, version
            )

            # Determine the encoding for this model, as in the original code
            if model_str in ["o1-preview", "o1-mini", "claude-3.5-sonnet", "o1"]:
                model_encoding = tiktoken.encoding_for_model("gpt-4o")
            elif model_str in ["deepseek-chat"]:
                model_encoding = tiktoken.encoding_for_model("cl100k_base")
            else:
                # for the rest (like "gpt-4o-mini", "gpt-4o"), use the model_str
                model_encoding = tiktoken.encoding_for_model(model_str)

            # Initialize token counters if not done before
            if model_str not in TOKENS_IN:
                TOKENS_IN[model_str] = 0
                TOKENS_OUT[model_str] = 0

            # Count tokens
            TOKENS_IN[model_str] += len(model_encoding.encode(system_prompt + prompt))
            TOKENS_OUT[model_str] += len(model_encoding.encode(answer))

            # Print cost if requested
            if print_cost:
                print(f"Current experiment cost = ${curr_cost_est()}, "
                      f"** Approximate values, may not reflect true cost")

            return answer

        except Exception as e:
            # On any exception, sleep and retry
            print("Inference Exception:", e)
            time.sleep(timeout)
            continue

    # If we reach here, we've tried `tries` times
    raise Exception("Max retries: timeout")


# Example usage
# print(query_model(model_str="o1-mini", prompt="hi", system_prompt="hey"))
