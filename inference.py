# inference.py
import os
import time
import json
import tiktoken
import anthropic
import google.generativeai as genai
from openai import OpenAI

TOKENS_IN = dict()
TOKENS_OUT = dict()

def _encoding_for(model_id: str):
    try:
        # Known OpenAI families
        if any(model_id.startswith(p) for p in ["o1", "o3", "gpt-4", "gpt-5", "gpt-4o"]):
            return tiktoken.encoding_for_model("gpt-4o")
        return tiktoken.encoding_for_model(model_id)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")

def curr_cost_est():
    # Keep as an approximation; unknown models won't be counted
    costmap_in = {
        "gpt-4o": 2.50 / 1_000_000,
        "gpt-4o-mini": 0.150 / 1_000_000,
        "o1-preview": 15.00 / 1_000_000,
        "o1-mini": 3.00 / 1_000_000,
        "claude-3-5-sonnet": 3.00 / 1_000_000,
        "deepseek-chat": 1.00 / 1_000_000,
        "o1": 15.00 / 1_000_000,
        "o3-mini": 1.10 / 1_000_000,
        # Extend as needed
    }
    costmap_out = {
        "gpt-4o": 10.00 / 1_000_000,
        "gpt-4o-mini": 0.6 / 1_000_000,
        "o1-preview": 60.00 / 1_000_000,
        "o1-mini": 12.00 / 1_000_000,
        "claude-3-5-sonnet": 12.00 / 1_000_000,
        "deepseek-chat": 5.00 / 1_000_000,
        "o1": 60.00 / 1_000_000,
        "o3-mini": 4.40 / 1_000_000,
    }
    total = 0.0
    for k, v in TOKENS_IN.items():
        if k in costmap_in:
            total += costmap_in[k] * v
    for k, v in TOKENS_OUT.items():
        if k in costmap_out:
            total += costmap_out[k] * v
    return total


def _track_tokens(model_str, system_prompt, prompt, answer, print_cost=True):
    try:
        enc = _encoding_for(model_str)
        TOKENS_IN.setdefault(model_str, 0)
        TOKENS_OUT.setdefault(model_str, 0)
        TOKENS_IN[model_str] += len(enc.encode((system_prompt or "") + (prompt or "")))
        TOKENS_OUT[model_str] += len(enc.encode(answer or ""))
        if print_cost:
            print(f"Current experiment cost = ${curr_cost_est()}, **Approximate only**")
    except Exception as e:
        if print_cost:
            print(f"Cost approximation error: {e}")


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
    version="1.5"
):
    """
    Generic model router:
    - OpenAI: any 'gpt-*' (incl. gpt-5*) and 'o*' models
    - Anthropic: any 'claude-*' model id
    - Google: any 'gemini-*' model id (incl. gemini-2.5-*)
    - DeepSeek (explicit): 'deepseek-chat'
    """
    # Resolve API keys from env when necessary
    if openai_api_key is None and os.getenv("OPENAI_API_KEY"):
        openai_api_key = os.getenv("OPENAI_API_KEY")
    if anthropic_api_key is None and os.getenv("ANTHROPIC_API_KEY"):
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if gemini_api_key is None and os.getenv("GEMINI_API_KEY"):
        gemini_api_key = os.getenv("GEMINI_API_KEY")

    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    if anthropic_api_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
    if gemini_api_key:
        os.environ["GEMINI_API_KEY"] = gemini_api_key

    normalized = (model_str or "").strip()

    for _ in range(tries):
        try:
            # -------------------------
            # OpenAI families (GPT-4*, GPT-5*, o1/o3 etc.)
            # -------------------------
            if normalized.startswith("gpt-") or normalized.startswith("o1") or normalized.startswith("o3"):
                client = OpenAI()
                # Some o* models may prefer single 'user' message; keep conservative merge where needed
                if normalized.startswith("o1") or normalized.startswith("o3"):
                    messages = [{"role": "user", "content": (system_prompt or "") + (prompt or "")}]
                else:
                    messages = [
                        {"role": "system", "content": system_prompt or ""},
                        {"role": "user", "content": prompt or ""}
                    ]
                kwargs = {"model": normalized, "messages": messages}
                if temp is not None:
                    kwargs["temperature"] = temp
                completion = client.chat.completions.create(**kwargs)
                answer = completion.choices[0].message.content

            # -------------------------
            # Anthropic Claude (any)
            # -------------------------
            elif normalized.startswith("claude-"):
                client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
                message = client.messages.create(
                    model=normalized,
                    system=system_prompt or "",
                    messages=[{"role": "user", "content": prompt or ""}],
                    temperature=temp if temp is not None else 0.0,
                )
                # Content is a list of content blocks—coalesce to text
                answer = ""
                try:
                    for blk in message.content:
                        if getattr(blk, "type", "") == "text":
                            answer += blk.text
                        elif isinstance(blk, dict) and blk.get("type") == "text":
                            answer += blk.get("text", "")
                except Exception:
                    answer = json.loads(message.to_json())["content"][0]["text"]

            # -------------------------
            # Google Gemini (incl. 2.5)
            # -------------------------
            elif normalized.startswith("gemini-"):
                genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
                model = genai.GenerativeModel(model_name=normalized, system_instruction=system_prompt or "")
                if temp is not None:
                    cfg = genai.types.GenerationConfig(temperature=temp)
                    answer = model.generate_content(prompt or "", generation_config=cfg).text
                else:
                    answer = model.generate_content(prompt or "").text

            # -------------------------
            # DeepSeek explicit
            # -------------------------
            elif normalized == "deepseek-chat":
                deepseek_client = OpenAI(
                    api_key=os.getenv("DEEPSEEK_API_KEY"),
                    base_url="https://api.deepseek.com/v1"
                )
                messages = [
                    {"role": "system", "content": system_prompt or ""},
                    {"role": "user", "content": prompt or ""}
                ]
                kwargs = {"model": "deepseek-chat", "messages": messages}
                if temp is not None:
                    kwargs["temperature"] = temp
                completion = deepseek_client.chat.completions.create(**kwargs)
                answer = completion.choices[0].message.content

            else:
                # Default: try OpenAI with the provided model id
                client = OpenAI()
                messages = [
                    {"role": "system", "content": system_prompt or ""},
                    {"role": "user", "content": prompt or ""}
                ]
                kwargs = {"model": normalized, "messages": messages}
                if temp is not None:
                    kwargs["temperature"] = temp
                completion = client.chat.completions.create(**kwargs)
                answer = completion.choices[0].message.content

            _track_tokens(normalized, system_prompt, prompt, answer, print_cost=print_cost)
            return answer

        except Exception as e:
            print("Inference Exception:", e)
            time.sleep(timeout)
            continue
    raise Exception("Max retries: timeout")
