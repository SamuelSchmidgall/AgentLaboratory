import time, tiktoken
from openai import OpenAI
import openai
import os, anthropic, json
import google.generativeai as genai

TOKENS_IN = dict()
TOKENS_OUT = dict()

encoding = tiktoken.get_encoding("cl100k_base")

def curr_cost_est():
    costmap_in = {
        "gpt-4o": 2.50 / 1000000,
        "gpt-4o-mini": 0.150 / 1000000,
        "o1-preview": 15.00 / 1000000,
        "o1-mini": 3.00 / 1000000,
        "claude-3-5-sonnet": 3.00 / 1000000,
        "deepseek-chat": 1.00 / 1000000,
        "o1": 15.00 / 1000000,
        "gemini-1.0-pro-latest": 0.00 / 1000000, # Gemini Pro is currently free, adjust if pricing changes
        "gemini-2.0-flash-thinking-exp-01-21": 0.00 / 1000000, # Gemini Flash is also currently free, adjust if pricing changes
    }
    costmap_out = {
        "gpt-4o": 10.00/ 1000000,
        "gpt-4o-mini": 0.6 / 1000000,
        "o1-preview": 60.00 / 1000000,
        "o1-mini": 12.00 / 1000000,
        "claude-3-5-sonnet": 12.00 / 1000000,
        "deepseek-chat": 5.00 / 1000000,
        "o1": 60.00 / 1000000,
        "gemini-1.0-pro-latest": 0.00 / 1000000, # Gemini Pro is currently free, adjust if pricing changes
        "gemini-2.0-flash-thinking-exp-01-21": 0.00 / 1000000, # Gemini Flash is also currently free, adjust if pricing changes
    }
    return sum([costmap_in[_]*TOKENS_IN[_] for _ in TOKENS_IN]) + sum([costmap_out[_]*TOKENS_OUT[_] for _ in TOKENS_OUT])

def query_model(model_str, prompt, system_prompt, openai_api_key=None, anthropic_api_key=None, google_api_key=None, tries=5, timeout=5.0, temp=None, print_cost=True, version="1.5"):
    preloaded_openai_api = os.getenv('OPENAI_API_KEY')
    preloaded_google_api = os.getenv('GOOGLE_API_KEY') # Check for Google API key

    if openai_api_key is None and preloaded_openai_api is not None:
        openai_api_key = preloaded_openai_api
    if google_api_key is None and preloaded_google_api is not None: # Use preloaded Google API key if available
        google_api_key = preloaded_google_api

    if openai_api_key is None and anthropic_api_key is None and google_api_key is None: # Check for Google API key as well
        raise Exception("No API key provided in query_model function (OpenAI, Anthropic, or Google required)")

    if openai_api_key is not None:
        openai.api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
    if anthropic_api_key is not None:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
    if google_api_key is not None: # Configure Gemini with Google API key
        genai.configure(api_key=google_api_key)
        os.environ["GOOGLE_API_KEY"] = google_api_key


    for _ in range(tries):
        try:
            if model_str == "gpt-4o-mini" or model_str == "gpt4omini" or model_str == "gpt-4omini" or model_str == "gpt4o-mini":
                model_str = "gpt-4o-mini"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                if version == "0.28":
                    if temp is None:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages
                        )
                    else:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages, temperature=temp
                        )
                else:
                    client = OpenAI()
                    if temp is None:
                        completion = client.chat.completions.create(
                            model="gpt-4o-mini-2024-07-18", messages=messages, )
                    else:
                        completion = client.chat.completions.create(
                            model="gpt-4o-mini-2024-07-18", messages=messages, temperature=temp)
                answer = completion.choices[0].message.content
            elif model_str == "claude-3.5-sonnet":
                client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                message = client.messages.create(
                    model="claude-3-5-sonnet-latest",
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}])
                answer = json.loads(message.to_json())["content"][0]["text"]
            elif model_str == "gpt4o" or model_str == "gpt-4o":
                model_str = "gpt-4o"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                if version == "0.28":
                    if temp is None:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages
                        )
                    else:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages, temperature=temp)
                else:
                    client = OpenAI()
                    if temp is None:
                        completion = client.chat.completions.create(
                            model="gpt-4o-2024-08-06", messages=messages, )
                    else:
                        completion = client.chat.completions.create(
                            model="gpt-4o-2024-08-06", messages=messages, temperature=temp)
                answer = completion.choices[0].message.content
            elif model_str == "deepseek-chat":
                model_str = "deepseek-chat"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                if version == "0.28":
                    raise Exception("Please upgrade your OpenAI version to use DeepSeek client")
                else:
                    deepseek_client = OpenAI(
                        api_key=os.getenv('DEEPSEEK_API_KEY'),
                        base_url="https://api.deepseek.com/v1"
                    )
                    if temp is None:
                        completion = deepseek_client.chat.completions.create(
                            model="deepseek-chat",
                            messages=messages)
                    else:
                        completion = deepseek_client.chat.completions.create(
                            model="deepseek-chat",
                            messages=messages,
                            temperature=temp)
                answer = completion.choices[0].message.content
            elif model_str == "o1-mini":
                model_str = "o1-mini"
                messages = [
                    {"role": "user", "content": system_prompt + prompt}]
                if version == "0.28":
                    completion = openai.ChatCompletion.create(
                        model=f"{model_str}",  # engine = "deployment_name".
                        messages=messages)
                else:
                    client = OpenAI()
                    completion = client.chat.completions.create(
                        model="o1-mini-2024-09-12", messages=messages)
                answer = completion.choices[0].message.content
            elif model_str == "o1":
                model_str = "o1"
                messages = [
                    {"role": "user", "content": system_prompt + prompt}]
                if version == "0.28":
                    completion = openai.ChatCompletion.create(
                        model=f"{model_str}",  # engine = "deployment_name".
                        messages=messages)
                else:
                    client = OpenAI()
                    completion = client.chat.completions.create(
                        model="o1-2024-12-17", messages=messages)
                answer = completion.choices[0].message.content
            elif model_str == "o1-preview":
                model_str = "o1-preview"
                messages = [
                    {"role": "user", "content": system_prompt + prompt}]
                if version == "0.28":
                    completion = openai.ChatCompletion.create(
                        model=f"{model_str}",  # engine = "deployment_name".
                        messages=messages)
                else:
                    client = OpenAI()
                    completion = client.chat.completions.create(
                        model="o1-preview", messages=messages)
                answer = completion.choices[0].message.content
            elif model_str == "gemini-1.0-pro-latest": # Gemini Pro
                print(f"DEBUG: Gemini Pro - model_str: {model_str}") # DEBUG
                print(f"DEBUG: Gemini Pro - system_prompt: {system_prompt}") # DEBUG
                print(f"DEBUG: Gemini Pro - prompt: {prompt}") # DEBUG
                model = genai.GenerativeModel('gemini-1.0-pro-latest')
                print("DEBUG: Gemini Pro - GenerativeModel instantiated") # DEBUG
                time.sleep(10) # ADD SLEEP HERE - RATE LIMITING TEST
                try: # DEBUG - Add try-except around generate_content
                    response = model.generate_content([system_prompt, prompt])
                    print("DEBUG: Gemini Pro - response generated successfully") # DEBUG
                    answer = response.text
                except Exception as gemini_e: # DEBUG - Catch potential exceptions
                    print(f"DEBUG: Gemini Pro - Exception during generate_content: {gemini_e}, Exception: {gemini_e}") # DEBUG
                    raise gemini_e # DEBUG - Re-raise the exception to be caught in the outer loop

            elif model_str == "gemini-2.0-flash-thinking-exp-01-21": # Gemini Flash
                print(f"DEBUG: Gemini Flash - model_str: {model_str}") # DEBUG
                print(f"DEBUG: Gemini Flash - system_prompt: {system_prompt}") # DEBUG
                print(f"DEBUG: Gemini Flash - prompt: {prompt}") # DEBUG
                model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp-01-21')
                print("DEBUG: Gemini Flash - GenerativeModel instantiated") # DEBUG
                time.sleep(10)  # ADD SLEEP HERE - RATE LIMITING TEST
                try: # DEBUG - Add try-except around generate_content
                    response = model.generate_content([system_prompt, prompt])
                    print("DEBUG: Gemini Flash - response generated successfully") # DEBUG
                    answer = response.text
                except Exception as gemini_e: # DEBUG - Catch potential exceptions
                    print(f"DEBUG: Gemini Flash - Exception during generate_content: {gemini_e}, Exception: {gemini_e}") # DEBUG
                    raise gemini_e # DEBUG - Re-raise the exception to be caught in the outer loop


            # TODO use model.count_tokens instead for
            if model_str in ["o1-preview", "o1-mini", "claude-3.5-sonnet", "o1", "gemini-1.0-pro-latest", "gemini-2.0-flash-thinking-exp-01-21"]:
                encoding = tiktoken.encoding_for_model("gpt-4o")
            elif model_str in ["deepseek-chat"]:
                encoding = tiktoken.encoding_for_model("cl100k_base")
            else:
                encoding = tiktoken.encoding_for_model(model_str)


            if model_str not in TOKENS_IN:
                TOKENS_IN[model_str] = 0
                TOKENS_OUT[model_str] = 0
            TOKENS_IN[model_str] += len(encoding.encode(system_prompt + prompt))
            TOKENS_OUT[model_str] += len(encoding.encode(answer))
            if print_cost:
                print(f"Current experiment cost = ${curr_cost_est()}, ** Approximate values, may not reflect true cost")
            return answer
        except Exception as e:
            print("Inference Exception:", e)
            time.sleep(timeout)
            continue
    raise Exception("Max retries: timeout")


#print(query_model(model_str="o1-mini", prompt="hi", system_prompt="hey"))


#print(query_model(model_str="o1-mini", prompt="hi", system_prompt="hey"))