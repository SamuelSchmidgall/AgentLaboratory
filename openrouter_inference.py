import os
import requests

def query_openrouter_model(model_str, prompt, system_prompt, openrouter_api_key=None, temp=None):
    if openrouter_api_key is None:
        openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
    if openrouter_api_key is None:
        raise Exception("No OpenRouter API key provided")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model_str,  # e.g. "openrouter/phi-3-mini-128k-instruct"
        "messages": messages,
    }
    if temp is not None:
        data["temperature"] = temp
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=data,
        timeout=60
    )
    if response.status_code != 200:
        raise Exception(f"OpenRouter API error: {response.text}")
    completion = response.json()
    return completion["choices"][0]["message"]["content"] 