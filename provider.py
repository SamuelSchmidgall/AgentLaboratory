import anthropic
import openai
from openai import OpenAI


class OpenaiProvider:
    @staticmethod
    def get_response(
        api_key: str,
        model_name: str,
        user_prompt: str,
        system_prompt: str,
        temperature: float = None,
        base_url: str | None = None,
    ) -> str:
        openai.api_key = api_key
        client_config = {
            "api_key": api_key,
        }

        if base_url:
            openai.base_url = base_url
            client_config["base_url"] = base_url

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        version = openai.__version__

        if version == "0.28":
            if temperature is None:
                completion = openai.ChatCompletion.create(
                    model=model_name,
                    messages=messages,
                )
            else:
                completion = openai.ChatCompletion.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                )
        else:
            client = OpenAI(**client_config)
            if temperature is None:
                completion = client.chat.completions.create(
                    model=model_name, messages=messages)
            else:
                completion = client.chat.completions.create(
                    model=model_name, messages=messages, temperature=temperature)

        return completion.choices[0].message.content

class AnthropicProvider:
    @staticmethod
    def get_response(
        api_key: str,
        model_name: str,
        user_prompt: str,
        system_prompt: str,
        temperature: float = None,
    ) -> str:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model_name,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens = 8192 if 'sonnet' in model_name else 4096,
            temperature=temperature,
        )
        return message.content[0].text
