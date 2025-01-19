from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Mapping
import tiktoken
from openai import OpenAI
import anthropic
import json
import os
import time


@dataclass
class Message:
    role: str
    content: str


@dataclass
class ModelConfig:
    model_name: str
    input_cost_per_1k: float
    output_cost_per_1k: float
    provider: str
    encoding_name: str = "cl100k_base"


class TokenCounter:
    def __init__(self):
        self.tokens_in: Dict[str, int] = {}
        self.tokens_out: Dict[str, int] = {}

    def update_counts(self, model_name: str, input_tokens: int, output_tokens: int):
        if model_name not in self.tokens_in:
            self.tokens_in[model_name] = 0
            self.tokens_out[model_name] = 0
        self.tokens_in[model_name] += input_tokens
        self.tokens_out[model_name] += output_tokens

    def calculate_cost(self, model_configs: Dict[str, ModelConfig]) -> float:
        total_cost = 0.0
        for model_name, tokens in self.tokens_in.items():
            if model_name in model_configs:
                config = model_configs[model_name]
                input_cost = (tokens * config.input_cost_per_1k) / 1_000_000
                output_cost = (self.tokens_out[model_name] * config.output_cost_per_1k) / 1_000_000
                total_cost += input_cost + output_cost
        return total_cost


class LLMStrategy(ABC):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.encoding = tiktoken.get_encoding(config.encoding_name)

    @abstractmethod
    def query(self, prompt: str, system_prompt: str, api_key: Optional[str] = None, temperature: Optional[float] = None) -> str:
        pass

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))


class OpenAIStrategy(LLMStrategy):
    def __init__(self, config: ModelConfig, default_api_key: Optional[str] = None):
        super().__init__(config)
        self.default_api_key = default_api_key

    def query(self, prompt: str, system_prompt: str, api_key: Optional[str] = None, temperature: Optional[float] = None) -> str:
        used_key = api_key or self.default_api_key
        if not used_key:
            raise ValueError("No API key provided for OpenAI API")

        client = OpenAI(api_key=used_key)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        completion = client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=temperature,
        )
        return completion.choices[0].message.content


class AnthropicStrategy(LLMStrategy):
    def __init__(self, config: ModelConfig, default_api_key: Optional[str] = None):
        super().__init__(config)
        self.default_api_key = default_api_key

    def query(self, prompt: str, system_prompt: str, api_key: Optional[str] = None, temperature: Optional[float] = None) -> str:
        used_key = api_key or self.default_api_key
        if not used_key:
            raise ValueError("No API key provided for Anthropic API")

        client = anthropic.Anthropic(api_key=used_key)
        message = client.messages.create(
            model=self.config.model_name,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        return json.loads(message.to_json())["content"][0]["text"]


class DeepseekStrategy(LLMStrategy):
    def __init__(self, config: ModelConfig, default_api_key: Optional[str] = None):
        super().__init__(config)
        self.default_api_key = default_api_key

    def query(self, prompt: str, system_prompt: str, api_key: Optional[str] = None, temperature: Optional[float] = None) -> str:
        used_key = api_key or self.default_api_key
        if not used_key:
            raise ValueError("No API key provided for Deepseek API")

        client = OpenAI(
            api_key=used_key,
            base_url="https://api.deepseek.com/v1"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        completion = client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=temperature,
        )
        return completion.choices[0].message.content