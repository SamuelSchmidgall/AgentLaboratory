from typing import Optional, Dict
from .models import (
    LLMStrategy,
    ModelConfig,
    OpenAIStrategy,
    AnthropicStrategy,
    DeepseekStrategy
)

class LLMStrategyFactory:
    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        self.api_keys = api_keys or {}
        self.model_configs = {
            "gpt-4o": ModelConfig("gpt-4o-2024-08-06", 2.50, 10.00, "openai"),
            "gpt-4o-mini": ModelConfig("gpt-4o-mini-2024-07-18", 0.15, 0.60, "openai"),
            "claude-3-5-sonnet": ModelConfig("claude-3-5-sonnet-latest", 3.00, 12.00, "anthropic"),
            "deepseek-chat": ModelConfig("deepseek-chat", 1.00, 5.00, "deepseek"),
            "o1-mini": ModelConfig("o1-mini-2024-09-12", 3.00, 12.00, "openai"),
            "o1": ModelConfig("o1-2024-12-17", 15.00, 60.00, "openai"),
            "o1-preview": ModelConfig("o1-preview", 15.00, 60.00, "openai"),
        }

    def create_strategy(self, model_name: str) -> LLMStrategy:
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")

        config = self.model_configs[model_name]
        provider = config.provider
        default_api_key = self.api_keys.get(provider)

        if provider == "openai":
            return OpenAIStrategy(config, default_api_key)
        elif provider == "anthropic":
            return AnthropicStrategy(config, default_api_key)
        elif provider == "deepseek":
            return DeepseekStrategy(config, default_api_key)
        else:
            raise ValueError(f"No strategy implementation for provider: {provider}")
