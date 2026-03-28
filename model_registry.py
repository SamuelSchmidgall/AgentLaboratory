import json
import os
from datetime import datetime, timezone, timedelta

from config import GOOGLE_GENERATIVE_API_BASE_URL, DEEPSEEK_API_BASE_URL, OLLAMA_API_BASE_URL

STALENESS_DAYS = 7
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "models_pricing.json")

# Base URLs per provider (non-OpenAI providers that need a custom base_url)
PROVIDER_BASE_URLS = {
    "deepseek": DEEPSEEK_API_BASE_URL,
    "google": GOOGLE_GENERATIVE_API_BASE_URL,
}


class ModelNotFoundError(Exception):
    def __init__(self, model_name, available_models):
        self.model_name = model_name
        self.available_models = available_models
        super().__init__(
            f"Model '{model_name}' not found. Available models: {', '.join(sorted(available_models))}"
        )


class ModelRegistry:
    def __init__(self, config_path=None, auto_refresh=True):
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.models = {}
        self.last_updated = None
        self.tokens_in = {}
        self.tokens_out = {}
        self._alias_map = {}
        self._load()
        if auto_refresh and self.is_stale():
            self._try_refresh()

    def _load(self):
        try:
            with open(self.config_path, "r") as f:
                data = json.load(f)
            self.models = data.get("models", {})
            self.last_updated = data.get("last_updated")
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            print(f"Warning: Could not load {self.config_path}, using default models.")
            self.models = DEFAULT_MODELS
            self.last_updated = None
            self._save()
        self._build_alias_map()

    def _build_alias_map(self):
        self._alias_map = {}
        for name, info in self.models.items():
            self._alias_map[name] = name
            for alias in info.get("aliases", []):
                self._alias_map[alias] = name

    def _save(self):
        data = {
            "last_updated": self.last_updated or datetime.now(timezone.utc).isoformat(),
            "models": self.models,
        }
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump(data, f, indent=2)

    def _try_refresh(self):
        try:
            from model_fetcher import update_models_pricing
            updated = update_models_pricing(self.config_path)
            if updated:
                self._load()
        except Exception as e:
            print(f"Warning: Auto-refresh failed ({e}), using cached data.")

    def is_stale(self):
        if self.last_updated is None:
            return True
        try:
            last = datetime.fromisoformat(self.last_updated)
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            return datetime.now(timezone.utc) - last > timedelta(days=STALENESS_DAYS)
        except (ValueError, TypeError):
            return True

    def resolve_alias(self, name):
        if name in self._alias_map:
            return self._alias_map[name]
        # Anthropic models: match by startswith (e.g. claude-3-5-sonnet-20241022)
        for canonical, info in self.models.items():
            if info.get("provider") == "anthropic" and name.startswith(canonical):
                return canonical
        raise ModelNotFoundError(name, list(self.models.keys()))

    def get_model(self, name_or_alias):
        canonical = self.resolve_alias(name_or_alias)
        return self.models[canonical]

    def get_provider(self, name_or_alias):
        return self.get_model(name_or_alias)["provider"]

    def get_api_model_name(self, name_or_alias):
        model = self.get_model(name_or_alias)
        # For anthropic, if user passed a full versioned name, use it directly
        if model["provider"] == "anthropic" and name_or_alias != self.resolve_alias(name_or_alias):
            if name_or_alias.startswith(self.resolve_alias(name_or_alias)):
                return name_or_alias
        return model["api_model_name"]

    def get_base_url(self, name_or_alias):
        provider = self.get_provider(name_or_alias)
        return PROVIDER_BASE_URLS.get(provider)

    def get_cost_input(self, name_or_alias):
        cost = self.get_model(name_or_alias).get("cost_per_million_input")
        if cost is None:
            return None
        return cost / 1_000_000

    def get_cost_output(self, name_or_alias):
        cost = self.get_model(name_or_alias).get("cost_per_million_output")
        if cost is None:
            return None
        return cost / 1_000_000

    def list_models(self, provider=None):
        if provider:
            return [name for name, info in self.models.items() if info["provider"] == provider]
        return list(self.models.keys())

    def get_canonical_for_cost(self, name_or_alias):
        """Return canonical name used as key in tokens_in/tokens_out dicts."""
        return self.resolve_alias(name_or_alias)

    def curr_cost_est(self):
        total = 0.0
        for model_name, count in self.tokens_in.items():
            cost = self.get_cost_input(model_name)
            if cost is not None:
                total += cost * count
        for model_name, count in self.tokens_out.items():
            cost = self.get_cost_output(model_name)
            if cost is not None:
                total += cost * count
        return total


# Hardcoded fallback — used only when the JSON file is missing or corrupt
DEFAULT_MODELS = {
    "gpt-4o": {
        "provider": "openai",
        "api_model_name": "gpt-4o-2024-08-06",
        "aliases": ["gpt4o"],
        "cost_per_million_input": 2.50,
        "cost_per_million_output": 10.00,
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "api_model_name": "gpt-4o-mini-2024-07-18",
        "aliases": ["gpt4omini", "gpt-4omini", "gpt4o-mini"],
        "cost_per_million_input": 0.150,
        "cost_per_million_output": 0.60,
    },
    "o1-mini": {
        "provider": "openai",
        "api_model_name": "o1-mini-2024-09-12",
        "aliases": [],
        "cost_per_million_input": 1.10,
        "cost_per_million_output": 4.40,
    },
    "o3-mini": {
        "provider": "openai",
        "api_model_name": "o3-mini-2025-01-31",
        "aliases": [],
        "cost_per_million_input": 1.10,
        "cost_per_million_output": 4.40,
    },
    "claude-3-5-sonnet": {
        "provider": "anthropic",
        "api_model_name": "claude-3-5-sonnet",
        "aliases": [],
        "cost_per_million_input": 3.00,
        "cost_per_million_output": 15.00,
    },
    "gemini-2.5-flash": {
        "provider": "google",
        "api_model_name": "gemini-2.5-flash",
        "aliases": [],
        "cost_per_million_input": 0.30,
        "cost_per_million_output": 2.50,
    },
    "deepseek-chat": {
        "provider": "deepseek",
        "api_model_name": "deepseek-chat",
        "aliases": [],
        "cost_per_million_input": 0.27,
        "cost_per_million_output": 1.10,
    },
}
