# Model Pricing Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace all hardcoded model pricing, aliases, and routing logic with a data-driven registry backed by `configs/models_pricing.json` and a web-based fetch utility.

**Architecture:** A `ModelRegistry` class loads model definitions from a JSON config file and provides lookup/routing/cost methods. A `ModelFetcher` utility discovers models via provider APIs and scrapes pricing from docs pages. On startup, stale data (>7 days) triggers an auto-refresh attempt with graceful fallback. `inference.py`'s ~250-line if/elif chain is replaced by ~15 lines of registry-driven code.

**Tech Stack:** Python, requests, beautifulsoup4, json, tiktoken (existing)

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `configs/models_pricing.json` | Create | Cached model/pricing data, single source of truth |
| `model_registry.py` | Create | Load JSON, resolve aliases, route to provider, track costs |
| `model_fetcher.py` | Create | API model discovery + pricing page scraping per provider |
| `update_models.py` | Create | CLI entry point for manual refresh |
| `inference.py` | Modify | Replace if/elif chain + costmaps with registry lookups |
| `provider.py` | Modify | Add provider dispatcher function |
| `AgentLaboratoryWebUI/config_gradio.py` | Modify | Populate dropdown from registry |
| `AgentLaboratoryWebUI/app.py` | Modify | Use registry for model list |
| `requirements.txt` | Modify | Add requests, beautifulsoup4 |
| `tests/test_model_registry.py` | Create | Unit tests for registry |
| `tests/test_model_fetcher.py` | Create | Unit tests for fetcher |

---

### Task 1: Create `configs/models_pricing.json` with all current models

**Files:**
- Create: `configs/models_pricing.json`

- [ ] **Step 1: Create the configs directory**

```bash
mkdir -p configs
```

- [ ] **Step 2: Create `configs/models_pricing.json` with all current model data**

This JSON contains every model currently hardcoded in `inference.py` lines 16-75 and 104-335. All pricing comes from the existing `costmap_in`/`costmap_out` dicts. The `api_model_name` comes from the model name strings passed to provider calls. The `aliases` come from the if/elif conditions.

```json
{
  "last_updated": "2026-03-28T00:00:00Z",
  "models": {
    "gpt-4o": {
      "provider": "openai",
      "api_model_name": "gpt-4o-2024-08-06",
      "aliases": ["gpt4o", "gpt-4o"],
      "cost_per_million_input": 2.50,
      "cost_per_million_output": 10.00
    },
    "gpt-4o-mini": {
      "provider": "openai",
      "api_model_name": "gpt-4o-mini-2024-07-18",
      "aliases": ["gpt4omini", "gpt-4omini", "gpt4o-mini"],
      "cost_per_million_input": 0.150,
      "cost_per_million_output": 0.60
    },
    "gpt-4.1": {
      "provider": "openai",
      "api_model_name": "gpt-4.1",
      "aliases": ["gpt-4-1"],
      "cost_per_million_input": 3.00,
      "cost_per_million_output": 12.00
    },
    "gpt-4.1-mini": {
      "provider": "openai",
      "api_model_name": "gpt-4.1-mini",
      "aliases": ["gpt-4-1-mini"],
      "cost_per_million_input": 0.80,
      "cost_per_million_output": 3.20
    },
    "gpt-4.1-nano": {
      "provider": "openai",
      "api_model_name": "gpt-4.1-nano",
      "aliases": ["gpt-4-1-nano"],
      "cost_per_million_input": 0.20,
      "cost_per_million_output": 0.80
    },
    "gpt-5.2": {
      "provider": "openai",
      "api_model_name": "gpt-5.2",
      "aliases": ["gpt5.2", "gpt-5-2"],
      "cost_per_million_input": 1.75,
      "cost_per_million_output": 14.00
    },
    "gpt-5.2-pro": {
      "provider": "openai",
      "api_model_name": "gpt-5.2-pro",
      "aliases": ["gpt5.2-pro", "gpt-5-2-pro"],
      "cost_per_million_input": 21.00,
      "cost_per_million_output": 168.00
    },
    "gpt-5-mini": {
      "provider": "openai",
      "api_model_name": "gpt-5-mini",
      "aliases": ["gpt5-mini", "gpt5mini"],
      "cost_per_million_input": 0.25,
      "cost_per_million_output": 2.00
    },
    "o1": {
      "provider": "openai",
      "api_model_name": "o1-2024-12-17",
      "aliases": [],
      "cost_per_million_input": 15.00,
      "cost_per_million_output": 60.00
    },
    "o1-preview": {
      "provider": "openai",
      "api_model_name": "o1-preview-2024-12-17",
      "aliases": [],
      "cost_per_million_input": 15.00,
      "cost_per_million_output": 60.00
    },
    "o1-mini": {
      "provider": "openai",
      "api_model_name": "o1-mini-2024-09-12",
      "aliases": [],
      "cost_per_million_input": 1.10,
      "cost_per_million_output": 4.40
    },
    "o3-mini": {
      "provider": "openai",
      "api_model_name": "o3-mini-2025-01-31",
      "aliases": [],
      "cost_per_million_input": 1.10,
      "cost_per_million_output": 4.40
    },
    "o4-mini": {
      "provider": "openai",
      "api_model_name": "o4-mini",
      "aliases": [],
      "cost_per_million_input": 4.00,
      "cost_per_million_output": 16.00
    },
    "claude-4.5-opus": {
      "provider": "anthropic",
      "api_model_name": "claude-4.5-opus",
      "aliases": [],
      "cost_per_million_input": 5.00,
      "cost_per_million_output": 25.00
    },
    "claude-4.5-sonnet": {
      "provider": "anthropic",
      "api_model_name": "claude-4.5-sonnet",
      "aliases": [],
      "cost_per_million_input": 3.00,
      "cost_per_million_output": 15.00
    },
    "claude-4.5-haiku": {
      "provider": "anthropic",
      "api_model_name": "claude-4.5-haiku",
      "aliases": [],
      "cost_per_million_input": 1.00,
      "cost_per_million_output": 5.00
    },
    "claude-4.1-opus": {
      "provider": "anthropic",
      "api_model_name": "claude-4.1-opus",
      "aliases": [],
      "cost_per_million_input": 15.00,
      "cost_per_million_output": 75.00
    },
    "claude-4-opus": {
      "provider": "anthropic",
      "api_model_name": "claude-4-opus",
      "aliases": [],
      "cost_per_million_input": 15.00,
      "cost_per_million_output": 75.00
    },
    "claude-4-sonnet": {
      "provider": "anthropic",
      "api_model_name": "claude-4-sonnet",
      "aliases": [],
      "cost_per_million_input": 3.00,
      "cost_per_million_output": 15.00
    },
    "claude-3-7-sonnet": {
      "provider": "anthropic",
      "api_model_name": "claude-3-7-sonnet",
      "aliases": [],
      "cost_per_million_input": 3.00,
      "cost_per_million_output": 15.00
    },
    "claude-3-5-sonnet": {
      "provider": "anthropic",
      "api_model_name": "claude-3-5-sonnet",
      "aliases": [],
      "cost_per_million_input": 3.00,
      "cost_per_million_output": 15.00
    },
    "claude-3-5-haiku": {
      "provider": "anthropic",
      "api_model_name": "claude-3-5-haiku",
      "aliases": [],
      "cost_per_million_input": 0.80,
      "cost_per_million_output": 4.00
    },
    "deepseek-chat": {
      "provider": "deepseek",
      "api_model_name": "deepseek-chat",
      "aliases": [],
      "cost_per_million_input": 0.27,
      "cost_per_million_output": 1.10
    },
    "gemini-3.0-pro": {
      "provider": "google",
      "api_model_name": "gemini-3-pro-preview",
      "aliases": ["gemini-3-pro", "gemini-3.0-pro-preview"],
      "cost_per_million_input": 2.00,
      "cost_per_million_output": 12.00
    },
    "gemini-3.0-flash": {
      "provider": "google",
      "api_model_name": "gemini-3-flash-preview",
      "aliases": ["gemini-3-flash", "gemini-3.0-flash-preview"],
      "cost_per_million_input": 0.50,
      "cost_per_million_output": 3.00
    },
    "gemini-2.5-pro": {
      "provider": "google",
      "api_model_name": "gemini-2.5-pro",
      "aliases": [],
      "cost_per_million_input": 1.25,
      "cost_per_million_output": 10.00
    },
    "gemini-2.5-flash": {
      "provider": "google",
      "api_model_name": "gemini-2.5-flash",
      "aliases": [],
      "cost_per_million_input": 0.30,
      "cost_per_million_output": 2.50
    },
    "gemini-2.5-flash-lite": {
      "provider": "google",
      "api_model_name": "gemini-2.5-flash-lite",
      "aliases": [],
      "cost_per_million_input": 0.10,
      "cost_per_million_output": 0.40
    },
    "gemini-2.0-flash": {
      "provider": "google",
      "api_model_name": "gemini-2.0-flash",
      "aliases": [],
      "cost_per_million_input": null,
      "cost_per_million_output": null
    }
  }
}
```

- [ ] **Step 3: Commit**

```bash
git add configs/models_pricing.json
git commit -m "feat: add configs/models_pricing.json with all current model data"
```

---

### Task 2: Create `model_registry.py`

**Files:**
- Create: `model_registry.py`
- Create: `tests/test_model_registry.py`

- [ ] **Step 1: Write tests for ModelRegistry**

Create `tests/test_model_registry.py`:

```python
import json
import os
import pytest
from unittest.mock import patch
from datetime import datetime, timezone

# We'll import after creating the module
# from model_registry import ModelRegistry, ModelNotFoundError


SAMPLE_CONFIG = {
    "last_updated": "2026-03-28T00:00:00Z",
    "models": {
        "gpt-4o": {
            "provider": "openai",
            "api_model_name": "gpt-4o-2024-08-06",
            "aliases": ["gpt4o"],
            "cost_per_million_input": 2.50,
            "cost_per_million_output": 10.00
        },
        "claude-3-5-sonnet": {
            "provider": "anthropic",
            "api_model_name": "claude-3-5-sonnet-20241022",
            "aliases": ["claude-3-5-sonnet-latest"],
            "cost_per_million_input": 3.00,
            "cost_per_million_output": 15.00
        },
        "deepseek-chat": {
            "provider": "deepseek",
            "api_model_name": "deepseek-chat",
            "aliases": [],
            "cost_per_million_input": 0.27,
            "cost_per_million_output": 1.10
        },
        "gemini-2.5-flash": {
            "provider": "google",
            "api_model_name": "gemini-2.5-flash",
            "aliases": [],
            "cost_per_million_input": 0.30,
            "cost_per_million_output": 2.50
        },
        "null-cost-model": {
            "provider": "openai",
            "api_model_name": "null-cost-model",
            "aliases": [],
            "cost_per_million_input": null,
            "cost_per_million_output": null
        }
    }
}


@pytest.fixture
def sample_config_path(tmp_path):
    config_path = tmp_path / "configs" / "models_pricing.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(SAMPLE_CONFIG))
    return str(config_path)


@pytest.fixture
def registry(sample_config_path):
    from model_registry import ModelRegistry
    return ModelRegistry(config_path=sample_config_path, auto_refresh=False)


class TestModelResolution:
    def test_resolve_canonical_name(self, registry):
        model = registry.get_model("gpt-4o")
        assert model["provider"] == "openai"
        assert model["api_model_name"] == "gpt-4o-2024-08-06"

    def test_resolve_alias(self, registry):
        model = registry.get_model("gpt4o")
        assert model["api_model_name"] == "gpt-4o-2024-08-06"

    def test_unknown_model_raises(self, registry):
        from model_registry import ModelNotFoundError
        with pytest.raises(ModelNotFoundError):
            registry.get_model("nonexistent-model")

    def test_resolve_alias_method(self, registry):
        assert registry.resolve_alias("gpt4o") == "gpt-4o"

    def test_resolve_canonical_returns_same(self, registry):
        assert registry.resolve_alias("gpt-4o") == "gpt-4o"

    def test_anthropic_startswith_matching(self, registry):
        """Anthropic models with suffixes like -latest or -20241022 should match."""
        model = registry.get_model("claude-3-5-sonnet-20241022")
        assert model["provider"] == "anthropic"


class TestProviderRouting:
    def test_get_provider(self, registry):
        assert registry.get_provider("gpt-4o") == "openai"
        assert registry.get_provider("claude-3-5-sonnet") == "anthropic"
        assert registry.get_provider("deepseek-chat") == "deepseek"
        assert registry.get_provider("gemini-2.5-flash") == "google"

    def test_get_api_model_name(self, registry):
        assert registry.get_api_model_name("gpt-4o") == "gpt-4o-2024-08-06"

    def test_get_base_url(self, registry):
        assert registry.get_base_url("gpt-4o") is None
        assert "deepseek" in registry.get_base_url("deepseek-chat")
        assert "generativelanguage" in registry.get_base_url("gemini-2.5-flash")


class TestCostEstimation:
    def test_get_cost_input(self, registry):
        assert registry.get_cost_input("gpt-4o") == 2.50 / 1_000_000

    def test_get_cost_output(self, registry):
        assert registry.get_cost_output("gpt-4o") == 10.00 / 1_000_000

    def test_null_cost_returns_none(self, registry):
        assert registry.get_cost_input("null-cost-model") is None
        assert registry.get_cost_output("null-cost-model") is None

    def test_curr_cost_est_empty(self, registry):
        assert registry.curr_cost_est() == 0.0

    def test_curr_cost_est_with_tokens(self, registry):
        registry.tokens_in["gpt-4o"] = 1000
        registry.tokens_out["gpt-4o"] = 500
        cost = registry.curr_cost_est()
        expected = 1000 * (2.50 / 1_000_000) + 500 * (10.00 / 1_000_000)
        assert abs(cost - expected) < 1e-10


class TestListModels:
    def test_list_all(self, registry):
        models = registry.list_models()
        assert "gpt-4o" in models
        assert "claude-3-5-sonnet" in models

    def test_list_by_provider(self, registry):
        models = registry.list_models(provider="openai")
        assert "gpt-4o" in models
        assert "claude-3-5-sonnet" not in models


class TestFallback:
    def test_missing_file_uses_defaults(self, tmp_path):
        from model_registry import ModelRegistry
        bad_path = str(tmp_path / "nonexistent" / "models_pricing.json")
        reg = ModelRegistry(config_path=bad_path, auto_refresh=False)
        # Should have loaded DEFAULT_MODELS
        assert len(reg.list_models()) > 0

    def test_corrupt_file_uses_defaults(self, tmp_path):
        from model_registry import ModelRegistry
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json{{{")
        reg = ModelRegistry(config_path=str(bad_file), auto_refresh=False)
        assert len(reg.list_models()) > 0


class TestStaleness:
    def test_is_stale_when_old(self, sample_config_path):
        from model_registry import ModelRegistry
        # Set last_updated to 30 days ago
        with open(sample_config_path) as f:
            data = json.load(f)
        data["last_updated"] = "2026-02-01T00:00:00Z"
        with open(sample_config_path, "w") as f:
            json.dump(data, f)
        reg = ModelRegistry(config_path=sample_config_path, auto_refresh=False)
        assert reg.is_stale() is True

    def test_is_not_stale_when_fresh(self, sample_config_path):
        from model_registry import ModelRegistry
        with open(sample_config_path) as f:
            data = json.load(f)
        data["last_updated"] = datetime.now(timezone.utc).isoformat()
        with open(sample_config_path, "w") as f:
            json.dump(data, f)
        reg = ModelRegistry(config_path=sample_config_path, auto_refresh=False)
        assert reg.is_stale() is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd d:/GitHub/AgentLaboratory && uv run pytest tests/test_model_registry.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'model_registry'`

- [ ] **Step 3: Implement `model_registry.py`**

Create `model_registry.py` in the project root:

```python
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


# Hardcoded fallback — all models from configs/models_pricing.json
# Used only when the JSON file is missing or corrupt
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd d:/GitHub/AgentLaboratory && uv run pytest tests/test_model_registry.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add model_registry.py tests/test_model_registry.py
git commit -m "feat: add ModelRegistry with alias resolution, provider routing, and cost tracking"
```

---

### Task 3: Add provider dispatcher to `provider.py`

**Files:**
- Modify: `provider.py:92-127` (add after `AnthropicProvider`)

- [ ] **Step 1: Add `get_provider_response` dispatcher function**

Append to `provider.py` after the `AnthropicProvider` class:

```python
def get_provider_response(provider, api_key, model_name, user_prompt, system_prompt, temperature=None, base_url=None):
    """Dispatch to the correct provider based on provider string."""
    if provider == "anthropic":
        return AnthropicProvider.get_response(
            api_key=api_key,
            model_name=model_name,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
        )
    else:
        # openai, google, deepseek, ollama all use OpenAI-compatible API
        return OpenaiProvider.get_response(
            api_key=api_key,
            model_name=model_name,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            base_url=base_url,
        )
```

- [ ] **Step 2: Commit**

```bash
git add provider.py
git commit -m "feat: add get_provider_response dispatcher to provider.py"
```

---

### Task 4: Rewrite `inference.py` to use `ModelRegistry`

**Files:**
- Modify: `inference.py` (full rewrite of lines 1-378)

- [ ] **Step 1: Rewrite `inference.py`**

Replace the entire file with:

```python
import os
import tiktoken
import time

from config import OLLAMA_API_BASE_URL
from model_registry import ModelRegistry
from provider import get_provider_response
from utils import remove_thinking_process

# Global registry instance
registry = ModelRegistry()

encoding = tiktoken.get_encoding("cl100k_base")


def curr_cost_est():
    return registry.curr_cost_est()


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

    if (preloaded_openai_api is None and
        preload_anthropic_api is None and
        preload_google_api is None and
        preload_deepseek_api is None):
        raise Exception("No API key provided in query_model function")

    # Handle Ollama passthrough
    if preloaded_openai_api == "ollama":
        return _query_ollama(model_str, prompt, system_prompt, tries, timeout, temp)

    for _ in range(tries):
        try:
            # Resolve model via registry
            canonical = registry.get_canonical_for_cost(model_str)
            provider = registry.get_provider(model_str)
            api_model_name = registry.get_api_model_name(model_str)
            base_url = registry.get_base_url(model_str)

            # Determine API key based on provider
            api_key_map = {
                "openai": os.getenv('OPENAI_API_KEY'),
                "anthropic": os.getenv('ANTHROPIC_API_KEY'),
                "google": os.getenv('GOOGLE_API_KEY'),
                "deepseek": os.getenv('DEEPSEEK_API_KEY'),
            }
            api_key = api_key_map.get(provider)
            if api_key is None:
                raise Exception(f"No API key set for provider '{provider}'")

            answer = get_provider_response(
                provider=provider,
                api_key=api_key,
                model_name=api_model_name,
                user_prompt=prompt,
                system_prompt=system_prompt,
                temperature=temp,
                base_url=base_url,
            )

            answer = remove_thinking_process(answer)

            # Cost estimation
            try:
                try:
                    model_encoding = tiktoken.encoding_for_model(canonical)
                except KeyError:
                    model_encoding = tiktoken.encoding_for_model("gpt-4o")
                if canonical not in registry.tokens_in:
                    registry.tokens_in[canonical] = 0
                    registry.tokens_out[canonical] = 0
                registry.tokens_in[canonical] += len(model_encoding.encode(system_prompt + prompt))
                registry.tokens_out[canonical] += len(model_encoding.encode(answer))
                if print_cost:
                    print(f"Current experiment cost = ${curr_cost_est()}, ** Approximate values, may not reflect true cost")
            except Exception as e:
                if print_cost:
                    print(f"Cost approximation has an error? {e}")

            return answer
        except Exception as e:
            print("Inference Exception:", e)
            time.sleep(timeout)
            continue
    raise Exception("Max retries: timeout")


def _query_ollama(model_str, prompt, system_prompt, tries, timeout, temp):
    """Handle Ollama models — bypass registry, pass model string directly."""
    from provider import OpenaiProvider
    for _ in range(tries):
        try:
            answer = OpenaiProvider.get_response(
                api_key="ollama",
                model_name=model_str,
                user_prompt=prompt,
                system_prompt=system_prompt,
                temperature=temp,
                base_url=OLLAMA_API_BASE_URL,
            )
            return remove_thinking_process(answer)
        except Exception as e:
            print("Inference Exception:", e)
            time.sleep(timeout)
            continue
    raise Exception("Max retries: timeout")
```

- [ ] **Step 2: Verify the project still works**

```bash
cd d:/GitHub/AgentLaboratory && uv run python -c "from inference import query_model, curr_cost_est; print('Import OK')"
```

Expected: `Import OK`

- [ ] **Step 3: Commit**

```bash
git add inference.py
git commit -m "refactor: rewrite inference.py to use ModelRegistry instead of hardcoded if/elif chain"
```

---

### Task 5: Create `model_fetcher.py`

**Files:**
- Create: `model_fetcher.py`
- Create: `tests/test_model_fetcher.py`

- [ ] **Step 1: Write tests for model_fetcher**

Create `tests/test_model_fetcher.py`:

```python
import json
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone


SAMPLE_OPENAI_MODELS_RESPONSE = {
    "data": [
        {"id": "gpt-4o", "object": "model"},
        {"id": "gpt-4o-mini", "object": "model"},
        {"id": "text-embedding-ada-002", "object": "model"},
        {"id": "dall-e-3", "object": "model"},
    ]
}


SAMPLE_ANTHROPIC_MODELS_RESPONSE = {
    "data": [
        {"id": "claude-3-5-sonnet-20241022", "type": "model"},
        {"id": "claude-3-5-haiku-20241022", "type": "model"},
    ]
}


SAMPLE_GOOGLE_MODELS_RESPONSE = {
    "models": [
        {"name": "models/gemini-2.5-flash", "displayName": "Gemini 2.5 Flash"},
        {"name": "models/embedding-001", "displayName": "Embedding 001"},
    ]
}


class TestFetchModelsFromAPI:
    @patch("model_fetcher.requests.get")
    def test_fetch_openai_models(self, mock_get):
        from model_fetcher import fetch_models_from_api
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = SAMPLE_OPENAI_MODELS_RESPONSE
        mock_get.return_value = mock_resp

        models = fetch_models_from_api("openai", api_key="test-key")
        # Should filter out embedding and dall-e models
        assert "gpt-4o" in models
        assert "gpt-4o-mini" in models
        assert "text-embedding-ada-002" not in models
        assert "dall-e-3" not in models

    @patch("model_fetcher.requests.get")
    def test_fetch_returns_empty_on_failure(self, mock_get):
        from model_fetcher import fetch_models_from_api
        mock_get.side_effect = Exception("Network error")
        models = fetch_models_from_api("openai", api_key="test-key")
        assert models == []

    @patch("model_fetcher.requests.get")
    def test_fetch_google_models(self, mock_get):
        from model_fetcher import fetch_models_from_api
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = SAMPLE_GOOGLE_MODELS_RESPONSE
        mock_get.return_value = mock_resp

        models = fetch_models_from_api("google", api_key="test-key")
        assert "gemini-2.5-flash" in models
        assert "embedding-001" not in models


class TestMergeModels:
    def test_merge_adds_new_model(self):
        from model_fetcher import merge_discovered_models
        existing = {
            "last_updated": "2026-03-01T00:00:00Z",
            "models": {}
        }
        discovered = {"openai": ["gpt-4o-new"]}
        result = merge_discovered_models(existing, discovered)
        assert "gpt-4o-new" in result["models"]
        assert result["models"]["gpt-4o-new"]["provider"] == "openai"
        assert result["models"]["gpt-4o-new"]["cost_per_million_input"] is None

    def test_merge_keeps_existing(self):
        from model_fetcher import merge_discovered_models
        existing = {
            "last_updated": "2026-03-01T00:00:00Z",
            "models": {
                "gpt-4o": {
                    "provider": "openai",
                    "api_model_name": "gpt-4o",
                    "aliases": [],
                    "cost_per_million_input": 2.50,
                    "cost_per_million_output": 10.00
                }
            }
        }
        discovered = {"openai": ["gpt-4o"]}
        result = merge_discovered_models(existing, discovered)
        # Existing model should keep its pricing
        assert result["models"]["gpt-4o"]["cost_per_million_input"] == 2.50

    def test_merge_does_not_remove_existing(self):
        from model_fetcher import merge_discovered_models
        existing = {
            "last_updated": "2026-03-01T00:00:00Z",
            "models": {
                "gpt-4o": {
                    "provider": "openai",
                    "api_model_name": "gpt-4o",
                    "aliases": [],
                    "cost_per_million_input": 2.50,
                    "cost_per_million_output": 10.00
                }
            }
        }
        discovered = {"openai": ["gpt-4o-new"]}
        result = merge_discovered_models(existing, discovered)
        assert "gpt-4o" in result["models"]


class TestUpdateModelsPricing:
    @patch("model_fetcher.fetch_pricing")
    @patch("model_fetcher.fetch_models_from_api")
    def test_update_writes_file(self, mock_fetch_models, mock_fetch_pricing, tmp_path):
        from model_fetcher import update_models_pricing
        config_path = str(tmp_path / "models_pricing.json")
        existing = {
            "last_updated": "2026-01-01T00:00:00Z",
            "models": {}
        }
        with open(config_path, "w") as f:
            json.dump(existing, f)

        mock_fetch_models.return_value = ["gpt-4o"]
        mock_fetch_pricing.return_value = {}

        result = update_models_pricing(config_path, force=True)
        assert result is True

        with open(config_path) as f:
            data = json.load(f)
        assert data["last_updated"] != "2026-01-01T00:00:00Z"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd d:/GitHub/AgentLaboratory && uv run pytest tests/test_model_fetcher.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'model_fetcher'`

- [ ] **Step 3: Implement `model_fetcher.py`**

Create `model_fetcher.py`:

```python
import json
import os
import re
from datetime import datetime, timezone, timedelta

import requests
from bs4 import BeautifulSoup

from config import GOOGLE_GENERATIVE_API_BASE_URL, DEEPSEEK_API_BASE_URL

FETCH_TIMEOUT = 10

# Model ID prefixes to exclude from API discovery (embeddings, image gen, etc.)
EXCLUDED_PREFIXES = (
    "text-embedding", "embedding", "dall-e", "tts-", "whisper",
    "davinci", "babbage", "curie", "ada",
)

# Provider API endpoints for model discovery
PROVIDER_API_ENDPOINTS = {
    "openai": "https://api.openai.com/v1/models",
    "anthropic": "https://api.anthropic.com/v1/models",
    "google": f"{GOOGLE_GENERATIVE_API_BASE_URL}models",
    "deepseek": f"{DEEPSEEK_API_BASE_URL}/models",
}

# Provider pricing page URLs
PROVIDER_PRICING_URLS = {
    "openai": "https://platform.openai.com/docs/pricing",
    "anthropic": "https://docs.anthropic.com/en/docs/about-claude/models",
    "google": "https://ai.google.dev/gemini-api/docs/pricing",
    "deepseek": "https://api-docs.deepseek.com/quick_start/pricing",
}


def fetch_models_from_api(provider, api_key):
    """Fetch available model IDs from a provider's API. Returns list of model ID strings."""
    endpoint = PROVIDER_API_ENDPOINTS.get(provider)
    if not endpoint:
        return []

    try:
        headers = {}
        params = {}
        if provider == "openai" or provider == "deepseek":
            headers["Authorization"] = f"Bearer {api_key}"
        elif provider == "anthropic":
            headers["x-api-key"] = api_key
            headers["anthropic-version"] = "2023-06-01"
        elif provider == "google":
            params["key"] = api_key

        resp = requests.get(endpoint, headers=headers, params=params, timeout=FETCH_TIMEOUT)
        if resp.status_code != 200:
            print(f"Warning: {provider} API returned status {resp.status_code}")
            return []

        data = resp.json()

        if provider == "google":
            # Google returns {"models": [{"name": "models/gemini-2.5-flash", ...}]}
            raw_ids = [m["name"].replace("models/", "") for m in data.get("models", [])]
        else:
            # OpenAI/Anthropic/DeepSeek return {"data": [{"id": "model-name", ...}]}
            raw_ids = [m["id"] for m in data.get("data", [])]

        # Filter out non-chat models
        return [mid for mid in raw_ids if not mid.startswith(EXCLUDED_PREFIXES)]

    except Exception as e:
        print(f"Warning: Failed to fetch models from {provider}: {e}")
        return []


def fetch_pricing(provider):
    """Scrape pricing page for a provider. Returns dict of {model_name: {input: float, output: float}}.
    Returns empty dict on failure."""
    url = PROVIDER_PRICING_URLS.get(provider)
    if not url:
        return {}

    try:
        resp = requests.get(url, timeout=FETCH_TIMEOUT, headers={"User-Agent": "AgentLaboratory/1.0"})
        if resp.status_code != 200:
            print(f"Warning: {provider} pricing page returned status {resp.status_code}")
            return {}

        soup = BeautifulSoup(resp.text, "html.parser")

        if provider == "openai":
            return _parse_openai_pricing(soup)
        elif provider == "anthropic":
            return _parse_anthropic_pricing(soup)
        elif provider == "google":
            return _parse_google_pricing(soup)
        elif provider == "deepseek":
            return _parse_deepseek_pricing(soup)

    except Exception as e:
        print(f"Warning: Failed to fetch pricing for {provider}: {e}")
        return {}


def _parse_price_str(text):
    """Extract a numeric price from a string like '$2.50' or '$0.30 / 1M tokens'."""
    match = re.search(r'\$?([\d.]+)', text.strip())
    if match:
        return float(match.group(1))
    return None


def _parse_openai_pricing(soup):
    """Parse OpenAI pricing page. Returns {model: {input: float, output: float}}."""
    pricing = {}
    # Look for tables with pricing data
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all(["td", "th"])
            if len(cells) >= 3:
                model_name = cells[0].get_text(strip=True).lower()
                input_price = _parse_price_str(cells[1].get_text(strip=True))
                output_price = _parse_price_str(cells[2].get_text(strip=True))
                if input_price is not None and output_price is not None:
                    pricing[model_name] = {"input": input_price, "output": output_price}
    return pricing


def _parse_anthropic_pricing(soup):
    """Parse Anthropic pricing page."""
    pricing = {}
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all(["td", "th"])
            if len(cells) >= 3:
                model_name = cells[0].get_text(strip=True).lower()
                input_price = _parse_price_str(cells[1].get_text(strip=True))
                output_price = _parse_price_str(cells[2].get_text(strip=True))
                if input_price is not None and output_price is not None:
                    pricing[model_name] = {"input": input_price, "output": output_price}
    return pricing


def _parse_google_pricing(soup):
    """Parse Google pricing page."""
    pricing = {}
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all(["td", "th"])
            if len(cells) >= 3:
                model_name = cells[0].get_text(strip=True).lower()
                input_price = _parse_price_str(cells[1].get_text(strip=True))
                output_price = _parse_price_str(cells[2].get_text(strip=True))
                if input_price is not None and output_price is not None:
                    pricing[model_name] = {"input": input_price, "output": output_price}
    return pricing


def _parse_deepseek_pricing(soup):
    """Parse DeepSeek pricing page."""
    pricing = {}
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all(["td", "th"])
            if len(cells) >= 3:
                model_name = cells[0].get_text(strip=True).lower()
                input_price = _parse_price_str(cells[1].get_text(strip=True))
                output_price = _parse_price_str(cells[2].get_text(strip=True))
                if input_price is not None and output_price is not None:
                    pricing[model_name] = {"input": input_price, "output": output_price}
    return pricing


def merge_discovered_models(existing_data, discovered_by_provider, pricing_by_provider=None):
    """Merge newly discovered models into existing config data.

    Args:
        existing_data: Current JSON data dict
        discovered_by_provider: dict of {provider: [model_id, ...]}
        pricing_by_provider: dict of {provider: {model_name: {input: float, output: float}}}

    Returns:
        Updated data dict
    """
    if pricing_by_provider is None:
        pricing_by_provider = {}

    models = existing_data.get("models", {})

    for provider, model_ids in discovered_by_provider.items():
        provider_pricing = pricing_by_provider.get(provider, {})
        for model_id in model_ids:
            if model_id not in models:
                # New model — add with pricing if available
                price_info = provider_pricing.get(model_id, {})
                models[model_id] = {
                    "provider": provider,
                    "api_model_name": model_id,
                    "aliases": [],
                    "cost_per_million_input": price_info.get("input"),
                    "cost_per_million_output": price_info.get("output"),
                }
                if price_info:
                    print(f"  Added new model: {model_id} (with pricing)")
                else:
                    print(f"  Added new model: {model_id} (no pricing available)")
            else:
                # Existing model — update pricing if we scraped new data
                price_info = provider_pricing.get(model_id, {})
                if price_info:
                    models[model_id]["cost_per_million_input"] = price_info.get("input", models[model_id].get("cost_per_million_input"))
                    models[model_id]["cost_per_million_output"] = price_info.get("output", models[model_id].get("cost_per_million_output"))

    existing_data["models"] = models
    existing_data["last_updated"] = datetime.now(timezone.utc).isoformat()
    return existing_data


def update_models_pricing(config_path, force=False):
    """Main entry point: refresh models_pricing.json.

    Args:
        config_path: Path to models_pricing.json
        force: If True, skip staleness check

    Returns:
        True if update was performed, False if skipped
    """
    # Load existing data
    try:
        with open(config_path) as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = {"last_updated": None, "models": {}}

    # Check staleness
    if not force:
        last = existing_data.get("last_updated")
        if last:
            try:
                last_dt = datetime.fromisoformat(last)
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=timezone.utc)
                if datetime.now(timezone.utc) - last_dt < timedelta(days=7):
                    print("Model pricing data is fresh, skipping update.")
                    return False
            except (ValueError, TypeError):
                pass

    print("Updating model pricing data...")

    # Gather API keys
    api_keys = {
        "openai": os.getenv("OPENAI_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "google": os.getenv("GOOGLE_API_KEY"),
        "deepseek": os.getenv("DEEPSEEK_API_KEY"),
    }

    # Discover models via APIs
    discovered = {}
    for provider, key in api_keys.items():
        if key and key != "ollama":
            print(f"  Fetching models from {provider}...")
            models = fetch_models_from_api(provider, key)
            if models:
                discovered[provider] = models
                print(f"  Found {len(models)} models from {provider}")

    # Scrape pricing
    pricing = {}
    for provider in ["openai", "anthropic", "google", "deepseek"]:
        print(f"  Fetching pricing from {provider}...")
        provider_pricing = fetch_pricing(provider)
        if provider_pricing:
            pricing[provider] = provider_pricing
            print(f"  Got pricing for {len(provider_pricing)} models from {provider}")

    # Merge
    updated_data = merge_discovered_models(existing_data, discovered, pricing)

    # Write
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(updated_data, f, indent=2)

    print(f"Model pricing data updated at {config_path}")
    return True
```

- [ ] **Step 4: Run tests**

```bash
cd d:/GitHub/AgentLaboratory && uv run pytest tests/test_model_fetcher.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add model_fetcher.py tests/test_model_fetcher.py
git commit -m "feat: add model_fetcher with API discovery and pricing scraping"
```

---

### Task 6: Create `update_models.py` CLI entry point

**Files:**
- Create: `update_models.py`

- [ ] **Step 1: Create `update_models.py`**

```python
"""CLI tool to refresh model pricing data.

Usage:
    python update_models.py          # Update if stale (>7 days)
    python update_models.py --force  # Force update regardless of freshness
"""
import argparse
import os
import sys

from model_fetcher import update_models_pricing
from model_registry import DEFAULT_CONFIG_PATH


def main():
    parser = argparse.ArgumentParser(description="Update model pricing data")
    parser.add_argument("--force", action="store_true", help="Force update regardless of freshness")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to models_pricing.json")
    args = parser.parse_args()

    try:
        updated = update_models_pricing(args.config, force=args.force)
        if updated:
            print("Done.")
        else:
            print("No update needed. Use --force to override.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it runs**

```bash
cd d:/GitHub/AgentLaboratory && uv run python update_models.py --help
```

Expected: Shows help text with `--force` and `--config` options.

- [ ] **Step 3: Commit**

```bash
git add update_models.py
git commit -m "feat: add update_models.py CLI for manual pricing refresh"
```

---

### Task 7: Update `requirements.txt`

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add dependencies**

Add these lines to `requirements.txt`:

```
requests
beautifulsoup4
```

- [ ] **Step 2: Commit**

```bash
git add requirements.txt
git commit -m "chore: add requests and beautifulsoup4 to requirements.txt"
```

---

### Task 8: Update UI files to use registry

**Files:**
- Modify: `AgentLaboratoryWebUI/config_gradio.py:185-198`
- Modify: `AgentLaboratoryWebUI/app.py:75-90`

- [ ] **Step 1: Update `config_gradio.py`**

Replace the hardcoded `llm_backend_options` list (lines 185-198) with:

```python
from model_registry import ModelRegistry

_registry = ModelRegistry(auto_refresh=False)
llm_backend_options = _registry.list_models()
```

This replaces the static list:
```python
# OLD (remove):
# llm_backend_options = [
#     "o1", "o1-preview", "o1-mini", "o3-mini",
#     "gpt-4o", "gpt-4o-mini",
#     "deepseek-chat",
#     ...
# ]
```

- [ ] **Step 2: Update `app.py`**

In the Flask app, add an endpoint or use registry for model listing. The key change is replacing any hardcoded model list references with `ModelRegistry.list_models()`.

Find any place in `app.py` where model names are listed or validated and replace with registry lookups.

- [ ] **Step 3: Verify UI imports work**

```bash
cd d:/GitHub/AgentLaboratory && uv run python -c "from AgentLaboratoryWebUI.config_gradio import llm_backend_options; print(llm_backend_options[:5])"
```

Expected: Prints first 5 model names from the registry.

- [ ] **Step 4: Commit**

```bash
git add AgentLaboratoryWebUI/config_gradio.py AgentLaboratoryWebUI/app.py
git commit -m "refactor: use ModelRegistry for UI model dropdowns instead of hardcoded lists"
```

---

### Task 9: Integration test — end-to-end validation

**Files:**
- No new files — validation only

- [ ] **Step 1: Verify imports chain works**

```bash
cd d:/GitHub/AgentLaboratory && uv run python -c "
from inference import query_model, curr_cost_est
from model_registry import ModelRegistry
reg = ModelRegistry(auto_refresh=False)
print('Models:', len(reg.list_models()))
print('OpenAI models:', reg.list_models(provider='openai'))
print('Resolve gpt4o:', reg.resolve_alias('gpt4o'))
print('Provider for claude-3-5-sonnet:', reg.get_provider('claude-3-5-sonnet'))
print('API name for gpt-4o:', reg.get_api_model_name('gpt-4o'))
print('Cost est:', curr_cost_est())
print('All OK')
"
```

Expected: Prints model info and `All OK`.

- [ ] **Step 2: Verify update_models.py runs (dry run)**

```bash
cd d:/GitHub/AgentLaboratory && uv run python update_models.py
```

Expected: Either "No update needed" (if JSON is fresh) or attempts to fetch and prints progress.

- [ ] **Step 3: Final commit if any fixups needed**

```bash
git add -A && git commit -m "fix: integration fixups for model registry"
```

Only run this if fixups were needed.
