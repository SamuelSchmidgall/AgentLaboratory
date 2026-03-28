import json
import os
import pytest
from unittest.mock import patch
from datetime import datetime, timezone


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
            "cost_per_million_input": None,
            "cost_per_million_output": None
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
