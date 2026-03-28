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


SAMPLE_GOOGLE_MODELS_RESPONSE = {
    "models": [
        {"name": "models/gemini-2.5-flash", "displayName": "Gemini 2.5 Flash"},
        {"name": "models/embedding-001", "displayName": "Embedding 001"},
    ]
}


SAMPLE_AGGREGATOR_RESPONSE = {
    "results": [
        {
            "model": "gpt-4o",
            "provider_name": "OpenAI",
            "input_price_per_1m_tokens": 2.50,
            "output_price_per_1m_tokens": 10.00,
        },
        {
            "model": "claude-3-5-sonnet",
            "provider_name": "Anthropic",
            "input_price_per_1m_tokens": 3.00,
            "output_price_per_1m_tokens": 15.00,
        },
        {
            "model": "gemini-2.5-flash",
            "provider_name": "Google",
            "input_price_per_1m_tokens": 0.30,
            "output_price_per_1m_tokens": 2.50,
        },
        {
            "model": "some-unknown-provider-model",
            "provider_name": "SomeOtherProvider",
            "input_price_per_1m_tokens": 1.00,
            "output_price_per_1m_tokens": 2.00,
        },
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


class TestFetchPricingFromAggregator:
    @patch("model_fetcher.requests.get")
    def test_fetch_pricing(self, mock_get):
        from model_fetcher import fetch_pricing_from_aggregator
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = SAMPLE_AGGREGATOR_RESPONSE
        mock_get.return_value = mock_resp

        pricing = fetch_pricing_from_aggregator()
        assert "openai" in pricing
        assert "anthropic" in pricing
        assert "google" in pricing
        # Unknown providers should be filtered out
        assert "someotherprovider" not in pricing

        assert pricing["openai"]["gpt-4o"]["input"] == 2.50
        assert pricing["openai"]["gpt-4o"]["output"] == 10.00
        assert pricing["anthropic"]["claude-3-5-sonnet"]["input"] == 3.00

    @patch("model_fetcher.requests.get")
    def test_fetch_pricing_returns_empty_on_failure(self, mock_get):
        from model_fetcher import fetch_pricing_from_aggregator
        mock_get.side_effect = Exception("Network error")
        pricing = fetch_pricing_from_aggregator()
        assert pricing == {}

    @patch("model_fetcher.requests.get")
    def test_fetch_pricing_handles_bad_status(self, mock_get):
        from model_fetcher import fetch_pricing_from_aggregator
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_get.return_value = mock_resp
        pricing = fetch_pricing_from_aggregator()
        assert pricing == {}


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

    def test_merge_updates_pricing_for_existing_models(self):
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
        pricing = {"openai": {"gpt-4o": {"input": 3.00, "output": 12.00}}}
        result = merge_discovered_models(existing, {}, pricing)
        assert result["models"]["gpt-4o"]["cost_per_million_input"] == 3.00
        assert result["models"]["gpt-4o"]["cost_per_million_output"] == 12.00


class TestUpdateModelsPricing:
    @patch("model_fetcher.fetch_pricing_from_aggregator")
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
        mock_fetch_pricing.return_value = {
            "openai": {"gpt-4o": {"input": 2.50, "output": 10.00}}
        }

        result = update_models_pricing(config_path, force=True)
        assert result is True

        with open(config_path) as f:
            data = json.load(f)
        assert data["last_updated"] != "2026-01-01T00:00:00Z"
