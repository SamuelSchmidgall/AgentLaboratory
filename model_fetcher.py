import json
import os
import re
from datetime import datetime, timezone, timedelta

import requests

from config import GOOGLE_GENERATIVE_API_BASE_URL, DEEPSEEK_API_BASE_URL

FETCH_TIMEOUT = 15

# Pricing aggregator API — single source for all providers
PRICING_API_URL = "https://pricepertoken.com/api/pricing"

# Map pricepertoken.com provider_name to our internal provider names
PROVIDER_NAME_MAP = {
    "OpenAI": "openai",
    "Anthropic": "anthropic",
    "Google": "google",
    "Deepseek": "deepseek",
}

# Model ID prefixes to exclude (embeddings, image gen, etc.)
EXCLUDED_PREFIXES = (
    "text-embedding", "embedding", "dall-e", "tts-", "whisper",
    "davinci", "babbage", "curie", "ada", "text-ada", "text-davinci",
)

# Provider API endpoints for model discovery
PROVIDER_API_ENDPOINTS = {
    "openai": "https://api.openai.com/v1/models",
    "anthropic": "https://api.anthropic.com/v1/models",
    "google": f"{GOOGLE_GENERATIVE_API_BASE_URL}models",
    "deepseek": f"{DEEPSEEK_API_BASE_URL}/models",
}


def fetch_models_from_api(provider, api_key):
    """Fetch available model IDs from a provider's API. Returns list of model ID strings."""
    endpoint = PROVIDER_API_ENDPOINTS.get(provider)
    if not endpoint:
        return []

    try:
        headers = {}
        params = {}
        if provider in ("openai", "deepseek"):
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
            raw_ids = [m["name"].replace("models/", "") for m in data.get("models", [])]
        else:
            raw_ids = [m["id"] for m in data.get("data", [])]

        # Filter out non-chat models
        return [mid for mid in raw_ids if not mid.startswith(EXCLUDED_PREFIXES)]

    except Exception as e:
        print(f"Warning: Failed to fetch models from {provider}: {e}")
        return []


def fetch_pricing_from_aggregator():
    """Fetch pricing for all providers from pricepertoken.com.

    Returns:
        dict of {provider: {model_id: {input: float, output: float}}}
        where input/output are per-million-token prices in USD.
    """
    try:
        resp = requests.get(PRICING_API_URL, timeout=FETCH_TIMEOUT, headers={
            "User-Agent": "Mozilla/5.0 (compatible; AgentLaboratory/1.0)",
        })
        if resp.status_code != 200:
            print(f"Warning: Pricing API returned status {resp.status_code}")
            return {}

        data = resp.json()
        results = data.get("results", [])
        print(f"  Pricing API returned {len(results)} models")

        pricing = {}
        for entry in results:
            provider_name = entry.get("provider_name", "")
            provider = PROVIDER_NAME_MAP.get(provider_name)
            if provider is None:
                continue

            model_id = entry.get("model", "")
            input_price = entry.get("input_price_per_1m_tokens")
            output_price = entry.get("output_price_per_1m_tokens")

            if not model_id or input_price is None or output_price is None:
                continue

            if provider not in pricing:
                pricing[provider] = {}

            pricing[provider][model_id] = {
                "input": float(input_price),
                "output": float(output_price),
            }

        for p, models in pricing.items():
            print(f"  {p}: {len(models)} models with pricing")

        return pricing

    except Exception as e:
        print(f"Warning: Failed to fetch pricing from aggregator: {e}")
        return {}


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
                price_info = provider_pricing.get(model_id, {})
                if price_info:
                    models[model_id]["cost_per_million_input"] = price_info.get("input", models[model_id].get("cost_per_million_input"))
                    models[model_id]["cost_per_million_output"] = price_info.get("output", models[model_id].get("cost_per_million_output"))

    # Also update pricing for existing models that weren't in discovered_by_provider
    # (e.g. models already in our JSON that have updated prices)
    for provider, provider_pricing in pricing_by_provider.items():
        for model_id, price_info in provider_pricing.items():
            if model_id in models and price_info:
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

    # Step 1: Fetch pricing from aggregator (no API keys needed!)
    print("  Fetching pricing from pricepertoken.com...")
    pricing = fetch_pricing_from_aggregator()

    # Step 2: Discover models via provider APIs (requires API keys)
    api_keys = {
        "openai": os.getenv("OPENAI_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "google": os.getenv("GOOGLE_API_KEY"),
        "deepseek": os.getenv("DEEPSEEK_API_KEY"),
    }

    discovered = {}
    for provider, key in api_keys.items():
        if key and key != "ollama":
            print(f"  Fetching models from {provider} API...")
            models = fetch_models_from_api(provider, key)
            if models:
                discovered[provider] = models
                print(f"  Found {len(models)} models from {provider}")
        else:
            print(f"  Skipping {provider} API (no API key set)")

    if not discovered and not pricing:
        print("Warning: No data fetched. Check your network connection.")
        # Still update timestamp to avoid repeated failures
        existing_data["last_updated"] = datetime.now(timezone.utc).isoformat()

    # Step 3: Merge
    updated_data = merge_discovered_models(existing_data, discovered, pricing)

    # Write
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(updated_data, f, indent=2)

    print(f"Model pricing data updated at {config_path}")
    return True
