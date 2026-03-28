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
            raw_ids = [m["name"].replace("models/", "") for m in data.get("models", [])]
        else:
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
