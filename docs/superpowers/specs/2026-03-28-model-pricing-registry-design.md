# Model Pricing Registry — Design Spec

## Problem

Model pricing, aliases, and routing logic are hardcoded across multiple files (`inference.py`, `config_gradio.py`, `app.py`). Adding or updating a model requires editing 3+ files and touching a ~250-line if/elif chain. Pricing drifts out of date silently.

## Solution

Replace all hardcoded model/pricing data with a single JSON file (`configs/models_pricing.json`) backed by a registry class and a web-based fetch utility that auto-refreshes pricing and discovers new models.

## Data Model — `configs/models_pricing.json`

```json
{
  "last_updated": "2026-03-28T12:00:00Z",
  "models": {
    "gpt-4o": {
      "provider": "openai",
      "api_model_name": "gpt-4o-2024-08-06",
      "aliases": ["gpt4o"],
      "cost_per_million_input": 2.50,
      "cost_per_million_output": 10.00
    }
  }
}
```

Fields:
- **`provider`**: One of `openai`, `anthropic`, `google`, `deepseek`, `ollama`. Determines which provider class handles the request.
- **`api_model_name`**: Actual model name sent to the API (handles version suffix mapping).
- **`aliases`**: Alternative user-facing names that resolve to this model.
- **`cost_per_million_input/output`**: USD pricing. `null` if unknown.

## Architecture

### New files

| File | Purpose |
|------|---------|
| `model_registry.py` | `ModelRegistry` class — loads JSON, resolves aliases, routes to provider, calculates costs |
| `model_fetcher.py` | Fetch utility — API model discovery + pricing page scraping per provider |
| `update_models.py` | CLI entry point: `python update_models.py [--force]` |
| `configs/models_pricing.json` | Cached model/pricing data (checked into repo with defaults) |

### Modified files

| File | Change |
|------|--------|
| `inference.py` | Replace if/elif chain + costmaps with `ModelRegistry` lookups (~250 lines → ~15) |
| `provider.py` | Add `get_response_by_provider(provider_name, ...)` dispatcher |
| `config_gradio.py` | Populate UI dropdown from `ModelRegistry.list_models()` |
| `app.py` | Same — use registry for defaults and model lists |

### Flow

```
App startup
  -> ModelRegistry loads configs/models_pricing.json
  -> Checks last_updated; if stale (>7 days):
      -> model_fetcher attempts refresh
      -> Success: updates JSON + reloads
      -> Failure: warns, continues with cached data
  -> inference.py uses registry for all lookups

Manual: python update_models.py [--force]
  -> model_fetcher fetches all providers
  -> Writes updated configs/models_pricing.json
```

## ModelRegistry API

```python
class ModelRegistry:
    def __init__(self, config_path="configs/models_pricing.json"):
        """Load JSON, check freshness, attempt refresh if stale."""

    def get_model(self, name_or_alias: str) -> dict:
        """Resolve alias, return model entry. Raises ModelNotFoundError if unknown."""

    def get_cost_input(self, model_name: str) -> float | None:
        """Return per-token input cost, or None if unknown."""

    def get_cost_output(self, model_name: str) -> float | None:
        """Return per-token output cost, or None if unknown."""

    def list_models(self, provider: str = None) -> list[str]:
        """List available model names, optionally filtered by provider."""

    def resolve_alias(self, name: str) -> str:
        """Return canonical model name from alias."""

    def get_provider(self, model_name: str) -> str:
        """Return provider string for routing."""

    def get_api_model_name(self, model_name: str) -> str:
        """Return the actual API model name to send."""

    def curr_cost_est(self) -> float:
        """Calculate cumulative cost from tracked tokens."""
```

## Fetch Strategy

### Model Discovery (via API)

| Provider | Endpoint | Auth |
|----------|----------|------|
| OpenAI | `GET /v1/models` | API key |
| Anthropic | `GET /v1/models` | API key |
| Google | `GET /v1beta/models` | API key |
| DeepSeek | `GET /v1/models` | API key |

Returns available model IDs. Filter to relevant ones (skip embeddings, fine-tunes). Skip provider if no API key is set.

### Pricing Scraping

| Provider | Source URL | Strategy |
|----------|-----------|----------|
| OpenAI | `https://platform.openai.com/docs/pricing` | Parse pricing table HTML |
| Anthropic | `https://docs.anthropic.com/en/docs/about-claude/models` | Parse model comparison table |
| Google | `https://ai.google.dev/gemini-api/docs/pricing` | Parse pricing table |
| DeepSeek | `https://api-docs.deepseek.com/quick_start/pricing` | Parse pricing table |

Each provider gets its own parser function in `model_fetcher.py`.

### Merge Logic

- **New model via API, no pricing scraped**: Add with `cost: null`, log notice.
- **Existing model, updated pricing**: Update values.
- **Existing model not in API response**: Keep (may be deprecated but usable).
- **Update `last_updated`**: On any successful fetch (even partial).

## Error Handling

| Scenario | Behavior |
|----------|----------|
| JSON file missing or corrupt | Fall back to `DEFAULT_MODELS` hardcoded in `model_registry.py` (contains all currently supported models with current pricing), write fresh JSON |
| Fetch timeout (>10s) | Abort fetch, warn, continue with cached data |
| Scraping fails (page format changed) | Log warning per provider, keep existing JSON values for that provider |
| Unknown model requested | `ModelNotFoundError` with list of available models |
| Model has `null` pricing | Skip in cost calculation, print notice |
| Ollama (`OPENAI_API_KEY == "ollama"`) | Bypass registry, pass model string directly |
| No API keys set for any provider | Skip all fetching, use cached JSON silently |

## Refresh Policy

- **Auto**: On startup, if `last_updated` is >7 days old, attempt fetch. Non-blocking with 10s timeout.
- **Manual**: `python update_models.py --force` bypasses staleness check.

## Dependencies

- `requests` — HTTP client for API calls and page fetches
- `beautifulsoup4` — HTML parsing for pricing pages
- Both added to `requirements.txt`

## Testing

- JSON ships pre-populated with current pricing as defaults
- Fetcher functions independently testable per provider
- Registry testable with mock JSON (no network)
