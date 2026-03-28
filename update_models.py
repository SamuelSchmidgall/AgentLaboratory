"""CLI tool to refresh model pricing data.

Usage:
    python update_models.py          # Update if stale (>7 days)
    python update_models.py --force  # Force update regardless of freshness
"""
import argparse
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
