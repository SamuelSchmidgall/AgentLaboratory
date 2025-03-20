import json
from pathlib import Path
import logging

class SettingsManager:
    def __init__(self):
        self.settings_dir = Path("settings")
        self.settings_file = self.settings_dir / "user_settings.json"
        self._ensure_settings_dir()

    def _ensure_settings_dir(self):
        """Ensure the settings directory exists"""
        try:
            self.settings_dir.mkdir(exist_ok=True)
        except Exception as e:
            logging.error(f"Failed to create settings directory: {e}")

    def save_settings(self, settings: dict):
        """Save settings to JSON file"""
        try:
            # Filter out empty API keys before saving
            filtered_settings = {
                k: v for k, v in settings.items()
                if not (k.endswith('_api_key') and not v)
            }
            
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(filtered_settings, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save settings: {e}")

    def load_settings(self) -> dict:
        """Load settings from JSON file"""
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load settings: {e}")
        return {}
