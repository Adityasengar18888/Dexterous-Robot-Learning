"""Configuration loader and schema validator for the manipulation platform."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict
import yaml


class ConfigDict(dict):
    """Dictionary subclass that enables attribute-style dot access."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict) and not isinstance(value, ConfigDict):
                self[key] = ConfigDict(value)

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as err:
            raise AttributeError(f"Config has no attribute '{key}'") from err

    def __setattr__(self, key: str, value: Any) -> None:
        if isinstance(value, dict) and not isinstance(value, ConfigDict):
            value = ConfigDict(value)
        self[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert back to a standard nested Python dictionary."""
        out = {}
        for key, value in self.items():
            if isinstance(value, ConfigDict):
                out[key] = value.to_dict()
            elif isinstance(value, list):
                out[key] = [v.to_dict() if isinstance(v, ConfigDict) else v for v in value]
            else:
                out[key] = value
        return out


def deep_update(base: dict, update: dict) -> dict:
    """Recursively update nested dictionaries without overwriting unmentioned keys."""
    res = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and key in res and isinstance(res[key], dict):
            res[key] = deep_update(res[key], value)
        else:
            res[key] = copy.deepcopy(value)
    return res


def load_config(config_path: str | Path, base_config_path: str | Path | None = None) -> ConfigDict:
    """
    Load a YAML configuration file, optionally merging with a base configuration.

    Args:
        config_path: Path to the target YAML file.
        base_config_path: Optional path to a base YAML file to extend.

    Returns:
        ConfigDict wrapping the merged configuration.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f) or {}

    if base_config_path is not None:
        base_file = Path(base_config_path)
        if not base_file.exists():
            raise FileNotFoundError(f"Base configuration file not found: {base_file}")
        with open(base_file, "r", encoding="utf-8") as f:
            base_data = yaml.safe_load(f) or {}
        config_data = deep_update(base_data, config_data)

    return ConfigDict(config_data)
