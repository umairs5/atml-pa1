"""Configuration loading for Task 3 experiments."""

from copy import deepcopy
from pathlib import Path

import yaml


def _merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(method_config: str | Path) -> dict:
    """Combine the shared Task 3 protocol with a method configuration."""
    with Path("task3/configs/base.yaml").open(encoding="utf-8") as handle:
        base = yaml.safe_load(handle)
    with Path(method_config).open(encoding="utf-8") as handle:
        method = yaml.safe_load(handle)
    return _merge(base, method)
