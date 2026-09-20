"""Small YAML configuration helpers for Task 2 experiments."""

from __future__ import annotations

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
    """Combine the shared protocol with one method-specific configuration."""
    with Path("task2/configs/base.yaml").open(encoding="utf-8") as handle:
        base = yaml.safe_load(handle)
    with Path(method_config).open(encoding="utf-8") as handle:
        method = yaml.safe_load(handle)
    return _merge(base, method)
