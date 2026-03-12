from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str, override_path: str = None) -> Dict[str, Any]:
    config_file = Path(config_path)
    default_file = config_file.parent / 'default.yaml'
    if override_path is None and config_file.name != 'default.yaml' and default_file.exists():
        with open(default_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        with open(config_file, 'r', encoding='utf-8') as f:
            overrides = yaml.safe_load(f)
        config = deep_update(config, overrides)
    else:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    if override_path:
        with open(override_path, 'r', encoding='utf-8') as f:
            overrides = yaml.safe_load(f)
        config = deep_update(config, overrides)
    return config
