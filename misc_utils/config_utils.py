from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


def _load_yaml_mapping(config_path: Path) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f'Config file must contain a top-level mapping: {config_path}')
    return data


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
    config_data = _load_yaml_mapping(config_file)
    inherit_default = bool(config_data.pop('_inherit_default', True))
    if override_path is None and config_file.name != 'default.yaml' and default_file.exists() and inherit_default:
        config = _load_yaml_mapping(default_file)
        config = deep_update(config, config_data)
    else:
        config = config_data
    if override_path:
        overrides = _load_yaml_mapping(Path(override_path))
        overrides.pop('_inherit_default', None)
        config = deep_update(config, overrides)
    return config
