from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch


def load_compatible_checkpoint(module: torch.nn.Module, checkpoint_path: str) -> Dict[str, int]:
    checkpoint_file = Path(checkpoint_path)
    if any(ch in checkpoint_path for ch in '*?[]'):
        matches = sorted(checkpoint_file.parent.glob(checkpoint_file.name))
        if not matches:
            raise FileNotFoundError(f'No checkpoint matched pattern: {checkpoint_path}')
        checkpoint_file = matches[0]

    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    module_state = module.state_dict()

    compatible: Dict[str, torch.Tensor] = {}
    skipped_missing = 0
    skipped_shape = 0
    for key, value in state_dict.items():
        target = module_state.get(key)
        if target is None:
            skipped_missing += 1
            continue
        if value.shape != target.shape:
            skipped_shape += 1
            continue
        compatible[key] = value

    module.load_state_dict(compatible, strict=False)
    return {
        'loaded': len(compatible),
        'skipped_missing': skipped_missing,
        'skipped_shape': skipped_shape,
    }


def maybe_load_conditioned_curve_init(module: torch.nn.Module, config: Dict) -> Dict[str, int] | None:
    model_cfg = config.get('model', {})
    arch = str(model_cfg.get('arch', 'dab_curve_detr')).lower()
    if arch != 'dab_cond_curve_detr':
        return None
    if not bool(model_cfg.get('conditioned_curve_init_enabled', True)):
        return None
    checkpoint_path = model_cfg.get('conditioned_curve_init_checkpoint')
    if not checkpoint_path:
        return None
    return load_compatible_checkpoint(module, str(checkpoint_path))
