from typing import Tuple


def get_curve_coordinate_range(config) -> Tuple[float, float]:
    model_cfg = config.get('model', {}) if isinstance(config, dict) else {}
    curve_min = float(model_cfg.get('curve_coord_min', 0.0))
    curve_max = float(model_cfg.get('curve_coord_max', 1.0))
    if curve_max <= curve_min:
        raise ValueError(f'curve_coord_max must be greater than curve_coord_min, got {curve_min}..{curve_max}')
    return curve_min, curve_max


def curve_external_to_internal(curves, config):
    curve_min, curve_max = get_curve_coordinate_range(config)
    scale = curve_max - curve_min
    return (curves - curve_min) / scale


def curve_internal_to_external(curves, config):
    curve_min, curve_max = get_curve_coordinate_range(config)
    scale = curve_max - curve_min
    return curves * scale + curve_min
