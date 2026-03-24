from models.dab_curve_detr import DABCurveDETR
from models.parametric_detr import ParametricDETR


def build_model(config):
    model_cfg = config.get('model', {})
    arch = str(model_cfg.get('arch', 'parametric_detr')).lower()
    if arch == 'parametric_detr':
        return ParametricDETR(config)
    if arch == 'dab_curve_detr':
        return DABCurveDETR(config)
    raise ValueError(f'Unsupported model.arch: {arch}')


__all__ = ['ParametricDETR', 'DABCurveDETR', 'build_model']
