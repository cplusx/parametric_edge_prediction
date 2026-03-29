from models.dab_curve_detr import DABCurveDETR
from models.dab_endpoint_detr import DABEndpointDETR


def build_model(config):
    model_cfg = config.get('model', {})
    arch = str(model_cfg.get('arch', 'dab_curve_detr')).lower()
    if arch == 'dab_curve_detr':
        return DABCurveDETR(config)
    if arch == 'dab_endpoint_detr':
        return DABEndpointDETR(config)
    raise ValueError(f'Unsupported model.arch: {arch}. Supported: dab_curve_detr, dab_endpoint_detr.')


__all__ = ['DABCurveDETR', 'DABEndpointDETR', 'build_model']
