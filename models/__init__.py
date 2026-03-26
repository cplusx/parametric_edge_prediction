from models.dab_curve_detr import DABCurveDETR


def build_model(config):
    model_cfg = config.get('model', {})
    arch = str(model_cfg.get('arch', 'dab_curve_detr')).lower()
    if arch != 'dab_curve_detr':
        raise ValueError(f'Unsupported model.arch: {arch}. Only dab_curve_detr is supported.')
    return DABCurveDETR(config)


__all__ = ['DABCurveDETR', 'build_model']
