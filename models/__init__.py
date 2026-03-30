def build_model(config):
    model_cfg = config.get('model', {})
    arch = str(model_cfg.get('arch', 'dab_curve_detr')).lower()
    if arch == 'dab_curve_detr':
        from models.dab_curve_detr import DABCurveDETR
        return DABCurveDETR(config)
    if arch == 'dab_endpoint_detr':
        from models.dab_endpoint_detr import DABEndpointDETR
        return DABEndpointDETR(config)
    if arch == 'endpoint_flow_matching':
        from models.endpoint_flow_matching import EndpointFlowMatchingModel
        return EndpointFlowMatchingModel.from_config_dict(config)
    raise ValueError(f'Unsupported model.arch: {arch}. Supported: dab_curve_detr, dab_endpoint_detr, endpoint_flow_matching.')

__all__ = ['build_model']
