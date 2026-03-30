def build_loss_computer(config):
    arch = str(config.get('model', {}).get('arch', 'dab_curve_detr')).lower()
    if arch == 'dab_curve_detr':
        from models.losses.composite import ParametricEdgeLossComputer
        return ParametricEdgeLossComputer(config)
    if arch == 'dab_endpoint_detr':
        from models.losses.endpoint_composite import EndpointLossComputer
        return EndpointLossComputer(config)
    if arch == 'endpoint_flow_matching':
        from models.losses.endpoint_flow_matching import EndpointFlowMatchingLossComputer
        return EndpointFlowMatchingLossComputer(config)
    raise ValueError(f'Unsupported model.arch for loss selection: {arch}')


def compute_losses(outputs, targets, config):
    return build_loss_computer(config)(outputs, targets)

__all__ = ['build_loss_computer', 'compute_losses']
