def build_datamodule(config):
    arch = str(config.get('model', {}).get('arch', 'dab_curve_detr')).lower()
    if arch == 'dab_endpoint_detr':
        from edge_datasets.endpoint_datamodule import EndpointDetectionDataModule

        return EndpointDetectionDataModule(config)
    if arch == 'dab_curve_detr':
        from edge_datasets.parametric_edge_datamodule import ParametricEdgeDataModule

        return ParametricEdgeDataModule(config)
    raise ValueError(f'Unsupported model.arch for datamodule selection: {arch}')


__all__ = ['build_datamodule']
