from edge_datasets.endpoint_datamodule import EndpointDetectionDataModule
from edge_datasets.endpoint_dataset import LaionSyntheticEndpointDataset, ParametricEndpointDataset
from edge_datasets.laion_synthetic_dataset import LaionSyntheticEdgeDataset
from edge_datasets.parametric_edge_datamodule import ParametricEdgeDataModule
from edge_datasets.parametric_edge_dataset import ParametricEdgeDataset


def build_datamodule(config):
    arch = str(config.get('model', {}).get('arch', 'dab_curve_detr')).lower()
    if arch == 'dab_endpoint_detr':
        return EndpointDetectionDataModule(config)
    if arch == 'dab_curve_detr':
        return ParametricEdgeDataModule(config)
    raise ValueError(f'Unsupported model.arch for datamodule selection: {arch}')


__all__ = [
    'EndpointDetectionDataModule',
    'LaionSyntheticEdgeDataset',
    'LaionSyntheticEndpointDataset',
    'ParametricEdgeDataModule',
    'ParametricEdgeDataset',
    'ParametricEndpointDataset',
    'build_datamodule',
]
