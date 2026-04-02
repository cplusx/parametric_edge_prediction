from pathlib import Path
from typing import Callable, Dict, Sequence

from torch.utils.data import ConcatDataset

from edge_datasets.endpoint_dataset import (
    LaionSyntheticEndpointDataset,
    ParametricEndpointDataset,
    endpoint_detection_collate,
)
from edge_datasets.laion_synthetic_dataset import discover_laion_synthetic_samples
from edge_datasets.parametric_edge_datamodule import ParametricEdgeDataModule, RepeatedDataset


def _walk_train_datasets(dataset, fn: Callable[[object], None]) -> None:
    if hasattr(dataset, 'base_dataset') and isinstance(dataset, RepeatedDataset):
        _walk_train_datasets(dataset.base_dataset, fn)
        return
    if isinstance(dataset, ConcatDataset):
        for sub_dataset in dataset.datasets:
            _walk_train_datasets(sub_dataset, fn)
        return
    fn(dataset)


class EndpointDetectionDataModule(ParametricEdgeDataModule):
    def _build_dataset(self, edge_paths: Sequence[Path], split: str, train_augment: bool, common: Dict) -> ParametricEndpointDataset:
        return ParametricEndpointDataset(
            edge_paths,
            split=split,
            train_augment=train_augment,
            input_root=self._split_input_roots(split),
            endpoint_dedupe_distance_px=float(self.config['data'].get('endpoint_dedupe_distance_px', 2.0)),
            **common,
        )

    def _build_laion_dataset(self, dataset_cfg: Dict, split: str, train_augment: bool, common: Dict):
        sample_records = discover_laion_synthetic_samples(
            data_root=Path(dataset_cfg['data_root']),
            cache_root=Path(dataset_cfg.get('cache_root', Path(dataset_cfg['data_root']) / 'laion_edge_v2_bezier_cache_fast')),
            image_root=Path(dataset_cfg['image_root']) if dataset_cfg.get('image_root') is not None else None,
            edge_root=Path(dataset_cfg['edge_root']) if dataset_cfg.get('edge_root') is not None else None,
            batches=dataset_cfg.get('batches'),
            batch_glob=str(dataset_cfg.get('batch_glob', 'batch*')),
            quantize=int(dataset_cfg.get('quantize', 4)),
            max_samples=dataset_cfg.get('max_samples'),
            selection_seed=dataset_cfg.get('selection_seed'),
            selection_offset=int(dataset_cfg.get('selection_offset', 0)),
        )
        if not sample_records:
            raise FileNotFoundError(f'No LAION synthetic samples found for config: {dataset_cfg}')
        return LaionSyntheticEndpointDataset(
            sample_records=sample_records,
            cache_root=Path(dataset_cfg.get('cache_root', Path(dataset_cfg['data_root']) / 'laion_edge_v2_bezier_cache_fast')),
            image_size=int(common['image_size']),
            version_name=str(common['version_name']),
            target_degree=int(common['target_degree']),
            min_curve_length=float(common['min_curve_length']),
            max_targets=int(common['max_targets']),
            split=split,
            train_augment=train_augment,
            augment_cfg=dict(common['augment_cfg']),
            rgb_input=bool(common['rgb_input']),
            endpoint_dedupe_distance_px=float(self.config['data'].get('endpoint_dedupe_distance_px', 2.0)),
            skip_missing_bezier_cache=bool(dataset_cfg.get('skip_missing_bezier_cache', self.config['data'].get('skip_missing_bezier_cache', False))),
        )

    def _loader_kwargs(self) -> Dict:
        loader_kwargs = super()._loader_kwargs()
        loader_kwargs['collate_fn'] = endpoint_detection_collate
        return loader_kwargs

    def setup(self, stage=None) -> None:
        super().setup(stage=stage)
        self.train_sampler = None
        model_cfg = self.config.get('model', {})
        if str(model_cfg.get('arch', '')).lower() != 'endpoint_flow_matching':
            return
        if self.train_dataset is None:
            return
        def configure(dataset) -> None:
            if hasattr(dataset, 'configure_curriculum'):
                dataset.configure_curriculum(
                    enabled=True,
                    start_points=int(model_cfg.get('curriculum_start_points', 150)),
                    max_points=int(model_cfg.get('curriculum_max_points', 250)),
                    points_per_epoch=int(model_cfg.get('curriculum_points_per_epoch', 10)),
                    global_skip_points=int(model_cfg.get('curriculum_global_skip_points', 400)),
                )

        _walk_train_datasets(self.train_dataset, configure)

    def train_dataloader(self):
        return super().train_dataloader()

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)
        if self.train_dataset is None:
            return

        def apply_epoch(dataset) -> None:
            if hasattr(dataset, 'set_epoch'):
                dataset.set_epoch(int(epoch))

        _walk_train_datasets(self.train_dataset, apply_epoch)
