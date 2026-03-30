from pathlib import Path
from typing import Dict, List, Sequence

from torch.utils.data import ConcatDataset

from edge_datasets.endpoint_dataset import (
    LaionSyntheticEndpointDataset,
    ParametricEndpointDataset,
    endpoint_detection_collate,
)
from edge_datasets.laion_synthetic_dataset import discover_laion_synthetic_samples
from edge_datasets.parametric_edge_datamodule import DistributedCurriculumSampler, ParametricEdgeDataModule, RepeatedDataset


def _dataset_curriculum_counts(dataset) -> List[int]:
    if hasattr(dataset, 'get_curriculum_counts'):
        return list(dataset.get_curriculum_counts())
    if isinstance(dataset, RepeatedDataset):
        base_counts = _dataset_curriculum_counts(dataset.base_dataset)
        return base_counts * int(dataset.repeat_factor)
    if isinstance(dataset, ConcatDataset):
        counts: List[int] = []
        for sub_dataset in dataset.datasets:
            counts.extend(_dataset_curriculum_counts(sub_dataset))
        return counts
    raise TypeError(f'Unsupported dataset type for curriculum counts: {type(dataset)!r}')


class EndpointDetectionDataModule(ParametricEdgeDataModule):
    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.train_curriculum_counts: List[int] = []

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
            image_size=int(common['image_size']),
            target_degree=int(common['target_degree']),
            min_curve_length=float(common['min_curve_length']),
            max_targets=int(common['max_targets']),
            split=split,
            train_augment=train_augment,
            augment_cfg=dict(common['augment_cfg']),
            rgb_input=bool(common['rgb_input']),
            endpoint_dedupe_distance_px=float(self.config['data'].get('endpoint_dedupe_distance_px', 2.0)),
            curriculum_cache_root=Path(dataset_cfg.get('cache_root', Path(dataset_cfg['data_root']) / 'laion_edge_v2_bezier_cache_fast')),
        )

    def _loader_kwargs(self) -> Dict:
        loader_kwargs = super()._loader_kwargs()
        loader_kwargs['collate_fn'] = endpoint_detection_collate
        return loader_kwargs

    def setup(self, stage=None) -> None:
        super().setup(stage=stage)
        self.train_sampler = None
        self.train_curriculum_counts = []
        model_cfg = self.config.get('model', {})
        if str(model_cfg.get('arch', '')).lower() != 'endpoint_flow_matching':
            return
        if self.train_dataset is None:
            return
        counts = _dataset_curriculum_counts(self.train_dataset)
        if not counts:
            raise ValueError('Curriculum sampler received an empty train dataset.')
        self.train_curriculum_counts = counts

    def train_dataloader(self):
        model_cfg = self.config.get('model', {})
        if str(model_cfg.get('arch', '')).lower() == 'endpoint_flow_matching':
            trainer_cfg = self.config.get('trainer', {})
            if bool(trainer_cfg.get('use_distributed_sampler', False)):
                raise ValueError(
                    'endpoint_flow_matching requires trainer.use_distributed_sampler=false '
                    'so the curriculum sampler remains in control of per-rank sample selection.'
                )
            if self.train_sampler is None:
                if not self.train_curriculum_counts:
                    raise ValueError('endpoint_flow_matching requires precomputed curriculum counts before building train_dataloader.')
                self.train_sampler = DistributedCurriculumSampler(
                    self.train_dataset,
                    self.train_curriculum_counts,
                    start_points=int(model_cfg.get('curriculum_start_points', 100)),
                    max_points=int(model_cfg.get('curriculum_max_points', 200)),
                    points_per_epoch=int(model_cfg.get('curriculum_points_per_epoch', 10)),
                    shuffle=True,
                    seed=int(self.config.get('data', {}).get('split_seed', 42)),
                    drop_last=False,
                )
        return super().train_dataloader()
