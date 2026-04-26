from typing import Dict

from edge_datasets.conditioned_curve_dataset import (
    ConditionedCurveDataset,
    LaionSyntheticConditionedCurveDataset,
    conditioned_curve_collate,
)
from edge_datasets.laion_synthetic_dataset import discover_laion_bezier_samples
from edge_datasets.parametric_edge_datamodule import ParametricEdgeDataModule


class ConditionedCurveDataModule(ParametricEdgeDataModule):
    def _build_dataset(self, edge_paths, split: str, train_augment: bool, common: Dict) -> ConditionedCurveDataset:
        return ConditionedCurveDataset(
            edge_paths,
            split=split,
            train_augment=train_augment,
            input_root=self._split_input_roots(split),
            endpoint_dedupe_distance_px=float(self.config['data'].get('endpoint_dedupe_distance_px', 2.0)),
            endpoint_closed_curve_threshold_px=float(self.config['data'].get('endpoint_closed_curve_threshold_px', 2.0)),
            **common,
        )

    def _build_laion_dataset(self, dataset_cfg: Dict, split: str, train_augment: bool, common: Dict):
        sample_records = discover_laion_bezier_samples(
            data_root=dataset_cfg['data_root'],
            image_root=dataset_cfg.get('image_root'),
            bezier_root=dataset_cfg.get('bezier_root', dataset_cfg.get('edge_root')),
            entry_cache_path=dataset_cfg.get('entry_cache_path'),
            batches=dataset_cfg.get('batches'),
            batch_glob=str(dataset_cfg.get('batch_glob', 'batch*')),
            max_samples=dataset_cfg.get('max_samples'),
            selection_seed=dataset_cfg.get('selection_seed'),
            selection_offset=int(dataset_cfg.get('selection_offset', 0)),
        )
        if not sample_records:
            raise FileNotFoundError(f'No LAION synthetic samples found for config: {dataset_cfg}')
        return LaionSyntheticConditionedCurveDataset(
            sample_records=sample_records,
            image_size=int(common['image_size']),
            target_degree=int(common['target_degree']),
            max_targets=int(common['max_targets']),
            split=split,
            train_augment=train_augment,
            augment_cfg=dict(common['augment_cfg']),
            rgb_input=bool(common['rgb_input']),
            endpoint_dedupe_distance_px=float(self.config['data'].get('endpoint_dedupe_distance_px', 2.0)),
            endpoint_closed_curve_threshold_px=float(self.config['data'].get('endpoint_closed_curve_threshold_px', 2.0)),
        )

    def _loader_kwargs(self) -> Dict:
        kwargs = super()._loader_kwargs()
        kwargs['collate_fn'] = conditioned_curve_collate
        return kwargs
