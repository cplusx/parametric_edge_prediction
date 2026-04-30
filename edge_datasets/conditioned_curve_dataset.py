from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from edge_datasets.graph_pipeline import prepare_eval_curve_sample, prepare_training_curve_sample
from misc_utils.bezier_target_utils import (
    ensure_target_cache,
    load_cached_targets,
    load_compact_bezier_targets,
    load_image_array_original,
    resolve_input_path,
)
from misc_utils.endpoint_target_utils import curves_to_endpoint_clusters_with_incidence


_CONDITIONED_CURVE_RETRY_EXCEPTIONS = (FileNotFoundError, OSError, ValueError, KeyError, EOFError)


def _build_condition_points(
    curves: np.ndarray,
    image_size: np.ndarray,
    *,
    endpoint_dedupe_distance_px: float,
    endpoint_closed_curve_threshold_px: float,
) -> np.ndarray:
    endpoint_targets = curves_to_endpoint_clusters_with_incidence(
        curves=curves,
        image_size=image_size,
        dedupe_distance_px=endpoint_dedupe_distance_px,
        closed_curve_threshold_px=endpoint_closed_curve_threshold_px,
    )
    return np.asarray(endpoint_targets['points'], dtype=np.float32)


def _pad_condition_points(condition_points_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(condition_points_list)
    max_count = max((int(points.shape[0]) for points in condition_points_list), default=0)
    if max_count <= 0:
        empty_points = torch.zeros((batch_size, 0, 2), dtype=torch.float32)
        empty_mask = torch.zeros((batch_size, 0), dtype=torch.bool)
        return empty_points, empty_mask

    padded_points = torch.zeros((batch_size, max_count, 2), dtype=torch.float32)
    padding_mask = torch.ones((batch_size, max_count), dtype=torch.bool)
    for batch_idx, points in enumerate(condition_points_list):
        count = int(points.shape[0])
        if count <= 0:
            continue
        padded_points[batch_idx, :count] = points
        padding_mask[batch_idx, :count] = False
    return padded_points, padding_mask


class ConditionedCurveDataset(Dataset):
    def __init__(
        self,
        curve_paths: Sequence[Path],
        cache_root: Path,
        image_size: int,
        version_name: str,
        input_root: Optional[Sequence[Path]] = None,
        rgb_input: bool = False,
        target_degree: int = 3,
        min_curve_length: float = 3.0,
        max_targets: int = 128,
        split: str = 'train',
        train_augment: bool = False,
        augment_cfg: Optional[Dict] = None,
        endpoint_dedupe_distance_px: float = 2.0,
        endpoint_closed_curve_threshold_px: float = 2.0,
    ) -> None:
        self.curve_paths = [Path(path) for path in curve_paths]
        self.cache_root = Path(cache_root)
        self.image_size = int(image_size)
        self.version_name = version_name
        self.input_root = [Path(root) for root in input_root] if input_root is not None else None
        self.rgb_input = bool(rgb_input)
        self.target_degree = int(target_degree)
        self.min_curve_length = float(min_curve_length)
        self.max_targets = int(max_targets)
        self.split = str(split)
        self.train_augment = bool(train_augment)
        self.augment_cfg = dict(augment_cfg or {})
        self.endpoint_dedupe_distance_px = float(endpoint_dedupe_distance_px)
        self.endpoint_closed_curve_threshold_px = float(endpoint_closed_curve_threshold_px)

    def __len__(self) -> int:
        return len(self.curve_paths)

    def _load_curve_targets(self, path: Path) -> Dict:
        if path.suffix == '.npz':
            try:
                return load_compact_bezier_targets(
                    path,
                    target_degree=self.target_degree,
                    max_targets=self.max_targets,
                )
            except KeyError:
                pass
        cache_path = ensure_target_cache(
            edge_path=path,
            cache_root=self.cache_root,
            version_name=self.version_name,
            target_degree=self.target_degree,
            min_curve_length=self.min_curve_length,
        )
        return load_cached_targets(cache_path)

    def _build_item(self, index: int) -> Dict:
        curve_path = self.curve_paths[index]
        target_cache = self._load_curve_targets(curve_path)
        image_path = resolve_input_path(curve_path, self.input_root)
        image = load_image_array_original(image_path, rgb=self.rgb_input)
        curves = np.asarray(target_cache['curves'], dtype=np.float32)
        rng = np.random.default_rng((torch.initial_seed() + index) % (2 ** 32))
        if self.split == 'train' and self.train_augment:
            image_hwc, target_data = prepare_training_curve_sample(
                image=image,
                curves=curves,
                image_size=self.image_size,
                max_targets=self.max_targets,
                augment_cfg=self.augment_cfg,
                rng=rng,
            )
        else:
            image_hwc, target_data = prepare_eval_curve_sample(
                image=image,
                curves=curves,
                image_size=self.image_size,
                max_targets=self.max_targets,
            )
        image_chw = np.transpose(image_hwc, (2, 0, 1))
        curve_array = np.asarray(target_data['curves'], dtype=np.float32)
        condition_points = _build_condition_points(
            curve_array,
            np.asarray(target_data['image_size'], dtype=np.int64),
            endpoint_dedupe_distance_px=self.endpoint_dedupe_distance_px,
            endpoint_closed_curve_threshold_px=self.endpoint_closed_curve_threshold_px,
        )
        curves_tensor = torch.from_numpy(curve_array).float()
        labels = torch.ones((curves_tensor.shape[0],), dtype=torch.long)
        return {
            'image': torch.from_numpy(image_chw).float(),
            'condition_points': torch.from_numpy(condition_points).float(),
            'target': {
                'labels': labels,
                'curves': curves_tensor,
                'image_size': torch.from_numpy(np.asarray(target_data['image_size'], dtype=np.int64)).long(),
                'sample_id': curve_path.stem,
                'bezier_path': str(curve_path),
                'input_path': str(image_path),
            },
        }

    def __getitem__(self, index: int) -> Dict:
        try:
            return self._build_item(index)
        except _CONDITIONED_CURVE_RETRY_EXCEPTIONS:
            dataset_len = len(self.curve_paths)
            if dataset_len <= 1:
                raise
            redirected_index = int(np.random.randint(0, dataset_len - 1))
            if redirected_index >= int(index):
                redirected_index += 1
            return self[redirected_index]


class LaionSyntheticConditionedCurveDataset(Dataset):
    def __init__(
        self,
        sample_records: Sequence[Dict],
        image_size: int,
        target_degree: int,
        max_targets: int,
        split: str,
        train_augment: bool,
        augment_cfg: Optional[Dict],
        rgb_input: bool,
        endpoint_dedupe_distance_px: float = 2.0,
        endpoint_closed_curve_threshold_px: float = 2.0,
    ) -> None:
        self.sample_records = list(sample_records)
        self.image_size = int(image_size)
        self.target_degree = int(target_degree)
        self.max_targets = int(max_targets)
        self.split = str(split)
        self.train_augment = bool(train_augment)
        self.augment_cfg = dict(augment_cfg or {})
        self.rgb_input = bool(rgb_input)
        self.endpoint_dedupe_distance_px = float(endpoint_dedupe_distance_px)
        self.endpoint_closed_curve_threshold_px = float(endpoint_closed_curve_threshold_px)

    def __len__(self) -> int:
        return len(self.sample_records)

    @staticmethod
    def _rng_for_index(index: int) -> np.random.Generator:
        return np.random.default_rng((torch.initial_seed() + index) % (2 ** 32))

    def _build_item(self, index: int) -> Dict:
        record = self.sample_records[index]
        target_cache = load_compact_bezier_targets(
            Path(record['bezier_path']),
            target_degree=self.target_degree,
            max_targets=self.max_targets,
        )
        image = load_image_array_original(Path(record['image_path']), rgb=self.rgb_input)
        curves = np.asarray(target_cache['curves'], dtype=np.float32)
        rng = self._rng_for_index(index)
        if self.split == 'train' and self.train_augment:
            image_hwc, target_data = prepare_training_curve_sample(
                image=image,
                curves=curves,
                image_size=self.image_size,
                max_targets=self.max_targets,
                augment_cfg=self.augment_cfg,
                rng=rng,
            )
        else:
            image_hwc, target_data = prepare_eval_curve_sample(
                image=image,
                curves=curves,
                image_size=self.image_size,
                max_targets=self.max_targets,
            )
        image_chw = np.transpose(image_hwc, (2, 0, 1))
        curve_array = np.asarray(target_data['curves'], dtype=np.float32)
        condition_points = _build_condition_points(
            curve_array,
            np.asarray(target_data['image_size'], dtype=np.int64),
            endpoint_dedupe_distance_px=self.endpoint_dedupe_distance_px,
            endpoint_closed_curve_threshold_px=self.endpoint_closed_curve_threshold_px,
        )
        curves_tensor = torch.from_numpy(curve_array).float()
        labels = torch.ones((curves_tensor.shape[0],), dtype=torch.long)
        sample_id = f"{record['batch_name']}_{record['image_id']}"
        return {
            'image': torch.from_numpy(image_chw).float(),
            'condition_points': torch.from_numpy(condition_points).float(),
            'target': {
                'labels': labels,
                'curves': curves_tensor,
                'image_size': torch.from_numpy(np.asarray(target_data['image_size'], dtype=np.int64)).long(),
                'sample_id': sample_id,
                'input_path': str(record['image_path']),
                'bezier_path': str(record['bezier_path']),
                'dataset_name': 'laion_synthetic',
            },
        }

    def __getitem__(self, index: int) -> Dict:
        dataset_len = len(self.sample_records)
        try:
            return self._build_item(index)
        except _CONDITIONED_CURVE_RETRY_EXCEPTIONS:
            if dataset_len <= 1:
                raise
            redirected_index = int(np.random.randint(0, dataset_len - 1))
            if redirected_index >= int(index):
                redirected_index += 1
            return self[redirected_index]


def conditioned_curve_collate(batch: List[Dict]) -> Dict:
    images = torch.stack([item['image'] for item in batch], dim=0)
    targets = [item['target'] for item in batch]
    condition_points, condition_padding_mask = _pad_condition_points([item['condition_points'] for item in batch])
    return {
        'images': images,
        'targets': targets,
        'model_inputs': {
            'condition_points': condition_points,
            'condition_padding_mask': condition_padding_mask,
        },
    }
