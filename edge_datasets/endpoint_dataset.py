from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from edge_datasets.graph_pipeline import (
    prepare_eval_endpoint_sample_with_mask,
    prepare_training_endpoint_sample_with_mask,
)
from misc_utils.bezier_target_utils import (
    ensure_target_cache,
    load_binary_edge_annotation,
    load_cached_targets,
    load_image_array_original,
    resolve_input_path,
)
from misc_utils.endpoint_target_utils import curves_to_unique_endpoints

_ENDPOINT_SAMPLE_RETRY_EXCEPTIONS = (FileNotFoundError, OSError, ValueError, KeyError, EOFError)


def _endpoint_target_from_curve_target(target_data: Dict, sample_id: str, edge_path: str, input_path: str) -> Dict:
    image_chw = np.transpose(target_data['image_hwc'], (2, 0, 1))
    edge_chw = np.transpose(target_data['edge_hwc'], (2, 0, 1))
    if 'points' in target_data:
        points = np.asarray(target_data['points'], dtype=np.float32)
    else:
        points = curves_to_unique_endpoints(
            target_data['curves'],
            image_size=target_data['image_size'],
            dedupe_distance_px=float(target_data.get('endpoint_dedupe_distance_px', 2.0)),
        )
    point_tensor = torch.from_numpy(points).float()
    labels = torch.zeros((point_tensor.shape[0],), dtype=torch.long)
    return {
        'image': torch.from_numpy(image_chw).float(),
        'target': {
            'labels': labels,
            'points': point_tensor,
            'image_size': torch.from_numpy(target_data['image_size']).long(),
            'edge_mask': torch.from_numpy(edge_chw).float(),
            'num_targets': torch.tensor(point_tensor.shape[0], dtype=torch.long),
            'sample_id': sample_id,
            'edge_path': edge_path,
            'input_path': input_path,
            'curriculum_direct_accept': torch.tensor(1.0, dtype=torch.float32),
            'curriculum_redirected_request': torch.tensor(0.0, dtype=torch.float32),
            'curriculum_rejected_candidates': torch.tensor(0.0, dtype=torch.float32),
        },
    }


def _empty_endpoint_item(
    *,
    image_size: int,
    rgb_input: bool,
    sample_id: str,
    edge_path: str,
    input_path: str,
) -> Dict:
    channels = 3 if rgb_input else 1
    return {
        'image': torch.zeros((channels, image_size, image_size), dtype=torch.float32),
        'target': {
            'labels': torch.zeros((0,), dtype=torch.long),
            'points': torch.zeros((0, 2), dtype=torch.float32),
            'image_size': torch.tensor([image_size, image_size], dtype=torch.long),
            'edge_mask': torch.zeros((1, image_size, image_size), dtype=torch.float32),
            'num_targets': torch.tensor(0, dtype=torch.long),
            'sample_id': sample_id,
            'edge_path': edge_path,
            'input_path': input_path,
            'curriculum_direct_accept': torch.tensor(0.0, dtype=torch.float32),
            'curriculum_redirected_request': torch.tensor(1.0, dtype=torch.float32),
            'curriculum_rejected_candidates': torch.tensor(0.0, dtype=torch.float32),
        },
    }


class ParametricEndpointDataset(Dataset):
    def __init__(
        self,
        edge_paths: Sequence[Path],
        cache_root: Path,
        image_size: int,
        version_name: str,
        input_root: Optional[Sequence[Path]] = None,
        rgb_input: bool = False,
        target_degree: int = 5,
        min_curve_length: float = 3.0,
        max_targets: int = 128,
        split: str = 'train',
        train_augment: bool = False,
        augment_cfg: Optional[Dict] = None,
        endpoint_dedupe_distance_px: float = 2.0,
    ) -> None:
        self.edge_paths = [Path(path) for path in edge_paths]
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
        self.curriculum_enabled = False
        self.curriculum_start_points = 0
        self.curriculum_max_points = 0
        self.curriculum_points_per_epoch = 0
        self.curriculum_global_skip_points = 0
        self.current_epoch = 0
        self._raw_endpoint_count_cache: Dict[int, int] = {}
        self._target_cache_path_cache: Dict[int, Path] = {}

    def __len__(self) -> int:
        return len(self.edge_paths)

    def configure_curriculum(
        self,
        *,
        enabled: bool,
        start_points: int,
        max_points: int,
        points_per_epoch: int,
        global_skip_points: int = 0,
    ) -> None:
        self.curriculum_enabled = bool(enabled)
        self.curriculum_start_points = int(start_points)
        self.curriculum_max_points = int(max_points)
        self.curriculum_points_per_epoch = int(points_per_epoch)
        self.curriculum_global_skip_points = int(global_skip_points)

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def _current_curriculum_cap(self) -> int:
        if not self.curriculum_enabled:
            return int(1e9)
        cap = self.curriculum_start_points + self.current_epoch * self.curriculum_points_per_epoch
        return max(1, min(int(cap), self.curriculum_max_points))

    def _rng_for_index(self, index: int) -> np.random.Generator:
        return np.random.default_rng((torch.initial_seed() + index) % (2 ** 32))

    @staticmethod
    def _random_redirect_index(rng: np.random.Generator, current_index: int, dataset_len: int) -> int:
        if dataset_len <= 1:
            return int(current_index)
        redirected = int(rng.integers(0, dataset_len - 1))
        if redirected >= current_index:
            redirected += 1
        return redirected

    def _redirected_item_once(self, redirected_index: int) -> Dict:
        try:
            redirected_item = self._build_item(redirected_index)
        except _ENDPOINT_SAMPLE_RETRY_EXCEPTIONS:
            edge_path = self.edge_paths[int(redirected_index) % len(self.edge_paths)]
            redirected_item = _empty_endpoint_item(
                image_size=self.image_size,
                rgb_input=self.rgb_input,
                sample_id=f'{edge_path.stem}_redirect_fallback',
                edge_path=str(edge_path),
                input_path='',
            )
        redirected_item['target']['curriculum_direct_accept'] = torch.tensor(0.0, dtype=torch.float32)
        redirected_item['target']['curriculum_redirected_request'] = torch.tensor(1.0, dtype=torch.float32)
        redirected_item['target']['curriculum_rejected_candidates'] = (
            redirected_item['target']['curriculum_rejected_candidates'] + 1.0
        )
        return redirected_item

    def _target_cache_path(self, index: int) -> Path:
        cached = self._target_cache_path_cache.get(int(index))
        if cached is not None:
            return cached
        edge_path = self.edge_paths[index]
        cache_path = ensure_target_cache(
            edge_path=edge_path,
            cache_root=self.cache_root,
            version_name=self.version_name,
            target_degree=self.target_degree,
            min_curve_length=self.min_curve_length,
        )
        self._target_cache_path_cache[int(index)] = cache_path
        return cache_path

    def _build_item(self, index: int) -> Dict:
        edge_path = self.edge_paths[index]
        cache_path = self._target_cache_path(index)
        target_cache = load_cached_targets(cache_path)
        image_path = resolve_input_path(edge_path, self.input_root)
        image = load_image_array_original(image_path, rgb=self.rgb_input)
        edge_mask = load_binary_edge_annotation(edge_path).astype(np.float32)[..., None] / 255.0
        curves = np.asarray(target_cache['curves'], dtype=np.float32)
        rng = self._rng_for_index(index)
        if self.split == 'train' and self.train_augment:
            image_hwc, edge_hwc, target_data = prepare_training_endpoint_sample_with_mask(
                image=image,
                mask=edge_mask,
                curves=curves,
                image_size=self.image_size,
                augment_cfg=self.augment_cfg,
                rng=rng,
                dedupe_distance_px=self.endpoint_dedupe_distance_px,
            )
        else:
            image_hwc, edge_hwc, target_data = prepare_eval_endpoint_sample_with_mask(
                image=image,
                mask=edge_mask,
                curves=curves,
                image_size=self.image_size,
                dedupe_distance_px=self.endpoint_dedupe_distance_px,
            )
        target_data = {
            **target_data,
            'image_hwc': image_hwc,
            'edge_hwc': edge_hwc,
            'endpoint_dedupe_distance_px': self.endpoint_dedupe_distance_px,
        }
        return _endpoint_target_from_curve_target(
            target_data=target_data,
            sample_id=edge_path.stem,
            edge_path=str(edge_path),
            input_path=str(image_path),
        )

    def _raw_endpoint_count(self, index: int) -> int:
        cached = self._raw_endpoint_count_cache.get(int(index))
        if cached is not None:
            return int(cached)
        cache_path = self._target_cache_path(index)
        target_data = load_cached_targets(cache_path)
        points = curves_to_unique_endpoints(
            target_data['curves'],
            image_size=target_data['image_size'],
            dedupe_distance_px=self.endpoint_dedupe_distance_px,
        )
        count = int(points.shape[0])
        self._raw_endpoint_count_cache[int(index)] = count
        return count

    def __getitem__(self, index: int) -> Dict:
        if not (self.split == 'train' and self.train_augment and self.curriculum_enabled):
            return self._build_item(index)
        cap = self._current_curriculum_cap()
        dataset_len = len(self.edge_paths)
        rng = self._rng_for_index(index + self.current_epoch * max(1, dataset_len))
        if self.curriculum_global_skip_points > 0:
            try:
                raw_point_count = self._raw_endpoint_count(index)
            except _ENDPOINT_SAMPLE_RETRY_EXCEPTIONS:
                raw_point_count = None
            if raw_point_count is None or raw_point_count > self.curriculum_global_skip_points:
                redirected_index = self._random_redirect_index(rng, int(index), dataset_len)
                return self._redirected_item_once(redirected_index)
        try:
            item = self._build_item(index)
        except _ENDPOINT_SAMPLE_RETRY_EXCEPTIONS:
            redirected_index = self._random_redirect_index(rng, int(index), dataset_len)
            return self._redirected_item_once(redirected_index)
        point_count = int(item['target']['points'].shape[0])
        if 0 < point_count <= cap:
            item['target']['curriculum_direct_accept'] = torch.tensor(1.0, dtype=torch.float32)
            item['target']['curriculum_redirected_request'] = torch.tensor(0.0, dtype=torch.float32)
            item['target']['curriculum_rejected_candidates'] = torch.tensor(0.0, dtype=torch.float32)
            return item
        redirected_index = self._random_redirect_index(rng, int(index), dataset_len)
        return self._redirected_item_once(redirected_index)


class LaionSyntheticEndpointDataset(Dataset):
    def __init__(
        self,
        sample_records: Sequence[Dict],
        cache_root: Path,
        image_size: int,
        version_name: str,
        target_degree: int,
        min_curve_length: float,
        max_targets: int,
        split: str,
        train_augment: bool,
        augment_cfg: Optional[Dict],
        rgb_input: bool,
        endpoint_dedupe_distance_px: float = 2.0,
    ) -> None:
        self.sample_records = list(sample_records)
        self.cache_root = Path(cache_root)
        self.image_size = int(image_size)
        self.version_name = str(version_name)
        self.target_degree = int(target_degree)
        self.min_curve_length = float(min_curve_length)
        self.max_targets = int(max_targets)
        self.split = str(split)
        self.train_augment = bool(train_augment)
        self.augment_cfg = dict(augment_cfg or {})
        self.rgb_input = bool(rgb_input)
        self.endpoint_dedupe_distance_px = float(endpoint_dedupe_distance_px)
        self.curriculum_enabled = False
        self.curriculum_start_points = 0
        self.curriculum_max_points = 0
        self.curriculum_points_per_epoch = 0
        self.curriculum_global_skip_points = 0
        self.current_epoch = 0
        self._raw_endpoint_count_cache: Dict[int, int] = {}
        self._target_cache_path_cache: Dict[int, Path] = {}

    def __len__(self) -> int:
        return len(self.sample_records)

    def configure_curriculum(
        self,
        *,
        enabled: bool,
        start_points: int,
        max_points: int,
        points_per_epoch: int,
        global_skip_points: int = 0,
    ) -> None:
        self.curriculum_enabled = bool(enabled)
        self.curriculum_start_points = int(start_points)
        self.curriculum_max_points = int(max_points)
        self.curriculum_points_per_epoch = int(points_per_epoch)
        self.curriculum_global_skip_points = int(global_skip_points)

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def _current_curriculum_cap(self) -> int:
        if not self.curriculum_enabled:
            return int(1e9)
        cap = self.curriculum_start_points + self.current_epoch * self.curriculum_points_per_epoch
        return max(1, min(int(cap), self.curriculum_max_points))

    def _rng_for_index(self, index: int) -> np.random.Generator:
        return np.random.default_rng((torch.initial_seed() + index) % (2 ** 32))

    @staticmethod
    def _random_redirect_index(rng: np.random.Generator, current_index: int, dataset_len: int) -> int:
        if dataset_len <= 1:
            return int(current_index)
        redirected = int(rng.integers(0, dataset_len - 1))
        if redirected >= current_index:
            redirected += 1
        return redirected

    def _redirected_item_once(self, redirected_index: int) -> Dict:
        try:
            redirected_item = self._build_item(redirected_index)
        except _ENDPOINT_SAMPLE_RETRY_EXCEPTIONS:
            record = self.sample_records[int(redirected_index) % len(self.sample_records)]
            sample_id = f"{record['batch_name']}_{record['image_id']}_redirect_fallback"
            redirected_item = _empty_endpoint_item(
                image_size=self.image_size,
                rgb_input=self.rgb_input,
                sample_id=sample_id,
                edge_path=str(record['edge_path']),
                input_path=str(record['image_path']),
            )
        redirected_item['target']['curriculum_direct_accept'] = torch.tensor(0.0, dtype=torch.float32)
        redirected_item['target']['curriculum_redirected_request'] = torch.tensor(1.0, dtype=torch.float32)
        redirected_item['target']['curriculum_rejected_candidates'] = (
            redirected_item['target']['curriculum_rejected_candidates'] + 1.0
        )
        return redirected_item

    def _target_cache_path(self, index: int) -> Path:
        cached = self._target_cache_path_cache.get(int(index))
        if cached is not None:
            return cached
        record = self.sample_records[index]
        cache_path = ensure_target_cache(
            edge_path=Path(record['edge_path']),
            cache_root=self.cache_root,
            version_name=self.version_name,
            target_degree=self.target_degree,
            min_curve_length=self.min_curve_length,
        )
        self._target_cache_path_cache[int(index)] = cache_path
        return cache_path

    def _build_item(self, index: int) -> Dict:
        record = self.sample_records[index]
        target_cache = load_cached_targets(self._target_cache_path(index))
        image = load_image_array_original(Path(record['image_path']), rgb=self.rgb_input)
        edge_mask = load_binary_edge_annotation(Path(record['edge_path'])).astype(np.float32)[..., None] / 255.0
        curves = np.asarray(target_cache['curves'], dtype=np.float32)
        rng = self._rng_for_index(index)
        if self.split == 'train' and self.train_augment:
            image_hwc, edge_hwc, target_data = prepare_training_endpoint_sample_with_mask(
                image=image,
                mask=edge_mask,
                curves=curves,
                image_size=self.image_size,
                augment_cfg=self.augment_cfg,
                rng=rng,
                dedupe_distance_px=self.endpoint_dedupe_distance_px,
            )
        else:
            image_hwc, edge_hwc, target_data = prepare_eval_endpoint_sample_with_mask(
                image=image,
                mask=edge_mask,
                curves=curves,
                image_size=self.image_size,
                dedupe_distance_px=self.endpoint_dedupe_distance_px,
            )
        target_data = {
            **target_data,
            'image_hwc': image_hwc,
            'edge_hwc': edge_hwc,
            'endpoint_dedupe_distance_px': self.endpoint_dedupe_distance_px,
        }
        sample_id = f"{record['batch_name']}_{record['image_id']}"
        return _endpoint_target_from_curve_target(
            target_data=target_data,
            sample_id=sample_id,
            edge_path=str(record['edge_path']),
            input_path=str(record['image_path']),
        )

    def _raw_endpoint_count(self, index: int) -> int:
        cached = self._raw_endpoint_count_cache.get(int(index))
        if cached is not None:
            return int(cached)
        cache_path = self._target_cache_path(index)
        target_data = load_cached_targets(cache_path)
        points = curves_to_unique_endpoints(
            target_data['curves'],
            image_size=target_data['image_size'],
            dedupe_distance_px=self.endpoint_dedupe_distance_px,
        )
        count = int(points.shape[0])
        self._raw_endpoint_count_cache[int(index)] = count
        return count

    def __getitem__(self, index: int) -> Dict:
        if not (self.split == 'train' and self.train_augment and self.curriculum_enabled):
            return self._build_item(index)
        cap = self._current_curriculum_cap()
        dataset_len = len(self.sample_records)
        rng = self._rng_for_index(index + self.current_epoch * max(1, dataset_len))
        if self.curriculum_global_skip_points > 0:
            try:
                raw_point_count = self._raw_endpoint_count(index)
            except _ENDPOINT_SAMPLE_RETRY_EXCEPTIONS:
                raw_point_count = None
            if raw_point_count is None or raw_point_count > self.curriculum_global_skip_points:
                redirected_index = self._random_redirect_index(rng, int(index), dataset_len)
                return self._redirected_item_once(redirected_index)
        try:
            item = self._build_item(index)
        except _ENDPOINT_SAMPLE_RETRY_EXCEPTIONS:
            redirected_index = self._random_redirect_index(rng, int(index), dataset_len)
            return self._redirected_item_once(redirected_index)
        point_count = int(item['target']['points'].shape[0])
        if 0 < point_count <= cap:
            item['target']['curriculum_direct_accept'] = torch.tensor(1.0, dtype=torch.float32)
            item['target']['curriculum_redirected_request'] = torch.tensor(0.0, dtype=torch.float32)
            item['target']['curriculum_rejected_candidates'] = torch.tensor(0.0, dtype=torch.float32)
            return item
        redirected_index = self._random_redirect_index(rng, int(index), dataset_len)
        return self._redirected_item_once(redirected_index)


def endpoint_detection_collate(batch: List[Dict]) -> Dict:
    images = torch.stack([item['image'] for item in batch], dim=0)
    targets = [item['target'] for item in batch]
    return {'images': images, 'targets': targets}
