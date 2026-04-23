from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from edge_datasets.graph_pipeline import (
    prepare_eval_endpoint_sample,
    prepare_training_endpoint_sample,
)
from misc_utils.bezier_target_utils import (
    load_compact_bezier_targets,
    load_image_array_original,
    resolve_input_path,
)
from misc_utils.endpoint_target_utils import curves_to_unique_endpoints

_ENDPOINT_SAMPLE_RETRY_EXCEPTIONS = (FileNotFoundError, OSError, ValueError, KeyError, EOFError)


def _endpoint_target_from_curve_target(
    target_data: Dict,
    *,
    sample_id: str,
    input_path: str,
    bezier_path: str,
    dataset_name: str,
) -> Dict:
    image_chw = np.transpose(target_data['image_hwc'], (2, 0, 1))
    curves = np.asarray(target_data.get('curves', np.zeros((0, 2, 2), dtype=np.float32)), dtype=np.float32)
    if 'points' in target_data:
        points = np.asarray(target_data['points'], dtype=np.float32)
    else:
        points = curves_to_unique_endpoints(
            curves,
            image_size=target_data['image_size'],
            dedupe_distance_px=float(target_data.get('endpoint_dedupe_distance_px', 2.0)),
        )
    curve_tensor = torch.from_numpy(curves).float()
    point_tensor = torch.from_numpy(points).float()
    labels = torch.zeros((point_tensor.shape[0],), dtype=torch.long)
    return {
        'image': torch.from_numpy(image_chw).float(),
        'target': {
            'labels': labels,
            'curves': curve_tensor,
            'points': point_tensor,
            'image_size': torch.from_numpy(target_data['image_size']).long(),
            'sample_id': sample_id,
            'input_path': input_path,
            'bezier_path': bezier_path,
            'dataset_name': dataset_name,
        },
    }


class ParametricEndpointDataset(Dataset):
    def __init__(
        self,
        curve_paths: Sequence[Path],
        image_size: int,
        input_root: Optional[Sequence[Path]] = None,
        rgb_input: bool = False,
        target_degree: int = 5,
        max_targets: int = 128,
        split: str = 'train',
        train_augment: bool = False,
        augment_cfg: Optional[Dict] = None,
        endpoint_dedupe_distance_px: float = 2.0,
    ) -> None:
        self.curve_paths = [Path(path) for path in curve_paths]
        self.image_size = int(image_size)
        self.input_root = [Path(root) for root in input_root] if input_root is not None else None
        self.rgb_input = bool(rgb_input)
        self.target_degree = int(target_degree)
        self.max_targets = int(max_targets)
        self.split = str(split)
        self.train_augment = bool(train_augment)
        self.augment_cfg = dict(augment_cfg or {})
        self.endpoint_dedupe_distance_px = float(endpoint_dedupe_distance_px)

    def __len__(self) -> int:
        return len(self.curve_paths)

    def _load_curve_targets(self, index: int) -> Dict:
        return load_compact_bezier_targets(
            self.curve_paths[index],
            target_degree=self.target_degree,
            max_targets=self.max_targets,
        )

    @staticmethod
    def _rng_for_index(index: int) -> np.random.Generator:
        return np.random.default_rng((torch.initial_seed() + index) % (2 ** 32))

    def _build_item(self, index: int) -> Dict:
        curve_path = self.curve_paths[index]
        target_cache = self._load_curve_targets(index)
        image_path = resolve_input_path(curve_path, self.input_root)
        image = load_image_array_original(image_path, rgb=self.rgb_input)
        curves = np.asarray(target_cache['curves'], dtype=np.float32)
        rng = self._rng_for_index(index)
        if self.split == 'train' and self.train_augment:
            image_hwc, target_data = prepare_training_endpoint_sample(
                image=image,
                curves=curves,
                image_size=self.image_size,
                augment_cfg=self.augment_cfg,
                rng=rng,
                dedupe_distance_px=self.endpoint_dedupe_distance_px,
                max_targets=self.max_targets,
            )
        else:
            image_hwc, target_data = prepare_eval_endpoint_sample(
                image=image,
                curves=curves,
                image_size=self.image_size,
                dedupe_distance_px=self.endpoint_dedupe_distance_px,
                max_targets=self.max_targets,
            )
        target_data = {
            **target_data,
            'image_hwc': image_hwc,
            'endpoint_dedupe_distance_px': self.endpoint_dedupe_distance_px,
        }
        return _endpoint_target_from_curve_target(
            target_data,
            sample_id=curve_path.stem,
            input_path=str(image_path),
            bezier_path=str(curve_path),
            dataset_name='parametric_edge',
        )

    def __getitem__(self, index: int) -> Dict:
        try:
            return self._build_item(index)
        except _ENDPOINT_SAMPLE_RETRY_EXCEPTIONS:
            dataset_len = len(self.curve_paths)
            if dataset_len <= 1:
                raise
            redirected_index = int(np.random.randint(0, dataset_len - 1))
            if redirected_index >= int(index):
                redirected_index += 1
            return self[redirected_index]


class LaionSyntheticEndpointDataset(Dataset):
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

    def __len__(self) -> int:
        return len(self.sample_records)

    def _load_curve_targets(self, index: int) -> Dict:
        record = self.sample_records[index]
        return load_compact_bezier_targets(
            Path(record['bezier_path']),
            target_degree=self.target_degree,
            max_targets=self.max_targets,
        )

    @staticmethod
    def _rng_for_index(index: int) -> np.random.Generator:
        return np.random.default_rng((torch.initial_seed() + index) % (2 ** 32))

    def _build_item(self, index: int) -> Dict:
        record = self.sample_records[index]
        target_cache = self._load_curve_targets(index)
        image = load_image_array_original(Path(record['image_path']), rgb=self.rgb_input)
        curves = np.asarray(target_cache['curves'], dtype=np.float32)
        rng = self._rng_for_index(index)
        if self.split == 'train' and self.train_augment:
            image_hwc, target_data = prepare_training_endpoint_sample(
                image=image,
                curves=curves,
                image_size=self.image_size,
                augment_cfg=self.augment_cfg,
                rng=rng,
                dedupe_distance_px=self.endpoint_dedupe_distance_px,
                max_targets=self.max_targets,
            )
        else:
            image_hwc, target_data = prepare_eval_endpoint_sample(
                image=image,
                curves=curves,
                image_size=self.image_size,
                dedupe_distance_px=self.endpoint_dedupe_distance_px,
                max_targets=self.max_targets,
            )
        target_data = {
            **target_data,
            'image_hwc': image_hwc,
            'endpoint_dedupe_distance_px': self.endpoint_dedupe_distance_px,
        }
        sample_id = f"{record['batch_name']}_{record['image_id']}"
        return _endpoint_target_from_curve_target(
            target_data,
            sample_id=sample_id,
            input_path=str(record['image_path']),
            bezier_path=str(record['bezier_path']),
            dataset_name='laion_synthetic',
        )

    def __getitem__(self, index: int) -> Dict:
        dataset_len = len(self.sample_records)
        try:
            return self._build_item(index)
        except _ENDPOINT_SAMPLE_RETRY_EXCEPTIONS:
            if dataset_len <= 1:
                raise
            redirected_index = int(np.random.randint(0, dataset_len - 1))
            if redirected_index >= int(index):
                redirected_index += 1
            return self[redirected_index]


def endpoint_detection_collate(batch: List[Dict]) -> Dict:
    images = torch.stack([item['image'] for item in batch], dim=0)
    targets = [item['target'] for item in batch]
    return {'images': images, 'targets': targets}
