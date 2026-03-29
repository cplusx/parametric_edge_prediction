from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from edge_datasets.graph_pipeline import prepare_eval_sample_with_mask, prepare_training_sample_with_mask
from misc_utils.bezier_target_utils import (
    ensure_graph_cache,
    load_binary_edge_annotation,
    load_cached_graph,
    load_image_array_original,
    resolve_input_path,
    unpack_polylines,
)
from misc_utils.endpoint_target_utils import curves_to_unique_endpoints


def _endpoint_target_from_curve_target(target_data: Dict, sample_id: str, edge_path: str, input_path: str) -> Dict:
    image_chw = np.transpose(target_data['image_hwc'], (2, 0, 1))
    edge_chw = np.transpose(target_data['edge_hwc'], (2, 0, 1))
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

    def __len__(self) -> int:
        return len(self.edge_paths)

    def __getitem__(self, index: int) -> Dict:
        edge_path = self.edge_paths[index]
        cache_path = ensure_graph_cache(edge_path=edge_path, cache_root=self.cache_root, version_name=self.version_name)
        graph_data = load_cached_graph(cache_path)
        image_path = resolve_input_path(edge_path, self.input_root)
        image = load_image_array_original(image_path, rgb=self.rgb_input)
        edge_mask = load_binary_edge_annotation(edge_path).astype(np.float32)[..., None] / 255.0
        polylines = unpack_polylines(graph_data['graph_points'], graph_data['graph_offsets'])
        rng = np.random.default_rng((torch.initial_seed() + index) % (2 ** 32))
        if self.split == 'train' and self.train_augment:
            image_hwc, edge_hwc, target_data = prepare_training_sample_with_mask(
                image=image,
                mask=edge_mask,
                polylines=polylines,
                image_size=self.image_size,
                target_degree=self.target_degree,
                min_curve_length=self.min_curve_length,
                max_targets=self.max_targets,
                augment_cfg=self.augment_cfg,
                rng=rng,
            )
        else:
            image_hwc, edge_hwc, target_data = prepare_eval_sample_with_mask(
                image=image,
                mask=edge_mask,
                polylines=polylines,
                image_size=self.image_size,
                target_degree=self.target_degree,
                min_curve_length=self.min_curve_length,
                max_targets=self.max_targets,
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


class LaionSyntheticEndpointDataset(Dataset):
    def __init__(
        self,
        sample_records: Sequence[Dict],
        image_size: int,
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
        self.image_size = int(image_size)
        self.target_degree = int(target_degree)
        self.min_curve_length = float(min_curve_length)
        self.max_targets = int(max_targets)
        self.split = str(split)
        self.train_augment = bool(train_augment)
        self.augment_cfg = dict(augment_cfg or {})
        self.rgb_input = bool(rgb_input)
        self.endpoint_dedupe_distance_px = float(endpoint_dedupe_distance_px)

    def __len__(self) -> int:
        return len(self.sample_records)

    def __getitem__(self, index: int) -> Dict:
        record = self.sample_records[index]
        graph_data = load_cached_graph(Path(record['cache_path']))
        image = load_image_array_original(Path(record['image_path']), rgb=self.rgb_input)
        edge_mask = load_binary_edge_annotation(Path(record['edge_path'])).astype(np.float32)[..., None] / 255.0
        polylines = unpack_polylines(graph_data['graph_points'], graph_data['graph_offsets'])
        rng = np.random.default_rng((torch.initial_seed() + index) % (2 ** 32))
        if self.split == 'train' and self.train_augment:
            image_hwc, edge_hwc, target_data = prepare_training_sample_with_mask(
                image=image,
                mask=edge_mask,
                polylines=polylines,
                image_size=self.image_size,
                target_degree=self.target_degree,
                min_curve_length=self.min_curve_length,
                max_targets=self.max_targets,
                augment_cfg=self.augment_cfg,
                rng=rng,
            )
        else:
            image_hwc, edge_hwc, target_data = prepare_eval_sample_with_mask(
                image=image,
                mask=edge_mask,
                polylines=polylines,
                image_size=self.image_size,
                target_degree=self.target_degree,
                min_curve_length=self.min_curve_length,
                max_targets=self.max_targets,
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


def endpoint_detection_collate(batch: List[Dict]) -> Dict:
    images = torch.stack([item['image'] for item in batch], dim=0)
    targets = [item['target'] for item in batch]
    return {'images': images, 'targets': targets}
