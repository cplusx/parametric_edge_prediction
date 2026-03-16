from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from edge_datasets.graph_pipeline import prepare_eval_sample, prepare_training_sample
from misc_utils.bezier_target_utils import ensure_graph_cache, load_cached_graph, load_image_array_original, resolve_input_path, unpack_polylines


class ParametricEdgeDataset(Dataset):
    def __init__(
        self,
        edge_paths: Sequence[Path],
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

    def __len__(self) -> int:
        return len(self.edge_paths)

    def __getitem__(self, index: int) -> Dict:
        edge_path = self.edge_paths[index]
        cache_path = ensure_graph_cache(
            edge_path=edge_path,
            cache_root=self.cache_root,
            version_name=self.version_name,
        )
        graph_data = load_cached_graph(cache_path)
        image_path = resolve_input_path(edge_path, self.input_root)
        image = load_image_array_original(image_path, rgb=self.rgb_input)
        polylines = unpack_polylines(graph_data['graph_points'], graph_data['graph_offsets'])
        rng = np.random.default_rng((torch.initial_seed() + index) % (2 ** 32))
        if self.split == 'train' and self.train_augment:
            image_hwc, target_data = prepare_training_sample(
                image=image,
                polylines=polylines,
                image_size=self.image_size,
                target_degree=self.target_degree,
                min_curve_length=self.min_curve_length,
                max_targets=self.max_targets,
                augment_cfg=self.augment_cfg,
                rng=rng,
            )
        else:
            image_hwc, target_data = prepare_eval_sample(
                image=image,
                polylines=polylines,
                image_size=self.image_size,
                target_degree=self.target_degree,
                min_curve_length=self.min_curve_length,
                max_targets=self.max_targets,
            )
        image_chw = np.transpose(image_hwc, (2, 0, 1))
        curves = torch.from_numpy(target_data['curves']).float()
        boxes = torch.from_numpy(target_data['curve_boxes']).float()
        lengths = torch.from_numpy(target_data['curve_lengths']).float()
        norm_lengths = torch.from_numpy(target_data.get('curve_norm_lengths', target_data['curve_lengths'] * 0.0)).float()
        curvatures = torch.from_numpy(target_data.get('curve_curvatures', target_data['curve_lengths'] * 0.0)).float()
        labels = torch.ones((curves.shape[0],), dtype=torch.long)
        return {
            'image': torch.from_numpy(image_chw).float(),
            'target': {
                'labels': labels,
                'curves': curves,
                'boxes': boxes,
                'curve_lengths': lengths,
                'curve_norm_lengths': norm_lengths[: curves.shape[0]],
                'curve_curvatures': curvatures[: curves.shape[0]],
                'image_size': torch.from_numpy(target_data['image_size']).long(),
                'num_targets': torch.tensor(curves.shape[0], dtype=torch.long),
                'sample_id': edge_path.stem,
                'edge_path': str(edge_path),
                'input_path': str(image_path),
            },
        }


def parametric_edge_collate(batch: List[Dict]) -> Dict:
    images = torch.stack([item['image'] for item in batch], dim=0)
    targets = [item['target'] for item in batch]
    return {'images': images, 'targets': targets}
