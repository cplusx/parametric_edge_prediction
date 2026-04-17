from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from edge_datasets.graph_pipeline import prepare_eval_curve_sample, prepare_training_curve_sample
from misc_utils.bezier_target_utils import ensure_target_cache, load_cached_targets, load_compact_bezier_targets, load_image_array_original, resolve_input_path


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

    def __getitem__(self, index: int) -> Dict:
        curve_path = self.edge_paths[index]
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
        curves = torch.from_numpy(target_data['curves']).float()
        labels = torch.ones((curves.shape[0],), dtype=torch.long)
        return {
            'image': torch.from_numpy(image_chw).float(),
            'target': {
                'labels': labels,
                'curves': curves,
                'image_size': torch.from_numpy(target_data['image_size']).long(),
                'sample_id': curve_path.stem,
                'bezier_path': str(curve_path),
                'input_path': str(image_path),
            },
        }


def parametric_edge_collate(batch: List[Dict]) -> Dict:
    images = torch.stack([item['image'] for item in batch], dim=0)
    targets = [item['target'] for item in batch]
    return {'images': images, 'targets': targets}
