from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from edge_datasets.graph_pipeline import prepare_eval_endpoint_sample_with_mask
from edge_datasets.endpoint_dataset import endpoint_detection_collate
from misc_utils.bezier_target_utils import load_binary_edge_annotation, load_cached_targets, load_image_array_original
from misc_utils.endpoint_target_utils import curves_to_unique_endpoints


def _build_endpoint_item(
    *,
    image_hwc: np.ndarray,
    edge_hwc: np.ndarray,
    points: np.ndarray,
    sample_id: str,
    edge_path: str,
    input_path: str,
) -> Dict:
    image_chw = np.transpose(image_hwc, (2, 0, 1))
    edge_chw = np.transpose(edge_hwc, (2, 0, 1))
    point_tensor = torch.from_numpy(np.asarray(points, dtype=np.float32)).float()
    labels = torch.zeros((point_tensor.shape[0],), dtype=torch.long)
    return {
        'image': torch.from_numpy(image_chw).float(),
        'target': {
            'labels': labels,
            'points': point_tensor,
            'image_size': torch.tensor([image_hwc.shape[0], image_hwc.shape[1]], dtype=torch.long),
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


class SingleLaionEndpointFlowOverfitDataset(Dataset):
    def __init__(
        self,
        *,
        image_path: Path,
        edge_path: Path,
        bezier_cache_path: Path,
        image_size: int = 256,
        rgb_input: bool = True,
        endpoint_dedupe_distance_px: float = 2.0,
        repeats: int = 800,
    ) -> None:
        self.image_path = Path(image_path)
        self.edge_path = Path(edge_path)
        self.bezier_cache_path = Path(bezier_cache_path)
        self.image_size = int(image_size)
        self.rgb_input = bool(rgb_input)
        self.endpoint_dedupe_distance_px = float(endpoint_dedupe_distance_px)
        self.repeats = int(repeats)

        target_cache = load_cached_targets(self.bezier_cache_path)
        image = load_image_array_original(self.image_path, rgb=self.rgb_input)
        edge_mask = load_binary_edge_annotation(self.edge_path).astype(np.float32)[..., None] / 255.0
        curves = np.asarray(target_cache['curves'], dtype=np.float32)
        image_hwc, edge_hwc, target_data = prepare_eval_endpoint_sample_with_mask(
            image=image,
            mask=edge_mask,
            curves=curves,
            image_size=self.image_size,
            dedupe_distance_px=self.endpoint_dedupe_distance_px,
        )
        points = np.asarray(
            target_data.get(
                'points',
                curves_to_unique_endpoints(
                    target_data['curves'],
                    image_size=target_data['image_size'],
                    dedupe_distance_px=self.endpoint_dedupe_distance_px,
                ),
            ),
            dtype=np.float32,
        )
        sample_id = f'{self.edge_path.parent.parent.parent.parent.name}_{self.edge_path.stem}'
        self.item = _build_endpoint_item(
            image_hwc=image_hwc,
            edge_hwc=edge_hwc,
            points=points,
            sample_id=sample_id,
            edge_path=str(self.edge_path),
            input_path=str(self.image_path),
        )

    def __len__(self) -> int:
        return self.repeats

    def __getitem__(self, index: int) -> Dict:
        del index
        return {
            'image': self.item['image'].clone(),
            'target': {
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in self.item['target'].items()
            },
        }


class SingleLaionEndpointFlowOverfitDataModule:
    def __init__(
        self,
        *,
        train_dataset: Dataset,
        val_dataset: Dataset,
        batch_size: int,
        val_batch_size: Optional[int] = None,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> None:
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = val_dataset
        self.batch_size = int(batch_size)
        self.val_batch_size = int(val_batch_size if val_batch_size is not None else batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)

    def setup(self, stage=None) -> None:
        del stage

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=bool(self.num_workers > 0),
            collate_fn=endpoint_detection_collate,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=bool(self.num_workers > 0),
            collate_fn=endpoint_detection_collate,
        )

    def test_dataloader(self):
        return self.val_dataloader()
