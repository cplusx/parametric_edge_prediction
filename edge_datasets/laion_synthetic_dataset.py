import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from edge_datasets.graph_pipeline import prepare_eval_sample_with_mask, prepare_training_sample_with_mask
from misc_utils.bezier_target_utils import load_binary_edge_annotation, load_cached_graph, load_image_array_original, unpack_polylines

SUPPORTED_IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')


def _laion_entry_cache_path(
    data_root: Path,
    cache_root: Path,
    image_root: Path,
    edge_root: Path,
    batch_glob: str,
    quantize: int,
    batches: Optional[Sequence[str]],
) -> Path:
    key_parts = [
        str(data_root),
        str(cache_root),
        str(image_root),
        str(edge_root),
        str(batch_glob),
        str(int(quantize)),
    ]
    if batches is not None:
        key_parts.extend(sorted(str(batch) for batch in batches))
    digest = hashlib.sha1('\n'.join(key_parts).encode('utf-8')).hexdigest()[:12]
    return data_root / f'laion_entry_cache_{digest}.txt'


def _read_laion_entry_cache(cache_path: Path) -> List[Dict[str, Path]]:
    if not cache_path.exists():
        return []
    records: List[Dict[str, Path]] = []
    with cache_path.open('r', encoding='utf-8') as handle:
        for raw_line in handle:
            line = raw_line.rstrip('\n')
            if not line:
                continue
            fields = line.split('\t')
            if len(fields) != 5:
                continue
            batch_name, image_id, image_path_str, edge_path_str, graph_cache_path_str = fields
            image_path = Path(image_path_str)
            edge_path = Path(edge_path_str)
            graph_cache_path = Path(graph_cache_path_str)
            if not image_path.exists() or not edge_path.exists() or not graph_cache_path.exists():
                continue
            records.append({
                'batch_name': batch_name,
                'image_id': image_id,
                'image_path': image_path,
                'edge_path': edge_path,
                'cache_path': graph_cache_path,
            })
    return records


def _write_laion_entry_cache(cache_path: Path, records: Sequence[Dict[str, Path]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(
                '\t'.join([
                    str(record['batch_name']),
                    str(record['image_id']),
                    str(record['image_path']),
                    str(record['edge_path']),
                    str(record['cache_path']),
                ])
            )
            handle.write('\n')


def select_laion_sample_records(
    sample_records: Sequence[Dict[str, Path]],
    max_samples: Optional[int] = None,
    selection_seed: Optional[int] = None,
    selection_offset: int = 0,
) -> List[Dict[str, Path]]:
    records = list(sample_records)
    if selection_seed is not None:
        rng = np.random.default_rng(int(selection_seed))
        order = rng.permutation(len(records)).tolist()
        records = [records[index] for index in order]
    offset = max(0, int(selection_offset))
    if offset:
        records = records[offset:]
    if max_samples is not None:
        records = records[: int(max_samples)]
    return records


def discover_laion_synthetic_samples(
    data_root: Path,
    cache_root: Path,
    image_root: Optional[Path] = None,
    edge_root: Optional[Path] = None,
    batches: Optional[Sequence[str]] = None,
    batch_glob: str = 'batch*',
    quantize: int = 4,
    max_samples: Optional[int] = None,
    selection_seed: Optional[int] = None,
    selection_offset: int = 0,
) -> List[Dict[str, Path]]:
    data_root = Path(data_root)
    cache_root = Path(cache_root)
    image_root = Path(image_root) if image_root is not None else data_root
    edge_root = Path(edge_root) if edge_root is not None else (data_root / 'laion_edge_v2')
    entry_cache_path = _laion_entry_cache_path(
        data_root=data_root,
        cache_root=cache_root,
        image_root=image_root,
        edge_root=edge_root,
        batch_glob=batch_glob,
        quantize=quantize,
        batches=batches,
    )
    cached_records = _read_laion_entry_cache(entry_cache_path)
    if cached_records:
        return select_laion_sample_records(
            cached_records,
            max_samples=max_samples,
            selection_seed=selection_seed,
            selection_offset=selection_offset,
        )
    batch_names = [str(batch) for batch in batches] if batches is not None else sorted(path.name for path in cache_root.glob(batch_glob) if path.is_dir())
    records: List[Dict[str, Path]] = []
    for batch_name in batch_names:
        cache_batch_dir = cache_root / batch_name
        if not cache_batch_dir.exists():
            continue
        for cache_path in sorted(cache_batch_dir.glob('*_graph.npz')):
            image_id = cache_path.stem.replace('_graph', '')
            edge_path = edge_root / batch_name / 'edges' / f'quantize_{int(quantize)}' / 'edge' / f'{image_id}.npz'
            if not edge_path.exists():
                continue
            image_path = None
            for suffix in SUPPORTED_IMAGE_SUFFIXES:
                candidate = image_root / batch_name / 'images' / f'{image_id}{suffix}'
                if candidate.exists():
                    image_path = candidate
                    break
            if image_path is None:
                continue
            records.append({
                'batch_name': batch_name,
                'image_id': image_id,
                'image_path': image_path,
                'edge_path': edge_path,
                'cache_path': cache_path,
            })
    if records:
        _write_laion_entry_cache(entry_cache_path, records)
    return select_laion_sample_records(
        records,
        max_samples=max_samples,
        selection_seed=selection_seed,
        selection_offset=selection_offset,
    )


class LaionSyntheticEdgeDataset(Dataset):
    def __init__(
        self,
        sample_records: Sequence[Dict[str, Path]],
        image_size: int,
        target_degree: int,
        min_curve_length: float,
        max_targets: int,
        split: str = 'train',
        train_augment: bool = False,
        augment_cfg: Optional[Dict] = None,
        rgb_input: bool = True,
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
        image_chw = np.transpose(image_hwc, (2, 0, 1))
        edge_chw = np.transpose(edge_hwc, (2, 0, 1))
        curves = torch.from_numpy(target_data['curves']).float()
        boxes = torch.from_numpy(target_data['curve_boxes']).float()
        lengths = torch.from_numpy(target_data['curve_lengths']).float()
        norm_lengths = torch.from_numpy(target_data.get('curve_norm_lengths', target_data['curve_lengths'] * 0.0)).float()
        curvatures = torch.from_numpy(target_data.get('curve_curvatures', target_data['curve_lengths'] * 0.0)).float()
        labels = torch.ones((curves.shape[0],), dtype=torch.long)
        sample_id = f"{record['batch_name']}_{record['image_id']}"
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
                'edge_mask': torch.from_numpy(edge_chw).float(),
                'num_targets': torch.tensor(curves.shape[0], dtype=torch.long),
                'sample_id': sample_id,
                'edge_path': str(record['edge_path']),
                'input_path': str(record['image_path']),
                'cache_path': str(record['cache_path']),
                'dataset_name': 'laion_synthetic',
            },
        }