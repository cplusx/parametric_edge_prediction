from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from edge_datasets.graph_pipeline import prepare_eval_curve_sample_with_mask, prepare_training_curve_sample_with_mask
from misc_utils.bezier_target_utils import ensure_target_cache, load_binary_edge_annotation, load_cached_targets, load_image_array_original

SUPPORTED_IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
LAION_ENTRY_CACHE_PROBE_COUNT = 32
LAION_ENTRY_CACHE_MIN_VALID_RATIO = 0.7
LAION_RUNTIME_FALLBACK_MIN_VALID_RATIO = 0.5
LAION_SAMPLE_LOAD_RETRY_LIMIT = 8
LAION_SAMPLE_LOAD_EXCEPTIONS = (FileNotFoundError, OSError, ValueError, KeyError, EOFError)
_LAION_ENTRY_CACHE_MEMO: Dict[Path, Dict[str, object]] = {}


def _laion_entry_cache_path(data_root: Path) -> Path:
    return data_root / 'laion_entry_cache.txt'


def _filter_laion_sample_records_by_batches(
    records: Sequence[Dict[str, Path]],
    batches: Optional[Sequence[str]],
) -> List[Dict[str, Path]]:
    if batches is None:
        return list(records)
    allowed_batches = {str(batch) for batch in batches}
    return [record for record in records if str(record['batch_name']) in allowed_batches]


def _record_paths_exist(record: Dict[str, Path]) -> bool:
    return Path(record['image_path']).exists() and Path(record['edge_path']).exists()


def _probe_record_indices(records: Sequence[Dict[str, Path]], probe_count: int) -> List[int]:
    if not records:
        return []
    capped_count = max(1, min(int(probe_count), len(records)))
    if capped_count >= len(records):
        return list(range(len(records)))
    if capped_count == 1:
        return [0]
    return sorted({int(round(position * (len(records) - 1) / (capped_count - 1))) for position in range(capped_count)})


def _probe_laion_sample_records(
    records: Sequence[Dict[str, Path]],
    probe_count: int,
) -> Dict[str, object]:
    probe_indices = _probe_record_indices(records, probe_count=probe_count)
    valid_indices = [record_index for record_index in probe_indices if _record_paths_exist(records[record_index])]
    ratio = float(len(valid_indices)) / float(len(probe_indices)) if probe_indices else 0.0
    return {
        'probe_indices': probe_indices,
        'valid_indices': valid_indices,
        'valid_ratio': ratio,
    }


def _read_laion_entry_cache(cache_path: Path) -> List[Dict[str, Path]]:
    if not cache_path.exists():
        return []
    cache_path = cache_path.resolve()
    cache_stat = cache_path.stat()
    memo_entry = _LAION_ENTRY_CACHE_MEMO.get(cache_path)
    if memo_entry is not None and memo_entry['mtime_ns'] == cache_stat.st_mtime_ns:
        return list(memo_entry['records'])
    records: List[Dict[str, Path]] = []
    with cache_path.open('r', encoding='utf-8') as handle:
        for raw_line in handle:
            line = raw_line.rstrip('\n')
            if not line:
                continue
            fields = line.split('\t')
            if len(fields) not in (4, 5):
                continue
            batch_name, image_id, image_path_str, edge_path_str = fields[:4]
            image_path = Path(image_path_str)
            edge_path = Path(edge_path_str)
            records.append({
                'batch_name': batch_name,
                'image_id': image_id,
                'image_path': image_path,
                'edge_path': edge_path,
            })
    probe_result = _probe_laion_sample_records(records, probe_count=LAION_ENTRY_CACHE_PROBE_COUNT)
    if not probe_result['valid_indices']:
        return []
    if float(probe_result['valid_ratio']) < LAION_ENTRY_CACHE_MIN_VALID_RATIO:
        return []
    _LAION_ENTRY_CACHE_MEMO[cache_path] = {
        'mtime_ns': cache_stat.st_mtime_ns,
        'records': tuple(records),
    }
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
                ])
            )
            handle.write('\n')


def _empty_laion_curve_item(record: Dict[str, Path], image_size: int, target_degree: int, rgb_input: bool) -> Dict:
    channels = 3 if rgb_input else 1
    control_points = int(target_degree) + 1
    sample_id = f"{record['batch_name']}_{record['image_id']}_empty_fallback"
    return {
        'image': torch.zeros((channels, image_size, image_size), dtype=torch.float32),
        'target': {
            'labels': torch.zeros((0,), dtype=torch.long),
            'curves': torch.zeros((0, control_points, 2), dtype=torch.float32),
            'boxes': torch.zeros((0, 4), dtype=torch.float32),
            'curve_lengths': torch.zeros((0,), dtype=torch.float32),
            'curve_norm_lengths': torch.zeros((0,), dtype=torch.float32),
            'curve_curvatures': torch.zeros((0,), dtype=torch.float32),
            'image_size': torch.tensor([image_size, image_size], dtype=torch.long),
            'edge_mask': torch.zeros((1, image_size, image_size), dtype=torch.float32),
            'num_targets': torch.tensor(0, dtype=torch.long),
            'sample_id': sample_id,
            'edge_path': str(record['edge_path']),
            'input_path': str(record['image_path']),
            'dataset_name': 'laion_synthetic',
        },
    }


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
    entry_cache_path = _laion_entry_cache_path(data_root=data_root)
    cached_records = _filter_laion_sample_records_by_batches(
        _read_laion_entry_cache(entry_cache_path),
        batches=batches,
    )
    if cached_records:
        return select_laion_sample_records(
            cached_records,
            max_samples=max_samples,
            selection_seed=selection_seed,
            selection_offset=selection_offset,
        )
    batch_names = sorted(path.name for path in edge_root.glob(batch_glob) if path.is_dir())
    records: List[Dict[str, Path]] = []
    for batch_name in batch_names:
        edge_batch_dir = edge_root / batch_name / 'edges' / f'quantize_{int(quantize)}' / 'edge'
        if not edge_batch_dir.exists():
            continue
        for edge_path in sorted(edge_batch_dir.glob('*.npz')):
            image_id = edge_path.stem
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
            })
    if records:
        _write_laion_entry_cache(entry_cache_path, records)
    records = _filter_laion_sample_records_by_batches(records, batches=batches)
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
        cache_root: Path,
        image_size: int,
        version_name: str,
        target_degree: int,
        min_curve_length: float,
        max_targets: int,
        split: str = 'train',
        train_augment: bool = False,
        augment_cfg: Optional[Dict] = None,
        rgb_input: bool = True,
    ) -> None:
        self.sample_records = list(sample_records)
        if not self.sample_records:
            raise ValueError('sample_records must not be empty')
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
        self.max_load_attempts = max(1, min(LAION_SAMPLE_LOAD_RETRY_LIMIT, len(self.sample_records)))
        probe_result = _probe_laion_sample_records(self.sample_records, probe_count=LAION_ENTRY_CACHE_PROBE_COUNT)
        self.known_good_indices = list(probe_result['valid_indices'])
        if not self.known_good_indices:
            raise FileNotFoundError('No readable LAION samples were found in the startup probe set.')
        if float(probe_result['valid_ratio']) < LAION_RUNTIME_FALLBACK_MIN_VALID_RATIO:
            raise RuntimeError(
                'Too many unreadable LAION samples in the startup probe set; rebuild the cache or verify dataset paths.'
            )
        self._target_cache_path_cache: Dict[int, Path] = {}

    def __len__(self) -> int:
        return len(self.sample_records)

    def _choose_fallback_index(self, base_index: int, attempt: int, tried_indices: set) -> Optional[int]:
        remaining_known_good = [index for index in self.known_good_indices if index not in tried_indices]
        if remaining_known_good:
            rng = np.random.default_rng((torch.initial_seed() + base_index * 9973 + attempt) % (2 ** 32))
            return int(remaining_known_good[int(rng.integers(len(remaining_known_good)))])
        remaining_indices = [index for index in range(len(self.sample_records)) if index not in tried_indices]
        if not remaining_indices:
            return None
        rng = np.random.default_rng((torch.initial_seed() + base_index * 9973 + attempt * 17) % (2 ** 32))
        return int(remaining_indices[int(rng.integers(len(remaining_indices)))])

    def _load_item_from_index(self, record_index: int) -> Dict:
        record = self.sample_records[record_index]
        cache_path = self._target_cache_path(record_index)
        target_cache = load_cached_targets(cache_path)
        image = load_image_array_original(Path(record['image_path']), rgb=self.rgb_input)
        edge_mask = load_binary_edge_annotation(Path(record['edge_path'])).astype(np.float32)[..., None] / 255.0
        curves = np.asarray(target_cache['curves'], dtype=np.float32)
        rng = np.random.default_rng((torch.initial_seed() + record_index) % (2 ** 32))
        if self.split == 'train' and self.train_augment:
            image_hwc, edge_hwc, target_data = prepare_training_curve_sample_with_mask(
                image=image,
                mask=edge_mask,
                curves=curves,
                image_size=self.image_size,
                max_targets=self.max_targets,
                augment_cfg=self.augment_cfg,
                rng=rng,
            )
        else:
            image_hwc, edge_hwc, target_data = prepare_eval_curve_sample_with_mask(
                image=image,
                mask=edge_mask,
                curves=curves,
                image_size=self.image_size,
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
        if record_index not in self.known_good_indices and _record_paths_exist(record):
            self.known_good_indices.append(record_index)
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
                'dataset_name': 'laion_synthetic',
            },
        }

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

    def __getitem__(self, index: int) -> Dict:
        tried_indices = set()
        candidate_index = int(index) % len(self.sample_records)
        attempt = 0
        while attempt < self.max_load_attempts:
            if candidate_index in tried_indices:
                next_index = self._choose_fallback_index(index, attempt, tried_indices)
                if next_index is None:
                    break
                candidate_index = next_index
            tried_indices.add(candidate_index)
            try:
                return self._load_item_from_index(candidate_index)
            except LAION_SAMPLE_LOAD_EXCEPTIONS:
                next_index = self._choose_fallback_index(index, attempt + 1, tried_indices)
                if next_index is None:
                    break
                else:
                    candidate_index = next_index
                attempt += 1
        fallback_indices = list(self.known_good_indices) + [i for i in range(len(self.sample_records)) if i not in self.known_good_indices]
        for fallback_index in fallback_indices:
            try:
                return self._load_item_from_index(int(fallback_index))
            except LAION_SAMPLE_LOAD_EXCEPTIONS:
                continue
        record = self.sample_records[int(index) % len(self.sample_records)]
        return _empty_laion_curve_item(
            record=record,
            image_size=self.image_size,
            target_degree=self.target_degree,
            rgb_input=self.rgb_input,
        )
