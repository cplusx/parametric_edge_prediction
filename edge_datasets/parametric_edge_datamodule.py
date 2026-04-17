import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pytorch_lightning as pl
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler

from edge_datasets.laion_synthetic_dataset import LaionSyntheticBezierDataset, discover_laion_bezier_samples
from edge_datasets.parametric_edge_dataset import ParametricEdgeDataset, parametric_edge_collate
from misc_utils.bezier_target_utils import image_id_from_stem


def grouped_split(edge_paths: Sequence[Path], val_ratio: float, test_ratio: float, seed: int) -> Tuple[List[Path], List[Path], List[Path]]:
    groups: Dict[str, List[Path]] = {}
    for edge_path in edge_paths:
        groups.setdefault(image_id_from_stem(edge_path.stem), []).append(edge_path)
    group_ids = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(group_ids)
    total = len(group_ids)
    val_count = max(1, int(round(total * val_ratio))) if total > 2 else 1
    test_count = max(1, int(round(total * test_ratio))) if total > 3 else 0
    val_ids = set(group_ids[:val_count])
    test_ids = set(group_ids[val_count: val_count + test_count])
    train_ids = set(group_ids[val_count + test_count:])
    if not train_ids:
        train_ids = set(group_ids)
        val_ids = set(group_ids[: min(1, total)])
        test_ids = set()
    train = [path for gid in sorted(train_ids) for path in sorted(groups[gid])]
    val = [path for gid in sorted(val_ids) for path in sorted(groups[gid])]
    test = [path for gid in sorted(test_ids) for path in sorted(groups[gid])]
    return train, val, test


def _resolve_globbed_paths(patterns) -> List[Path]:
    if patterns is None:
        return []
    if isinstance(patterns, (str, Path)):
        pattern_list = [str(patterns)]
    else:
        pattern_list = [str(pattern) for pattern in patterns]
    resolved: List[Path] = []
    for pattern in pattern_list:
        resolved.extend(sorted(Path().glob(pattern)))
    unique_paths = []
    seen = set()
    for path in resolved:
        resolved_path = path.resolve()
        if resolved_path in seen:
            continue
        seen.add(resolved_path)
        unique_paths.append(path)
    return unique_paths


class RepeatedDataset(Dataset):
    def __init__(self, base_dataset: Dataset, repeat_factor: int) -> None:
        if repeat_factor <= 0:
            raise ValueError('repeat_factor must be positive')
        self.base_dataset = base_dataset
        self.repeat_factor = int(repeat_factor)

    def __len__(self) -> int:
        return len(self.base_dataset) * self.repeat_factor

    def __getitem__(self, index: int):
        return self.base_dataset[index % len(self.base_dataset)]


class DistributedCurriculumSampler(Sampler[int]):
    def __init__(
        self,
        dataset: Dataset,
        sample_counts: Sequence[int],
        *,
        start_points: int,
        max_points: int,
        points_per_epoch: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> None:
        if len(sample_counts) != len(dataset):
            raise ValueError(
                f'sample_counts length {len(sample_counts)} does not match dataset length {len(dataset)}'
            )
        if num_replicas is None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                num_replicas = torch.distributed.get_world_size()
            else:
                num_replicas = 1
        if rank is None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
            else:
                rank = 0
        self.dataset = dataset
        self.sample_counts = [int(count) for count in sample_counts]
        self.start_points = int(start_points)
        self.max_points = int(max_points)
        self.points_per_epoch = int(points_per_epoch)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def current_cap(self) -> int:
        cap = self.start_points + self.epoch * self.points_per_epoch
        return max(1, min(int(cap), self.max_points))

    def _eligible_indices(self) -> List[int]:
        cap = self.current_cap()
        return [index for index, count in enumerate(self.sample_counts) if 0 < int(count) <= cap]

    def _compute_rank_indices(self) -> List[int]:
        indices = self._eligible_indices()
        if not indices:
            return []
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            order = torch.randperm(len(indices), generator=generator).tolist()
            indices = [indices[position] for position in order]
        if self.drop_last:
            total_size = (len(indices) // self.num_replicas) * self.num_replicas
            indices = indices[:total_size]
        else:
            total_size = int(math.ceil(len(indices) / self.num_replicas)) * self.num_replicas
            if total_size > len(indices):
                padding = indices[: total_size - len(indices)]
                indices = indices + padding
        return indices[self.rank:total_size:self.num_replicas]

    def _num_samples(self) -> int:
        num_indices = len(self._eligible_indices())
        if num_indices == 0:
            return 0
        if self.drop_last:
            total_size = (num_indices // self.num_replicas) * self.num_replicas
        else:
            total_size = int(math.ceil(num_indices / self.num_replicas)) * self.num_replicas
        return total_size // self.num_replicas

    def __iter__(self):
        return iter(self._compute_rank_indices())

    def __len__(self) -> int:
        return self._num_samples()


class ParametricEdgeDataModule(pl.LightningDataModule):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_sampler: Optional[Sampler[int]] = None

    def _split_input_roots(self, split: str) -> Optional[List[Path]]:
        data_cfg = self.config['data']
        split_keys = (f'{split}_input_roots', f'{split}_input_root')
        roots = None
        for split_key in split_keys:
            if data_cfg.get(split_key) is not None:
                roots = data_cfg.get(split_key)
                break
        if roots is None:
            roots = data_cfg.get('input_roots', data_cfg.get('input_root'))
        if roots is None:
            return None
        if isinstance(roots, (str, Path)):
            return [Path(roots)]
        return [Path(root) for root in roots]

    def _build_dataset(self, edge_paths: Sequence[Path], split: str, train_augment: bool, common: Dict) -> ParametricEdgeDataset:
        return ParametricEdgeDataset(
            edge_paths,
            split=split,
            train_augment=train_augment,
            input_root=self._split_input_roots(split),
            **common,
        )

    def _build_laion_dataset(self, dataset_cfg: Dict, split: str, train_augment: bool, common: Dict):
        sample_records = discover_laion_bezier_samples(
            data_root=Path(dataset_cfg['data_root']),
            image_root=Path(dataset_cfg['image_root']) if dataset_cfg.get('image_root') is not None else None,
            bezier_root=Path(dataset_cfg.get('bezier_root', dataset_cfg.get('edge_root'))) if dataset_cfg.get('bezier_root', dataset_cfg.get('edge_root')) is not None else None,
            entry_cache_path=Path(dataset_cfg['entry_cache_path']) if dataset_cfg.get('entry_cache_path') is not None else None,
            batches=dataset_cfg.get('batches'),
            batch_glob=str(dataset_cfg.get('batch_glob', 'batch*')),
            max_samples=dataset_cfg.get('max_samples'),
            selection_seed=dataset_cfg.get('selection_seed'),
            selection_offset=int(dataset_cfg.get('selection_offset', 0)),
        )
        if not sample_records:
            raise FileNotFoundError(f'No LAION synthetic samples found for config: {dataset_cfg}')
        return LaionSyntheticBezierDataset(
            sample_records=sample_records,
            image_size=int(common['image_size']),
            target_degree=int(common['target_degree']),
            max_targets=int(common['max_targets']),
            split=split,
            train_augment=train_augment,
            augment_cfg=dict(common['augment_cfg']),
            rgb_input=bool(common['rgb_input']),
        )

    def _build_optional_dataset_from_spec(self, dataset_cfg: Optional[Dict], split: str, train_augment: bool, common: Dict):
        if dataset_cfg is None:
            return None
        dataset_type = str(dataset_cfg.get('dataset_type', '')).lower()
        if dataset_type == 'laion_synthetic':
            return self._build_laion_dataset(dataset_cfg, split=split, train_augment=train_augment, common=common)
        raise ValueError(f'Unsupported dataset_type for {split}: {dataset_type}')

    def _build_extra_train_datasets(self, common: Dict) -> List:
        data_cfg = self.config['data']
        extra_cfgs = list(data_cfg.get('extra_train_datasets', []))
        datasets = []
        for extra_cfg in extra_cfgs:
            dataset_type = str(extra_cfg.get('dataset_type', '')).lower()
            if dataset_type != 'laion_synthetic':
                raise ValueError(f'Unsupported extra_train_datasets dataset_type: {dataset_type}')
            datasets.append(
                self._build_laion_dataset(
                    extra_cfg,
                    split='train',
                    train_augment=bool(data_cfg.get('train_augment', True)),
                    common=common,
                )
            )
        return datasets

    def _expand_train_dataset_for_virtual_epoch(self, dataset: Dataset) -> Dataset:
        trainer_cfg = self.config.get('trainer', {})
        effective_steps = trainer_cfg.get('effective_train_batches_per_epoch')
        if effective_steps is None:
            return dataset
        accumulate = max(1, int(trainer_cfg.get('accumulate_grad_batches', 1)))
        batch_size = int(self.config['data']['batch_size'])
        devices = max(1, int(trainer_cfg.get('devices', 1)))
        required_samples = int(effective_steps) * accumulate * batch_size * devices
        dataset_length = len(dataset)
        if dataset_length <= 0 or dataset_length >= required_samples:
            return dataset
        repeat_factor = int(math.ceil(required_samples / dataset_length))
        return RepeatedDataset(dataset, repeat_factor)

    def setup(self, stage: Optional[str] = None) -> None:
        data_cfg = self.config['data']
        include_primary_train_dataset = bool(data_cfg.get('include_primary_train_dataset', True))
        needs_primary_val_dataset = data_cfg.get('val_dataset') is None
        needs_primary_test_dataset = data_cfg.get('test_dataset') is None
        require_primary_split_paths = include_primary_train_dataset or needs_primary_val_dataset or needs_primary_test_dataset
        explicit_split_globs = any(
            data_cfg.get(key) is not None
            for key in ('train_edge_glob', 'train_edge_globs', 'val_edge_glob', 'val_edge_globs', 'test_edge_glob', 'test_edge_globs')
        )

        train_paths = []
        val_paths = []
        test_paths = []

        if not require_primary_split_paths:
            pass
        elif explicit_split_globs:
            train_paths = _resolve_globbed_paths(data_cfg.get('train_edge_globs', data_cfg.get('train_edge_glob')))
            val_paths = _resolve_globbed_paths(data_cfg.get('val_edge_globs', data_cfg.get('val_edge_glob')))
            test_paths = _resolve_globbed_paths(data_cfg.get('test_edge_globs', data_cfg.get('test_edge_glob')))
            if include_primary_train_dataset and not train_paths:
                raise FileNotFoundError('No training edge maps matched train_edge_glob/train_edge_globs')
            if needs_primary_val_dataset and not val_paths:
                raise FileNotFoundError('No validation edge maps matched val_edge_glob/val_edge_globs')
            if needs_primary_test_dataset and not test_paths:
                raise FileNotFoundError('No test edge maps matched test_edge_glob/test_edge_globs')
        else:
            edge_paths = sorted(Path().glob(data_cfg['edge_glob']))
            if not edge_paths:
                raise FileNotFoundError(f"No edge maps matched {data_cfg['edge_glob']}")
            if data_cfg.get('sample_names'):
                wanted = set(data_cfg['sample_names'])
                edge_paths = [path for path in edge_paths if path.name in wanted or path.stem in wanted]
                if not edge_paths:
                    raise FileNotFoundError(f"None of sample_names matched {data_cfg['sample_names']}")
            if data_cfg.get('limit_samples'):
                edge_paths = edge_paths[: int(data_cfg['limit_samples'])]

            if data_cfg.get('overfit_num_samples'):
                count = int(data_cfg['overfit_num_samples'])
                chosen_paths = edge_paths[:count]
                train_paths = chosen_paths
                val_paths = chosen_paths
                test_paths = chosen_paths
            else:
                train_paths, val_paths, test_paths = grouped_split(
                    edge_paths=edge_paths,
                    val_ratio=float(data_cfg.get('val_ratio', 0.1)),
                    test_ratio=float(data_cfg.get('test_ratio', 0.0)),
                    seed=int(data_cfg.get('split_seed', 42)),
                )
                if not train_paths:
                    train_paths = edge_paths
                    val_paths = val_paths or edge_paths[:1]
                    test_paths = test_paths or []

        if require_primary_split_paths and data_cfg.get('overfit_num_samples') and explicit_split_globs:
            count = int(data_cfg['overfit_num_samples'])
            combined_paths = train_paths + val_paths + test_paths
            chosen_paths = combined_paths[:count]
            train_paths = chosen_paths
            val_paths = chosen_paths
            test_paths = chosen_paths

        common = dict(
            cache_root=Path(data_cfg.get('cache_dir', '.')),
            image_size=int(data_cfg['image_size']),
            version_name=data_cfg.get('target_version', 'v5_anchor_consistent'),
            rgb_input=bool(data_cfg.get('rgb_input', False)),
            target_degree=int(data_cfg.get('target_degree', 3)),
            min_curve_length=float(data_cfg.get('min_curve_length', 3.0)),
            max_targets=int(data_cfg.get('max_targets', 128)),
            augment_cfg=dict(data_cfg.get('augment', {})),
        )
        primary_train_dataset = None
        if include_primary_train_dataset:
            primary_train_dataset = self._build_dataset(
                train_paths,
                split='train',
                train_augment=bool(data_cfg.get('train_augment', True)),
                common=common,
            )
        extra_train_datasets = self._build_extra_train_datasets(common)
        train_datasets = []
        if primary_train_dataset is not None:
            train_datasets.append(primary_train_dataset)
        train_datasets.extend(extra_train_datasets)
        if not train_datasets:
            raise ValueError('Training dataset list is empty. Enable primary train data or configure extra_train_datasets.')
        self.train_dataset = train_datasets[0] if len(train_datasets) == 1 else ConcatDataset(train_datasets)
        self.train_dataset = self._expand_train_dataset_for_virtual_epoch(self.train_dataset)
        self.val_dataset = self._build_optional_dataset_from_spec(data_cfg.get('val_dataset'), split='val', train_augment=False, common=common)
        if self.val_dataset is None:
            self.val_dataset = self._build_dataset(
                val_paths or train_paths[:1],
                split='val',
                train_augment=False,
                common=common,
            )
        self.test_dataset = self._build_optional_dataset_from_spec(data_cfg.get('test_dataset'), split='test', train_augment=False, common=common)
        if self.test_dataset is None:
            self.test_dataset = self._build_dataset(
                test_paths or val_paths or train_paths[:1],
                split='test',
                train_augment=False,
                common=common,
            )

    def _loader_kwargs(self) -> Dict:
        num_workers = int(self.config['data'].get('num_workers', 0))
        return {
            'num_workers': num_workers,
            'collate_fn': parametric_edge_collate,
            'pin_memory': bool(self.config['data'].get('pin_memory', True)),
            'persistent_workers': bool(num_workers > 0 and self.config['data'].get('persistent_workers', True)),
        }

    def set_epoch(self, epoch: int) -> None:
        if self.train_sampler is not None and hasattr(self.train_sampler, 'set_epoch'):
            self.train_sampler.set_epoch(int(epoch))

    def train_dataloader(self) -> DataLoader:
        shuffle = self.train_sampler is None
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.config['data']['batch_size']),
            shuffle=shuffle,
            sampler=self.train_sampler,
            **self._loader_kwargs(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=int(self.config['data'].get('val_batch_size', self.config['data']['batch_size'])),
            shuffle=False,
            **self._loader_kwargs(),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=int(self.config['data'].get('val_batch_size', self.config['data']['batch_size'])),
            shuffle=False,
            **self._loader_kwargs(),
        )
