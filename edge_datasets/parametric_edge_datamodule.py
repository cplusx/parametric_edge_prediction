import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pytorch_lightning as pl
from torch.utils.data import DataLoader

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


class ParametricEdgeDataModule(pl.LightningDataModule):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

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

    def setup(self, stage: Optional[str] = None) -> None:
        data_cfg = self.config['data']
        explicit_split_globs = any(
            data_cfg.get(key) is not None
            for key in ('train_edge_glob', 'train_edge_globs', 'val_edge_glob', 'val_edge_globs', 'test_edge_glob', 'test_edge_globs')
        )

        if explicit_split_globs:
            train_paths = _resolve_globbed_paths(data_cfg.get('train_edge_globs', data_cfg.get('train_edge_glob')))
            val_paths = _resolve_globbed_paths(data_cfg.get('val_edge_globs', data_cfg.get('val_edge_glob')))
            test_paths = _resolve_globbed_paths(data_cfg.get('test_edge_globs', data_cfg.get('test_edge_glob')))
            if not train_paths:
                raise FileNotFoundError('No training edge maps matched train_edge_glob/train_edge_globs')
            if not val_paths:
                raise FileNotFoundError('No validation edge maps matched val_edge_glob/val_edge_globs')
            if not test_paths:
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

        if data_cfg.get('overfit_num_samples') and explicit_split_globs:
            count = int(data_cfg['overfit_num_samples'])
            combined_paths = train_paths + val_paths + test_paths
            chosen_paths = combined_paths[:count]
            train_paths = chosen_paths
            val_paths = chosen_paths
            test_paths = chosen_paths

        common = dict(
            cache_root=Path(data_cfg['cache_dir']),
            image_size=int(data_cfg['image_size']),
            version_name=data_cfg.get('target_version', 'v5_anchor_consistent'),
            rgb_input=bool(data_cfg.get('rgb_input', False)),
            target_degree=int(data_cfg.get('target_degree', 3)),
            min_curve_length=float(data_cfg.get('min_curve_length', 3.0)),
            max_targets=int(data_cfg.get('max_targets', 128)),
            augment_cfg=dict(data_cfg.get('augment', {})),
        )
        self.train_dataset = self._build_dataset(
            train_paths,
            split='train',
            train_augment=bool(data_cfg.get('train_augment', True)),
            common=common,
        )
        self.val_dataset = self._build_dataset(
            val_paths or train_paths[:1],
            split='val',
            train_augment=False,
            common=common,
        )
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

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.config['data']['batch_size']),
            shuffle=True,
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
