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


class ParametricEdgeDataModule(pl.LightningDataModule):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        data_cfg = self.config['data']
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

        common = dict(
            cache_root=Path(data_cfg['cache_dir']),
            image_size=int(data_cfg['image_size']),
            version_name=data_cfg.get('target_version', 'v5_anchor_consistent'),
            input_root=Path(data_cfg['input_root']) if data_cfg.get('input_root') else None,
            rgb_input=bool(data_cfg.get('rgb_input', False)),
            target_degree=int(data_cfg.get('target_degree', 3)),
            min_curve_length=float(data_cfg.get('min_curve_length', 3.0)),
            max_targets=int(data_cfg.get('max_targets', 128)),
        )
        self.train_dataset = ParametricEdgeDataset(train_paths, **common)
        self.val_dataset = ParametricEdgeDataset(val_paths or train_paths[:1], **common)
        self.test_dataset = ParametricEdgeDataset(test_paths or val_paths or train_paths[:1], **common)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.config['data']['batch_size']),
            shuffle=True,
            num_workers=int(self.config['data'].get('num_workers', 0)),
            collate_fn=parametric_edge_collate,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=int(self.config['data'].get('val_batch_size', self.config['data']['batch_size'])),
            shuffle=False,
            num_workers=int(self.config['data'].get('num_workers', 0)),
            collate_fn=parametric_edge_collate,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=int(self.config['data'].get('val_batch_size', self.config['data']['batch_size'])),
            shuffle=False,
            num_workers=int(self.config['data'].get('num_workers', 0)),
            collate_fn=parametric_edge_collate,
        )
