import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from callbacks.endpoint_flow_visualizer import EndpointFlowVisualizer
from edge_datasets.endpoint_flow_overfit_dataset import (
    SingleLaionEndpointFlowOverfitDataModule,
    SingleLaionEndpointFlowOverfitDataset,
)
from misc_utils.config_utils import load_config
from pl_trainer.parametric_edge_trainer import ParametricEdgeLightningModule
from train import resolve_runtime_scaling


def _read_first_cached_record(entry_cache_path: Path, sample_key: Optional[str] = None) -> Dict[str, str]:
    with entry_cache_path.open('r', encoding='utf-8') as handle:
        for raw_line in handle:
            line = raw_line.rstrip('\n')
            if not line:
                continue
            fields = line.split('\t')
            if len(fields) < 5:
                continue
            batch_name, image_id, image_path, edge_path, bezier_cache_path = fields[:5]
            key = f'{batch_name}_{image_id}'
            if sample_key is not None and key != sample_key:
                continue
            if not Path(image_path).exists() or not Path(edge_path).exists() or not Path(bezier_cache_path).exists():
                continue
            return {
                'sample_key': key,
                'image_path': image_path,
                'edge_path': edge_path,
                'bezier_cache_path': bezier_cache_path,
            }
    if sample_key is None:
        raise FileNotFoundError(f'No cached LAION sample found in {entry_cache_path}')
    raise FileNotFoundError(f'Specified sample_key={sample_key} not found with existing cache in {entry_cache_path}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Overfit endpoint_flow_matching on a single cached LAION sample.')
    parser.add_argument('--config', default='configs/parametric_edge/laion_endpoint_flow_matching.yaml')
    parser.add_argument('--entry-cache', required=True)
    parser.add_argument('--sample-key', default=None)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--train-repeats', type=int, default=800)
    parser.add_argument('--val-repeats', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--effective-batch-size', type=int, default=64)
    parser.add_argument('--max-epochs', type=int, default=10)
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--train-vis-every', type=int, default=100)
    parser.add_argument('--val-vis-every', type=int, default=1)
    args = parser.parse_args()

    record = _read_first_cached_record(Path(args.entry_cache), sample_key=args.sample_key)

    config = load_config(args.config)
    config['data']['batch_size'] = int(args.batch_size)
    config['data']['val_batch_size'] = int(args.batch_size)
    config['data']['num_workers'] = int(args.num_workers)
    config['trainer']['default_root_dir'] = str(Path(args.output_dir))
    config['trainer']['max_epochs'] = int(args.max_epochs)
    config['trainer']['devices'] = int(args.devices)
    config['trainer']['strategy'] = 'auto'
    config['trainer']['effective_batch_size'] = int(args.effective_batch_size)
    config['trainer']['accumulate_grad_batches'] = 'auto'
    config['trainer']['limit_train_batches'] = 1.0
    config['trainer']['limit_val_batches'] = 1.0
    config['trainer']['effective_train_batches_per_epoch'] = None
    config['trainer']['run_test_after_fit'] = False
    config['callbacks']['visualization_every_n_train_steps'] = int(args.train_vis_every)
    config['callbacks']['visualization_every_n_epochs'] = int(args.val_vis_every)
    config['logging']['wandb']['enabled'] = False
    config = resolve_runtime_scaling(config)

    if int(config.get('data', {}).get('num_workers', 0)) > 0:
        torch.multiprocessing.set_sharing_strategy('file_system')

    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('high')

    root_dir = Path(config['trainer']['default_root_dir'])
    root_dir.mkdir(parents=True, exist_ok=True)
    (root_dir / 'metadata.json').write_text(json.dumps({
        'sample_key': record['sample_key'],
        'image_path': record['image_path'],
        'edge_path': record['edge_path'],
        'bezier_cache_path': record['bezier_cache_path'],
        'batch_size': int(args.batch_size),
        'effective_batch_size': int(args.effective_batch_size),
        'train_repeats': int(args.train_repeats),
        'val_repeats': int(args.val_repeats),
        'max_epochs': int(args.max_epochs),
        'train_vis_every': int(args.train_vis_every),
    }, indent=2))

    train_dataset = SingleLaionEndpointFlowOverfitDataset(
        image_path=Path(record['image_path']),
        edge_path=Path(record['edge_path']),
        bezier_cache_path=Path(record['bezier_cache_path']),
        image_size=int(config['data']['image_size']),
        rgb_input=bool(config['data'].get('rgb_input', True)),
        endpoint_dedupe_distance_px=float(config['data'].get('endpoint_dedupe_distance_px', 2.0)),
        repeats=int(args.train_repeats),
    )
    val_dataset = SingleLaionEndpointFlowOverfitDataset(
        image_path=Path(record['image_path']),
        edge_path=Path(record['edge_path']),
        bezier_cache_path=Path(record['bezier_cache_path']),
        image_size=int(config['data']['image_size']),
        rgb_input=bool(config['data'].get('rgb_input', True)),
        endpoint_dedupe_distance_px=float(config['data'].get('endpoint_dedupe_distance_px', 2.0)),
        repeats=int(args.val_repeats),
    )
    datamodule = SingleLaionEndpointFlowOverfitDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=int(config['data']['batch_size']),
        val_batch_size=int(config['data'].get('val_batch_size', config['data']['batch_size'])),
        num_workers=int(config['data'].get('num_workers', 0)),
        pin_memory=bool(config['data'].get('pin_memory', True)),
    )

    model = ParametricEdgeLightningModule(config)
    loggers = [CSVLogger(save_dir=str(root_dir), name='csv_logs')]
    checkpoint = ModelCheckpoint(
        dirpath=root_dir / 'checkpoints',
        save_top_k=1,
        save_last=True,
        monitor='val_loss_main',
        mode='min',
        filename='best-{epoch:03d}-{val_loss_main:.4f}',
        auto_insert_metric_name=False,
    )
    visualizer = EndpointFlowVisualizer(
        val_every_n_epochs=int(config['callbacks'].get('visualization_every_n_epochs', 1)),
        train_every_n_steps=int(config['callbacks'].get('visualization_every_n_train_steps', 100)),
        inference_steps=int(config['callbacks'].get('visualization_inference_steps', 20)),
        guidance_scales=tuple(config['callbacks'].get('visualization_guidance_scales', [1.0])),
    )
    trainer = pl.Trainer(
        default_root_dir=str(root_dir),
        max_epochs=int(config['trainer']['max_epochs']),
        accelerator=config['trainer'].get('accelerator', 'gpu'),
        devices=config['trainer'].get('devices', 1),
        strategy='auto',
        precision=config['trainer'].get('precision', '32-true'),
        accumulate_grad_batches=int(config['trainer'].get('accumulate_grad_batches', 1)),
        log_every_n_steps=int(config['trainer'].get('log_every_n_steps', 10)),
        limit_train_batches=config['trainer'].get('limit_train_batches', 1.0),
        limit_val_batches=config['trainer'].get('limit_val_batches', 1.0),
        gradient_clip_val=float(config['trainer'].get('gradient_clip_val', 0.1)),
        callbacks=[checkpoint, LearningRateMonitor(logging_interval='epoch'), visualizer],
        logger=loggers,
        deterministic=bool(config['trainer'].get('deterministic', False)),
        benchmark=bool(config['trainer'].get('benchmark', True)),
        num_sanity_val_steps=int(config['trainer'].get('num_sanity_val_steps', 0)),
        check_val_every_n_epoch=int(config['trainer'].get('check_val_every_n_epoch', 1)),
        use_distributed_sampler=False,
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    main()
