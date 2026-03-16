import argparse
from pathlib import Path
from typing import List

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import WandbLogger

from callbacks.training_visualizer import ParametricEdgeVisualizer
from edge_datasets.parametric_edge_datamodule import ParametricEdgeDataModule
from misc_utils.config_utils import load_config
from pl_trainer.parametric_edge_trainer import ParametricEdgeLightningModule


def load_pretrained_with_query_expansion(model: ParametricEdgeLightningModule, checkpoint_path: str) -> None:
    checkpoint_file = Path(checkpoint_path)
    if any(ch in checkpoint_path for ch in '*?[]'):
        matches = sorted(checkpoint_file.parent.glob(checkpoint_file.name))
        if not matches:
            raise FileNotFoundError(f'No checkpoint matched pattern: {checkpoint_path}')
        checkpoint_file = matches[0]
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    model_state = model.state_dict()
    updated = {}
    for key, value in state_dict.items():
        if key not in model_state:
            continue
        target = model_state[key]
        if value.shape == target.shape:
            updated[key] = value
            continue
        if key.endswith('model.query_embed.weight') and value.ndim == 2 and target.ndim == 2 and value.shape[1] == target.shape[1]:
            count = min(value.shape[0], target.shape[0])
            merged = target.clone()
            merged[:count] = value[:count]
            if count < target.shape[0]:
                mean_query = value[:count].mean(dim=0, keepdim=True)
                merged[count:] = mean_query
            updated[key] = merged
    model.load_state_dict(updated, strict=False)


def build_loggers(config, root_dir: Path) -> List:
    logging_cfg = config.get('logging', {})
    loggers: List = []

    if bool(logging_cfg.get('csv', True)):
        loggers.append(CSVLogger(save_dir=str(root_dir), name='csv_logs'))

    wandb_cfg = logging_cfg.get('wandb', {})
    if bool(wandb_cfg.get('enabled', False)):
        save_dir = Path(wandb_cfg.get('save_dir', root_dir / 'wandb'))
        save_dir.mkdir(parents=True, exist_ok=True)
        loggers.append(WandbLogger(
            project=wandb_cfg.get('project', 'parametric-edge-prediction'),
            name=wandb_cfg.get('name'),
            save_dir=str(save_dir),
            offline=bool(wandb_cfg.get('offline', False)),
            log_model=bool(wandb_cfg.get('log_model', False)),
            tags=wandb_cfg.get('tags'),
            group=wandb_cfg.get('group'),
            job_type=wandb_cfg.get('job_type'),
            notes=wandb_cfg.get('notes'),
            config=config,
        ))

    return loggers


def main() -> None:
    parser = argparse.ArgumentParser(description='Train a DETR-style parametric edge detector.')
    parser.add_argument('--config', default='configs/parametric_edge/default.yaml')
    parser.add_argument('--override-config', default=None)
    parser.add_argument('--resume-from', default=None)
    args = parser.parse_args()

    config = load_config(args.config, args.override_config)
    root_dir = Path(config['trainer']['default_root_dir'])
    root_dir.mkdir(parents=True, exist_ok=True)

    datamodule = ParametricEdgeDataModule(config)
    model = ParametricEdgeLightningModule(config)
    if config['model'].get('pretrained_checkpoint'):
        load_pretrained_with_query_expansion(model, config['model']['pretrained_checkpoint'])
    loggers = build_loggers(config, root_dir)
    checkpoint = ModelCheckpoint(
        dirpath=root_dir / 'checkpoints',
        save_top_k=1,
        save_last=True,
        monitor='val_loss_main',
        mode='min',
        filename='best-{epoch:03d}-{val_loss_main:.4f}',
        auto_insert_metric_name=False,
    )
    visualizer = ParametricEdgeVisualizer(
        every_n_epochs=int(config['callbacks'].get('visualization_every_n_epochs', 1)),
        max_score_curves=int(config['callbacks'].get('visualization_max_curves', 24)),
        score_threshold=float(config['callbacks'].get('visualization_score_threshold', 0.3)),
    )
    trainer = pl.Trainer(
        default_root_dir=str(root_dir),
        max_epochs=int(config['trainer']['max_epochs']),
        accelerator=config['trainer'].get('accelerator', 'auto'),
        devices=config['trainer'].get('devices', 1),
        strategy=config['trainer'].get('strategy', 'auto'),
        precision=config['trainer'].get('precision', '32-true'),
        accumulate_grad_batches=int(config['trainer'].get('accumulate_grad_batches', 1)),
        log_every_n_steps=int(config['trainer'].get('log_every_n_steps', 1)),
        limit_train_batches=config['trainer'].get('limit_train_batches', 1.0),
        limit_val_batches=config['trainer'].get('limit_val_batches', 1.0),
        overfit_batches=config['trainer'].get('overfit_batches', 0.0),
        gradient_clip_val=float(config['trainer'].get('gradient_clip_val', 0.1)),
        callbacks=[checkpoint, LearningRateMonitor(logging_interval='epoch'), visualizer],
        logger=loggers if loggers else False,
        deterministic=bool(config['trainer'].get('deterministic', True)),
        benchmark=bool(config['trainer'].get('benchmark', False)),
        num_sanity_val_steps=int(config['trainer'].get('num_sanity_val_steps', 2)),
        check_val_every_n_epoch=int(config['trainer'].get('check_val_every_n_epoch', 1)),
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume_from)
    if bool(config['trainer'].get('run_test_after_fit', True)):
        trainer.test(model=model, datamodule=datamodule, ckpt_path='best')


if __name__ == '__main__':
    main()
