import argparse
from pathlib import Path
from typing import Any, Dict, List

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import WandbLogger

from callbacks.training_visualizer import ParametricEdgeVisualizer
from callbacks.tracked_curve_visualizer import TrackedCurveVisualizer
from edge_datasets import build_datamodule
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


def resolve_trainer_strategy(config) -> str:
    trainer_cfg = config.get('trainer', {})
    strategy = trainer_cfg.get('strategy', 'auto')
    devices = trainer_cfg.get('devices', 1)
    if int(devices) == 1 and str(strategy).startswith('deepspeed'):
        return 'auto'
    return strategy


def resolve_limit_train_batches(config):
    trainer_cfg = config.get('trainer', {})
    if trainer_cfg.get('effective_train_batches_per_epoch') is None:
        return trainer_cfg.get('limit_train_batches', 1.0)
    effective_steps = int(trainer_cfg['effective_train_batches_per_epoch'])
    accumulate = max(1, int(trainer_cfg.get('accumulate_grad_batches', 1)))
    return effective_steps * accumulate


def _format_runtime_values(value: Any, context: Dict[str, Any]) -> Any:
    if isinstance(value, str):
        try:
            return value.format(**context)
        except (KeyError, IndexError, ValueError):
            return value
    if isinstance(value, list):
        return [_format_runtime_values(item, context) for item in value]
    if isinstance(value, dict):
        return {key: _format_runtime_values(item, context) for key, item in value.items()}
    return value


def resolve_runtime_scaling(config: Dict[str, Any]) -> Dict[str, Any]:
    trainer_cfg = config.setdefault('trainer', {})
    data_cfg = config.get('data', {})

    accelerator = str(trainer_cfg.get('accelerator', 'auto'))
    devices = trainer_cfg.get('devices', 'auto')
    if isinstance(devices, str) and devices.lower() == 'auto':
        if accelerator == 'gpu' or (accelerator == 'auto' and torch.cuda.is_available()):
            devices = max(1, torch.cuda.device_count())
        else:
            devices = 1
        trainer_cfg['devices'] = devices

    if isinstance(devices, str):
        devices = int(devices)
        trainer_cfg['devices'] = devices

    effective_batch_size = trainer_cfg.get('effective_batch_size', trainer_cfg.get('global_batch_size'))
    accumulate = trainer_cfg.get('accumulate_grad_batches', 1)
    if isinstance(accumulate, str) and accumulate.lower() == 'auto':
        if effective_batch_size is None:
            raise ValueError('trainer.effective_batch_size is required when accumulate_grad_batches is auto')
        per_device_batch = int(data_cfg.get('batch_size', 1))
        denom = per_device_batch * int(devices)
        if denom <= 0:
            raise ValueError(f'Invalid per-device batch/device count for auto scaling: batch={per_device_batch}, devices={devices}')
        if int(effective_batch_size) % denom != 0:
            raise ValueError(
                f'Effective batch size {effective_batch_size} is not divisible by per-device batch {per_device_batch} * devices {devices}'
            )
        accumulate = int(effective_batch_size) // denom
        trainer_cfg['accumulate_grad_batches'] = accumulate

    global_batch_size = int(devices) * int(data_cfg.get('batch_size', 1)) * int(trainer_cfg.get('accumulate_grad_batches', 1))
    context = {
        'devices': int(devices),
        'world_size': int(devices),
        'num_devices': int(devices),
        'batch_size': int(data_cfg.get('batch_size', 1)),
        'accumulate_grad_batches': int(trainer_cfg.get('accumulate_grad_batches', 1)),
        'effective_batch_size': effective_batch_size,
        'global_batch_size': global_batch_size,
    }
    return _format_runtime_values(config, context)


def main() -> None:
    parser = argparse.ArgumentParser(description='Train a DETR-style parametric edge detector.')
    parser.add_argument('--config', default='configs/parametric_edge/default.yaml')
    parser.add_argument('--override-config', default=None)
    parser.add_argument('--resume-from', default=None)
    args = parser.parse_args()

    config = load_config(args.config, args.override_config)
    config = resolve_runtime_scaling(config)
    seed = config.get('trainer', {}).get('seed')
    if seed is not None:
        pl.seed_everything(int(seed), workers=True)
    torch.set_float32_matmul_precision('high')
    root_dir = Path(config['trainer']['default_root_dir'])
    root_dir.mkdir(parents=True, exist_ok=True)

    datamodule = build_datamodule(config)
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
    arch = str(config.get('model', {}).get('arch', 'dab_curve_detr')).lower()
    if arch == 'dab_endpoint_detr':
        from callbacks.endpoint_visualizer import ParametricEndpointVisualizer
        visualizer = ParametricEndpointVisualizer(
            val_every_n_epochs=int(config['callbacks'].get('visualization_every_n_epochs', 1)),
            train_every_n_steps=int(config['callbacks'].get('visualization_every_n_train_steps', 0)),
            max_score_points=int(config['callbacks'].get('visualization_max_points', 64)),
            score_threshold=float(config['callbacks'].get('visualization_score_threshold', 0.3)),
        )
    elif arch == 'endpoint_flow_matching':
        from callbacks.endpoint_flow_visualizer import EndpointFlowVisualizer
        visualizer = EndpointFlowVisualizer(
            val_every_n_epochs=int(config['callbacks'].get('visualization_every_n_epochs', 1)),
            inference_steps=int(config['callbacks'].get('visualization_inference_steps', 20)),
            guidance_scales=tuple(config['callbacks'].get('visualization_guidance_scales', [1.0, 3.0, 5.0, 7.0])),
        )
    else:
        visualizer = ParametricEdgeVisualizer(
            val_every_n_epochs=int(config['callbacks'].get('visualization_every_n_epochs', 1)),
            train_every_n_steps=int(config['callbacks'].get('visualization_every_n_train_steps', 0)),
            max_score_curves=int(config['callbacks'].get('visualization_max_curves', 24)),
            score_threshold=float(config['callbacks'].get('visualization_score_threshold', 0.3)),
        )
    callbacks = [checkpoint, LearningRateMonitor(logging_interval='epoch'), visualizer]
    tracked_curve_cfg = config.get('callbacks', {}).get('tracked_curve')
    if tracked_curve_cfg and bool(tracked_curve_cfg.get('enabled', False)):
        callbacks.append(
            TrackedCurveVisualizer(
                sample_id=str(tracked_curve_cfg['sample_id']),
                target_idx=int(tracked_curve_cfg['target_idx']),
                every_n_epochs=int(tracked_curve_cfg.get('every_n_epochs', 10)),
                output_subdir=str(tracked_curve_cfg.get('output_subdir', 'tracked_curve')),
            )
        )
    trainer = pl.Trainer(
        default_root_dir=str(root_dir),
        max_epochs=int(config['trainer']['max_epochs']),
        accelerator=config['trainer'].get('accelerator', 'auto'),
        devices=config['trainer'].get('devices', 1),
        strategy=resolve_trainer_strategy(config),
        precision=config['trainer'].get('precision', '32-true'),
        accumulate_grad_batches=int(config['trainer'].get('accumulate_grad_batches', 1)),
        log_every_n_steps=int(config['trainer'].get('log_every_n_steps', 1)),
        limit_train_batches=resolve_limit_train_batches(config),
        limit_val_batches=config['trainer'].get('limit_val_batches', 1.0),
        overfit_batches=config['trainer'].get('overfit_batches', 0.0),
        gradient_clip_val=float(config['trainer'].get('gradient_clip_val', 0.1)),
        callbacks=callbacks,
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
