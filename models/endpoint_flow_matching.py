from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from misc_utils.endpoint_flow_utils import match_uniform_points_to_targets, sample_uniform_points, scale_points_to_flow
from models.dab_curve_detr import DABEncoder, DABResNetBackbone

try:
    from diffusers.configuration_utils import ConfigMixin, register_to_config
    from diffusers.models.embeddings import TimestepEmbedding, Timesteps
    from diffusers.models.modeling_utils import ModelMixin
    from diffusers.utils import BaseOutput
except ImportError as exc:  # pragma: no cover - exercised on cluster env
    raise ImportError(
        'diffusers is required for endpoint_flow_matching. Install it in the runtime environment.'
    ) from exc


@dataclass
class EndpointFlowMatchingOutput(BaseOutput):
    sample: torch.FloatTensor


class EndpointFlowDecoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, memory: torch.Tensor, memory_key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        residual = x
        x_norm = self.self_norm(x)
        x = residual + self.self_attn(x_norm, x_norm, x_norm, need_weights=False)[0]

        residual = x
        x_norm = self.cross_norm(x)
        x = residual + self.cross_attn(
            x_norm,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )[0]

        residual = x
        x_norm = self.ffn_norm(x)
        x = residual + self.ffn(x_norm)
        return x


class EndpointFlowMatchingModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        *,
        input_channels: int = 3,
        hidden_dim: int = 256,
        num_points: int = 300,
        nheads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        encoder_dim_feedforward: int = 2048,
        decoder_dim_feedforward: int = 2048,
        dropout: float = 0.0,
        dab_backbone_name: str = 'resnet50',
        dab_backbone_pretrained: bool = True,
        dab_backbone_dilation: bool = False,
        backbone_input_norm: str = 'imagenet',
        gradient_checkpointing: bool = True,
        cond_drop_rate: float = 0.1,
        num_train_timesteps: int = 1000,
        scheduler_shift: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_points = int(num_points)
        self.num_train_timesteps = int(num_train_timesteps)
        self.cond_drop_rate = float(cond_drop_rate)
        self.gradient_checkpointing = bool(gradient_checkpointing)

        backbone_config = {
            'model': {
                'input_channels': int(input_channels),
                'hidden_dim': self.hidden_dim,
                'dab_backbone_name': str(dab_backbone_name),
                'dab_backbone_pretrained': bool(dab_backbone_pretrained),
                'dab_backbone_dilation': bool(dab_backbone_dilation),
                'backbone_input_norm': str(backbone_input_norm),
            }
        }
        self.backbone = DABResNetBackbone(backbone_config)
        self.input_proj = nn.Conv2d(self.backbone.num_channels, self.hidden_dim, kernel_size=1)
        self.encoder = DABEncoder(
            d_model=self.hidden_dim,
            nhead=int(nheads),
            dim_feedforward=int(encoder_dim_feedforward),
            dropout=float(dropout),
            num_layers=int(num_encoder_layers),
            gradient_checkpointing=bool(gradient_checkpointing),
        )
        self.point_proj = nn.Linear(2, self.hidden_dim)
        self.time_proj = Timesteps(self.hidden_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embed = TimestepEmbedding(self.hidden_dim, self.hidden_dim)
        self.decoder_layers = nn.ModuleList(
            [
                EndpointFlowDecoderLayer(
                    hidden_dim=self.hidden_dim,
                    num_heads=int(nheads),
                    dim_feedforward=int(decoder_dim_feedforward),
                    dropout=float(dropout),
                )
                for _ in range(int(num_decoder_layers))
            ]
        )
        self.final_norm = nn.LayerNorm(self.hidden_dim)
        self.velocity_head = nn.Linear(self.hidden_dim, 2)
        self.current_epoch = 0

    @classmethod
    def from_config_dict(cls, config: Dict) -> 'EndpointFlowMatchingModel':
        model_cfg = dict(config['model'])
        model_cfg.pop('arch', None)
        return cls(**model_cfg)

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def _encode_image(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        src, mask, pos = self.backbone(images)
        src = self.input_proj(src)
        pos = pos.to(src.dtype)
        src_flat = src.flatten(2).permute(0, 2, 1)
        pos_flat = pos.flatten(2).permute(0, 2, 1)
        mask_flat = mask.flatten(1)
        memory = self.encoder(src_flat, pos=pos_flat, key_padding_mask=mask_flat)
        return memory, mask_flat

    def _decode(self, hidden: torch.Tensor, memory: torch.Tensor, memory_key_padding_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.decoder_layers:
            if self.gradient_checkpointing and self.training:
                hidden = checkpoint(layer, hidden, memory, memory_key_padding_mask, use_reentrant=False)
            else:
                hidden = layer(hidden, memory, memory_key_padding_mask)
        return self.final_norm(hidden)

    def predict(self, sample: torch.Tensor, timestep: torch.Tensor, image_cond: torch.Tensor) -> EndpointFlowMatchingOutput:
        memory, memory_key_padding_mask = self._encode_image(image_cond)
        timestep = timestep.to(device=sample.device)
        if timestep.ndim == 0:
            timestep = timestep[None].expand(sample.shape[0])
        timestep_proj = self.time_proj(timestep)
        timestep_emb = self.time_embed(timestep_proj.to(dtype=sample.dtype)).unsqueeze(1)
        hidden = self.point_proj(sample) + timestep_emb
        hidden = self._decode(hidden, memory, memory_key_padding_mask)
        velocity = self.velocity_head(hidden)
        return EndpointFlowMatchingOutput(sample=velocity)

    def forward(self, images: torch.Tensor, targets=None) -> Dict[str, torch.Tensor]:
        if targets is None:
            raise ValueError('EndpointFlowMatchingModel.forward requires targets for training/validation. Use the pipeline for inference.')
        batch_size = images.shape[0]
        source_points_ext = sample_uniform_points(
            batch_size,
            self.num_points,
            device=images.device,
            dtype=images.dtype,
        )
        target_points_ext, valid_mask = match_uniform_points_to_targets(source_points_ext, targets)
        source_points = scale_points_to_flow(source_points_ext)
        target_points = scale_points_to_flow(target_points_ext).to(device=images.device, dtype=images.dtype)
        valid_mask = valid_mask.to(device=images.device)
        timestep_indices = torch.randint(0, self.num_train_timesteps, (batch_size,), device=images.device)
        timesteps = timestep_indices.to(dtype=images.dtype)
        tau = (timestep_indices.to(dtype=images.dtype) + 0.5) / float(self.num_train_timesteps)
        tau = tau.view(batch_size, 1, 1)
        noisy_points = (1.0 - tau) * source_points + tau * target_points
        image_cond = images
        cond_drop_mask = torch.zeros((batch_size,), dtype=torch.bool, device=images.device)
        if self.training and self.cond_drop_rate > 0.0:
            cond_drop_mask = torch.rand((batch_size,), device=images.device) < self.cond_drop_rate
            if cond_drop_mask.any():
                image_cond = images.clone()
                image_cond[cond_drop_mask] = 0.0
        prediction = self.predict(noisy_points, timesteps, image_cond)
        return {
            'pred_velocity': prediction.sample,
            'target_velocity': target_points - source_points,
            'valid_mask': valid_mask.to(dtype=source_points.dtype),
            'timestep_indices': timestep_indices,
            'source_points': source_points,
            'target_points': target_points,
            'noisy_points': noisy_points,
            'pred_group_count': 1,
            'cond_drop_mask': cond_drop_mask,
        }
