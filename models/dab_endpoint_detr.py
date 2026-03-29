from typing import Dict, List, Optional, Tuple

import math
import torch
from torch import nn

from models.curve_coordinates import curve_external_to_internal
from models.dab_curve_detr import DABCurveDecoder, DABEncoder, DABResNetBackbone, inverse_sigmoid


class DABEndpointDETR(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.config = config
        model_cfg = config['model']
        self.hidden_dim = int(model_cfg['hidden_dim'])
        self.num_queries = int(model_cfg['num_queries'])
        self.num_encoder_layers = int(model_cfg.get('num_encoder_layers', 6))
        self.num_decoder_layers = int(model_cfg.get('num_decoder_layers', 6))
        self.num_heads = int(model_cfg.get('nheads', 8))
        self.encoder_ffn_dim = int(model_cfg.get('encoder_dim_feedforward', model_cfg.get('dim_feedforward', self.hidden_dim * 4)))
        self.decoder_ffn_dim = int(model_cfg.get('decoder_dim_feedforward', model_cfg.get('dim_feedforward', self.hidden_dim * 4)))
        self.dropout = float(model_cfg.get('dropout', 0.0))
        self.object_bias = float(model_cfg.get('object_bias', -2.0))
        self.no_object_bias = float(model_cfg.get('no_object_bias', 2.0))
        self.keep_query_pos = bool(model_cfg.get('dab_keep_query_pos', False))
        self.query_scale_type = str(model_cfg.get('dab_query_scale_type', 'cond_elewise'))
        self.query_modulator_mode = str(model_cfg.get('dab_query_modulation_mode', 'sine_proj'))
        self.force_legacy_cross_attn = bool(model_cfg.get('dab_force_legacy_cross_attn', False))
        self.gradient_checkpointing = bool(model_cfg.get('gradient_checkpointing', False))
        self.dn_enabled = bool(model_cfg.get('dn_enabled', False))
        self.num_dn_groups = int(model_cfg.get('dn_num_groups', 0))
        self.dn_use_cdn = bool(model_cfg.get('dn_use_cdn', False))
        self.dn_noise_scale = float(model_cfg.get('dn_noise_scale', 0.04))
        self.dn_use_label_embed = bool(model_cfg.get('dn_use_label_embed', False))
        self.dn_label_noise_ratio = float(model_cfg.get('dn_label_noise_ratio', 0.0))
        self.aux_weight = float(config['loss'].get('aux_weight', 0.0))
        self.aux_layer_stride = max(1, int(config['loss'].get('aux_layer_stride', 1)))
        self.aux_last_n_layers = max(0, int(config['loss'].get('aux_last_n_layers', 0)))

        self.backbone = DABResNetBackbone(config)
        self.input_proj = nn.Conv2d(self.backbone.num_channels, self.hidden_dim, kernel_size=1)
        self.encoder = DABEncoder(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.encoder_ffn_dim,
            dropout=self.dropout,
            num_layers=self.num_encoder_layers,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        self.decoder = DABCurveDecoder(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.decoder_ffn_dim,
            dropout=self.dropout,
            num_layers=self.num_decoder_layers,
            curve_dim=2,
            keep_query_pos=self.keep_query_pos,
            query_scale_type=self.query_scale_type,
            curve_embed_diff_each_layer=False,
            query_modulator_mode=self.query_modulator_mode,
            force_legacy_cross_attn=self.force_legacy_cross_attn,
            gradient_checkpointing=self.gradient_checkpointing,
        )

        self.class_embed = nn.Linear(self.hidden_dim, 2)
        self.query_content_embed = nn.Embedding(self.num_queries, self.hidden_dim)
        self.refpoint_embed = nn.Embedding(self.num_queries, 2)
        self.dn_content_embed = nn.Embedding(1, self.hidden_dim)
        self.dn_label_embed = nn.Embedding(2, self.hidden_dim) if self.dn_use_label_embed else None

        nn.init.constant_(self.class_embed.bias[0], self.object_bias)
        nn.init.constant_(self.class_embed.bias[1], self.no_object_bias)
        self._init_reference_points()
        self.current_epoch = 0

    def _init_reference_points(self) -> None:
        with torch.no_grad():
            side = int(math.ceil(math.sqrt(float(self.num_queries))))
            ys = torch.linspace(0.05, 0.95, side)
            xs = torch.linspace(0.05, 0.95, side)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
            refs = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)[: self.num_queries].clamp(1e-4, 1.0 - 1e-4)
            self.refpoint_embed.weight.copy_(inverse_sigmoid(refs))

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def _selected_aux_layer_indices(self) -> List[int]:
        if self.aux_weight <= 0.0:
            return []
        aux_layers = max(0, self.num_decoder_layers - 1)
        indices = list(range(aux_layers))
        if self.aux_last_n_layers > 0:
            indices = indices[-self.aux_last_n_layers:]
        return indices[:: self.aux_layer_stride]

    def _set_aux_outputs(self, outputs_class: torch.Tensor, outputs_points: torch.Tensor, dn_count: int) -> List[Dict[str, torch.Tensor]]:
        aux_indices = self._selected_aux_layer_indices()
        return [
            {
                'pred_logits': outputs_class[layer_idx][:, dn_count:],
                'pred_points': outputs_points[layer_idx][:, dn_count:],
            }
            for layer_idx in aux_indices
        ]

    def _build_dn_query_content(self, dn_labels: torch.Tensor) -> torch.Tensor:
        base = self.dn_content_embed.weight[0].view(1, 1, -1).expand(dn_labels.shape[0], dn_labels.shape[1], -1)
        if self.dn_label_embed is None:
            return base
        return base + self.dn_label_embed(dn_labels)

    def _maybe_noisy_dn_input_labels(self, dn_labels: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if self.dn_label_embed is None or self.dn_label_noise_ratio <= 0.0:
            return dn_labels
        noisy_labels = dn_labels.clone()
        flip_mask = (torch.rand_like(noisy_labels, dtype=torch.float32) < self.dn_label_noise_ratio) & valid_mask
        noisy_labels[flip_mask] = 1 - noisy_labels[flip_mask]
        return noisy_labels

    def _build_dn_queries(
        self,
        targets: Optional[List[dict]],
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        if (not self.training) or targets is None or (not self.dn_enabled) or self.num_dn_groups <= 0:
            return None, None, None
        if self.dn_use_cdn:
            raise NotImplementedError('CDN is not implemented for the endpoint-only branch yet.')
        batch_size = len(targets)
        max_targets = max((target['points'].shape[0] for target in targets), default=0)
        if max_targets <= 0:
            return None, None, None

        pad_size = max_targets * self.num_dn_groups
        dn_ref_points = torch.full((batch_size, pad_size, 2), 0.5, dtype=torch.float32, device=device)
        dn_target_points = torch.zeros((batch_size, pad_size, 2), dtype=torch.float32, device=device)
        dn_mask = torch.zeros((batch_size, pad_size), dtype=torch.bool, device=device)
        dn_labels = torch.zeros((batch_size, pad_size), dtype=torch.long, device=device)
        eps = 1e-4

        for batch_idx, target in enumerate(targets):
            target_points_ext = target['points'].to(device)
            if target_points_ext.numel() == 0:
                continue
            target_points_int = curve_external_to_internal(target_points_ext, self.config)
            count = target_points_int.shape[0]
            for group_idx in range(self.num_dn_groups):
                start = group_idx * max_targets
                end = start + count
                noisy = target_points_int + torch.randn_like(target_points_int) * self.dn_noise_scale
                dn_ref_points[batch_idx, start:end] = noisy.clamp(eps, 1.0 - eps)
                dn_target_points[batch_idx, start:end] = target_points_ext
                dn_mask[batch_idx, start:end] = True

        dn_input_labels = self._maybe_noisy_dn_input_labels(dn_labels, dn_mask)
        dn_content = self._build_dn_query_content(dn_input_labels)
        dn_meta = {
            'mask': dn_mask,
            'labels': dn_labels,
            'points': dn_target_points,
            'count': pad_size,
            'pad_size': pad_size,
            'single_pad': max_targets,
            'num_dn_groups': self.num_dn_groups,
            'use_cdn': False,
        }
        dn_ref_unsigmoid = inverse_sigmoid(dn_ref_points)
        return dn_content, dn_ref_unsigmoid, dn_meta

    def _build_dn_attention_mask(self, dn_meta: Optional[Dict[str, torch.Tensor]], device: torch.device) -> Optional[torch.Tensor]:
        if dn_meta is None or int(dn_meta.get('pad_size', 0)) <= 0:
            return None
        pad_size = int(dn_meta['pad_size'])
        single_pad = int(dn_meta['single_pad'])
        num_dn_groups = int(dn_meta['num_dn_groups'])
        total_queries = pad_size + self.num_queries
        attn_mask = torch.zeros((total_queries, total_queries), dtype=torch.bool, device=device)
        attn_mask[pad_size:, :pad_size] = True
        for group_idx in range(num_dn_groups):
            start = group_idx * single_pad
            end = start + single_pad
            if start > 0:
                attn_mask[start:end, :start] = True
            if end < pad_size:
                attn_mask[start:end, end:pad_size] = True
        return attn_mask

    def forward(self, images: torch.Tensor, targets=None) -> Dict[str, torch.Tensor]:
        src, mask, pos = self.backbone(images)
        src = self.input_proj(src)
        pos = pos.to(src.dtype)

        batch_size = src.shape[0]
        src_flat = src.flatten(2).permute(0, 2, 1)
        pos_flat = pos.flatten(2).permute(0, 2, 1)
        mask_flat = mask.flatten(1)

        memory = self.encoder(src_flat, pos=pos_flat, key_padding_mask=mask_flat)
        matching_query_content = self.query_content_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        matching_ref_points_unsigmoid = self.refpoint_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        dn_query_content, dn_ref_points_unsigmoid, dn_meta = self._build_dn_queries(targets, images.device)
        if dn_query_content is not None and dn_ref_points_unsigmoid is not None:
            query_content = torch.cat([dn_query_content, matching_query_content], dim=1)
            ref_points_unsigmoid = torch.cat([dn_ref_points_unsigmoid, matching_ref_points_unsigmoid], dim=1)
        else:
            query_content = matching_query_content
            ref_points_unsigmoid = matching_ref_points_unsigmoid
            dn_meta = None
        self_attn_mask = self._build_dn_attention_mask(dn_meta, images.device)

        hs, references = self.decoder(
            tgt=query_content,
            memory=memory,
            memory_key_padding_mask=mask_flat,
            pos=pos_flat,
            ref_curves_unsigmoid=ref_points_unsigmoid,
            self_attn_mask=self_attn_mask,
        )
        outputs_class = self.class_embed(hs)

        reference_before_sigmoid = inverse_sigmoid(references)
        outputs_points = []
        for layer_idx in range(hs.shape[0]):
            tmp = self.decoder._curve_embed_for_layer(layer_idx, hs[layer_idx])
            tmp = tmp + reference_before_sigmoid[layer_idx]
            outputs_points.append(tmp.sigmoid())
        outputs_points = torch.stack(outputs_points, dim=0)
        dn_count = int(dn_meta['count']) if dn_meta is not None else 0

        out = {
            'pred_logits': outputs_class[-1][:, dn_count:],
            'pred_points': outputs_points[-1][:, dn_count:],
            'pred_group_count': 1,
        }
        if dn_count > 0:
            out['dn_pred_logits'] = outputs_class[-1][:, :dn_count]
            out['dn_pred_points'] = outputs_points[-1][:, :dn_count]
            out['dn_meta'] = dn_meta
        aux_outputs = self._set_aux_outputs(outputs_class, outputs_points, dn_count=dn_count)
        if aux_outputs:
            out['aux_outputs'] = aux_outputs
        return out
