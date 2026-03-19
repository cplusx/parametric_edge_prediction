from typing import Dict, List, Optional, Sequence, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from misc_utils.train_utils import build_dn_queries
from models.position_encoding import PositionEmbeddingSine


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x / (1.0 - x))


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int) -> None:
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        layers = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DINOv2PyramidBackbone(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        model_cfg = config['model']
        self.image_size = int(config['data']['image_size'])
        self.input_channels = int(model_cfg.get('input_channels', 3))
        self.hidden_dim = int(model_cfg['hidden_dim'])
        self.num_feature_levels = int(model_cfg.get('num_feature_levels', 4))
        self.backbone_name = str(model_cfg.get('backbone_name', 'vit_small_patch14_dinov2'))
        self.pretrained = bool(model_cfg.get('backbone_pretrained', True))
        self.out_indices = tuple(model_cfg.get('backbone_out_indices', (2, 5, 8, 11)))
        self.normalize_mode = str(model_cfg.get('backbone_input_norm', 'imagenet'))

        self.backbone = timm.create_model(self.backbone_name, pretrained=self.pretrained, img_size=self.image_size)
        embed_dim = int(self.backbone.embed_dim)
        self.intermediate_proj = nn.ModuleList([nn.Conv2d(embed_dim, self.hidden_dim, kernel_size=1) for _ in self.out_indices])
        self.merge_proj = nn.Conv2d(self.hidden_dim * len(self.out_indices), self.hidden_dim, kernel_size=1)
        self.downsample_blocks = nn.ModuleList()
        for _ in range(max(0, self.num_feature_levels - 1)):
            self.downsample_blocks.append(nn.Sequential(
                nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, self.hidden_dim),
                nn.GELU(),
            ))
        self.position_encoding = PositionEmbeddingSine(num_pos_feats=self.hidden_dim // 2)
        self.level_embed = nn.Parameter(torch.randn(self.num_feature_levels, self.hidden_dim))

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer('pixel_mean', mean, persistent=False)
        self.register_buffer('pixel_std', std, persistent=False)

    def _normalize_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        elif images.shape[1] != 3:
            raise ValueError('DINOv2 backbone expects 1 or 3 input channels')
        if self.normalize_mode == 'imagenet':
            return (images - self.pixel_mean) / self.pixel_std
        return images

    def forward(self, images: torch.Tensor) -> List[torch.Tensor]:
        images = self._normalize_images(images)
        intermediates = self.backbone.forward_intermediates(images, indices=self.out_indices, norm=False, intermediates_only=True)
        projected = [proj(feat) for proj, feat in zip(self.intermediate_proj, intermediates)]
        base = self.merge_proj(torch.cat(projected, dim=1))
        feats = [base]
        while len(feats) < self.num_feature_levels:
            feats.append(self.downsample_blocks[len(feats) - 1](feats[-1]))
        return feats[: self.num_feature_levels]


class MultiScaleDeformableAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_levels: int, num_points: int, offset_scale: float) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError('hidden_dim must be divisible by num_heads')
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = hidden_dim // num_heads
        self.offset_scale = offset_scale

        self.sampling_offsets = nn.Linear(hidden_dim, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(hidden_dim, num_heads * num_levels * num_points)
        self.value_proj = nn.ModuleList([nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1) for _ in range(num_levels)])
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query: torch.Tensor, reference_points: torch.Tensor, multi_scale_feats: Sequence[torch.Tensor]) -> torch.Tensor:
        batch_size, num_queries, _ = query.shape
        offsets = self.sampling_offsets(query).view(batch_size, num_queries, self.num_heads, self.num_levels, self.num_points, 2)
        offsets = torch.tanh(offsets)
        attn_weights = self.attention_weights(query).view(batch_size, num_queries, self.num_heads, self.num_levels, self.num_points)
        attn_weights = F.softmax(attn_weights.view(batch_size, num_queries, self.num_heads, -1), dim=-1)
        attn_weights = attn_weights.view(batch_size, num_queries, self.num_heads, self.num_levels, self.num_points)

        outputs = []
        for level_idx, feat in enumerate(multi_scale_feats):
            bsz, channels, height, width = feat.shape
            value = self.value_proj[level_idx](feat).view(bsz, self.num_heads, self.head_dim, height, width)
            value = value.reshape(bsz * self.num_heads, self.head_dim, height, width)

            scale = self.offset_scale * float(2 ** level_idx)
            normalizer = torch.tensor([max(width - 1, 1), max(height - 1, 1)], device=query.device, dtype=query.dtype).view(1, 1, 1, 1, 2)
            sampling_locations = reference_points[:, :, None, None, :] + (offsets[:, :, :, level_idx] * scale) / normalizer
            grid = sampling_locations.permute(0, 2, 1, 3, 4).reshape(bsz * self.num_heads, num_queries, self.num_points, 2)
            grid = grid * 2.0 - 1.0
            sampled = F.grid_sample(value, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            sampled = sampled.view(bsz, self.num_heads, self.head_dim, num_queries, self.num_points)
            sampled = sampled.permute(0, 3, 1, 4, 2)
            level_weights = attn_weights[:, :, :, level_idx].unsqueeze(-1)
            outputs.append((sampled * level_weights).sum(dim=3))

        output = torch.stack(outputs, dim=0).sum(dim=0)
        output = output.reshape(batch_size, num_queries, self.hidden_dim)
        return self.output_proj(output)


class DeformableDecoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_levels: int, num_points: int, ffn_dim: int, dropout: float, offset_scale: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = MultiScaleDeformableAttention(hidden_dim, num_heads, num_levels, num_points, offset_scale)
        self.linear1 = nn.Linear(hidden_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        query_pos: torch.Tensor,
        reference_points: torch.Tensor,
        multi_scale_feats: Sequence[torch.Tensor],
        self_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = tgt + query_pos
        attn = self.self_attn(q, q, tgt, attn_mask=self_attn_mask, need_weights=False)[0]
        tgt = self.norm1(tgt + self.dropout(attn))
        cross = self.cross_attn(tgt + query_pos, reference_points, multi_scale_feats)
        tgt = self.norm2(tgt + self.dropout(cross))
        ffn = self.linear2(self.dropout(F.gelu(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout(ffn))
        return tgt


class ParametricDETR(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.config = config
        model_cfg = config['model']
        self.hidden_dim = int(model_cfg['hidden_dim'])
        self.num_queries = int(model_cfg['num_queries'])
        self.num_feature_levels = int(model_cfg.get('num_feature_levels', 4))
        self.num_dn_groups = int(model_cfg.get('dn_num_groups', 0))
        self.dn_noise_scale = float(model_cfg.get('dn_noise_scale', 0.04))
        self.target_degree = int(model_cfg.get('target_degree', config['data'].get('target_degree', 5)))
        self.num_control_points = self.target_degree + 1
        self.curve_dim = self.num_control_points * 2
        self.num_encoder_layers = int(model_cfg.get('num_encoder_layers', 2))
        self.num_decoder_layers = int(model_cfg.get('num_decoder_layers', 6))
        self.num_heads = int(model_cfg.get('nheads', 8))
        self.ffn_dim = int(model_cfg.get('dim_feedforward', self.hidden_dim * 4))
        self.dropout = float(model_cfg.get('dropout', 0.0))
        self.num_sampling_points = int(model_cfg.get('num_sampling_points', 4))
        self.deformable_offset_scale = float(model_cfg.get('deformable_offset_scale', 2.0))
        self.object_bias = float(model_cfg.get('object_bias', -2.0))
        self.no_object_bias = float(model_cfg.get('no_object_bias', 2.0))
        self.query_source = str(model_cfg.get('query_source', 'two_stage'))
        self.curve_parameterization = str(model_cfg.get('curve_parameterization', 'endpoint_offsets'))
        self.curve_delta_scale = float(model_cfg.get('curve_delta_scale', 0.35))
        self.interior_delta_ratio = float(model_cfg.get('interior_delta_ratio', 0.35))
        self.group_proposal_pool_factor = float(model_cfg.get('group_proposal_pool_factor', 1.0))
        self.group_proposal_split_strategy = str(model_cfg.get('group_proposal_split_strategy', 'random_aux_chunk'))
        self.group_addback_enabled = (
            float(config['loss'].get('one_to_many_weight', 0.0)) > 0.0
            or float(config['loss'].get('topk_positive_weight', 0.0)) > 0.0
        )
        self.group_detr_num_groups = max(
            1,
            int(config['loss'].get('group_detr_num_groups', 1)) if self.group_addback_enabled else 1,
        )
        if self.query_source not in {'two_stage', 'learned'}:
            raise ValueError(f'Unsupported query_source: {self.query_source}')
        if self.curve_parameterization not in {'endpoint_offsets', 'full_offsets', 'anchor_delta'}:
            raise ValueError(f'Unsupported curve_parameterization: {self.curve_parameterization}')
        if self.group_proposal_split_strategy not in {'score_chunk', 'random_aux_chunk'}:
            raise ValueError(f'Unsupported group_proposal_split_strategy: {self.group_proposal_split_strategy}')

        self.backbone = DINOv2PyramidBackbone(config)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.num_heads,
                dim_feedforward=self.ffn_dim,
                dropout=self.dropout,
                batch_first=True,
                activation='gelu',
            ),
            num_layers=self.num_encoder_layers,
        )
        self.decoder_layers = nn.ModuleList([
            DeformableDecoderLayer(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                num_levels=self.num_feature_levels,
                num_points=self.num_sampling_points,
                ffn_dim=self.ffn_dim,
                dropout=self.dropout,
                offset_scale=self.deformable_offset_scale,
            )
            for _ in range(self.num_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(self.hidden_dim)

        self.curve_anchor_embed = MLP(self.curve_dim, self.hidden_dim, self.hidden_dim, num_layers=2)
        self.query_content_embed = nn.Embedding(self.num_queries * self.group_detr_num_groups, self.hidden_dim)
        self.learned_query_curve_anchors = nn.Embedding(self.num_queries * self.group_detr_num_groups, self.curve_dim)
        self.query_fuse = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.proposal_score_head = nn.Linear(self.hidden_dim, 1)
        self.proposal_ref_head = MLP(self.hidden_dim, self.hidden_dim, 2, num_layers=3)
        self.proposal_curve_head = MLP(self.hidden_dim, self.hidden_dim, self.curve_dim, num_layers=3)
        self.proposal_memory_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.curve_to_ref_head = MLP(self.curve_dim, self.hidden_dim, 2, num_layers=3)

        self.dn_content_embed = nn.Embedding(1, self.hidden_dim)

        self.class_head = nn.Linear(self.hidden_dim, 2)
        self.curve_head = MLP(self.hidden_dim, self.hidden_dim, self.curve_dim, num_layers=3)
        self.curve_anchor_heads = nn.ModuleList([MLP(self.hidden_dim, self.hidden_dim, self.curve_dim, num_layers=3) for _ in range(self.num_decoder_layers)])

        nn.init.constant_(self.class_head.bias[0], self.object_bias)
        nn.init.constant_(self.class_head.bias[1], self.no_object_bias)
        with torch.no_grad():
            side = int(torch.ceil(torch.sqrt(torch.tensor(float(self.num_queries)))).item())
            ys = torch.linspace(0.05, 0.95, side)
            xs = torch.linspace(0.05, 0.95, side)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
            refs = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)[: self.num_queries].clamp(1e-4, 1.0 - 1e-4)
            init_curve = refs.unsqueeze(1).expand(-1, self.num_control_points, -1).reshape(self.num_queries, self.curve_dim)
            repeated_init_curve = init_curve.repeat(self.group_detr_num_groups, 1)
            self.learned_query_curve_anchors.weight.copy_(inverse_sigmoid(repeated_init_curve))

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def _active_group_count(self, targets: Optional[List[dict]]) -> int:
        del targets
        if not self.training or self.group_detr_num_groups <= 1:
            return 1
        return self.group_detr_num_groups

    def _flatten_multi_scale(self, feats: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        tokens = []
        coords = []
        memory_feats = []
        for level_idx, feat in enumerate(feats):
            feat = feat + self.backbone.position_encoding(feat) + self.backbone.level_embed[level_idx].view(1, -1, 1, 1)
            memory_feats.append(feat)
            bsz, _, height, width = feat.shape
            tokens.append(feat.flatten(2).transpose(1, 2))
            ys = torch.linspace(0.0, 1.0, height, device=feat.device, dtype=feat.dtype)
            xs = torch.linspace(0.0, 1.0, width, device=feat.device, dtype=feat.dtype)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
            coords.append(torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1))
        memory = torch.cat(tokens, dim=1)
        memory_coords = torch.cat(coords, dim=0)
        return memory, memory_feats, memory_coords

    def _split_memory_to_levels(self, memory: torch.Tensor, feats: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        outputs = []
        start = 0
        for feat in feats:
            bsz, channels, height, width = feat.shape
            length = height * width
            level_tokens = memory[:, start:start + length]
            outputs.append(level_tokens.transpose(1, 2).reshape(bsz, channels, height, width))
            start += length
        return outputs

    def _curve_anchor_to_attn_ref(self, curve_anchor: torch.Tensor) -> torch.Tensor:
        mins = curve_anchor.amin(dim=2)
        maxs = curve_anchor.amax(dim=2)
        center = ((mins + maxs) * 0.5).clamp(1e-4, 1.0 - 1e-4)
        ref_delta = self.curve_to_ref_head(curve_anchor.reshape(curve_anchor.shape[0], curve_anchor.shape[1], self.curve_dim))
        return torch.sigmoid(inverse_sigmoid(center) + ref_delta)

    def _build_main_queries(
        self,
        memory: torch.Tensor,
        memory_coords: torch.Tensor,
        batch_size: int,
        targets: Optional[List[dict]],
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if self.query_source == 'learned':
            return self._build_learned_queries(batch_size, targets)

        proposal_logits = self.proposal_score_head(memory).squeeze(-1)
        proposal_ref = torch.sigmoid(inverse_sigmoid(memory_coords.unsqueeze(0).expand(batch_size, -1, -1)) + self.proposal_ref_head(memory))
        proposal_curve_base = proposal_ref.unsqueeze(2).expand(-1, -1, self.num_control_points, -1).reshape(batch_size, -1, self.curve_dim)
        proposal_curve = torch.sigmoid(inverse_sigmoid(proposal_curve_base) + self.proposal_curve_head(memory))
        group_count = self._active_group_count(targets)
        pool_target = max(self.num_queries, int(round(self.group_proposal_pool_factor * group_count * self.num_queries)))
        pool_k = min(pool_target, proposal_logits.shape[1])
        topk_idx = proposal_logits.topk(pool_k, dim=1).indices
        gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        proposal_memory_pool = torch.gather(memory, 1, gather_idx)
        proposal_anchor_pool = torch.gather(proposal_curve, 1, topk_idx.unsqueeze(-1).expand(-1, -1, self.curve_dim))
        proposal_anchor_pool = proposal_anchor_pool.reshape(batch_size, pool_k, self.num_control_points, 2)

        if self.group_proposal_split_strategy == 'random_aux_chunk' and group_count > 1 and pool_k > self.num_queries:
            remain_start = min(self.num_queries, pool_k)
            remain_len = max(0, pool_k - remain_start)
            if remain_len > 0:
                randomized_memory_tail = []
                randomized_anchor_tail = []
                for batch_idx in range(batch_size):
                    perm = torch.randperm(remain_len, device=memory.device)
                    randomized_memory_tail.append(proposal_memory_pool[batch_idx, remain_start:][perm])
                    randomized_anchor_tail.append(proposal_anchor_pool[batch_idx, remain_start:][perm])
                proposal_memory_pool = torch.cat(
                    [proposal_memory_pool[:, :remain_start], torch.stack(randomized_memory_tail, dim=0)],
                    dim=1,
                )
                proposal_anchor_pool = torch.cat(
                    [proposal_anchor_pool[:, :remain_start], torch.stack(randomized_anchor_tail, dim=0)],
                    dim=1,
                )

        group_contents = []
        group_anchors = []
        for group_idx in range(group_count):
            start = group_idx * self.num_queries
            if self.group_proposal_split_strategy == 'score_chunk':
                chunk_start = min(group_idx * self.num_queries, pool_k)
                chunk_end = min(chunk_start + self.num_queries, pool_k)
            elif self.group_proposal_split_strategy == 'random_aux_chunk':
                if group_idx == 0:
                    chunk_start = 0
                    chunk_end = min(self.num_queries, pool_k)
                else:
                    chunk_start = min(self.num_queries + (group_idx - 1) * self.num_queries, pool_k)
                    chunk_end = min(chunk_start + self.num_queries, pool_k)
            else:
                raise ValueError(f'Unsupported group_proposal_split_strategy: {self.group_proposal_split_strategy}')
            chunk_len = max(0, chunk_end - chunk_start)

            learned = self.query_content_embed.weight[start:start + chunk_len].unsqueeze(0).expand(batch_size, -1, -1)
            if chunk_len > 0:
                proposal_memory = proposal_memory_pool[:, chunk_start:chunk_end]
                query_content = self.query_fuse(torch.cat([self.proposal_memory_proj(proposal_memory), learned], dim=-1))
                query_anchor = proposal_anchor_pool[:, chunk_start:chunk_end]
            else:
                query_content = memory.new_zeros((batch_size, 0, self.hidden_dim))
                query_anchor = proposal_anchor_pool[:, :0]
            if chunk_len < self.num_queries:
                remain = self.num_queries - chunk_len
                extra_content = self.query_content_embed.weight[start + chunk_len:start + self.num_queries].unsqueeze(0).expand(batch_size, -1, -1)
                extra_anchor = self.learned_query_curve_anchors.weight[start + chunk_len:start + self.num_queries].sigmoid().unsqueeze(0).expand(batch_size, -1, -1)
                extra_anchor = extra_anchor.reshape(batch_size, remain, self.num_control_points, 2)
                query_content = torch.cat([query_content, extra_content], dim=1)
                query_anchor = torch.cat([query_anchor, extra_anchor], dim=1)
            group_contents.append(query_content)
            group_anchors.append(query_anchor)
        return torch.cat(group_contents, dim=1), torch.cat(group_anchors, dim=1), group_count

    def _build_learned_queries(
        self,
        batch_size: int,
        targets: Optional[List[dict]],
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        group_count = self._active_group_count(targets)
        total_queries = group_count * self.num_queries
        query_content = self.query_content_embed.weight[:total_queries].unsqueeze(0).expand(batch_size, -1, -1)
        query_anchor = self.learned_query_curve_anchors.weight[:total_queries].sigmoid()
        query_anchor = query_anchor.reshape(1, total_queries, self.num_control_points, 2).expand(batch_size, -1, -1, -1)
        return query_content, query_anchor, group_count

    def _build_group_self_attn_mask(self, dn_count: int, group_count: int, device: torch.device) -> Optional[torch.Tensor]:
        if group_count <= 1:
            return None
        total_queries = dn_count + group_count * self.num_queries
        mask = torch.zeros((total_queries, total_queries), dtype=torch.bool, device=device)
        base = dn_count
        for source_group in range(group_count):
            src_start = base + source_group * self.num_queries
            src_end = src_start + self.num_queries
            for target_group in range(group_count):
                if source_group == target_group:
                    continue
                tgt_start = base + target_group * self.num_queries
                tgt_end = tgt_start + self.num_queries
                mask[src_start:src_end, tgt_start:tgt_end] = True
        return mask

    def _build_dn_queries(self, targets: Optional[List[dict]], device: torch.device) -> Tuple[Optional[torch.Tensor], Optional[Dict], Optional[torch.Tensor]]:
        if not self.training or targets is None or self.num_dn_groups <= 0:
            return None, None, None
        dn_curves, dn_targets, dn_mask, dn_labels = build_dn_queries(targets, self.num_dn_groups, self.dn_noise_scale, device)
        if dn_curves.shape[1] == 0:
            return None, None, None
        dn_curve_points = dn_curves.reshape(dn_curves.shape[0], dn_curves.shape[1], self.num_control_points, 2)
        dn_target_points = dn_targets.reshape(dn_targets.shape[0], dn_targets.shape[1], self.num_control_points, 2)
        dn_content = self.dn_content_embed.weight[0].view(1, 1, -1).expand(dn_curves.shape[0], dn_curves.shape[1], -1)
        dn_meta = {
            'mask': dn_mask,
            'labels': dn_labels,
            'curves': dn_target_points,
            'noisy_curves': dn_curve_points,
            'count': dn_curves.shape[1],
        }
        return dn_content, dn_meta, dn_curve_points

    def _decode_to_output(
        self,
        hidden: torch.Tensor,
        curve_anchor: torch.Tensor,
        ref_points: torch.Tensor,
        dn_count: int = 0,
        query_group_count: int = 1,
    ) -> Dict[str, torch.Tensor]:
        logits = self.class_head(hidden)
        curve_raw = self.curve_head(hidden)
        anchor_logits = inverse_sigmoid(curve_anchor.reshape(hidden.shape[0], hidden.shape[1], self.curve_dim))
        curves = torch.sigmoid(anchor_logits + self._curve_delta_logits(curve_raw)).reshape(hidden.shape[0], hidden.shape[1], self.num_control_points, 2)

        main_logits = logits[:, dn_count:]
        main_curves = curves[:, dn_count:]
        main_hidden = hidden[:, dn_count:]
        main_refs = ref_points[:, dn_count:]
        main_anchors = curve_anchor[:, dn_count:]

        grouped_logits = main_logits.reshape(main_logits.shape[0], query_group_count, self.num_queries, -1)
        grouped_curves = main_curves.reshape(main_curves.shape[0], query_group_count, self.num_queries, self.num_control_points, 2)
        grouped_hidden = main_hidden.reshape(main_hidden.shape[0], query_group_count, self.num_queries, -1)
        grouped_refs = main_refs.reshape(main_refs.shape[0], query_group_count, self.num_queries, 2)
        grouped_anchors = main_anchors.reshape(main_anchors.shape[0], query_group_count, self.num_queries, self.num_control_points, 2)

        output = {
            'pred_logits': grouped_logits[:, 0],
            'pred_curves': grouped_curves[:, 0],
            'pred_query_hidden': grouped_hidden[:, 0],
            'pred_ref_points': grouped_refs[:, 0],
            'pred_curve_anchors': grouped_anchors[:, 0],
            'group_pred_logits': grouped_logits,
            'group_pred_curves': grouped_curves,
            'group_pred_query_hidden': grouped_hidden,
            'group_pred_ref_points': grouped_refs,
            'group_pred_curve_anchors': grouped_anchors,
            'pred_group_count': query_group_count,
            'pred_queries_per_group': self.num_queries,
        }
        if dn_count > 0:
            output['dn_pred_logits'] = logits[:, :dn_count]
            output['dn_pred_curves'] = curves[:, :dn_count]
            output['dn_pred_curve_anchors'] = curve_anchor[:, :dn_count]
        return output

    def _curve_delta_logits(self, curve_raw: torch.Tensor) -> torch.Tensor:
        raw_points = curve_raw.reshape(curve_raw.shape[0], curve_raw.shape[1], self.num_control_points, 2)
        if self.curve_parameterization in {'full_offsets', 'anchor_delta'}:
            scale = torch.full_like(raw_points, self.curve_delta_scale)
        else:
            scale = torch.full_like(raw_points, self.curve_delta_scale * self.interior_delta_ratio)
            scale[:, :, 0] = self.curve_delta_scale
            scale[:, :, -1] = self.curve_delta_scale
        return (raw_points * scale).reshape(curve_raw.shape[0], curve_raw.shape[1], self.curve_dim)

    def _selected_aux_layer_indices(self) -> List[int]:
        loss_cfg = self.config['loss']
        aux_weight = float(loss_cfg.get('aux_weight', 0.0))
        if aux_weight <= 0.0:
            return []
        aux_layers = max(0, self.num_decoder_layers - 1)
        if aux_layers == 0:
            return []
        stride = max(1, int(loss_cfg.get('aux_layer_stride', 1)))
        indices = list(range(0, aux_layers, stride))
        if not indices:
            return []
        return indices

    def forward(self, images: torch.Tensor, targets: Optional[List[dict]] = None) -> Dict[str, torch.Tensor]:
        feats = self.backbone(images)
        memory_tokens, feat_maps, memory_coords = self._flatten_multi_scale(feats)
        memory_tokens = self.encoder(memory_tokens)
        encoded_feats = self._split_memory_to_levels(memory_tokens, feat_maps)

        main_queries, main_curve_anchors, query_group_count = self._build_main_queries(memory_tokens, memory_coords, images.shape[0], targets)
        dn_queries, dn_meta, dn_curve_anchors = self._build_dn_queries(targets, images.device)
        if dn_queries is not None:
            queries = torch.cat([dn_queries, main_queries], dim=1)
            curve_anchors = torch.cat([dn_curve_anchors, main_curve_anchors], dim=1)
        else:
            queries = main_queries
            curve_anchors = main_curve_anchors
        self_attn_mask = self._build_group_self_attn_mask(0 if dn_meta is None else dn_meta['count'], query_group_count, images.device)

        hidden = torch.zeros_like(queries)
        decoder_states = []
        decoder_anchors = []
        decoder_refs = []
        current_anchor = curve_anchors
        for layer_idx, layer in enumerate(self.decoder_layers):
            current_ref = self._curve_anchor_to_attn_ref(current_anchor)
            query_pos = self.curve_anchor_embed(current_anchor.reshape(current_anchor.shape[0], current_anchor.shape[1], -1))
            hidden = layer(hidden + queries, query_pos, current_ref, encoded_feats, self_attn_mask=self_attn_mask)
            hidden = self.decoder_norm(hidden)
            decoder_states.append(hidden)
            decoder_anchors.append(current_anchor)
            decoder_refs.append(current_ref)
            anchor_delta = self.curve_anchor_heads[layer_idx](hidden)
            updated_anchor = inverse_sigmoid(current_anchor.reshape(current_anchor.shape[0], current_anchor.shape[1], self.curve_dim)) + anchor_delta
            current_anchor = torch.sigmoid(updated_anchor).reshape(current_anchor.shape[0], current_anchor.shape[1], self.num_control_points, 2)

        dn_count = dn_meta['count'] if dn_meta is not None else 0
        final = self._decode_to_output(decoder_states[-1], decoder_anchors[-1], decoder_refs[-1], dn_count=dn_count, query_group_count=query_group_count)
        aux_indices = self._selected_aux_layer_indices()
        if aux_indices:
            final['aux_outputs'] = [
                self._decode_to_output(
                    decoder_states[layer_idx],
                    decoder_anchors[layer_idx],
                    decoder_refs[layer_idx],
                    dn_count=dn_count,
                    query_group_count=query_group_count,
                )
                for layer_idx in aux_indices
            ]
        else:
            final['aux_outputs'] = []
        if dn_meta is not None:
            final['dn_meta'] = dn_meta
        return final
