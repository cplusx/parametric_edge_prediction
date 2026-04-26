from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.dab_curve_detr import (
    CurveCrossAttention,
    DABCurveDETR,
    DABCurveDecoder,
    DABCurveDecoderLayer,
    MLP,
    gen_sineembed_for_curve_position,
    inverse_sigmoid,
)
from models.curve_coordinates import curve_external_to_internal


class EndpointConditioningBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float, keep_query_pos: bool) -> None:
        super().__init__()
        self.keep_query_pos = keep_query_pos
        self.token_mlp = MLP(d_model, d_model, d_model, 3)
        self.qcontent_proj = nn.Linear(d_model, d_model)
        self.qpos_proj = nn.Linear(d_model, d_model)
        self.kcontent_proj = nn.Linear(d_model, d_model)
        self.kpos_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.cross_attn = CurveCrossAttention(d_model=d_model, nhead=nhead, dropout=dropout)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        tgt: torch.Tensor,
        query_pos: torch.Tensor,
        query_modulated: torch.Tensor,
        endpoint_sine: Optional[torch.Tensor],
        endpoint_padding_mask: Optional[torch.Tensor],
        *,
        is_first: bool,
        force_legacy_cross_attn: bool = False,
    ) -> torch.Tensor:
        if endpoint_sine is None or endpoint_sine.shape[1] == 0:
            return tgt

        endpoint_tokens = self.token_mlp(endpoint_sine)
        safe_padding_mask = endpoint_padding_mask
        if safe_padding_mask is not None:
            endpoint_tokens = endpoint_tokens.masked_fill(safe_padding_mask.unsqueeze(-1), 0.0)
            has_valid_condition = (~safe_padding_mask).any(dim=1)
            if not bool(has_valid_condition.all()):
                safe_padding_mask = safe_padding_mask.clone()
                safe_padding_mask[~has_valid_condition, 0] = False
            valid_scale = has_valid_condition[:, None, None].to(dtype=tgt.dtype)
        else:
            valid_scale = None

        q_content = self.qcontent_proj(tgt)
        if is_first or self.keep_query_pos:
            q_base = q_content + self.qpos_proj(query_pos)
        else:
            q_base = q_content
        q = torch.cat([q_base, query_modulated], dim=-1)
        k = torch.cat([
            self.kcontent_proj(endpoint_tokens),
            self.kpos_proj(endpoint_tokens),
        ], dim=-1)
        v = self.v_proj(endpoint_tokens)
        tgt2 = self.cross_attn(
            q,
            k,
            v,
            key_padding_mask=safe_padding_mask,
            use_sdpa=not force_legacy_cross_attn,
        )
        tgt2 = self.out_proj(tgt2)
        if valid_scale is not None:
            tgt2 = tgt2 * valid_scale
        return self.norm(tgt + self.dropout(tgt2))


class DABConditionedCurveDecoder(DABCurveDecoder):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        num_layers: int,
        curve_dim: int,
        keep_query_pos: bool,
        query_scale_type: str,
        curve_embed_diff_each_layer: bool,
        query_modulator_mode: str,
        force_legacy_cross_attn: bool = False,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_layers=num_layers,
            curve_dim=curve_dim,
            keep_query_pos=keep_query_pos,
            query_scale_type=query_scale_type,
            curve_embed_diff_each_layer=curve_embed_diff_each_layer,
            query_modulator_mode=query_modulator_mode,
            force_legacy_cross_attn=force_legacy_cross_attn,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.layers = nn.ModuleList([
            DABCurveDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                keep_query_pos=keep_query_pos,
            )
            for _ in range(num_layers)
        ])
        self.endpoint_condition_layers = nn.ModuleList([
            EndpointConditioningBlock(
                d_model=d_model,
                nhead=nhead,
                dropout=dropout,
                keep_query_pos=keep_query_pos,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor],
        pos: torch.Tensor,
        ref_curves_unsigmoid: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        condition_points: Optional[torch.Tensor] = None,
        condition_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output = tgt
        reference_curves = ref_curves_unsigmoid.sigmoid()
        ref_curves = [reference_curves]
        intermediate = []
        endpoint_sine = None
        if condition_points is not None and condition_points.numel() > 0:
            endpoint_sine = gen_sineembed_for_curve_position(condition_points, self.d_model)

        for layer_idx, layer in enumerate(self.layers):
            full_curve_sine = gen_sineembed_for_curve_position(reference_curves, self.d_model)
            query_pos = self.ref_curve_head(full_curve_sine)
            if self.query_scale_type != 'fix_elewise':
                if layer_idx == 0:
                    pos_transformation = 1
                else:
                    pos_transformation = self.query_scale(output)
            else:
                pos_transformation = self.query_scale.weight[layer_idx].view(1, 1, -1)
            query_modulated = self.query_modulator(
                decoder_state=output,
                full_curve_sine_embed=full_curve_sine,
                reference_curves=reference_curves,
                pos_transformation=pos_transformation,
                layer_idx=layer_idx,
            )
            is_first = layer_idx == 0
            output = layer(
                tgt=output,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
                query_modulated=query_modulated,
                self_attn_mask=self_attn_mask,
                is_first=is_first,
                force_legacy_cross_attn=self.force_legacy_cross_attn,
            )
            output = self.endpoint_condition_layers[layer_idx](
                tgt=output,
                query_pos=query_pos,
                query_modulated=query_modulated,
                endpoint_sine=endpoint_sine,
                endpoint_padding_mask=condition_padding_mask,
                is_first=is_first,
                force_legacy_cross_attn=self.force_legacy_cross_attn,
            )
            tmp = self._curve_embed_for_layer(layer_idx, output)
            tmp = tmp + inverse_sigmoid(reference_curves)
            new_reference_curves = tmp.sigmoid()
            if layer_idx != self.num_layers - 1:
                ref_curves.append(new_reference_curves)
            reference_curves = new_reference_curves.detach()
            intermediate.append(self.norm(output))

        return torch.stack(intermediate, dim=0), torch.stack(ref_curves, dim=0)


class DABConditionedCurveDETR(DABCurveDETR):
    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.decoder = DABConditionedCurveDecoder(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ffn_dim,
            dropout=self.dropout,
            num_layers=self.num_decoder_layers,
            curve_dim=self.curve_dim,
            keep_query_pos=self.keep_query_pos,
            query_scale_type=self.query_scale_type,
            curve_embed_diff_each_layer=self.curve_embed_diff_each_layer,
            query_modulator_mode=self.query_modulator_mode,
            force_legacy_cross_attn=self.force_legacy_cross_attn,
            gradient_checkpointing=self.gradient_checkpointing,
        )

    def forward(
        self,
        images: torch.Tensor,
        targets=None,
        condition_points: Optional[torch.Tensor] = None,
        condition_padding_mask: Optional[torch.Tensor] = None,
        **unused_kwargs,
    ) -> Dict[str, torch.Tensor]:
        del unused_kwargs
        src, mask, pos = self.backbone(images)
        src = self.input_proj(src)
        pos = pos.to(src.dtype)

        batch_size = src.shape[0]
        src_flat = src.flatten(2).permute(0, 2, 1)
        pos_flat = pos.flatten(2).permute(0, 2, 1)
        mask_flat = mask.flatten(1)

        memory = self.encoder(src_flat, pos=pos_flat, key_padding_mask=mask_flat)
        matching_query_content = self.query_content_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        matching_ref_curves_unsigmoid = self.refpoint_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        dn_query_content, dn_ref_curves_unsigmoid, dn_meta = self._build_dn_queries(targets, images.device)
        if dn_query_content is not None and dn_ref_curves_unsigmoid is not None:
            query_content = torch.cat([dn_query_content, matching_query_content], dim=1)
            ref_curves_unsigmoid = torch.cat([dn_ref_curves_unsigmoid, matching_ref_curves_unsigmoid], dim=1)
        else:
            query_content = matching_query_content
            ref_curves_unsigmoid = matching_ref_curves_unsigmoid
            dn_meta = None
        self_attn_mask = self._build_dn_attention_mask(dn_meta, images.device)

        if condition_points is not None:
            condition_points = condition_points.to(device=images.device, dtype=src.dtype)
            condition_points = curve_external_to_internal(condition_points, self.config)
        if condition_padding_mask is not None:
            condition_padding_mask = condition_padding_mask.to(device=images.device, dtype=torch.bool)

        hs, references = self.decoder(
            tgt=query_content,
            memory=memory,
            memory_key_padding_mask=mask_flat,
            pos=pos_flat,
            ref_curves_unsigmoid=ref_curves_unsigmoid,
            self_attn_mask=self_attn_mask,
            condition_points=condition_points,
            condition_padding_mask=condition_padding_mask,
        )
        outputs_class = self.class_embed(hs)

        reference_before_sigmoid = inverse_sigmoid(references)
        outputs_curves = []
        for layer_idx in range(hs.shape[0]):
            tmp = self.decoder._curve_embed_for_layer(layer_idx, hs[layer_idx])
            tmp = tmp + reference_before_sigmoid[layer_idx]
            outputs_curves.append(tmp.sigmoid())
        outputs_curves = torch.stack(outputs_curves, dim=0)
        dn_count = int(dn_meta['count']) if dn_meta is not None else 0

        out = {
            'pred_logits': outputs_class[-1][:, dn_count:],
            'pred_curves': outputs_curves[-1][:, dn_count:].reshape(batch_size, self.num_queries, self.num_control_points, 2),
            'pred_group_count': 1,
        }
        if dn_count > 0:
            out['dn_pred_logits'] = outputs_class[-1][:, :dn_count]
            out['dn_pred_curves'] = outputs_curves[-1][:, :dn_count].reshape(batch_size, dn_count, self.num_control_points, 2)
            out['dn_meta'] = dn_meta
        aux_outputs = self._set_aux_outputs(outputs_class, outputs_curves, dn_count=dn_count)
        if aux_outputs:
            out['aux_outputs'] = aux_outputs
        return out
