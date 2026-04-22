import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from models.curve_coordinates import curve_external_to_internal
from models.curve_query_initializers import build_curve_query_initializer
from models.position_encoding import PositionEmbeddingSine


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x / (1.0 - x))


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        layers = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FrozenBatchNorm2d(nn.Module):
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.register_buffer('weight', torch.ones(num_features))
        self.register_buffer('bias', torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-5
        weight = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        running_mean = self.running_mean.reshape(1, -1, 1, 1)
        running_var = self.running_var.reshape(1, -1, 1, 1)
        scale = weight * (running_var + eps).rsqrt()
        bias = bias - running_mean * scale
        return x * scale + bias


def _build_resnet(name: str, pretrained: bool, dilation: bool) -> nn.Module:
    weights_map = {
        'resnet18': 'ResNet18_Weights',
        'resnet34': 'ResNet34_Weights',
        'resnet50': 'ResNet50_Weights',
        'resnet101': 'ResNet101_Weights',
    }
    if name not in weights_map:
        raise ValueError(f'Unsupported DAB backbone: {name}')
    kwargs = {
        'replace_stride_with_dilation': [False, False, dilation],
        'norm_layer': FrozenBatchNorm2d,
    }
    constructor = getattr(torchvision.models, name)
    weights_enum = getattr(torchvision.models, weights_map[name], None)
    try:
        weights = weights_enum.DEFAULT if (pretrained and weights_enum is not None) else None
        return constructor(weights=weights, **kwargs)
    except TypeError:
        return constructor(pretrained=pretrained, **kwargs)


class DABResNetBackbone(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        model_cfg = config['model']
        self.input_channels = int(model_cfg.get('input_channels', 3))
        self.hidden_dim = int(model_cfg['hidden_dim'])
        self.backbone_name = str(model_cfg.get('dab_backbone_name', 'resnet50'))
        self.pretrained = bool(model_cfg.get('dab_backbone_pretrained', True))
        self.dilation = bool(model_cfg.get('dab_backbone_dilation', False))
        self.normalize_mode = str(model_cfg.get('backbone_input_norm', 'imagenet'))

        backbone = _build_resnet(self.backbone_name, self.pretrained, self.dilation)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.num_channels = 512 if self.backbone_name in {'resnet18', 'resnet34'} else 2048

        self.position_encoding = PositionEmbeddingSine(num_pos_feats=self.hidden_dim // 2)

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer('pixel_mean', mean, persistent=False)
        self.register_buffer('pixel_std', std, persistent=False)

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        elif images.shape[1] != 3:
            raise ValueError('DABCurveDETR expects 1 or 3 input channels')
        if self.normalize_mode == 'imagenet':
            images = (images - self.pixel_mean) / self.pixel_std
        return images

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self._normalize(images)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        mask = torch.zeros((x.shape[0], x.shape[-2], x.shape[-1]), dtype=torch.bool, device=x.device)
        pos = self.position_encoding(x).to(x.dtype)
        return x, mask, pos


def gen_sineembed_for_curve_position(pos_tensor: torch.Tensor, d_model: int) -> torch.Tensor:
    scale = 2 * math.pi
    dim_t = torch.arange(d_model // 2, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / max(d_model // 2, 1))
    pieces = []
    for coord_idx in range(pos_tensor.shape[-1]):
        coord = pos_tensor[..., coord_idx] * scale
        coord = coord[..., None] / dim_t
        coord = torch.stack((coord[..., 0::2].sin(), coord[..., 1::2].cos()), dim=-1).flatten(-2)
        pieces.append(coord)
    return torch.cat(pieces, dim=-1)


class DABEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, pos: Optional[torch.Tensor], key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        q = k = src if pos is None else src + pos
        src2 = self.self_attn(q, k, value=src, key_padding_mask=key_padding_mask, need_weights=False)[0]
        src = self.norm1(src + self.dropout(src2))
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = self.norm2(src + self.dropout(src2))
        return src


class DABEncoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, num_layers: int, gradient_checkpointing: bool = False) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            DABEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.norm = nn.LayerNorm(d_model)
        self.gradient_checkpointing = gradient_checkpointing

    def _should_checkpoint(self, *tensors: torch.Tensor) -> bool:
        if not self.gradient_checkpointing or not self.training or not torch.is_grad_enabled():
            return False
        return any(tensor.requires_grad for tensor in tensors if isinstance(tensor, torch.Tensor))

    def forward(self, src: torch.Tensor, pos: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        output = src
        for layer in self.layers:
            pos_scale = self.query_scale(output)
            scaled_pos = pos * pos_scale
            if self._should_checkpoint(output, scaled_pos):
                def layer_forward(current_output: torch.Tensor, current_pos: torch.Tensor) -> torch.Tensor:
                    return layer(current_output, pos=current_pos, key_padding_mask=key_padding_mask)

                output = activation_checkpoint(
                    layer_forward,
                    output,
                    scaled_pos,
                    use_reentrant=False,
                    determinism_check='none',
                )
            else:
                output = layer(output, pos=scaled_pos, key_padding_mask=key_padding_mask)
        return self.norm(output)


class CurveQueryModulator(nn.Module):
    def __init__(self, d_model: int, curve_dim: int, mode: str = 'sine_proj') -> None:
        super().__init__()
        self.d_model = d_model
        # Number of 2D points in the full curve parameterization, including both endpoints.
        self.num_curve_points = curve_dim // 2
        self.full_input_dim = self.num_curve_points * d_model
        self.mode = mode
        self.proj = nn.Linear(self.full_input_dim, d_model)

    def forward(
        self,
        decoder_state: torch.Tensor,
        full_curve_sine_embed: torch.Tensor,
        reference_curves: torch.Tensor,
        pos_transformation: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        del decoder_state, reference_curves, layer_idx
        base = full_curve_sine_embed
        if isinstance(pos_transformation, torch.Tensor):
            if pos_transformation.shape[-1] == 1:
                base = base * pos_transformation
            else:
                base = base * pos_transformation.repeat(1, 1, self.num_curve_points)
        return self.proj(base)


class CurveCrossAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError('d_model must be divisible by nhead')
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)

    def _build_attn_mask(
        self,
        batch_size: int,
        num_queries: int,
        num_tokens: int,
        key_padding_mask: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        additive_mask: Optional[torch.Tensor] = None

        if attn_mask is not None:
            additive_mask = attn_mask.to(device=device)
            if additive_mask.dtype == torch.bool:
                bool_mask = additive_mask
                additive_mask = torch.zeros(bool_mask.shape, device=device, dtype=dtype)
                additive_mask = additive_mask.masked_fill(~bool_mask, float('-inf'))
            else:
                additive_mask = additive_mask.to(dtype=dtype)

            if additive_mask.dim() == 2:
                additive_mask = additive_mask.unsqueeze(0).unsqueeze(0)
            elif additive_mask.dim() == 3:
                additive_mask = additive_mask.unsqueeze(1)
            elif additive_mask.dim() != 4:
                raise ValueError('attn_mask must have shape [Q, K], [B, Q, K], or [B, H, Q, K]')

            if additive_mask.shape[-2:] != (num_queries, num_tokens):
                raise ValueError('attn_mask has incompatible query/token dimensions')

            if additive_mask.shape[0] not in {1, batch_size}:
                raise ValueError('attn_mask batch dimension must be 1 or equal to query batch size')

            if additive_mask.shape[1] not in {1, self.nhead}:
                raise ValueError('attn_mask head dimension must be 1 or equal to nhead')

        if key_padding_mask is not None:
            if key_padding_mask.shape != (batch_size, num_tokens):
                raise ValueError('key_padding_mask has incompatible shape')
            padding_bias = torch.zeros((batch_size, 1, 1, num_tokens), device=device, dtype=dtype)
            padding_bias = padding_bias.masked_fill(key_padding_mask[:, None, None, :], float('-inf'))
            additive_mask = padding_bias if additive_mask is None else additive_mask + padding_bias

        return additive_mask

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        use_sdpa: bool = True,
    ) -> torch.Tensor:
        batch_size, num_queries, _ = query.shape
        num_tokens = key.shape[1]
        # MPS + SDPA is unstable here: validation hit a shape mismatch in cross-attention.
        # Fall back to the explicit matmul path on MPS for both train/eval.
        if query.device.type == 'mps':
            use_sdpa = False
        q = query.view(batch_size, num_queries, self.nhead, self.head_dim * 2).transpose(1, 2)
        k = key.view(batch_size, num_tokens, self.nhead, self.head_dim * 2).transpose(1, 2)
        v = value.view(batch_size, num_tokens, self.nhead, self.head_dim).transpose(1, 2)
        merged_mask = self._build_attn_mask(
            batch_size,
            num_queries,
            num_tokens,
            key_padding_mask,
            attn_mask,
            device=query.device,
            dtype=q.dtype,
        )
        if use_sdpa:
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=merged_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.head_dim * 2))
            if merged_mask is not None:
                scores = scores + merged_mask
            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            output = torch.matmul(attn, v)
        output = output.transpose(1, 2).reshape(batch_size, num_queries, self.d_model)
        return self.out_proj(output)


class DABCurveDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, keep_query_pos: bool = False) -> None:
        super().__init__()
        self.keep_query_pos = keep_query_pos

        self.sa_qcontent_proj = nn.Linear(d_model, d_model)
        self.sa_qpos_proj = nn.Linear(d_model, d_model)
        self.sa_kcontent_proj = nn.Linear(d_model, d_model)
        self.sa_kpos_proj = nn.Linear(d_model, d_model)
        self.sa_v_proj = nn.Linear(d_model, d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.cross_attn = CurveCrossAttention(d_model=d_model, nhead=nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor],
        pos: torch.Tensor,
        query_pos: torch.Tensor,
        query_modulated: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor],
        is_first: bool,
        force_legacy_cross_attn: bool = False,
    ) -> torch.Tensor:
        q_content = self.sa_qcontent_proj(tgt)
        q_pos = self.sa_qpos_proj(query_pos)
        k_content = self.sa_kcontent_proj(tgt)
        k_pos = self.sa_kpos_proj(query_pos)
        v = self.sa_v_proj(tgt)
        q = q_content + q_pos
        k = k_content + k_pos
        tgt2 = self.self_attn(q, k, value=v, attn_mask=self_attn_mask, need_weights=False)[0]
        tgt = self.norm1(tgt + self.dropout(tgt2))

        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)
        k_pos = self.ca_kpos_proj(pos)
        if is_first or self.keep_query_pos:
            q_base = q_content + self.ca_qpos_proj(query_pos)
        else:
            q_base = q_content
        q = torch.cat([q_base, query_modulated], dim=-1)
        k = torch.cat([k_content, k_pos], dim=-1)
        tgt2 = self.cross_attn(
            q,
            k,
            v,
            key_padding_mask=memory_key_padding_mask,
            use_sdpa=not force_legacy_cross_attn,
        )
        tgt = self.norm2(tgt + self.dropout(tgt2))

        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout(tgt2))
        return tgt


class DABCurveDecoder(nn.Module):
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
        super().__init__()
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
        self.num_layers = num_layers
        self.curve_dim = curve_dim
        self.d_model = d_model
        self.query_scale_type = query_scale_type
        if query_scale_type == 'cond_elewise':
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        elif query_scale_type == 'cond_scalar':
            self.query_scale = MLP(d_model, d_model, 1, 2)
        elif query_scale_type == 'fix_elewise':
            self.query_scale = nn.Embedding(num_layers, d_model)
        else:
            raise ValueError(f'Unsupported query_scale_type: {query_scale_type}')
        self.ref_curve_head = MLP(curve_dim * (d_model // 2), d_model, d_model, 2)
        self.query_modulator = CurveQueryModulator(d_model=d_model, curve_dim=curve_dim, mode=query_modulator_mode)
        if curve_embed_diff_each_layer:
            self.curve_embed = nn.ModuleList([MLP(d_model, d_model, curve_dim, 3) for _ in range(num_layers)])
        else:
            self.curve_embed = MLP(d_model, d_model, curve_dim, 3)
        self.curve_embed_diff_each_layer = curve_embed_diff_each_layer
        self.norm = nn.LayerNorm(d_model)
        self.force_legacy_cross_attn = force_legacy_cross_attn
        self.gradient_checkpointing = gradient_checkpointing

    def _should_checkpoint(self, *tensors: torch.Tensor) -> bool:
        if not self.gradient_checkpointing or not self.training or not torch.is_grad_enabled():
            return False
        return any(tensor.requires_grad for tensor in tensors if isinstance(tensor, torch.Tensor))

    def _curve_embed_for_layer(self, layer_idx: int, hidden: torch.Tensor) -> torch.Tensor:
        if self.curve_embed_diff_each_layer:
            return self.curve_embed[layer_idx](hidden)
        return self.curve_embed(hidden)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor],
        pos: torch.Tensor,
        ref_curves_unsigmoid: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output = tgt
        reference_curves = ref_curves_unsigmoid.sigmoid()
        ref_curves = [reference_curves]
        intermediate = []

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
            tmp = self._curve_embed_for_layer(layer_idx, output)
            tmp = tmp + inverse_sigmoid(reference_curves)
            new_reference_curves = tmp.sigmoid()
            if layer_idx != self.num_layers - 1:
                ref_curves.append(new_reference_curves)
            reference_curves = new_reference_curves.detach()
            intermediate.append(self.norm(output))

        return torch.stack(intermediate, dim=0), torch.stack(ref_curves, dim=0)


class DABCurveDETR(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.config = config
        model_cfg = config['model']
        self.hidden_dim = int(model_cfg['hidden_dim'])
        self.num_queries = int(model_cfg['num_queries'])
        self.target_degree = int(model_cfg.get('target_degree', config['data'].get('target_degree', 5)))
        self.num_control_points = self.target_degree + 1
        self.curve_dim = self.num_control_points * 2
        self.num_encoder_layers = int(model_cfg.get('num_encoder_layers', 6))
        self.num_decoder_layers = int(model_cfg.get('num_decoder_layers', 6))
        self.num_heads = int(model_cfg.get('nheads', 8))
        self.ffn_dim = int(model_cfg.get('dim_feedforward', self.hidden_dim * 4))
        self.dropout = float(model_cfg.get('dropout', 0.0))
        self.object_bias = float(model_cfg.get('object_bias', -2.0))
        self.no_object_bias = float(model_cfg.get('no_object_bias', 2.0))
        self.keep_query_pos = bool(model_cfg.get('dab_keep_query_pos', False))
        self.query_scale_type = str(model_cfg.get('dab_query_scale_type', 'cond_elewise'))
        self.curve_embed_diff_each_layer = bool(model_cfg.get('dab_curve_embed_diff_each_layer', False))
        self.query_modulator_mode = str(model_cfg.get('dab_query_modulation_mode', 'sine_proj'))
        self.force_legacy_cross_attn = bool(model_cfg.get('dab_force_legacy_cross_attn', False))
        self.gradient_checkpointing = bool(model_cfg.get('gradient_checkpointing', False))
        self.dn_enabled = bool(model_cfg.get('dn_enabled', False))
        self.num_dn_groups = int(model_cfg.get('dn_num_groups', 0))
        self.dn_use_cdn = bool(model_cfg.get('dn_use_cdn', False))
        self.dn_noise_scale = float(model_cfg.get('dn_noise_scale', 0.04))
        self.dn_negative_noise_bias = float(model_cfg.get('dn_negative_noise_bias', 1.0))
        self.dn_use_label_embed = bool(model_cfg.get('dn_use_label_embed', False))
        self.dn_label_noise_ratio = float(model_cfg.get('dn_label_noise_ratio', 0.0))
        self.aux_weight = float(config['loss'].get('aux_weight', 0.0))
        self.aux_layer_stride = max(1, int(config['loss'].get('aux_layer_stride', 1)))

        self.backbone = DABResNetBackbone(config)
        self.input_proj = nn.Conv2d(self.backbone.num_channels, self.hidden_dim, kernel_size=1)
        self.encoder = DABEncoder(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ffn_dim,
            dropout=self.dropout,
            num_layers=self.num_encoder_layers,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        self.decoder = DABCurveDecoder(
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

        self.class_embed = nn.Linear(self.hidden_dim, 2)
        self.query_content_embed = nn.Embedding(self.num_queries, self.hidden_dim)
        self.refpoint_embed = nn.Embedding(self.num_queries, self.curve_dim)
        self.dn_content_embed = nn.Embedding(1, self.hidden_dim)
        self.dn_label_embed = nn.Embedding(2, self.hidden_dim) if self.dn_use_label_embed else None
        self.curve_query_initializer = build_curve_query_initializer(config)

        nn.init.constant_(self.class_embed.bias[0], self.object_bias)
        nn.init.constant_(self.class_embed.bias[1], self.no_object_bias)
        self._init_reference_curves()

    def _init_reference_curves(self) -> None:
        with torch.no_grad():
            init_curve = self.curve_query_initializer.initialize(
                num_queries=self.num_queries,
                num_control_points=self.num_control_points,
                device=self.refpoint_embed.weight.device,
                dtype=self.refpoint_embed.weight.dtype,
            ).reshape(self.num_queries, self.curve_dim)
            self.refpoint_embed.weight.copy_(inverse_sigmoid(init_curve))

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def _selected_aux_layer_indices(self) -> List[int]:
        if self.aux_weight <= 0.0:
            return []
        aux_layers = max(0, self.num_decoder_layers - 1)
        return list(range(0, aux_layers, self.aux_layer_stride))

    def _set_aux_outputs(self, outputs_class: torch.Tensor, outputs_curves: torch.Tensor, dn_count: int) -> List[Dict[str, torch.Tensor]]:
        aux_indices = self._selected_aux_layer_indices()
        return [
            {
                'pred_logits': outputs_class[layer_idx][:, dn_count:],
                'pred_curves': outputs_curves[layer_idx][:, dn_count:].reshape(outputs_curves.shape[1], outputs_curves.shape[2] - dn_count, self.num_control_points, 2),
            }
            for layer_idx in aux_indices
        ]

    def _build_dn_query_content(self, dn_labels: torch.Tensor) -> torch.Tensor:
        base = self.dn_content_embed.weight[0].view(1, 1, -1).expand(dn_labels.shape[0], dn_labels.shape[1], -1)
        if self.dn_label_embed is None:
            return base
        return base + self.dn_label_embed(dn_labels)

    def _curve_noise_extent(self, target_curves_int: torch.Tensor) -> torch.Tensor:
        curve_min = target_curves_int.amin(dim=1, keepdim=True)
        curve_max = target_curves_int.amax(dim=1, keepdim=True)
        curve_extent = (curve_max - curve_min).clamp_min(0.05)
        return curve_extent.expand(-1, self.num_control_points, -1)

    def _sample_dn_noisy_curves(self, target_curves_int: torch.Tensor, *, negative: bool) -> torch.Tensor:
        eps = 1e-4
        # Keep the original DN behavior for non-CDN training so existing
        # denoising configurations remain comparable to earlier runs.
        if not self.dn_use_cdn:
            noisy = target_curves_int + torch.randn_like(target_curves_int) * self.dn_noise_scale
            return noisy.clamp(eps, 1.0 - eps)
        rand_sign = torch.where(
            torch.rand_like(target_curves_int) < 0.5,
            -torch.ones_like(target_curves_int),
            torch.ones_like(target_curves_int),
        )
        rand_part = torch.rand_like(target_curves_int)
        if negative:
            rand_part = rand_part + self.dn_negative_noise_bias
        offset = rand_sign * rand_part * self._curve_noise_extent(target_curves_int) * self.dn_noise_scale
        return (target_curves_int + offset).clamp(eps, 1.0 - eps)

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
        batch_size = len(targets)
        max_targets = max((target['curves'].shape[0] for target in targets), default=0)
        if max_targets <= 0:
            return None, None, None

        positive_pad = max_targets
        group_pad = positive_pad * (2 if self.dn_use_cdn else 1)
        pad_size = group_pad * self.num_dn_groups

        dn_ref_curves = torch.full(
            (batch_size, pad_size, self.num_control_points, 2),
            0.5,
            dtype=torch.float32,
            device=device,
        )
        dn_target_curves = torch.zeros(
            (batch_size, pad_size, self.num_control_points, 2),
            dtype=torch.float32,
            device=device,
        )
        dn_mask = torch.zeros((batch_size, pad_size), dtype=torch.bool, device=device)
        dn_curve_mask = torch.zeros((batch_size, pad_size), dtype=torch.bool, device=device)
        dn_labels = torch.ones((batch_size, pad_size), dtype=torch.long, device=device)

        for batch_idx, target in enumerate(targets):
            target_curves_ext = target['curves'].to(device)
            if target_curves_ext.numel() == 0:
                continue
            target_curves_int = curve_external_to_internal(target_curves_ext, self.config)
            count = target_curves_int.shape[0]
            for group_idx in range(self.num_dn_groups):
                group_start = group_idx * group_pad
                pos_start = group_start
                pos_end = pos_start + count
                positive_noisy = self._sample_dn_noisy_curves(target_curves_int, negative=False)
                dn_ref_curves[batch_idx, pos_start:pos_end] = positive_noisy
                dn_target_curves[batch_idx, pos_start:pos_end] = target_curves_ext
                dn_mask[batch_idx, pos_start:pos_end] = True
                dn_curve_mask[batch_idx, pos_start:pos_end] = True
                dn_labels[batch_idx, pos_start:pos_end] = 0
                if self.dn_use_cdn:
                    neg_start = group_start + positive_pad
                    neg_end = neg_start + count
                    negative_noisy = self._sample_dn_noisy_curves(target_curves_int, negative=True)
                    dn_ref_curves[batch_idx, neg_start:neg_end] = negative_noisy
                    dn_target_curves[batch_idx, neg_start:neg_end] = target_curves_ext
                    dn_mask[batch_idx, neg_start:neg_end] = True
                    dn_labels[batch_idx, neg_start:neg_end] = 1

        dn_input_labels = self._maybe_noisy_dn_input_labels(dn_labels, dn_mask)
        dn_content = self._build_dn_query_content(dn_input_labels)
        dn_meta = {
            'mask': dn_mask,
            'curve_mask': dn_curve_mask,
            'labels': dn_labels,
            'curves': dn_target_curves,
            'count': pad_size,
            'pad_size': pad_size,
            'single_pad': group_pad,
            'positive_pad': positive_pad,
            'num_dn_groups': self.num_dn_groups,
            'use_cdn': self.dn_use_cdn,
        }
        dn_ref_unsigmoid = inverse_sigmoid(dn_ref_curves.reshape(batch_size, pad_size, self.curve_dim))
        return dn_content, dn_ref_unsigmoid, dn_meta

    def _build_dn_attention_mask(self, dn_meta: Optional[Dict[str, torch.Tensor]], device: torch.device) -> Optional[torch.Tensor]:
        if dn_meta is None or int(dn_meta.get('pad_size', 0)) <= 0:
            return None
        pad_size = int(dn_meta['pad_size'])
        single_pad = int(dn_meta['single_pad'])
        num_dn_groups = int(dn_meta['num_dn_groups'])
        total_queries = pad_size + self.num_queries
        attn_mask = torch.zeros((total_queries, total_queries), dtype=torch.bool, device=device)
        # Matching queries cannot attend to denoising queries.
        attn_mask[pad_size:, :pad_size] = True
        # Different denoising groups cannot see each other.
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

        hs, references = self.decoder(
            tgt=query_content,
            memory=memory,
            memory_key_padding_mask=mask_flat,
            pos=pos_flat,
            ref_curves_unsigmoid=ref_curves_unsigmoid,
            self_attn_mask=self_attn_mask,
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
