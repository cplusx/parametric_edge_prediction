from typing import Dict, List

import torch

from models.curve_distances import curve_distance_type_from_config, curve_loss_name_from_config
from models.curve_coordinates import curve_external_to_internal
from models.losses.matched import MatchedCurveLoss, classification_loss_name_from_config
from models.losses.regularizers import DenoisingLoss
from models.matcher import HungarianCurveMatcher


class ParametricEdgeLossComputer:
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.matched_curve_loss = MatchedCurveLoss(config)
        self.denoising_loss = DenoisingLoss(config)
        self.matcher = HungarianCurveMatcher.from_config(config)

    def _add_weighted_term_logs(self, log_values: Dict, loss_cfg: Dict) -> None:
        class_loss_name = classification_loss_name_from_config(self.config)
        curve_loss_name = curve_loss_name_from_config(self.config)
        curve_weight_key = 'emd_weight' if curve_distance_type_from_config(self.config) == 'emd' else 'chamfer_weight'
        term_weight_keys = {
            class_loss_name: 'ce_weight',
            curve_loss_name: curve_weight_key,
            'loss_curve': 'curve_weight',
        }
        for raw_key, term_weight_key in term_weight_keys.items():
            if raw_key not in log_values:
                continue
            value = loss_cfg.get(term_weight_key)
            if value is None and raw_key == 'loss_curve':
                value = loss_cfg.get(curve_weight_key)
            if value is None:
                value = 0.0
            inner_weight = float(value)
            log_values[f'{raw_key}_weighted'] = log_values[raw_key] * inner_weight

    @torch.no_grad()
    def _compute_matching_cost_logs(self, outputs: Dict, targets: List[dict], indices: List) -> Dict[str, torch.Tensor]:
        pred_logits = outputs['pred_logits']
        pred_curves = outputs['pred_curves']
        curve_distance_type = curve_distance_type_from_config(self.config)
        selected: Dict[str, List[torch.Tensor]] = {
            'matching_cost_curve_raw': [],
            'matching_cost_curve': [],
            'matching_cost_edge_prob_raw': [],
            'matching_cost_edge_prob': [],
            'matching_cost_total': [],
        }
        if curve_distance_type == 'emd':
            selected['matching_cost_emd_raw'] = []
            selected['matching_cost_emd'] = []
        else:
            selected['matching_cost_chamfer_raw'] = []
            selected['matching_cost_chamfer'] = []
        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            tgt_curves = targets[batch_idx]['curves'][tgt_idx].to(pred_curves.device)
            tgt_curves = curve_external_to_internal(tgt_curves, self.config)
            matched_logits = pred_logits[batch_idx, src_idx]
            matched_curves = pred_curves[batch_idx, src_idx]
            local_idx = torch.arange(tgt_idx.numel(), device=src_idx.device)
            components = self.matcher.build_cost_components(
                logits=matched_logits,
                curves=matched_curves,
                tgt_curves=tgt_curves,
            )
            selected['matching_cost_curve_raw'].append(components['curve_raw'][local_idx, local_idx])
            selected['matching_cost_curve'].append(components['curve'][local_idx, local_idx])
            selected['matching_cost_edge_prob_raw'].append(components['edge_prob_raw'][local_idx, local_idx])
            selected['matching_cost_edge_prob'].append(components['edge_prob'][local_idx, local_idx])
            selected['matching_cost_total'].append(components['total'][local_idx, local_idx])
            if curve_distance_type == 'emd':
                selected['matching_cost_emd_raw'].append(components['emd_raw'][local_idx, local_idx])
                selected['matching_cost_emd'].append(components['emd'][local_idx, local_idx])
            else:
                selected['matching_cost_chamfer_raw'].append(components['chamfer_raw'][local_idx, local_idx])
                selected['matching_cost_chamfer'].append(components['chamfer'][local_idx, local_idx])

        zero = pred_curves.sum() * 0.0
        out: Dict[str, torch.Tensor] = {}
        for key, chunks in selected.items():
            out[key] = torch.cat(chunks, dim=0).mean().detach() if chunks else zero.detach()
        return out

    def __call__(self, outputs: Dict, targets: List[dict]) -> Dict:
        loss_cfg = self.config['loss']
        aux_reuse_main_matching = bool(loss_cfg.get('aux_reuse_main_matching', False))
        indices = self.matcher(
            logits=outputs['pred_logits'],
            curves=outputs['pred_curves'],
            targets=targets,
        )
        base = self.matched_curve_loss(outputs['pred_curves'], outputs['pred_logits'], targets, indices, outputs)
        total = base['loss_total']
        log_values = {key: value.detach() for key, value in base.items() if key != 'loss_total'}
        self._add_weighted_term_logs(log_values, loss_cfg)
        log_values.update(self._compute_matching_cost_logs(outputs, targets, indices))

        aux_weight = float(loss_cfg.get('aux_weight', 0.0))
        if aux_weight > 0.0:
            for level_idx, aux in enumerate(outputs.get('aux_outputs', [])):
                aux_indices = indices if aux_reuse_main_matching else self.matcher(
                    logits=aux['pred_logits'],
                    curves=aux['pred_curves'],
                    targets=targets,
                )
                aux_losses = self.matched_curve_loss(aux['pred_curves'], aux['pred_logits'], targets, aux_indices, aux)
                total = total + aux_weight * aux_losses['loss_total']
                log_values[f'loss_aux_{level_idx}'] = aux_losses['loss_total'].detach()

        dn = self.denoising_loss(outputs)
        total = total + dn['loss_dn']
        log_values['loss_dn_ce'] = dn['loss_dn_ce'].detach()
        log_values['loss_dn_curve'] = dn['loss_dn_curve'].detach()
        log_values['loss_main'] = base['loss_total'].detach()
        return {'loss': total, 'matching': indices, **log_values}


def build_loss_computer(config: Dict) -> ParametricEdgeLossComputer:
    return ParametricEdgeLossComputer(config)


def compute_losses(outputs: Dict, targets: List[dict], config: Dict) -> Dict:
    return build_loss_computer(config)(outputs, targets)
