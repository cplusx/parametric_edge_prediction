from typing import Dict, List

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
        term_weight_keys = {
            class_loss_name: 'ce_weight',
            'loss_chamfer': 'chamfer_weight',
        }
        for raw_key, term_weight_key in term_weight_keys.items():
            if raw_key not in log_values:
                continue
            inner_weight = float(loss_cfg.get(term_weight_key, 0.0))
            log_values[f'{raw_key}_weighted'] = log_values[raw_key] * inner_weight

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

        aux_weight = float(loss_cfg.get('aux_weight', 0.5))
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
