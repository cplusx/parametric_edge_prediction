from typing import Dict, List

from models.endpoint_matcher import HungarianPointMatcher
from models.losses.endpoint_matched import MatchedPointLoss
from models.losses.endpoint_regularizers import EndpointDenoisingLoss


class EndpointLossComputer:
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.matched_point_loss = MatchedPointLoss(config)
        self.denoising_loss = EndpointDenoisingLoss(config)
        self.matcher = HungarianPointMatcher.from_config(config)

    def __call__(self, outputs: Dict, targets: List[dict]) -> Dict:
        loss_cfg = self.config['loss']
        aux_reuse_main_matching = bool(loss_cfg.get('aux_reuse_main_matching', False))
        indices = self.matcher(
            logits=outputs['pred_logits'],
            points=outputs['pred_points'],
            targets=targets,
        )
        base = self.matched_point_loss(outputs['pred_points'], outputs['pred_logits'], targets, indices, outputs)
        total = base['loss_total']
        log_values = {key: value.detach() for key, value in base.items() if key != 'loss_total'}

        aux_weight = float(loss_cfg.get('aux_weight', 0.0))
        for level_idx, aux in enumerate(outputs.get('aux_outputs', [])):
            aux_indices = indices if aux_reuse_main_matching else self.matcher(
                logits=aux['pred_logits'],
                points=aux['pred_points'],
                targets=targets,
            )
            aux_losses = self.matched_point_loss(aux['pred_points'], aux['pred_logits'], targets, aux_indices, aux)
            total = total + aux_weight * aux_losses['loss_total']
            log_values[f'loss_aux_{level_idx}'] = aux_losses['loss_total'].detach()

        dn_losses = self.denoising_loss(outputs)
        total = total + dn_losses['loss_dn']
        log_values['loss_main'] = base['loss_total'].detach()
        log_values['loss_dn_ce'] = dn_losses['loss_dn_ce'].detach()
        log_values['loss_dn_point'] = dn_losses['loss_dn_point'].detach()
        log_values['loss_dn_curve'] = dn_losses['loss_dn_point'].detach()
        return {'loss': total, 'matching': indices, **log_values}


def build_endpoint_loss_computer(config: Dict) -> EndpointLossComputer:
    return EndpointLossComputer(config)
