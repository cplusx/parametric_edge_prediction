from typing import Dict, List

from models.losses.matched import MatchedCurveLoss
from models.losses.regularizers import CountLoss, DenoisingLoss, DistinctQueryLoss, OneToManyLoss, PositiveObjectLoss, TopKPositiveLoss
from models.matcher import hungarian_curve_matching


class ParametricEdgeLossComputer:
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.matched_curve_loss = MatchedCurveLoss(config)
        self.positive_object_loss = PositiveObjectLoss(config)
        self.one_to_many_loss = OneToManyLoss(config, self.matched_curve_loss, self.positive_object_loss)
        self.topk_positive_loss = TopKPositiveLoss(config)
        self.count_loss = CountLoss(config)
        self.distinct_query_loss = DistinctQueryLoss(config)
        self.denoising_loss = DenoisingLoss(config)

    def __call__(self, outputs: Dict, targets: List[dict]) -> Dict:
        loss_cfg = self.config['loss']
        indices = hungarian_curve_matching(
            logits=outputs['pred_logits'],
            curves=outputs['pred_curves'],
            targets=targets,
            control_cost=float(loss_cfg.get('control_cost', 5.0)),
            sample_cost=0.0,
            box_cost=float(loss_cfg.get('box_cost', 1.0)),
            giou_cost=float(loss_cfg.get('giou_cost', 1.0)),
            curve_distance_cost=float(loss_cfg.get('curve_distance_cost', 1.0)),
            curve_match_point_count=int(loss_cfg.get('curve_match_point_count', 4)),
            num_curve_samples=int(loss_cfg.get('num_curve_samples', 16)),
            active_counts=outputs.get('pred_active_counts'),
        )
        base = self.matched_curve_loss(outputs['pred_curves'], outputs['pred_logits'], targets, indices, outputs)
        total = base['loss_total']
        log_values = {key: value.detach() for key, value in base.items() if key != 'loss_total'}

        aux_weight = float(loss_cfg.get('aux_weight', 0.5))
        for level_idx, aux in enumerate(outputs.get('aux_outputs', [])):
            aux_indices = hungarian_curve_matching(
                logits=aux['pred_logits'],
                curves=aux['pred_curves'],
                targets=targets,
                control_cost=float(loss_cfg.get('control_cost', 5.0)),
                sample_cost=0.0,
                box_cost=float(loss_cfg.get('box_cost', 1.0)),
                giou_cost=float(loss_cfg.get('giou_cost', 1.0)),
                curve_distance_cost=float(loss_cfg.get('curve_distance_cost', 1.0)),
                curve_match_point_count=int(loss_cfg.get('curve_match_point_count', 4)),
                num_curve_samples=int(loss_cfg.get('num_curve_samples', 16)),
                active_counts=aux.get('pred_active_counts'),
            )
            aux_losses = self.matched_curve_loss(aux['pred_curves'], aux['pred_logits'], targets, aux_indices, aux)
            total = total + aux_weight * aux_losses['loss_total']
            log_values[f'loss_aux_{level_idx}'] = aux_losses['loss_total'].detach()

        one_to_many = self.one_to_many_loss(outputs, targets)
        total = total + float(loss_cfg.get('one_to_many_weight', 0.0)) * one_to_many['loss_om_total']
        for key, value in one_to_many.items():
            log_values[key] = value.detach()

        loss_topk_pos = self.topk_positive_loss(outputs, targets)
        total = total + float(loss_cfg.get('topk_positive_weight', 0.0)) * loss_topk_pos
        log_values['loss_topk_pos'] = loss_topk_pos.detach()

        loss_count = self.count_loss(outputs, targets)
        total = total + float(loss_cfg.get('count_weight', 0.0)) * loss_count
        log_values['loss_count'] = loss_count.detach()

        loss_distinct = self.distinct_query_loss(outputs)
        total = total + float(loss_cfg.get('distinct_weight', 0.0)) * loss_distinct
        log_values['loss_distinct'] = loss_distinct.detach()

        dn = self.denoising_loss(outputs)
        total = total + dn['loss_dn']
        log_values['loss_dn_ce'] = dn['loss_dn_ce'].detach()
        log_values['loss_dn_curve'] = dn['loss_dn_curve'].detach()
        return {'loss': total, 'matching': indices, **log_values}


def build_loss_computer(config: Dict) -> ParametricEdgeLossComputer:
    return ParametricEdgeLossComputer(config)


def compute_losses(outputs: Dict, targets: List[dict], config: Dict) -> Dict:
    return build_loss_computer(config)(outputs, targets)