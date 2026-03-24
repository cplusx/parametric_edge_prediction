from typing import Dict, List

from models.losses.matched import MatchedCurveLoss
from models.losses.regularizers import DenoisingLoss, DistinctQueryLoss, OneToManyLoss, PositiveObjectLoss
from models.matcher import hungarian_curve_matching


WEIGHTED_TERM_SPECS = {
    'main': {
        'prefix': 'loss',
        'outer_weight_key': None,
        'term_weight_keys': {
            'ce': 'ce_weight',
            'ctrl': 'ctrl_weight',
            'sample': 'sample_weight',
            'endpoint': 'endpoint_weight',
            'curve_dist': 'curve_distance_weight',
        },
    },
    'om': {
        'prefix': 'loss_om',
        'outer_weight_key': 'one_to_many_weight',
        'term_weight_keys': {
            'ce': 'one_to_many_ce_weight',
            'ctrl': 'one_to_many_ctrl_weight',
            'sample': 'one_to_many_sample_weight',
            'endpoint': 'one_to_many_endpoint_weight',
            'curve_dist': 'one_to_many_curve_distance_weight',
        },
    },
}


class ParametricEdgeLossComputer:
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.matched_curve_loss = MatchedCurveLoss(config)
        self.positive_object_loss = PositiveObjectLoss(config)
        self.one_to_many_loss = OneToManyLoss(config, self.matched_curve_loss, self.positive_object_loss)
        self.distinct_query_loss = DistinctQueryLoss(config)
        self.denoising_loss = DenoisingLoss(config)

    def _add_weighted_term_logs(self, log_values: Dict, scope: str, loss_cfg: Dict) -> None:
        spec = WEIGHTED_TERM_SPECS[scope]
        prefix = spec['prefix']
        outer_weight = 1.0
        if spec['outer_weight_key'] is not None:
            outer_weight = float(loss_cfg.get(spec['outer_weight_key'], 0.0))
        term_weight_defaults = spec.get('term_weight_defaults', {})
        for term_name, term_weight_key in spec['term_weight_keys'].items():
            raw_key = f'{prefix}_{term_name}'
            if raw_key not in log_values:
                continue
            inner_weight = float(loss_cfg.get(term_weight_key, term_weight_defaults.get(term_name, 0.0)))
            log_values[f'{raw_key}_weighted'] = log_values[raw_key] * (outer_weight * inner_weight)

    def __call__(self, outputs: Dict, targets: List[dict]) -> Dict:
        loss_cfg = self.config['loss']
        direction_invariant = bool(loss_cfg.get('direction_invariant', True))
        indices = hungarian_curve_matching(
            logits=outputs['pred_logits'],
            curves=outputs['pred_curves'],
            targets=targets,
            control_cost=float(loss_cfg.get('control_cost', 5.0)),
            sample_cost=float(loss_cfg.get('sample_cost', 2.0)),
            curve_distance_cost=float(loss_cfg.get('curve_distance_cost', 1.0)),
            curve_match_point_count=int(loss_cfg.get('curve_match_point_count', 4)),
            num_curve_samples=int(loss_cfg.get('num_curve_samples', 16)),
            direction_invariant=direction_invariant,
            config=self.config,
        )
        base = self.matched_curve_loss(outputs['pred_curves'], outputs['pred_logits'], targets, indices, outputs)
        total = base['loss_total']
        log_values = {key: value.detach() for key, value in base.items() if key != 'loss_total'}
        self._add_weighted_term_logs(log_values, 'main', loss_cfg)

        aux_weight = float(loss_cfg.get('aux_weight', 0.5))
        for level_idx, aux in enumerate(outputs.get('aux_outputs', [])):
            aux_indices = hungarian_curve_matching(
                logits=aux['pred_logits'],
                curves=aux['pred_curves'],
                targets=targets,
                control_cost=float(loss_cfg.get('control_cost', 5.0)),
                sample_cost=float(loss_cfg.get('sample_cost', 2.0)),
                curve_distance_cost=float(loss_cfg.get('curve_distance_cost', 1.0)),
                curve_match_point_count=int(loss_cfg.get('curve_match_point_count', 4)),
                num_curve_samples=int(loss_cfg.get('num_curve_samples', 16)),
                direction_invariant=direction_invariant,
                config=self.config,
            )
            aux_losses = self.matched_curve_loss(aux['pred_curves'], aux['pred_logits'], targets, aux_indices, aux)
            total = total + aux_weight * aux_losses['loss_total']
            log_values[f'loss_aux_{level_idx}'] = aux_losses['loss_total'].detach()

        one_to_many = self.one_to_many_loss(outputs, targets)
        total = total + float(loss_cfg.get('one_to_many_weight', 0.0)) * one_to_many['loss_om_total']
        for key, value in one_to_many.items():
            log_values[key] = value.detach()
        self._add_weighted_term_logs(log_values, 'om', loss_cfg)

        loss_distinct = self.distinct_query_loss(outputs, targets)
        total = total + float(loss_cfg.get('distinct_weight', 0.0)) * loss_distinct
        log_values['loss_distinct'] = loss_distinct.detach()

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
