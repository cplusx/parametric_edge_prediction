from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


_BUCKETS: List[Tuple[int, int]] = [
    (0, 100),
    (100, 300),
    (300, 500),
    (500, 700),
    (700, 900),
    (900, 1000),
]


class EndpointFlowMatchingLossComputer:
    def __init__(self, config: Dict) -> None:
        self.config = config
        loss_cfg = config.get('loss', {})
        self.velocity_weight = float(loss_cfg.get('velocity_weight', 1.0))
        self.presence_weight = float(loss_cfg.get('presence_weight', 1.0))

    def __call__(self, outputs: Dict, targets) -> Dict:
        del targets
        valid_mask = outputs['target_presence'].to(dtype=outputs['pred_velocity'].dtype)
        per_point_velocity = (outputs['pred_velocity'] - outputs['target_velocity']).pow(2).mean(dim=-1)
        valid_count = valid_mask.sum(dim=1).clamp_min(1.0)
        per_sample_velocity = (per_point_velocity * valid_mask).sum(dim=1) / valid_count
        velocity_loss = per_sample_velocity.mean()

        per_token_presence = F.binary_cross_entropy_with_logits(
            outputs['presence_logits'],
            outputs['target_presence'],
            reduction='none',
        )
        per_sample_presence = per_token_presence.mean(dim=1)
        presence_loss = per_sample_presence.mean()

        per_sample_total = self.velocity_weight * per_sample_velocity + self.presence_weight * per_sample_presence
        total = per_sample_total.mean()

        log_values = {
            'loss_main': total.detach(),
            'loss_velocity': velocity_loss.detach(),
            'loss_presence': presence_loss.detach(),
        }
        step_indices = outputs['timestep_indices']
        for start, end in _BUCKETS:
            bucket_mask = (step_indices >= start) & (step_indices < end)
            key = f'loss_bucket_{start:03d}_{end:04d}'
            if bucket_mask.any():
                log_values[key] = per_sample_total[bucket_mask].mean().detach()
            else:
                log_values[key] = total.detach() * 0.0

        return {'loss': total, **log_values}
