from typing import Dict, List, Tuple

import torch


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
        self.velocity_in_pixel_space = bool(loss_cfg.get('velocity_in_pixel_space', True))

    def __call__(self, outputs: Dict, targets) -> Dict:
        del targets
        if outputs.get('skip_batch', False):
            zero = outputs['loss_main_proxy']
            log_values = {
                'loss': zero,
                'loss_main': zero.detach(),
                'loss_velocity': zero.detach(),
                'curriculum_cap': outputs['curriculum_cap'].detach(),
                'kept_samples': outputs['kept_samples'].detach(),
                'skipped_samples': outputs['skipped_samples'].detach(),
                'curriculum_direct_accepts': outputs.get('curriculum_direct_accepts', zero).detach(),
                'curriculum_redirected_requests': outputs.get('curriculum_redirected_requests', zero).detach(),
                'curriculum_rejected_candidates': outputs.get('curriculum_rejected_candidates', zero).detach(),
            }
            for start, end in _BUCKETS:
                key = f'loss_bucket_{start:03d}_{end:04d}'
                log_values[key] = zero.detach()
            return log_values
        valid_mask = outputs['valid_mask'].to(dtype=outputs['pred_velocity'].dtype)
        pred_velocity = outputs['pred_velocity']
        target_velocity = outputs['target_velocity']
        if self.velocity_in_pixel_space:
            image_sizes = outputs['image_sizes'].to(device=pred_velocity.device, dtype=pred_velocity.dtype)
            velocity_scale = torch.stack((image_sizes[:, 1], image_sizes[:, 0]), dim=-1).unsqueeze(1) * 0.5
            pred_velocity = pred_velocity * velocity_scale
            target_velocity = target_velocity * velocity_scale
        per_point_velocity = (pred_velocity - target_velocity).pow(2).mean(dim=-1)
        valid_count = valid_mask.sum(dim=1).clamp_min(1.0)
        per_sample_velocity = (per_point_velocity * valid_mask).sum(dim=1) / valid_count
        velocity_loss = per_sample_velocity.mean()

        per_sample_total = self.velocity_weight * per_sample_velocity
        total = per_sample_total.mean()

        log_values = {
            'loss_main': total.detach(),
            'loss_velocity': velocity_loss.detach(),
            'curriculum_cap': outputs['curriculum_cap'].detach(),
            'kept_samples': outputs['kept_samples'].detach(),
            'skipped_samples': outputs['skipped_samples'].detach(),
            'curriculum_direct_accepts': outputs.get('curriculum_direct_accepts', total.detach() * 0.0).detach(),
            'curriculum_redirected_requests': outputs.get('curriculum_redirected_requests', total.detach() * 0.0).detach(),
            'curriculum_rejected_candidates': outputs.get('curriculum_rejected_candidates', total.detach() * 0.0).detach(),
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
