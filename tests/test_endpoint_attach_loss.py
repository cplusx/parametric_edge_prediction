import torch

from models.losses.endpoint_matched import MatchedPointLoss


def _config():
    return {
        'model': {'curve_coord_min': 0.0, 'curve_coord_max': 1.0},
        'loss': {
            'class_loss_type': 'focal',
            'focal_alpha': 0.5,
            'focal_gamma': 1.0,
            'endpoint_loss_type': 'attach',
            'ce_weight': 1.0,
            'point_weight': 5.0,
            'point_attach_weight': 1.0,
            'point_attach_low_degree_multiplier': 5.0,
            'point_attach_degree_threshold': 3,
            'point_attach_num_curve_samples': 8,
            'dynamic_class_balance': False,
        },
    }


def _target(points, curves, degree=None, loop_only=None, offsets=None, indices=None):
    point_count = points.shape[0]
    return {
        'points': points,
        'curves': curves,
        'point_degree': degree if degree is not None else torch.ones((point_count,), dtype=torch.long),
        'point_is_loop_only': loop_only if loop_only is not None else torch.zeros((point_count,), dtype=torch.bool),
        'point_curve_offsets': offsets if offsets is not None else torch.zeros((point_count + 1,), dtype=torch.long),
        'point_curve_indices': indices if indices is not None else torch.zeros((0,), dtype=torch.long),
    }


def _run_loss(target):
    pred_points = target['points'].clone().unsqueeze(0).requires_grad_(True)
    pred_logits = torch.tensor([[[6.0, -6.0]] * pred_points.shape[1]], dtype=torch.float32)
    indices = [(torch.arange(pred_points.shape[1]), torch.arange(pred_points.shape[1]))]
    losses = MatchedPointLoss(_config())(pred_points, pred_logits, [target], indices, {'pred_points': pred_points, 'pred_logits': pred_logits})
    assert torch.isfinite(losses['loss_total'])
    losses['loss_total'].backward()
    assert pred_points.grad is not None


def test_attach_loss_no_points_is_finite():
    target = _target(torch.zeros((0, 2)), torch.zeros((0, 3, 2)))
    pred_points = torch.zeros((1, 2, 2), requires_grad=True)
    pred_logits = torch.zeros((1, 2, 2))
    losses = MatchedPointLoss(_config())(pred_points, pred_logits, [target], [(torch.zeros((0,), dtype=torch.long), torch.zeros((0,), dtype=torch.long))], {})
    assert torch.isfinite(losses['loss_total'])


def test_attach_loss_no_curves_is_finite():
    target = _target(torch.tensor([[0.5, 0.5]], dtype=torch.float32), torch.zeros((0, 3, 2)))
    _run_loss(target)


def test_attach_loss_all_loop_only_is_finite():
    target = _target(
        torch.tensor([[0.5, 0.2]], dtype=torch.float32),
        torch.tensor([[[0.5, 0.2], [0.8, 0.5], [0.5, 0.2]]], dtype=torch.float32),
        degree=torch.tensor([2], dtype=torch.long),
        loop_only=torch.tensor([True]),
        offsets=torch.tensor([0, 1], dtype=torch.long),
        indices=torch.tensor([0], dtype=torch.long),
    )
    _run_loss(target)


def test_attach_loss_all_high_degree_is_finite():
    target = _target(
        torch.tensor([[0.5, 0.5]], dtype=torch.float32),
        torch.tensor([[[0.0, 0.5], [0.5, 0.5], [1.0, 0.5]]], dtype=torch.float32),
        degree=torch.tensor([3], dtype=torch.long),
        loop_only=torch.tensor([False]),
        offsets=torch.tensor([0, 1], dtype=torch.long),
        indices=torch.tensor([0], dtype=torch.long),
    )
    _run_loss(target)


def test_attach_loss_all_low_degree_is_finite():
    target = _target(
        torch.tensor([[0.5, 0.5]], dtype=torch.float32),
        torch.tensor([[[0.0, 0.5], [0.5, 0.5], [1.0, 0.5]]], dtype=torch.float32),
        degree=torch.tensor([2], dtype=torch.long),
        loop_only=torch.tensor([False]),
        offsets=torch.tensor([0, 1], dtype=torch.long),
        indices=torch.tensor([0], dtype=torch.long),
    )
    _run_loss(target)


def test_loop_only_point_loss_drops_l1_term():
    config = _config()
    config['loss']['ce_weight'] = 0.0
    target = _target(
        torch.tensor([[0.2, 0.5]], dtype=torch.float32),
        torch.tensor([[[0.2, 0.5], [0.5, 0.5], [0.8, 0.5]]], dtype=torch.float32),
        degree=torch.tensor([2], dtype=torch.long),
        loop_only=torch.tensor([True]),
        offsets=torch.tensor([0, 1], dtype=torch.long),
        indices=torch.tensor([0], dtype=torch.long),
    )
    pred_points = torch.tensor([[[0.5, 0.5]]], dtype=torch.float32, requires_grad=True)
    pred_logits = torch.tensor([[[6.0, -6.0]]], dtype=torch.float32)
    indices = [(torch.tensor([0], dtype=torch.long), torch.tensor([0], dtype=torch.long))]
    losses = MatchedPointLoss(config)(pred_points, pred_logits, [target], indices, {'pred_points': pred_points, 'pred_logits': pred_logits})
    assert losses['loss_point_l1'] > 0.0
    assert losses['loss_point_attach'] < 1e-5
    assert losses['loss_point'] < 1e-5
