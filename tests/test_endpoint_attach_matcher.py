import torch

from models.endpoint_matcher import HungarianPointMatcher


def _config(enabled=True):
    return {
        'model': {'curve_coord_min': 0.0, 'curve_coord_max': 1.0},
        'loss': {
            'point_cost': 1.0,
            'edge_prob_cost': 0.0,
            'point_attach_matching_enabled': enabled,
            'point_attach_matching_cost': 1.0,
            'point_attach_low_degree_multiplier': 5.0,
            'point_attach_degree_threshold': 3,
            'point_attach_num_curve_samples': 16,
        },
    }


def test_loop_only_matching_prefers_curve_distance_over_point_distance():
    target = {
        'points': torch.tensor([[0.0, 0.0], [0.5, 0.5]], dtype=torch.float32),
        'curves': torch.tensor(
            [
                [[0.8, 0.0], [0.8, 0.5], [0.8, 1.0]],
                [[0.5, 0.5], [0.6, 0.5], [0.7, 0.5]],
            ],
            dtype=torch.float32,
        ),
        'point_degree': torch.tensor([2, 2], dtype=torch.long),
        'point_is_loop_only': torch.tensor([True, False]),
        'point_curve_offsets': torch.tensor([0, 1, 2], dtype=torch.long),
        'point_curve_indices': torch.tensor([0, 1], dtype=torch.long),
    }
    logits = torch.zeros((1, 1, 2), dtype=torch.float32)
    pred_point = torch.tensor([[[0.8, 0.5]]], dtype=torch.float32)

    plain_indices = HungarianPointMatcher.from_config(_config(enabled=False))(logits=logits, points=pred_point, targets=[target])
    attach_indices = HungarianPointMatcher.from_config(_config(enabled=True))(logits=logits, points=pred_point, targets=[target])

    assert int(plain_indices[0][1][0]) == 1
    assert int(attach_indices[0][1][0]) == 0
