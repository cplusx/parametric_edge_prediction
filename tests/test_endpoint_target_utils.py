import numpy as np

from misc_utils.endpoint_target_utils import curves_to_endpoint_clusters_with_incidence


def _cluster_index_near(points, xy, tol=1e-4):
    distances = np.linalg.norm(points - np.asarray(xy, dtype=np.float32)[None, :], axis=1)
    matches = np.flatnonzero(distances <= tol)
    assert matches.size == 1
    return int(matches[0])


def _incident_count(targets, idx):
    offsets = targets['point_curve_offsets']
    return int(offsets[idx + 1] - offsets[idx])


def test_open_split_endpoint_degree_two():
    curves = np.asarray(
        [
            [[0.0, 0.5], [0.25, 0.5], [0.5, 0.5]],
            [[0.5, 0.5], [0.75, 0.5], [1.0, 0.5]],
        ],
        dtype=np.float32,
    )
    targets = curves_to_endpoint_clusters_with_incidence(curves, image_size=(101, 101), dedupe_distance_px=1.0)
    idx = _cluster_index_near(targets['points'], [0.5, 0.5])
    assert int(targets['point_degree'][idx]) == 2
    assert bool(targets['point_is_loop_only'][idx]) is False
    assert _incident_count(targets, idx) == 2


def test_open_junction_endpoint_degree_three():
    curves = np.asarray(
        [
            [[0.0, 0.5], [0.25, 0.5], [0.5, 0.5]],
            [[0.5, 0.5], [0.75, 0.5], [1.0, 0.5]],
            [[0.5, 0.5], [0.5, 0.75], [0.5, 1.0]],
        ],
        dtype=np.float32,
    )
    targets = curves_to_endpoint_clusters_with_incidence(curves, image_size=(101, 101), dedupe_distance_px=1.0)
    idx = _cluster_index_near(targets['points'], [0.5, 0.5])
    assert int(targets['point_degree'][idx]) == 3
    assert bool(targets['point_is_loop_only'][idx]) is False
    assert _incident_count(targets, idx) == 3


def test_closed_curve_loop_only_endpoint():
    curves = np.asarray(
        [
            [[0.5, 0.2], [0.8, 0.5], [0.5, 0.8], [0.2, 0.5], [0.5, 0.2]],
        ],
        dtype=np.float32,
    )
    targets = curves_to_endpoint_clusters_with_incidence(
        curves,
        image_size=(101, 101),
        dedupe_distance_px=1.0,
        closed_curve_threshold_px=1.0,
    )
    idx = _cluster_index_near(targets['points'], [0.5, 0.2])
    assert int(targets['point_degree'][idx]) == 2
    assert bool(targets['point_is_loop_only'][idx]) is True
    assert _incident_count(targets, idx) == 1
