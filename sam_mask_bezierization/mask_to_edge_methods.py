from __future__ import annotations

from typing import List, Sequence

import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from skimage.draw import line, polygon2mask
from skimage.measure import approximate_polygon, find_contours
from skimage.morphology import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    disk,
    remove_small_objects,
    remove_small_holes,
    skeletonize,
    thin,
)
from skimage.segmentation import find_boundaries


def polish_masks(masks: Sequence[dict], gap_threshold: int = 2) -> List[dict]:
    if not masks:
        return []
    masks = sorted(masks, key=lambda x: x["area"], reverse=True)
    mask_shape = masks[0]["segmentation"].shape
    label_image = np.zeros(mask_shape, dtype=np.int32)
    for i, mask in enumerate(masks):
        label = i + 1
        label_image[np.asarray(mask["segmentation"], dtype=bool)] = label
    unassigned = label_image == 0
    if np.any(unassigned):
        assigned_regions = label_image > 0
        dilated_regions = binary_dilation(
            assigned_regions, disk(max(int(gap_threshold), 1))
        )
        narrow_gaps = dilated_regions & unassigned
        _, indices = distance_transform_edt(~assigned_regions, return_indices=True)
        nearest_i = indices[0][narrow_gaps]
        nearest_j = indices[1][narrow_gaps]
        nearest_labels = label_image[nearest_i, nearest_j]
        label_image[narrow_gaps] = nearest_labels

        assigned_regions = label_image > 0
        closed_union = binary_closing(assigned_regions, footprint=disk(2))
        filled_union = ndimage.binary_fill_holes(closed_union)
        bubble_candidates = filled_union & (~assigned_regions)
        if np.any(bubble_candidates):
            cc, num_cc = ndimage.label(bubble_candidates.astype(np.uint8))
            for cc_idx in range(1, num_cc + 1):
                component = cc == cc_idx
                area = int(component.sum())
                if area == 0 or area > 256:
                    continue
                ys, xs = np.where(component)
                if ys.size == 0:
                    continue
                if (
                    ys.min() == 0
                    or xs.min() == 0
                    or ys.max() == label_image.shape[0] - 1
                    or xs.max() == label_image.shape[1] - 1
                ):
                    continue
                border = binary_dilation(component, footprint=disk(1)) & (~component)
                neighbor_labels = label_image[border]
                neighbor_labels = neighbor_labels[neighbor_labels > 0]
                if neighbor_labels.size == 0:
                    continue
                unique_labels = np.unique(neighbor_labels)
                if unique_labels.size > 6:
                    continue
                fill_label = int(np.bincount(neighbor_labels).argmax())
                label_image[component] = fill_label

    polished = []
    for i, mask in enumerate(masks):
        label = i + 1
        new_seg = label_image == label
        new_mask = mask.copy()
        new_mask["segmentation"] = new_seg
        new_mask["area"] = int(new_seg.sum())
        polished.append(new_mask)
    return polished


def suppress_near_duplicate_masks(
    masks: Sequence[dict],
    boundary_overlap_thresh: float = 0.85,
    boundary_dilation_radius: int = 2,
    min_area_keep: int = 64,
) -> List[dict]:
    if not masks:
        return []
    masks = sorted(masks, key=lambda x: x["area"], reverse=True)
    kept: List[dict] = []
    kept_dilated_boundaries: List[np.ndarray] = []
    footprint = disk(max(int(boundary_dilation_radius), 1))
    for mask in masks:
        seg = np.asarray(mask["segmentation"], dtype=bool)
        area = int(seg.sum())
        if area == 0:
            continue
        if area < int(min_area_keep):
            kept.append(mask)
            boundary = seg & ~binary_erosion(seg)
            kept_dilated_boundaries.append(
                binary_dilation(boundary, footprint=footprint)
            )
            continue
        boundary = seg & ~binary_erosion(seg)
        boundary_pixels = int(boundary.sum())
        if boundary_pixels == 0:
            continue
        is_duplicate = False
        for kept_boundary in kept_dilated_boundaries:
            overlap = int((boundary & kept_boundary).sum())
            if overlap / max(boundary_pixels, 1) >= float(boundary_overlap_thresh):
                is_duplicate = True
                break
        if is_duplicate:
            continue
        kept.append(mask)
        kept_dilated_boundaries.append(binary_dilation(boundary, footprint=footprint))
    return kept


def masks_to_label_image(masks: Sequence[dict]) -> np.ndarray:
    if not masks:
        return np.zeros((0, 0), dtype=np.int32)
    shape = masks[0]["segmentation"].shape
    label_image = np.zeros(shape, dtype=np.int32)
    for i, mask in enumerate(masks):
        label = i + 1
        label_image[np.asarray(mask["segmentation"], dtype=bool)] = label
    return label_image


def boundary_union_from_masks(masks: Sequence[dict], edge_width: int = 1) -> np.ndarray:
    if not masks:
        return np.zeros((0, 0), dtype=bool)
    mask_shape = masks[0]["segmentation"].shape
    edge_map = np.zeros(mask_shape, dtype=bool)
    selem = disk(max(edge_width // 2, 1))
    for mask in masks:
        segmentation = np.asarray(mask["segmentation"], dtype=bool)
        eroded = binary_erosion(segmentation)
        edges = segmentation & ~eroded
        if edge_width > 1:
            edges = binary_dilation(edges, footprint=selem)
        edge_map |= edges
    return edge_map.astype(bool)


def _rasterize_contours(contours: Sequence[np.ndarray], shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    canvas = np.zeros((h, w), dtype=bool)
    for contour in contours:
        contour = np.asarray(contour, dtype=np.float32)
        if contour.ndim != 2 or contour.shape[0] < 2:
            continue
        for p0, p1 in zip(contour[:-1], contour[1:]):
            r0, c0 = p0
            r1, c1 = p1
            rr, cc = line(
                int(np.clip(round(r0), 0, h - 1)),
                int(np.clip(round(c0), 0, w - 1)),
                int(np.clip(round(r1), 0, h - 1)),
                int(np.clip(round(c1), 0, w - 1)),
            )
            canvas[rr, cc] = True
    return canvas


def edge_map_from_masks_morphology(masks: Sequence[dict], edge_width: int = 1) -> np.ndarray:
    if not masks:
        return np.zeros((0, 0), dtype=bool)
    edge_map = boundary_union_from_masks(masks, edge_width=edge_width)
    return thin(edge_map).astype(bool)


def edge_map_from_boundary_band_centerline(
    masks: Sequence[dict],
    edge_width: int = 1,
    join_radius: int = 2,
    hole_area_threshold: int = 48,
) -> np.ndarray:
    if not masks:
        return np.zeros((0, 0), dtype=bool)
    boundary_union = boundary_union_from_masks(masks, edge_width=edge_width)
    if not np.any(boundary_union):
        return boundary_union.astype(bool)
    footprint = disk(max(int(join_radius), 1))
    band = binary_dilation(boundary_union, footprint=footprint)
    band = binary_closing(band, footprint=footprint)
    band = remove_small_holes(band, area_threshold=max(int(hole_area_threshold), 1))
    centerline = skeletonize(band).astype(bool)
    return thin(centerline).astype(bool)


def edge_map_from_local_ribbon_cleanup(
    masks: Sequence[dict],
    edge_width: int = 1,
    join_radius: int = 2,
    hole_area_threshold: int = 48,
    max_component_area: int = 160,
    roi_expand_radius: int = 6,
) -> np.ndarray:
    if not masks:
        return np.zeros((0, 0), dtype=bool)
    boundary_union = boundary_union_from_masks(masks, edge_width=edge_width)
    footprint = disk(max(int(join_radius), 1))
    band = binary_dilation(boundary_union, footprint=footprint)
    band_closed = binary_closing(band, footprint=footprint)
    filled_band = remove_small_holes(
        band_closed,
        area_threshold=max(int(hole_area_threshold), 1),
    )
    hole_pixels = filled_band & (~band_closed)
    cc, num_cc = ndimage.label(hole_pixels.astype(np.uint8))
    refined = boundary_union.copy()
    roi_footprint = disk(max(int(roi_expand_radius), 1))
    for cc_idx in range(1, num_cc + 1):
        component = cc == cc_idx
        area = int(component.sum())
        if area == 0 or area > int(max_component_area):
            continue
        roi = binary_dilation(component, footprint=roi_footprint)
        local_band = filled_band & roi
        local_centerline = skeletonize(local_band).astype(bool)
        refined[roi] = False
        refined |= local_centerline
    return thin(refined).astype(bool)


def edge_map_from_label_boundaries(label_image: np.ndarray) -> np.ndarray:
    if label_image.size == 0:
        return np.zeros((0, 0), dtype=bool)
    edges = find_boundaries(label_image, mode="inner", background=0)
    return thin(edges).astype(bool)


def edge_map_from_mask_contours(masks: Sequence[dict]) -> np.ndarray:
    if not masks:
        return np.zeros((0, 0), dtype=bool)
    shape = masks[0]["segmentation"].shape
    contours: List[np.ndarray] = []
    for mask in masks:
        seg = np.asarray(mask["segmentation"], dtype=np.uint8)
        contours.extend(find_contours(seg, level=0.5))
    return thin(_rasterize_contours(contours, shape)).astype(bool)


def edge_map_from_smoothed_mask_contours(
    masks: Sequence[dict],
    simplify_tolerance: float = 1.0,
) -> np.ndarray:
    if not masks:
        return np.zeros((0, 0), dtype=bool)
    shape = masks[0]["segmentation"].shape
    contours: List[np.ndarray] = []
    for mask in masks:
        seg = np.asarray(mask["segmentation"], dtype=np.uint8)
        for contour in find_contours(seg, level=0.5):
            contour = np.asarray(contour, dtype=np.float32)
            if contour.shape[0] < 6:
                continue
            simplified = approximate_polygon(contour, tolerance=float(simplify_tolerance))
            if simplified.shape[0] < 3:
                simplified = contour
            contours.append(simplified)
    return thin(_rasterize_contours(contours, shape)).astype(bool)


def smooth_masks_with_polygon_simplification(
    masks: Sequence[dict],
    simplify_tolerance: float = 1.0,
) -> List[dict]:
    if not masks:
        return []
    smoothed_masks: List[dict] = []
    for mask in masks:
        seg = np.asarray(mask["segmentation"], dtype=bool)
        contours = find_contours(seg.astype(np.uint8), level=0.5)
        if not contours:
            smoothed_masks.append(mask.copy())
            continue
        contour = max(contours, key=lambda c: c.shape[0])
        contour = np.asarray(contour, dtype=np.float32)
        simplified = approximate_polygon(contour, tolerance=float(simplify_tolerance))
        if simplified.shape[0] < 3:
            simplified = contour
        new_seg = polygon2mask(seg.shape, simplified).astype(bool)
        if new_seg.sum() == 0:
            new_seg = seg
        new_mask = mask.copy()
        new_mask["segmentation"] = new_seg
        new_mask["area"] = int(new_seg.sum())
        smoothed_masks.append(new_mask)
    return smoothed_masks


def _largest_contour(seg: np.ndarray) -> np.ndarray | None:
    contours = find_contours(seg.astype(np.uint8), level=0.5)
    if not contours:
        return None
    contour = max(contours, key=lambda c: c.shape[0])
    contour = np.asarray(contour, dtype=np.float32)
    if contour.shape[0] > 2 and np.linalg.norm(contour[0] - contour[-1]) < 1.5:
        contour = contour[:-1]
    return contour


def _corner_mask_for_closed_contour(
    contour: np.ndarray,
    lookahead: int = 3,
    corner_angle_deg: float = 35.0,
) -> np.ndarray:
    n = int(contour.shape[0])
    mask = np.zeros(n, dtype=bool)
    if n < max(8, 2 * lookahead + 1):
        return mask
    angle_threshold = np.deg2rad(float(corner_angle_deg))
    for idx in range(n):
        p_prev = contour[(idx - lookahead) % n]
        p_curr = contour[idx]
        p_next = contour[(idx + lookahead) % n]
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        norm1 = float(np.linalg.norm(v1))
        norm2 = float(np.linalg.norm(v2))
        if norm1 < 1e-6 or norm2 < 1e-6:
            continue
        cos_theta = float(np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0))
        turn_angle = float(np.arccos(cos_theta))
        if turn_angle >= angle_threshold:
            mask[idx] = True
    return mask


def _smooth_closed_contour_preserve_corners(
    contour: np.ndarray,
    corner_mask: np.ndarray,
    smoothing_iterations: int = 4,
) -> np.ndarray:
    smoothed = contour.astype(np.float32).copy()
    movable = ~corner_mask
    if not np.any(movable):
        return smoothed
    for _ in range(max(int(smoothing_iterations), 1)):
        prev_pts = np.roll(smoothed, 1, axis=0)
        next_pts = np.roll(smoothed, -1, axis=0)
        proposal = (prev_pts + 2.0 * smoothed + next_pts) / 4.0
        smoothed[movable] = proposal[movable]
    return smoothed


def smooth_masks_with_corner_preserving_contour_smoothing(
    masks: Sequence[dict],
    lookahead: int = 3,
    corner_angle_deg: float = 35.0,
    smoothing_iterations: int = 4,
) -> List[dict]:
    if not masks:
        return []
    smoothed_masks: List[dict] = []
    for mask in masks:
        seg = np.asarray(mask["segmentation"], dtype=bool)
        contour = _largest_contour(seg)
        if contour is None or contour.shape[0] < 8:
            smoothed_masks.append(mask.copy())
            continue
        corner_mask = _corner_mask_for_closed_contour(
            contour,
            lookahead=int(lookahead),
            corner_angle_deg=float(corner_angle_deg),
        )
        smoothed_contour = _smooth_closed_contour_preserve_corners(
            contour,
            corner_mask,
            smoothing_iterations=int(smoothing_iterations),
        )
        new_seg = polygon2mask(seg.shape, smoothed_contour).astype(bool)
        if new_seg.sum() == 0:
            new_seg = seg
        new_mask = mask.copy()
        new_mask["segmentation"] = new_seg
        new_mask["area"] = int(new_seg.sum())
        smoothed_masks.append(new_mask)
    return smoothed_masks


def _signed_turn_angles(contour: np.ndarray, lookahead: int = 1) -> np.ndarray:
    n = int(contour.shape[0])
    turns = np.zeros(n, dtype=np.float32)
    if n < max(5, 2 * lookahead + 1):
        return turns
    for idx in range(n):
        p_prev = contour[(idx - lookahead) % n]
        p_curr = contour[idx]
        p_next = contour[(idx + lookahead) % n]
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        norm1 = float(np.linalg.norm(v1))
        norm2 = float(np.linalg.norm(v2))
        if norm1 < 1e-6 or norm2 < 1e-6:
            continue
        cross = float(v1[0] * v2[1] - v1[1] * v2[0])
        dot = float(np.dot(v1, v2))
        turns[idx] = float(np.arctan2(cross, dot))
    return turns


def _dilate_cyclic_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or mask.size == 0:
        return mask
    expanded = mask.copy()
    idx = np.flatnonzero(mask)
    n = int(mask.size)
    for offset in range(1, int(radius) + 1):
        expanded[(idx + offset) % n] = True
        expanded[(idx - offset) % n] = True
    return expanded


def _taubin_smooth_closed_contour(
    contour: np.ndarray,
    iterations: int = 8,
    lambda_value: float = 0.5,
    mu_value: float = -0.53,
) -> np.ndarray:
    pts = contour.astype(np.float32).copy()
    if pts.shape[0] < 5:
        return pts

    def laplacian(x: np.ndarray) -> np.ndarray:
        return 0.5 * (np.roll(x, 1, axis=0) + np.roll(x, -1, axis=0)) - x

    for _ in range(max(int(iterations), 1)):
        pts = pts + float(lambda_value) * laplacian(pts)
        pts = pts + float(mu_value) * laplacian(pts)
    return pts


def _contour_unit_normals(contour: np.ndarray) -> np.ndarray:
    pts = np.asarray(contour, dtype=np.float32)
    tangent = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent_norm = np.maximum(tangent_norm, 1e-6)
    tangent = tangent / tangent_norm
    # Contours are in (row, col); rotate tangent by 90 degrees.
    normals = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    return normals.astype(np.float32)


def _connected_true_runs_cyclic(mask: np.ndarray) -> List[tuple[int, int]]:
    if mask.size == 0 or not np.any(mask):
        return []
    n = int(mask.size)
    runs: List[tuple[int, int]] = []
    visited = np.zeros(n, dtype=bool)
    for start in np.flatnonzero(mask):
        if visited[start]:
            continue
        end = start
        visited[end] = True
        while mask[(end + 1) % n] and not visited[(end + 1) % n]:
            end = (end + 1) % n
            visited[end] = True
        runs.append((int(start), int(end)))
    return runs


def detect_local_burr_mask(
    contour: np.ndarray,
    reference_contour: np.ndarray,
    fine_lookahead: int = 1,
    coarse_lookahead: int = 4,
    alternating_window_radius: int = 4,
    min_sign_changes: int = 3,
    fine_turn_deg: float = 8.0,
    strong_corner_deg: float = 35.0,
    displacement_thresh_px: float = 0.9,
    max_oscillation_ratio: float = 0.6,
    dilate_radius: int = 6,
) -> np.ndarray:
    n = int(contour.shape[0])
    if n < 16:
        return np.zeros(n, dtype=bool)

    fine_turn = _signed_turn_angles(contour, lookahead=int(fine_lookahead))
    coarse_turn = _signed_turn_angles(contour, lookahead=int(coarse_lookahead))
    fine_abs = np.abs(fine_turn)
    fine_thresh = np.deg2rad(float(fine_turn_deg))
    corner_thresh = np.deg2rad(float(strong_corner_deg))
    normals = _contour_unit_normals(reference_contour)
    normal_disp = np.sum((reference_contour - contour) * normals, axis=1)
    disp_abs = np.abs(normal_disp)

    candidate = (fine_abs >= fine_thresh) & (disp_abs >= float(displacement_thresh_px))
    sign = np.sign(normal_disp)
    selection = np.zeros(n, dtype=bool)
    radius = int(alternating_window_radius)
    max_net_turn = np.deg2rad(24.0)
    min_abs_turn = np.deg2rad(22.0)
    min_path_ratio = 1.045
    for idx in range(n):
        if not candidate[idx]:
            continue
        window_idx = (np.arange(idx - radius, idx + radius + 1) % n).astype(int)
        window_sign = sign[window_idx]
        window_sign = window_sign[np.abs(window_sign) > 0]
        if window_sign.size < 4:
            continue
        sign_changes = int(np.sum(window_sign[:-1] * window_sign[1:] < 0))
        window_turn = fine_turn[window_idx]
        oscillation_ratio = float(np.abs(window_turn.sum()) / (np.abs(window_turn).sum() + 1e-6))
        window_pts = contour[window_idx]
        path_len = float(np.linalg.norm(np.diff(window_pts, axis=0), axis=1).sum())
        chord_len = float(np.linalg.norm(window_pts[-1] - window_pts[0]))
        path_ratio = path_len / max(chord_len, 1e-6)
        net_turn = float(np.abs(window_turn.sum()))
        abs_turn = float(np.abs(window_turn).sum())
        oscillatory = sign_changes >= int(min_sign_changes) and oscillation_ratio <= float(max_oscillation_ratio)
        jagged_but_globally_smooth = (
            path_ratio >= min_path_ratio
            and net_turn <= max_net_turn
            and abs_turn >= min_abs_turn
        )
        if not (oscillatory or jagged_but_globally_smooth):
            continue
        if np.abs(coarse_turn[idx]) >= corner_thresh:
            continue
        selection[idx] = True
    return _dilate_cyclic_mask(selection, radius=int(dilate_radius))


def smooth_masks_with_local_taubin_burr_cleanup(
    masks: Sequence[dict],
    fine_lookahead: int = 1,
    coarse_lookahead: int = 4,
    alternating_window_radius: int = 4,
    min_sign_changes: int = 3,
    fine_turn_deg: float = 8.0,
    strong_corner_deg: float = 35.0,
    displacement_thresh_px: float = 0.9,
    max_oscillation_ratio: float = 0.6,
    dilate_radius: int = 6,
    taubin_iterations: int = 8,
    taubin_lambda: float = 0.5,
    taubin_mu: float = -0.53,
) -> tuple[List[dict], List[np.ndarray]]:
    if not masks:
        return [], []

    smoothed_masks: List[dict] = []
    burr_masks: List[np.ndarray] = []
    for mask in masks:
        seg = np.asarray(mask["segmentation"], dtype=bool)
        contour = _largest_contour(seg)
        if contour is None or contour.shape[0] < 16:
            smoothed_masks.append(mask.copy())
            burr_masks.append(np.zeros(seg.shape, dtype=bool))
            continue

        taubin_contour = _taubin_smooth_closed_contour(
            contour,
            iterations=int(taubin_iterations),
            lambda_value=float(taubin_lambda),
            mu_value=float(taubin_mu),
        )
        selected = detect_local_burr_mask(
            contour,
            taubin_contour,
            fine_lookahead=int(fine_lookahead),
            coarse_lookahead=int(coarse_lookahead),
            alternating_window_radius=int(alternating_window_radius),
            min_sign_changes=int(min_sign_changes),
            fine_turn_deg=float(fine_turn_deg),
            strong_corner_deg=float(strong_corner_deg),
            displacement_thresh_px=float(displacement_thresh_px),
            max_oscillation_ratio=float(max_oscillation_ratio),
            dilate_radius=int(dilate_radius),
        )
        if not np.any(selected):
            smoothed_masks.append(mask.copy())
            burr_masks.append(np.zeros(seg.shape, dtype=bool))
            continue
        refined_contour = contour.copy()
        refined_contour[selected] = taubin_contour[selected]
        new_seg = polygon2mask(seg.shape, refined_contour).astype(bool)
        if new_seg.sum() == 0:
            new_seg = seg
        new_mask = mask.copy()
        new_mask["segmentation"] = new_seg
        new_mask["area"] = int(new_seg.sum())
        smoothed_masks.append(new_mask)

        burr_canvas = np.zeros(seg.shape, dtype=bool)
        if np.any(selected):
            selected_runs = _connected_true_runs_cyclic(selected)
            for start, end in selected_runs:
                run_points = [contour[start]]
                idx = start
                while idx != end:
                    idx = (idx + 1) % contour.shape[0]
                    run_points.append(contour[idx])
                run_points = np.asarray(run_points, dtype=np.float32)
                burr_canvas |= _rasterize_contours([run_points], seg.shape)
        burr_masks.append(burr_canvas)
    return smoothed_masks, burr_masks


def edge_map_from_signed_distance_smoothed_masks(
    masks: Sequence[dict],
    gaussian_sigma: float = 1.0,
) -> np.ndarray:
    if not masks:
        return np.zeros((0, 0), dtype=bool)
    shape = masks[0]["segmentation"].shape
    contours: List[np.ndarray] = []
    for mask in masks:
        seg = np.asarray(mask["segmentation"], dtype=bool)
        if not np.any(seg):
            continue
        signed_distance = distance_transform_edt(seg) - distance_transform_edt(~seg)
        smoothed = ndimage.gaussian_filter(signed_distance.astype(np.float32), sigma=float(gaussian_sigma))
        refined_seg = smoothed > 0.0
        if refined_seg.sum() == 0:
            refined_seg = seg
        for contour in find_contours(refined_seg.astype(np.uint8), level=0.5):
            contour = np.asarray(contour, dtype=np.float32)
            if contour.shape[0] >= 6:
                contours.append(contour)
    return thin(_rasterize_contours(contours, shape)).astype(bool)


def edge_map_from_union_contours(masks: Sequence[dict]) -> np.ndarray:
    if not masks:
        return np.zeros((0, 0), dtype=bool)
    union = np.zeros(masks[0]["segmentation"].shape, dtype=bool)
    for mask in masks:
        union |= np.asarray(mask["segmentation"], dtype=bool)
    contours = find_contours(union.astype(np.uint8), level=0.5)
    return thin(_rasterize_contours(contours, union.shape)).astype(bool)


def postprocess_edge(mask: np.ndarray, closing_radius: int = 1, remove_small: int = 0) -> np.ndarray:
    edge = np.asarray(mask, dtype=bool)
    if int(closing_radius) > 0:
        edge = binary_closing(edge, footprint=disk(int(closing_radius)))
    if int(remove_small) > 0:
        edge = remove_small_objects(edge, min_size=int(remove_small), connectivity=2)
    return thin(edge).astype(bool)
