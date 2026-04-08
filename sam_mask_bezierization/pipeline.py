from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import cKDTree
from skimage.morphology import binary_dilation, disk, thin
from skimage.segmentation import find_boundaries

from bezierization.ablation_api import run_version
from bezierization.bezier_refiner_core import (
    render_piecewise_fits,
    sample_piecewise_bezier,
)

from sam_mask_bezierization.mask_to_edge_methods import (
    masks_to_label_image,
    polish_masks,
    suppress_near_duplicate_masks,
)

if TYPE_CHECKING:
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.sam2_image_predictor import SAM2ImagePredictor


def build_mask_generator(
    model_id: str = "facebook/sam2-hiera-large",
    points_per_side: int = 64,
    stability_score_thresh: float = 0.85,
    crop_n_layers: int = 1,
    points_per_batch: int = 128,
    crop_n_points_downscale_factor: int = 2,
) -> SAM2AutomaticMaskGenerator:
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    predictor = SAM2ImagePredictor.from_pretrained(
        model_id,
        compile_image_encoder=True,
    )
    return SAM2AutomaticMaskGenerator(
        predictor.model,
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=crop_n_layers,
        crop_n_points_downscale_factor=crop_n_points_downscale_factor,
    )


def generate_masks(mask_generator: SAM2AutomaticMaskGenerator, image_np: np.ndarray) -> list[dict]:
    if torch.cuda.is_available():
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
            return mask_generator.generate(image_np)
    with torch.inference_mode():
        return mask_generator.generate(image_np)


def _keep95_mask(seg: np.ndarray, keep_ratio: float = 0.95) -> tuple[np.ndarray, dict]:
    seg = np.asarray(seg, dtype=bool)
    labeled, num_cc = ndimage.label(seg.astype(np.uint8))
    total_area = int(seg.sum())
    if total_area == 0 or num_cc <= 1:
        return seg.copy(), {
            "num_components_before": int(num_cc),
            "num_components_after": int(num_cc),
            "kept_ratio": 1.0 if total_area > 0 else 0.0,
            "kept_sizes": [total_area] if total_area > 0 else [],
        }
    sizes = []
    for cc_idx in range(1, num_cc + 1):
        sizes.append((cc_idx, int((labeled == cc_idx).sum())))
    sizes.sort(key=lambda x: x[1], reverse=True)
    kept = np.zeros_like(seg, dtype=bool)
    kept_sizes: list[int] = []
    cumulative = 0
    for cc_idx, area in sizes:
        kept[labeled == cc_idx] = True
        kept_sizes.append(area)
        cumulative += area
        if cumulative / max(total_area, 1) >= float(keep_ratio):
            break
    _, num_cc_after = ndimage.label(kept.astype(np.uint8))
    return kept, {
        "num_components_before": int(num_cc),
        "num_components_after": int(num_cc_after),
        "kept_ratio": float(int(kept.sum()) / max(total_area, 1)),
        "kept_sizes": kept_sizes,
    }


def apply_keep95(masks: list[dict], keep_ratio: float = 0.95) -> tuple[list[dict], list[dict]]:
    out = []
    stats = []
    for idx, mask in enumerate(masks):
        kept_seg, stat = _keep95_mask(mask["segmentation"], keep_ratio=keep_ratio)
        new_mask = mask.copy()
        new_mask["segmentation"] = kept_seg
        new_mask["area"] = int(kept_seg.sum())
        out.append(new_mask)
        stat["mask_idx"] = int(idx)
        stat["area_after"] = int(kept_seg.sum())
        stats.append(stat)
    return out, stats


def detect_small_bubbles(raw_edge: np.ndarray, max_bubble_area: int = 20) -> np.ndarray:
    raw_edge = np.asarray(raw_edge, dtype=bool)
    background = ~raw_edge
    cc, num_cc = ndimage.label(background.astype(np.uint8))
    selected = np.zeros_like(raw_edge, dtype=bool)
    for cc_idx in range(1, num_cc + 1):
        component = cc == cc_idx
        area = int(component.sum())
        if area == 0 or area > int(max_bubble_area):
            continue
        ys, xs = np.where(component)
        if ys.size == 0:
            continue
        if (
            ys.min() == 0
            or xs.min() == 0
            or ys.max() == raw_edge.shape[0] - 1
            or xs.max() == raw_edge.shape[1] - 1
        ):
            continue
        selected |= component
    return selected


def repair_global_band_thin(raw_edge: np.ndarray, selected_bubbles: np.ndarray, band_radius: int = 1) -> np.ndarray:
    del band_radius
    filled = np.asarray(raw_edge, dtype=bool) | np.asarray(selected_bubbles, dtype=bool)
    return thin(filled).astype(bool)


def prune_tiny_edge_cc(edge: np.ndarray, pixel_limit: int = 15) -> tuple[np.ndarray, np.ndarray, list[dict[str, int]]]:
    edge = np.asarray(edge, dtype=bool)
    labels, num_cc = ndimage.label(edge.astype(np.uint8), structure=np.ones((3, 3), dtype=np.uint8))
    pruned = edge.copy()
    removed = np.zeros_like(edge, dtype=bool)
    records: list[dict[str, int]] = []
    for cc_id in range(1, num_cc + 1):
        cc_mask = labels == cc_id
        size = int(cc_mask.sum())
        if size >= int(pixel_limit):
            continue
        ys, xs = np.where(cc_mask)
        records.append(
            {
                "cc_id": int(cc_id),
                "size": size,
                "y_min": int(ys.min()),
                "x_min": int(xs.min()),
                "y_max": int(ys.max()),
                "x_max": int(xs.max()),
            }
        )
        pruned[cc_mask] = False
        removed |= cc_mask
    return pruned, removed, records


def _path_length(path: dict) -> float:
    pts = sample_piecewise_bezier(path["segments"], samples_per_segment=64)
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def _path_sampled_points(path: dict) -> np.ndarray:
    return sample_piecewise_bezier(path["segments"], samples_per_segment=64)


def _path_endpoints(path: dict) -> tuple[np.ndarray, np.ndarray]:
    pts = _path_sampled_points(path)
    if len(pts) == 0:
        zero = np.zeros(2, dtype=np.float64)
        return zero, zero
    return np.asarray(pts[0], dtype=np.float64), np.asarray(pts[-1], dtype=np.float64)


def _endpoint_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _connector_path(p0: np.ndarray, p1: np.ndarray) -> dict:
    cp = np.vstack([p0, p1]).astype(np.float64)
    seg = {
        "degree": 1,
        "control_points": cp,
        "t_values": np.array([0.0, 1.0], dtype=np.float64),
        "fitted_points": cp.copy(),
        "mean_error": 0.0,
        "max_error": 0.0,
        "points": cp.copy(),
    }
    return {"original_points": cp.copy(), "segments": [seg]}


def _rasterize_connector(shape: tuple[int, int], p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0 = int(round(float(p0[0]))), int(round(float(p0[1])))
    r1, c1 = int(round(float(p1[0]))), int(round(float(p1[1])))
    steps = max(abs(r1 - r0), abs(c1 - c0)) + 1
    rr = np.linspace(r0, r1, num=steps)
    cc = np.linspace(c0, c1, num=steps)
    rr = np.clip(np.round(rr).astype(int), 0, shape[0] - 1)
    cc = np.clip(np.round(cc).astype(int), 0, shape[1] - 1)
    mask[rr, cc] = True
    return mask


def _connector_is_useful(current_raster: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> bool:
    length = float(np.linalg.norm(p1 - p0))
    if length < 1.0:
        return False
    conn = _rasterize_connector(current_raster.shape, p0, p1)
    ys, xs = np.where(conn)
    if ys.size == 0:
        return False
    y0, y1 = max(int(ys.min()) - 2, 0), min(int(ys.max()) + 3, current_raster.shape[0])
    x0, x1 = max(int(xs.min()) - 2, 0), min(int(xs.max()) + 3, current_raster.shape[1])
    before = current_raster[y0:y1, x0:x1]
    after = before | conn[y0:y1, x0:x1]
    struct = np.ones((3, 3), dtype=np.uint8)
    n_before = int(ndimage.label(before.astype(np.uint8), structure=struct)[1])
    n_after = int(ndimage.label(after.astype(np.uint8), structure=struct)[1])
    return n_after < n_before


def _cluster_tiny_paths(paths: list[dict], max_length: float = 10.0, endpoint_tol: float = 3.1) -> list[list[int]]:
    lengths = [_path_length(p) for p in paths]
    tiny = [i for i, l in enumerate(lengths) if l <= max_length]
    if not tiny:
        return []
    endpoints = {i: _path_endpoints(paths[i]) for i in tiny}
    adj: dict[int, set[int]] = {i: set() for i in tiny}
    for i_idx, i in enumerate(tiny):
        s0, e0 = endpoints[i]
        for j in tiny[i_idx + 1:]:
            s1, e1 = endpoints[j]
            d = min(
                _endpoint_distance(s0, s1),
                _endpoint_distance(s0, e1),
                _endpoint_distance(e0, s1),
                _endpoint_distance(e0, e1),
            )
            if d <= endpoint_tol:
                adj[i].add(j)
                adj[j].add(i)
    seen: set[int] = set()
    clusters: list[list[int]] = []
    for i in tiny:
        if i in seen:
            continue
        stack = [i]
        comp: list[int] = []
        seen.add(i)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        clusters.append(sorted(comp))
    return clusters


def _find_large_attachments(
    paths: list[dict],
    cluster: list[int],
    max_length: float = 10.0,
    attach_tol: float = 1.6,
) -> list[np.ndarray]:
    lengths = [_path_length(p) for p in paths]
    cluster_set = set(cluster)
    cluster_endpoints: list[np.ndarray] = []
    for idx in cluster:
        s, e = _path_endpoints(paths[idx])
        cluster_endpoints.extend([s, e])
    attachments: list[np.ndarray] = []
    for idx, path in enumerate(paths):
        if idx in cluster_set or lengths[idx] <= max_length:
            continue
        s, e = _path_endpoints(path)
        for pt in (s, e):
            if any(_endpoint_distance(pt, cep) <= attach_tol for cep in cluster_endpoints):
                attachments.append(pt)
    unique: list[np.ndarray] = []
    for pt in attachments:
        if not any(_endpoint_distance(pt, other) <= 1.0 for other in unique):
            unique.append(pt)
    return unique


def _cluster_replacement_connectors(attachments: list[np.ndarray]) -> list[dict]:
    if len(attachments) <= 1:
        return []
    if len(attachments) == 2:
        return [_connector_path(attachments[0], attachments[1])]
    n = len(attachments)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = _endpoint_distance(attachments[i], attachments[j])
            dist[i, j] = d
            dist[j, i] = d
    mst = minimum_spanning_tree(dist).toarray()
    connectors: list[dict] = []
    for i in range(n):
        for j in range(n):
            if mst[i, j] > 0:
                connectors.append(_connector_path(attachments[i], attachments[j]))
    return connectors


def _cluster_bbox(paths: list[dict], cluster: list[int], margin: int = 4) -> tuple[int, int, int, int]:
    ys: list[int] = []
    xs: list[int] = []
    for idx in cluster:
        pts = _path_sampled_points(paths[idx])
        if len(pts) == 0:
            continue
        ys.extend(np.round(pts[:, 0]).astype(int).tolist())
        xs.extend(np.round(pts[:, 1]).astype(int).tolist())
    return max(min(ys) - margin, 0), max(ys) + margin + 1, max(min(xs) - margin, 0), max(xs) + margin + 1


def _accept_cluster_replacement(
    current_paths: list[dict],
    cluster: list[int],
    replacement: list[dict],
    shape: tuple[int, int],
) -> bool:
    if not replacement:
        return False
    cluster_set = set(cluster)
    kept = [p for i, p in enumerate(current_paths) if i not in cluster_set]
    before_full, _ = render_piecewise_fits(shape, current_paths)
    after_full, _ = render_piecewise_fits(shape, kept + replacement)
    y0, y1, x0, x1 = _cluster_bbox(current_paths, cluster)
    struct = np.ones((3, 3), dtype=np.uint8)
    before = before_full[y0:y1, x0:x1]
    after = after_full[y0:y1, x0:x1]
    n_before = int(ndimage.label(before.astype(np.uint8), structure=struct)[1])
    n_after = int(ndimage.label(after.astype(np.uint8), structure=struct)[1])
    before_len = sum(_path_length(current_paths[i]) for i in cluster)
    after_len = sum(_path_length(p) for p in replacement)
    return n_after <= n_before and after_len <= before_len + 2.5


def _path_with_snapped_endpoint(path: dict, endpoint: str, point: np.ndarray) -> dict:
    new_path = copy.deepcopy(path)
    point = np.asarray(point, dtype=np.float64)
    if endpoint == "start":
        new_path["segments"][0]["control_points"][0] = point
        if "points" in new_path["segments"][0]:
            new_path["segments"][0]["points"][0] = point
        if "fitted_points" in new_path["segments"][0]:
            new_path["segments"][0]["fitted_points"][0] = point
        if "original_points" in new_path and len(new_path["original_points"]) > 0:
            new_path["original_points"][0] = point
    else:
        new_path["segments"][-1]["control_points"][-1] = point
        if "points" in new_path["segments"][-1]:
            new_path["segments"][-1]["points"][-1] = point
        if "fitted_points" in new_path["segments"][-1]:
            new_path["segments"][-1]["fitted_points"][-1] = point
        if "original_points" in new_path and len(new_path["original_points"]) > 0:
            new_path["original_points"][-1] = point
    return new_path


def _tiny_path_attachments(
    paths: list[dict],
    tiny_idx: int,
    max_length: float = 6.0,
    attach_tol: float = 1.6,
) -> list[tuple[int, str, np.ndarray]]:
    tiny = paths[tiny_idx]
    if _path_length(tiny) > max_length:
        return []
    s, e = _path_endpoints(tiny)
    out: list[tuple[int, str, np.ndarray]] = []
    for idx, path in enumerate(paths):
        if idx == tiny_idx:
            continue
        ps, pe = _path_endpoints(path)
        if _endpoint_distance(s, ps) <= attach_tol or _endpoint_distance(e, ps) <= attach_tol:
            out.append((idx, "start", ps))
        if _endpoint_distance(s, pe) <= attach_tol or _endpoint_distance(e, pe) <= attach_tol:
            out.append((idx, "end", pe))
    dedup: list[tuple[int, str, np.ndarray]] = []
    seen: set[tuple[int, str]] = set()
    for idx, kind, pt in out:
        key = (idx, kind)
        if key in seen:
            continue
        dedup.append((idx, kind, pt))
        seen.add(key)
    return dedup


def _collapse_tiny_junction_paths(paths: list[dict]) -> tuple[list[dict], list[int]]:
    current = [copy.deepcopy(p) for p in paths]
    removed: list[int] = []
    lengths = [_path_length(p) for p in current]
    tiny_order = sorted([i for i, l in enumerate(lengths) if l <= 6.0], key=lambda i: lengths[i])
    for tiny_idx in tiny_order:
        if tiny_idx in removed:
            continue
        attachments = _tiny_path_attachments(current, tiny_idx, max_length=6.0, attach_tol=1.6)
        if len(attachments) < 2:
            continue
        tiny_pts = _path_sampled_points(current[tiny_idx])
        if len(tiny_pts) == 0:
            continue
        node = np.vstack([pt for _, _, pt in attachments] + [tiny_pts[len(tiny_pts) // 2]]).mean(axis=0)
        for idx, kind, _ in attachments:
            current[idx] = _path_with_snapped_endpoint(current[idx], kind, node)
        removed.append(tiny_idx)
    final = [p for i, p in enumerate(current) if i not in set(removed)]
    return final, removed


def postprocess_final_paths(paths: list[dict], source_edge: np.ndarray) -> tuple[list[dict], dict[str, Any]]:
    sampled = [_path_sampled_points(p) for p in paths]
    lengths = [_path_length(p) for p in paths]

    remove_idx: list[int] = []
    keep_idx: list[int] = []
    analysis_rows: list[dict[str, Any]] = []
    for i, pts in enumerate(sampled):
        if len(pts) == 0:
            keep_idx.append(i)
            continue
        others = [sampled[j] for j in range(len(sampled)) if j != i and len(sampled[j]) > 0]
        other = np.concatenate(others, axis=0)
        dists, _ = cKDTree(other).query(pts, k=1)
        mean_d = float(dists.mean())
        max_d = float(dists.max())
        is_tiny_redundant = lengths[i] <= 3.0 and mean_d <= 0.8 and max_d <= 1.6
        analysis_rows.append(
            {
                "path_idx": int(i),
                "length": lengths[i],
                "orig_point_count": int(len(paths[i]["original_points"])),
                "mean_dist_to_others": mean_d,
                "max_dist_to_others": max_d,
                "removed": bool(is_tiny_redundant),
            }
        )
        if is_tiny_redundant:
            remove_idx.append(i)
        else:
            keep_idx.append(i)

    kept = [paths[i] for i in keep_idx]
    removed = [paths[i] for i in remove_idx]
    current_raster, _ = render_piecewise_fits(source_edge.shape, kept)

    remaining_endpoints: list[tuple[int, str, np.ndarray]] = []
    for i, path in enumerate(kept):
        pts = _path_sampled_points(path)
        if len(pts) == 0:
            continue
        remaining_endpoints.append((i, "start", np.asarray(pts[0], dtype=np.float64)))
        remaining_endpoints.append((i, "end", np.asarray(pts[-1], dtype=np.float64)))

    connectors: list[dict] = []
    used_pairs: set[tuple[tuple[int, str], tuple[int, str]]] = set()
    for path in removed:
        pts = _path_sampled_points(path)
        if len(pts) == 0:
            continue
        rp0 = np.asarray(pts[0], dtype=np.float64)
        rp1 = np.asarray(pts[-1], dtype=np.float64)
        near0 = sorted(
            [(float(np.linalg.norm(ep - rp0)), idx, kind, ep) for idx, kind, ep in remaining_endpoints],
            key=lambda x: x[0],
        )
        near1 = sorted(
            [(float(np.linalg.norm(ep - rp1)), idx, kind, ep) for idx, kind, ep in remaining_endpoints],
            key=lambda x: x[0],
        )
        a = next((x for x in near0 if x[0] <= 2.5), None)
        b = next((x for x in near1 if x[0] <= 2.5 and (a is None or (x[1], x[2]) != (a[1], a[2]))), None)
        if a is None or b is None:
            continue
        pair = tuple(sorted(((a[1], a[2]), (b[1], b[2]))))
        if pair in used_pairs:
            continue
        p0 = np.asarray(a[3], dtype=np.float64)
        p1 = np.asarray(b[3], dtype=np.float64)
        if not _connector_is_useful(current_raster, p0, p1):
            continue
        used_pairs.add(pair)
        connector = _connector_path(p0, p1)
        connectors.append(connector)
        conn_raster, _ = render_piecewise_fits(source_edge.shape, [connector])
        current_raster |= conn_raster

    final_paths = kept + connectors

    cluster_replacements: list[dict] = []
    clusters = _cluster_tiny_paths(final_paths, max_length=10.0, endpoint_tol=3.1)
    final_remove_idx: set[int] = set()
    for cluster in clusters:
        if len(cluster) <= 1:
            continue
        attachments = _find_large_attachments(final_paths, cluster, max_length=10.0, attach_tol=1.6)
        if len(attachments) < 2:
            continue
        replacement = _cluster_replacement_connectors(attachments)
        if _accept_cluster_replacement(final_paths, cluster, replacement, source_edge.shape):
            final_remove_idx.update(cluster)
            cluster_replacements.extend(replacement)

    if final_remove_idx:
        final_paths = [p for i, p in enumerate(final_paths) if i not in final_remove_idx] + cluster_replacements

    snapped_removed_idx: list[int] = []
    final_paths, snapped_removed_idx = _collapse_tiny_junction_paths(final_paths)

    summary = {
        "original_path_count": len(paths),
        "original_segment_count": int(sum(len(p["segments"]) for p in paths)),
        "removed_path_indices": remove_idx,
        "removed_count": len(remove_idx),
        "connector_count": len(connectors),
        "cluster_removed_final_indices": sorted(final_remove_idx),
        "cluster_replacement_count": len(cluster_replacements),
        "junction_collapsed_final_indices": sorted(snapped_removed_idx),
        "junction_collapsed_count": len(snapped_removed_idx),
        "final_path_count": len(final_paths),
        "final_segment_count": int(sum(len(p["segments"]) for p in final_paths)),
        "path_analysis": analysis_rows,
        "removed_paths": removed,
        "connectors": connectors,
        "cluster_replacements": cluster_replacements,
    }
    return final_paths, summary


def draw_colored_curves_on_image(image_np: np.ndarray, paths: list[dict]) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(image_np)
    cmap = plt.colormaps.get_cmap("gist_rainbow")
    for idx, path in enumerate(paths):
        pts = _path_sampled_points(path)
        if len(pts) == 0:
            continue
        color = cmap(idx / max(len(paths), 1))
        ax.plot(pts[:, 1], pts[:, 0], color=color, linewidth=2)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return arr


def draw_endpoints_control_points_on_image(image_np: np.ndarray, paths: list[dict]) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(image_np)
    cmap = plt.colormaps.get_cmap("gist_rainbow")
    for idx, path in enumerate(paths):
        color = cmap(idx / max(len(paths), 1))
        for seg in path["segments"]:
            ctrl = np.asarray(seg["control_points"], dtype=np.float64)
            curve = sample_piecewise_bezier({"segments": [seg]}["segments"], samples_per_segment=64)
            if len(curve) > 0:
                ax.plot(curve[:, 1], curve[:, 0], color=color, linewidth=1.8, alpha=0.95)
            ax.plot(ctrl[:, 1], ctrl[:, 0], color=color, linestyle="--", linewidth=1.0, alpha=0.65)
            if len(ctrl) > 2:
                ax.scatter(ctrl[1:-1, 1], ctrl[1:-1, 0], s=18, c=[color], edgecolors="white", linewidths=0.6)
            ax.scatter([ctrl[0, 1], ctrl[-1, 1]], [ctrl[0, 0], ctrl[-1, 0]], s=24, c="red", edgecolors="yellow", linewidths=0.8)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return arr


def raster_to_rgb(raster: np.ndarray) -> np.ndarray:
    return np.repeat((np.asarray(raster, dtype=bool).astype(np.uint8) * 255)[..., None], 3, axis=2)


def run_single_image_final_strategy(
    image_path: str | Path,
    output_dir: str | Path,
    mask_generator: SAM2AutomaticMaskGenerator,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image)
    raw_masks = generate_masks(mask_generator, image_np)
    polished_masks = suppress_near_duplicate_masks(polish_masks(raw_masks, gap_threshold=2))
    kept_masks, keep95_stats = apply_keep95(polished_masks, keep_ratio=0.95)

    label_image = masks_to_label_image(kept_masks)
    raw_edge = find_boundaries(label_image, mode="inner", background=0).astype(bool)
    selected_bubbles = detect_small_bubbles(raw_edge)
    thinned_edge = repair_global_band_thin(raw_edge, selected_bubbles, band_radius=1)
    thinned_edge, removed_tiny_cc, removed_tiny_cc_records = prune_tiny_edge_cc(thinned_edge, pixel_limit=15)

    base_bezier_dir = output_dir / "base_bezier"
    base_result = run_version(
        "v5_anchor_consistent",
        output_dir=str(base_bezier_dir),
        image_array=(thinned_edge.astype(np.uint8) * 255),
        compute_raster=True,
        compute_summary=True,
        compute_metrics=True,
        include_debug_artifacts=True,
        max_segment_length=120.0,
        mean_error_threshold=0.75,
        max_error_threshold=2.5,
    )

    base_paths_obj = np.load(base_bezier_dir / "edge_map_bezier_control_points.npy", allow_pickle=True)
    base_paths = [p.item() if hasattr(p, "item") else p for p in base_paths_obj]
    final_paths, post_summary = postprocess_final_paths(base_paths, thinned_edge)
    final_raster, _ = render_piecewise_fits(thinned_edge.shape, final_paths)

    colored_overlay = draw_colored_curves_on_image(image_np, final_paths)
    ctrl_overlay = draw_endpoints_control_points_on_image(image_np, final_paths)
    raster_rgb = raster_to_rgb(final_raster)

    Image.fromarray(image_np).save(output_dir / "input.png")
    Image.fromarray(colored_overlay).save(output_dir / "colored_curves.png")
    Image.fromarray(ctrl_overlay).save(output_dir / "endpoints_control_points.png")
    Image.fromarray(raster_rgb).save(output_dir / "binarized_rasterized.png")
    np.save(output_dir / "final_paths.npy", np.array(final_paths, dtype=object), allow_pickle=True)
    Image.fromarray((thinned_edge.astype(np.uint8) * 255)).save(output_dir / "source_edge.png")
    Image.fromarray((removed_tiny_cc.astype(np.uint8) * 255)).save(output_dir / "removed_tiny_cc.png")

    sample_id = f"{Path(image_path).parent.parent.name}_{Path(image_path).stem}"
    summary = {
        "sample_id": sample_id,
        "image_path": str(image_path),
        "num_raw_masks": int(len(raw_masks)),
        "num_polished_masks": int(len(polished_masks)),
        "num_kept_masks": int(len(kept_masks)),
        "keep95_stats": keep95_stats,
        "base_segment_count": int(base_result["summary"]["segment_count"]),
        "base_path_count": int(base_result["summary"]["path_count"]),
        "final_segment_count": int(post_summary["final_segment_count"]),
        "final_path_count": int(post_summary["final_path_count"]),
        "removed_tiny_cc_count": int(len(removed_tiny_cc_records)),
        "removed_tiny_cc_pixels": int(removed_tiny_cc.sum()),
        "removed_tiny_cc_records": removed_tiny_cc_records,
        "removed_count": int(post_summary["removed_count"]),
        "connector_count": int(post_summary["connector_count"]),
        "cluster_replacement_count": int(post_summary["cluster_replacement_count"]),
        "junction_collapsed_count": int(post_summary["junction_collapsed_count"]),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return {
        "summary": summary,
        "input": image_np,
        "colored_curves": colored_overlay,
        "endpoints_control_points": ctrl_overlay,
        "binarized_rasterized": raster_rgb,
    }
