import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology, measure, util
from skimage.morphology import skeletonize, thin
import networkx as nx
from skimage.draw import line
from collections import Counter
from collections import defaultdict
from scipy.spatial import distance_matrix
from PIL import Image

from bezierization.fast_backend import (
    bernstein_basis as fast_bernstein_basis,
    chord_length_parameterize as fast_chord_length_parameterize,
    control_polygon_is_stable as fast_control_polygon_is_stable,
    evaluate_bezier as fast_evaluate_bezier,
    fit_error_metrics as fast_fit_error_metrics,
    path_length as fast_path_length,
    prepare_fit_context as fast_prepare_fit_context,
)

def build_graph(skeleton, connectivity=2):
    """
    Convert a skeletonized image to a NetworkX graph.
    
    Parameters:
    - skeleton: 2D binary numpy array.
    - connectivity: int, 1 for 4-connectivity, 2 for 8-connectivity.
    
    Returns:
    - G: NetworkX graph.
    """
    G = nx.Graph()
    rows, cols = skeleton.shape
    # Define neighbor offsets based on connectivity
    if connectivity == 1:
        neighbors = [(-1,0), (1,0), (0,-1), (0,1)]
    elif connectivity == 2:
        neighbors = [(-1,0), (1,0), (0,-1), (0,1),
                     (-1,-1), (-1,1), (1,-1), (1,1)]
    else:
        raise ValueError("Connectivity must be 1 or 2")
    
    # Add nodes and edges
    for y in range(rows):
        for x in range(cols):
            if skeleton[y, x]:
                G.add_node((y, x))
                for dy, dx in neighbors:
                    ny, nx_ = y + dy, x + dx
                    if 0 <= ny < rows and 0 <= nx_ < cols:
                        if skeleton[ny, nx_]:
                            G.add_edge((y, x), (ny, nx_))
    return G

def merge_close_subgraphs(G, dist_threshold=2.0):
    """
    Merge subgraphs in a NetworkX graph if they are within 'dist_threshold'.
    Effectively 'bridges' or unifies edges that are close together.
    
    Parameters:
    - G: NetworkX graph (undirected).
    - dist_threshold: float, distance below which two subgraphs get merged.
    
    Returns:
    - merged_graph: A new NetworkX graph with edges added
                    to connect components that are close.
    """
    # Make a copy so we don't mutate the original graph
    merged_graph = G.copy()
    
    # Step 1: Find connected components
    components = list(nx.connected_components(G))
    
    # Step 2: For each pair of distinct components, find the minimum distance
    # between any node in one component and any node in the other.
    # If min_dist < dist_threshold, add an edge between the closest nodes.
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            comp_i = components[i]
            comp_j = components[j]
            min_dist = float('inf')
            closest_pair = (None, None)
            
            for node_i in comp_i:
                for node_j in comp_j:
                    dist = np.linalg.norm(np.array(node_i) - np.array(node_j))
                    if dist < min_dist:
                        min_dist = dist
                        closest_pair = (node_i, node_j)
            
            if min_dist < dist_threshold:
                # Merge by adding an edge between the closest nodes.
                # This effectively unifies these two subgraphs.
                merged_graph.add_edge(closest_pair[0], closest_pair[1])
    
    return merged_graph

def get_edge_pixel_coords(labeled_image, label):
    """
    Returns a list of pixel coordinates belonging to 'label' in 'labeled_image'.
    """
    coords = np.argwhere(labeled_image == label)
    # coords is shape (N,2), each row [r,c]; convert to list of tuples if needed:
    # coords_list = [tuple(row) for row in coords]
    return coords

def measure_edge_distance(coords1, coords2):
    """
    Compute a basic measure of distance between two sets of edge coordinates.
    For simplicity, we'll use the average of minimal distances (a rough measure).
    """
    if coords1.size == 0 or coords2.size == 0:
        return np.inf
    dist_mat = distance_matrix(coords1, coords2)
    # E.g. take the average of the minimum distances in both directions
    d1 = dist_mat.min(axis=1).mean()  # average distance from coords1 to coords2
    d2 = dist_mat.min(axis=0).mean()  # average distance from coords2 to coords1
    return (d1 + d2) / 2.0

def average_edge(coords1, coords2):
    """
    Construct a naive 'average' path between coords1 and coords2.
    This is a very rough approach:
      - Combine both sets of points
      - Take the convex hull or bounding box
      - Possibly do a thinning or shortest path from start to end
    Here we just combine them, then we'll pick out a skeleton via connectivity.

    Returns: np.array of shape (N, 2) of the final path coordinates.
    """
    combined = np.vstack([coords1, coords2])
    # One naive approach: just take the unique coordinates and call that our set:
    combined_unique = np.unique(combined, axis=0)
    return combined_unique

def find_start_end_for_label(labeled_image, label, junctions):
    """
    Returns the (start_junction, end_junction) for the given 'label' by checking
    which junction(s) it touches. If it doesn't have exactly two distinct endpoints,
    returns None, None.
    
    'junctions' is typically a set or list of (row, col) coordinates
    that you recognized as junction points in your graph.
    """
    coords = get_edge_pixel_coords(labeled_image, label)
    coords_set = {tuple(c) for c in coords}

    # Which junction points are in this edge?
    junction_hits = coords_set.intersection(junctions)
    # For a normal "simple" edge (no cycles), we expect 2 endpoints if it
    # connects two distinct junctions. If more or fewer, might be a cycle or
    # a partial edge.
    if len(junction_hits) == 2:
        return tuple(sorted(junction_hits))
    else:
        return None, None

def merge_duplicate_edges_with_average(
    labeled_image,
    junctions,
    dist_threshold=3.0,
    background_label=0
):
    """
    Merges edges that share the same start and end junction
    and are 'close' (less than dist_threshold apart on average),
    by replacing them with a single 'average' edge.

    labeled_image: int 2D array from trace_edges (each edge has a unique >0 label).
    junctions: set or list of (r,c) junction coordinates.
    dist_threshold: float, threshold for deciding if two edges are close.
    background_label: int, usually 0 for background.

    Returns:
    - merged_labeled: updated 2D array with merges performed.
    """
    merged_labeled = labeled_image.copy()
    labels = np.unique(merged_labeled)
    labels = labels[labels != background_label]  # skip background

    # 1) For each label, find (start_junction, end_junction) + store coords
    edge_info = {}
    for lbl in labels:
        sj, ej = find_start_end_for_label(merged_labeled, lbl, junctions)
        if sj is not None and ej is not None:
            coords = get_edge_pixel_coords(merged_labeled, lbl)
            edge_info[lbl] = {
                'start': sj,
                'end': ej,
                'coords': coords
            }
        else:
            # Could be a cycle or something else, skip or store separately
            pass

    # 2) Group edges by their (start, end) pairs
    edges_by_pair = defaultdict(list)
    for lbl, info in edge_info.items():
        pair = (info['start'], info['end'])
        # To avoid direction issues, we ensure the start<end is consistent:
        # (It should already be sorted in find_start_end_for_label)
        edges_by_pair[pair].append(lbl)

    # We'll need to track which labels got merged & replaced
    used_labels = set()  # keep track of edges we've already merged

    # We'll generate new labels for merges
    new_label_counter = merged_labeled.max() + 1

    # 3) For each (start, end) pair, see if multiple edges exist
    for pair, lbl_list in edges_by_pair.items():
        if len(lbl_list) < 2:
            continue  # nothing to merge if there's only 1 edge

        # Potentially, we could do all-pairs merges. For simplicity,
        # we show a simple approach that merges the entire group
        # into one edge if *all* are close to each other, or merges
        # them pairwise if they pass the threshold. 
        # We'll do pairwise checks here:

        merged_edges = []
        while lbl_list:
            current_lbl = lbl_list.pop()
            if current_lbl in used_labels:
                continue

            current_coords = edge_info[current_lbl]['coords']
            to_merge = [current_lbl]
            # We collect edges that are "close" to current_lbl
            still_unassigned = []
            for other_lbl in lbl_list:
                if other_lbl in used_labels:
                    continue
                other_coords = edge_info[other_lbl]['coords']
                dist = measure_edge_distance(current_coords, other_coords)
                if dist < dist_threshold:
                    to_merge.append(other_lbl)
                else:
                    still_unassigned.append(other_lbl)
            # Update lbl_list to those not merged with current_lbl
            lbl_list = still_unassigned

            if len(to_merge) > 1:
                # Merge all in to_merge
                all_coords = []
                for lbl_m in to_merge:
                    used_labels.add(lbl_m)
                    all_coords.append(edge_info[lbl_m]['coords'])
                # Flatten
                all_coords = np.vstack(all_coords)

                # Construct an "average" path
                # For simplicity, let's just do "unique + minimal skeleton"
                combined_unique = np.unique(all_coords, axis=0)

                # Remove old labels from the image
                for lbl_m in to_merge:
                    merged_labeled[merged_labeled == lbl_m] = background_label

                # Now we create a new label for the merged edge
                new_label = new_label_counter
                new_label_counter += 1

                # Mark these coordinates in the image
                for (r, c) in combined_unique:
                    merged_labeled[r, c] = new_label

                merged_edges.append(new_label)
            else:
                # Only one label in to_merge => no merge needed
                # push it back if we want to keep it
                pass

    return merged_labeled

def identify_junctions_and_endpoints(G):
    """
    Identify junctions and endpoints in the graph.
    
    Parameters:
    - G: NetworkX graph.
    
    Returns:
    - junctions: list of node tuples with degree >=3.
    - endpoints: list of node tuples with degree ==1.
    """
    junctions = [node for node, degree in G.degree() if degree >=3]
    endpoints = [node for node, degree in G.degree() if degree ==1]
    return junctions, endpoints

def trace_edges(G, junctions, endpoints, height, width):
    """
    Trace each edge from endpoints or junctions to other endpoints or junctions.
    """
    labeled_image = np.zeros(
        (height, width),
        dtype=np.int32
    )
    
    label = 1
    visited = set()
    
    # Combine endpoints and junctions into a single list of "start" nodes.
    start_nodes = list(set(endpoints + junctions))
    
    for start_node in start_nodes:
        if start_node in visited:
            continue
        
        stack = [(start_node, [start_node])]
        
        while stack:
            current, path = stack.pop()
            visited.add(current)
            neighbors = list(G.neighbors(current))
            
            for neighbor in neighbors:
                if (neighbor in visited) and (neighbor != path[0]):
                    continue
                if (neighbor in junctions) or (neighbor in endpoints):
                    # We reached another endpoint or junction.
                    path_extended = path + [neighbor]
                    # Label the entire path.
                    rr, cc = zip(*path_extended)
                    labeled_image[rr, cc] = label
                    label += 1
                else:
                    stack.append((neighbor, path + [neighbor]))
    
    # Handle cycles
    cycles = [c for c in nx.cycle_basis(G) if len(c) > 0]
    for cycle in cycles:
        # Check if none of these cycle nodes were already labeled
        if any(node in visited for node in cycle):
            # If you prefer, skip if already visited
            # or do a partial labeling check
            continue
        # num_visited = 0
        # for node in cycle:
        #     if node in visited:
        #         num_visited += 1
        # if num_visited >= 4: # if there is only 1 visited node, we will label it, otherwise we skip
        #     continue
        rr, cc = zip(*cycle)
        labeled_image[rr, cc] = label
        label += 1
    
    return labeled_image

def split_connected_components(edge_map, connectivity=2):
    """
    Split connected components at junctions and assign unique labels to each edge.
    
    Parameters:
    - edge_map: 2D binary numpy array.
    - connectivity: int, 1 for 4-connectivity, 2 for 8-connectivity.
    
    Returns:
    - labeled_edges: 2D numpy array with unique labels for each edge.
    """
    # Step 1: Skeletonize the edge map
    # skeleton = skeletonize(edge_map).astype(np.uint8)
    skeleton = thin(edge_map).astype(np.uint8)
    
    # Step 2: Build graph from skeleton
    G = build_graph(skeleton, connectivity=connectivity)

    # G = merge_close_subgraphs(G, dist_threshold=5.0)
    
    # Step 3: Identify junctions and endpoints
    junctions, endpoints = identify_junctions_and_endpoints(G)
    
    # Step 4: Trace and label edges
    height, width = edge_map.shape[:2]
    labeled_edges = trace_edges(G, junctions, endpoints, height, width)

    # merged_labeled = merge_duplicate_edges_with_average(
    #     labeled_image=labeled_edges,
    #     junctions=junctions,
    #     dist_threshold=8.0
    # )
    
    # return merged_labeled, junctions, endpoints
    return labeled_edges, junctions, endpoints

def _edge_key(node_a, node_b):
    return tuple(sorted((tuple(node_a), tuple(node_b))))

def extract_ordered_edge_paths(edge_map, connectivity=2):
    """
    Extract ordered pixel paths between endpoints/junctions from a binary edge map.

    Returns:
    - paths: list of numpy arrays with shape (N, 2), in (row, col) order.
    - skeleton: thinned binary image.
    - graph: NetworkX graph built from the skeleton.
    - junctions: list of graph nodes with degree >= 3.
    - endpoints: list of graph nodes with degree == 1.
    """
    skeleton = thin(edge_map).astype(np.uint8)
    graph = build_graph(skeleton, connectivity=connectivity)
    junctions, endpoints = identify_junctions_and_endpoints(graph)
    anchors = set(junctions) | set(endpoints)
    visited_edges = set()
    paths = []

    for anchor in anchors:
        for neighbor in graph.neighbors(anchor):
            edge_key = _edge_key(anchor, neighbor)
            if edge_key in visited_edges:
                continue

            visited_edges.add(edge_key)
            path = [anchor, neighbor]
            prev_node = anchor
            current_node = neighbor

            while current_node not in anchors:
                next_nodes = [node for node in graph.neighbors(current_node) if node != prev_node]
                if not next_nodes:
                    break
                next_node = next_nodes[0]
                next_edge_key = _edge_key(current_node, next_node)
                if next_edge_key in visited_edges:
                    break
                visited_edges.add(next_edge_key)
                path.append(next_node)
                prev_node, current_node = current_node, next_node

            paths.append(np.asarray(path, dtype=np.int32))

    for component in nx.connected_components(graph):
        component_nodes = set(component)
        if component_nodes & anchors:
            continue

        component_subgraph = graph.subgraph(component_nodes)
        remaining_edges = [
            _edge_key(node_a, node_b)
            for node_a, node_b in component_subgraph.edges()
            if _edge_key(node_a, node_b) not in visited_edges
        ]
        if not remaining_edges:
            continue

        start_node = remaining_edges[0][0]
        path = [start_node]
        prev_node = None
        current_node = start_node

        while True:
            next_nodes = [node for node in component_subgraph.neighbors(current_node) if node != prev_node]
            if not next_nodes:
                break

            next_node = next_nodes[0]
            next_edge_key = _edge_key(current_node, next_node)
            if next_edge_key in visited_edges:
                break

            visited_edges.add(next_edge_key)
            path.append(next_node)
            prev_node, current_node = current_node, next_node
            if current_node == start_node:
                break

        if len(path) > 1:
            paths.append(np.asarray(path, dtype=np.int32))

    return paths, skeleton, graph, junctions, endpoints

def path_length(points):
    return fast_path_length(points)

def smooth_polyline(points, window=5):
    if len(points) < 3 or window <= 1:
        return points.astype(np.float64)
    radius = window // 2
    padded = np.pad(points.astype(np.float64), ((radius, radius), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    smoothed = np.zeros_like(points, dtype=np.float64)
    for dim in range(points.shape[1]):
        smoothed[:, dim] = np.convolve(padded[:, dim], kernel, mode="valid")
    smoothed[0] = points[0]
    smoothed[-1] = points[-1]
    return smoothed

def compute_turning_angles(points):
    if len(points) < 3:
        return np.zeros(len(points), dtype=np.float64)
    pts = smooth_polyline(points, window=min(7, len(points) if len(points) % 2 == 1 else len(points) - 1))
    angles = np.zeros(len(points), dtype=np.float64)
    for idx in range(1, len(points) - 1):
        vec_a = pts[idx] - pts[idx - 1]
        vec_b = pts[idx + 1] - pts[idx]
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            continue
        cosine = np.clip(np.dot(vec_a, vec_b) / (norm_a * norm_b), -1.0, 1.0)
        angles[idx] = np.degrees(np.arccos(cosine))
    return angles

def chord_length_parameterize(points):
    return fast_chord_length_parameterize(points)

def bernstein_basis(degree, t_values):
    return fast_bernstein_basis(degree, t_values)

def _evaluate_bezier_de_casteljau(control_points, t_values):
    control_points = np.asarray(control_points, dtype=np.float64)
    t_values = np.clip(np.asarray(t_values, dtype=np.float64), 0.0, 1.0)
    if not np.isfinite(control_points).all() or not np.isfinite(t_values).all():
        return np.full((len(t_values), control_points.shape[1]), np.nan, dtype=np.float64)
    work = np.broadcast_to(control_points, (len(t_values),) + control_points.shape).copy()
    one_minus_t = (1.0 - t_values)[:, None, None]
    t_column = t_values[:, None, None]
    with np.errstate(over='ignore', invalid='ignore'):
        for level in range(len(control_points) - 1, 0, -1):
            work[:, :level, :] = one_minus_t * work[:, :level, :] + t_column * work[:, 1:level + 1, :]
        evaluated = work[:, 0, :]
    if not np.isfinite(evaluated).all():
        return np.full((len(t_values), control_points.shape[1]), np.nan, dtype=np.float64)
    return evaluated

def evaluate_bezier(control_points, t_values, basis=None):
    control_points = np.asarray(control_points, dtype=np.float64)
    t_values = np.clip(np.asarray(t_values, dtype=np.float64), 0.0, 1.0)
    if not np.isfinite(control_points).all() or not np.isfinite(t_values).all():
        return np.full((len(t_values), control_points.shape[1]), np.nan, dtype=np.float64)
    if basis is None:
        basis = bernstein_basis(len(control_points) - 1, t_values)
    with np.errstate(over='ignore', invalid='ignore'):
        evaluated = fast_evaluate_bezier(control_points, basis)
    if np.isfinite(evaluated).all():
        return evaluated
    return _evaluate_bezier_de_casteljau(control_points, t_values)

def _control_polygon_is_stable(
    control_points,
    points,
    max_multiplier=100.0,
    max_offset_multiplier=25.0,
    data_scale=None,
    points_center=None,
):
    control_points = np.asarray(control_points, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    if not np.isfinite(control_points).all():
        return False
    if data_scale is None or points_center is None:
        _, _, data_scale, points_center = fast_prepare_fit_context(points)
    return fast_control_polygon_is_stable(
        control_points,
        points,
        data_scale=data_scale,
        points_center=points_center,
        max_multiplier=max_multiplier,
        max_offset_multiplier=max_offset_multiplier,
    )

def _prepare_fit_context(points):
    return fast_prepare_fit_context(points)


def fit_bezier_curve(points, degree, fit_context=None):
    """
    Least-squares Bezier fitting with fixed endpoints.
    """
    if fit_context is None:
        pts, t_values, data_scale, points_center = _prepare_fit_context(points)
    else:
        pts, t_values, data_scale, points_center = fit_context
    if len(pts) < 2:
        return None
    if degree < 1:
        raise ValueError("Bezier degree must be >= 1")
    degree = min(int(degree), max(1, len(pts) - 1))
    basis = bernstein_basis(degree, t_values)

    if degree == 1:
        control_points = np.vstack([pts[0], pts[-1]])
    else:
        endpoints = basis[:, [0, -1]]
        inner_basis = basis[:, 1:-1]
        if inner_basis.size == 0:
            return None
        rhs = pts - endpoints[:, [0]] * pts[0] - endpoints[:, [1]] * pts[-1]
        inner_control, _, rank, singular_values = np.linalg.lstsq(inner_basis, rhs, rcond=None)
        if singular_values.size == 0 or rank < inner_basis.shape[1]:
            return None
        smallest = float(singular_values[-1])
        if smallest <= 0.0:
            return None
        condition_number = float(singular_values[0] / smallest)
        if not np.isfinite(condition_number) or condition_number > 1e8:
            return None
        control_points = np.vstack([pts[0], inner_control, pts[-1]])

    if not np.isfinite(control_points).all():
        return None
    if not _control_polygon_is_stable(control_points, pts, data_scale=data_scale, points_center=points_center):
        return None
    fitted = evaluate_bezier(control_points, t_values, basis=basis)
    if not np.isfinite(fitted).all():
        return None
    return {
        "degree": degree,
        "control_points": control_points,
        "t_values": t_values,
        "fitted_points": fitted,
    }

def fit_error_metrics(points, fitted_points):
    return fast_fit_error_metrics(points, fitted_points)

def find_corner_candidates(points, angle_threshold_deg=55.0, min_index_gap=5, endpoint_margin=3):
    if len(points) < 3:
        return []
    angles = compute_turning_angles(points)
    raw_candidates = []
    for idx in range(max(1, endpoint_margin), min(len(points) - 1, len(points) - endpoint_margin)):
        if angles[idx] < angle_threshold_deg:
            continue
        left = max(1, idx - min_index_gap)
        right = min(len(points) - 1, idx + min_index_gap + 1)
        if angles[idx] >= angles[left:right].max():
            raw_candidates.append((angles[idx], idx))

    selected = []
    for _, idx in sorted(raw_candidates, reverse=True):
        if any(abs(idx - chosen) < min_index_gap for chosen in selected):
            continue
        selected.append(idx)
    return sorted(selected)

def split_polyline_by_corners(
    points,
    max_segment_length=96.0,
    angle_threshold_deg=55.0,
    min_index_gap=5,
    min_split_arc_length=8.0,
    prefer_anchor_for_length_split=False,
    length_split_lookahead=32.0,
    length_split_min_strength=0.68,
    split_extrema_window=5,
):
    """
    Produce stable split candidates from curvature and accumulated arclength.
    """
    if len(points) < 2:
        return [0]

    points = np.asarray(points, dtype=np.float64)
    cumulative_lengths = np.concatenate([
        [0.0],
        np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    ])
    split_indices = {0, len(points) - 1}
    corner_candidates = find_corner_candidates(
        points,
        angle_threshold_deg=angle_threshold_deg,
        min_index_gap=min_index_gap,
        endpoint_margin=max(3, min_index_gap // 2),
    )
    accepted_corner_indices = []
    anchor_strengths = compute_split_anchor_strengths(points, extrema_window=split_extrema_window)
    for idx in corner_candidates:
        arc_ok = True
        for existing in split_indices | set(accepted_corner_indices):
            if abs(cumulative_lengths[idx] - cumulative_lengths[existing]) < min_split_arc_length:
                arc_ok = False
                break
        if arc_ok and min(abs(idx - existing) for existing in split_indices | set(accepted_corner_indices)) >= min_index_gap:
            accepted_corner_indices.append(idx)
    split_indices.update(accepted_corner_indices)

    last_split = 0
    total_length = cumulative_lengths[-1]
    while True:
        target_length = cumulative_lengths[last_split] + max_segment_length
        if target_length >= total_length:
            break
        target_idx = int(np.searchsorted(cumulative_lengths, target_length, side="left"))
        chosen_idx = max(target_idx, last_split + min_index_gap)
        chosen_idx = min(chosen_idx, len(points) - 1)

        if prefer_anchor_for_length_split:
            lookahead_limit = min(total_length, target_length + max(0.0, float(length_split_lookahead)))
            candidate_indices = []
            for idx in range(chosen_idx, len(points) - 1):
                if cumulative_lengths[idx] > lookahead_limit:
                    break
                if idx - last_split < min_index_gap:
                    continue
                candidate_indices.append(idx)
            strong_candidates = [
                idx for idx in candidate_indices
                if anchor_strengths[idx] >= length_split_min_strength
            ]
            if strong_candidates:
                chosen_idx = max(
                    strong_candidates,
                    key=lambda idx: (anchor_strengths[idx], -abs(cumulative_lengths[idx] - target_length)),
                )

        split_indices.add(chosen_idx)
        if chosen_idx <= last_split:
            break
        last_split = chosen_idx

    split_indices.add(len(points) - 1)

    return sorted(split_indices)

def choose_best_bezier_fit(points, max_degree=5, mean_error_threshold=0.75, max_error_threshold=2.5):
    fit_context = _prepare_fit_context(points)
    best_fit = None
    for degree in range(1, max_degree + 1):
        fit = fit_bezier_curve(points, degree=degree, fit_context=fit_context)
        if fit is None:
            continue
        metrics = fit_error_metrics(points, fit["fitted_points"])
        fit["mean_error"] = metrics["mean_error"]
        fit["max_error"] = metrics["max_error"]
        if best_fit is None or fit["max_error"] < best_fit["max_error"]:
            best_fit = fit
        if fit["mean_error"] <= mean_error_threshold and fit["max_error"] <= max_error_threshold:
            return fit
    return best_fit


def compute_split_anchor_strengths(points, extrema_window=5):
    points = np.asarray(points, dtype=np.float64)
    n_points = len(points)
    strengths = np.zeros(n_points, dtype=np.float64)
    if n_points == 0:
        return strengths
    strengths[0] = 1.0
    strengths[-1] = 1.0
    if n_points < 3:
        return strengths

    angles = compute_turning_angles(points)
    if len(angles) == n_points:
        strengths = np.maximum(strengths, np.clip(angles / 120.0, 0.0, 1.0))

    window = max(2, int(extrema_window))
    for idx in range(1, n_points - 1):
        left = max(0, idx - window)
        right = min(n_points - 1, idx + window)
        for axis in range(2):
            left_delta = points[idx, axis] - points[left, axis]
            right_delta = points[right, axis] - points[idx, axis]
            if abs(left_delta) < 1e-6 or abs(right_delta) < 1e-6:
                continue
            if np.sign(left_delta) == np.sign(right_delta):
                continue
            balance = min(abs(left_delta), abs(right_delta)) / max(abs(left_delta), abs(right_delta))
            strengths[idx] = max(strengths[idx], 0.55 + 0.35 * balance)

    return np.clip(strengths, 0.0, 1.0)

def _fit_points_for_storage(points, max_degree, mean_error_threshold, max_error_threshold):
    fit = choose_best_bezier_fit(
        points,
        max_degree=max_degree,
        mean_error_threshold=mean_error_threshold,
        max_error_threshold=max_error_threshold,
    )
    if fit is None:
        return None
    fit["points"] = np.asarray(points, dtype=np.float64)
    return fit


def _is_closed_path(points, closure_threshold=1.5):
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return False
    return np.linalg.norm(points[0] - points[-1]) <= closure_threshold


def _is_smooth_open_path_candidate(
    points,
    max_segment_length,
    min_path_length=120.0,
    q90_turn_threshold=42.0,
    max_strong_anchor_count=4,
    strong_anchor_threshold=0.82,
    extrema_window=5,
):
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 8:
        return False
    if _is_closed_path(points):
        return False
    if path_length(points) < min_path_length:
        return False

    angles = compute_turning_angles(points)
    if len(angles) <= 2:
        return False
    interior = angles[1:-1]
    if len(interior) == 0:
        return False
    if float(np.quantile(interior, 0.9)) > q90_turn_threshold:
        return False

    anchor_strengths = compute_split_anchor_strengths(points, extrema_window=extrema_window)
    strong_count = int(np.sum(anchor_strengths[1:-1] >= strong_anchor_threshold))
    return strong_count <= max_strong_anchor_count


def _fit_smooth_path_with_consistent_splits(
    points,
    max_degree,
    mean_error_threshold,
    max_error_threshold,
    max_segment_length,
    min_points=6,
    extrema_window=5,
    target_length_factor=1.0,
    snap_window=40.0,
    snap_anchor_strength=0.6,
):
    points = np.asarray(points, dtype=np.float64)
    total_length = path_length(points)
    if total_length <= max_segment_length:
        return None

    target_length = max(1.0, max_segment_length * target_length_factor)
    segment_count = int(np.ceil(total_length / target_length))
    if segment_count <= 1:
        return None

    cumulative_lengths = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    anchor_strengths = compute_split_anchor_strengths(points, extrema_window=extrema_window)
    split_indices = [0]
    min_index_gap = max(4, min_points - 1)

    for split_num in range(1, segment_count):
        target_arc = total_length * split_num / segment_count
        target_idx = int(np.searchsorted(cumulative_lengths, target_arc, side="left"))
        target_idx = min(max(target_idx, split_indices[-1] + min_index_gap), len(points) - 2)
        chosen_idx = target_idx
        best_score = None
        for idx in range(split_indices[-1] + min_index_gap, len(points) - 1):
            if cumulative_lengths[idx] < target_arc - snap_window:
                continue
            if cumulative_lengths[idx] > target_arc + snap_window:
                break
            distance_penalty = abs(cumulative_lengths[idx] - target_arc) / max(snap_window, 1e-6)
            anchor_bonus = anchor_strengths[idx] if anchor_strengths[idx] >= snap_anchor_strength else 0.0
            score = distance_penalty - 0.45 * anchor_bonus
            if best_score is None or score < best_score:
                best_score = score
                chosen_idx = idx
        if chosen_idx <= split_indices[-1]:
            return None
        split_indices.append(chosen_idx)
    split_indices.append(len(points) - 1)

    segments = []
    for start_idx, end_idx in zip(split_indices[:-1], split_indices[1:]):
        sub_points = points[start_idx:end_idx + 1]
        if len(sub_points) < 2:
            return None
        fit = _fit_points_for_storage(
            sub_points,
            max_degree=max_degree,
            mean_error_threshold=mean_error_threshold,
            max_error_threshold=max_error_threshold,
        )
        if fit is None:
            return None
        if fit["mean_error"] > mean_error_threshold or fit["max_error"] > max_error_threshold:
            return None
        segments.append(fit)
    return segments


def _resample_polyline(points, sample_count=24):
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return np.zeros((sample_count, 2), dtype=np.float64)
    if len(points) == 1:
        return np.repeat(points, sample_count, axis=0)
    cumulative_lengths = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    total_length = cumulative_lengths[-1]
    if total_length < 1e-6:
        return np.repeat(points[:1], sample_count, axis=0)
    target_lengths = np.linspace(0.0, total_length, sample_count)
    rows = np.interp(target_lengths, cumulative_lengths, points[:, 0])
    cols = np.interp(target_lengths, cumulative_lengths, points[:, 1])
    return np.column_stack([rows, cols])


def _path_shape_descriptor(points, sample_count=24):
    sampled = _resample_polyline(points, sample_count=sample_count)
    if tuple(sampled[0]) > tuple(sampled[-1]):
        sampled = sampled[::-1]
    centered = sampled - sampled.mean(axis=0, keepdims=True)
    scale = max(path_length(sampled), 1e-6)
    return centered / scale


def _fit_path_to_target_ratios(
    points,
    target_ratios,
    max_degree,
    mean_error_threshold,
    max_error_threshold,
    min_points=6,
    extrema_window=5,
    snap_window=28.0,
    snap_anchor_strength=0.58,
):
    points = np.asarray(points, dtype=np.float64)
    cumulative_lengths = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    total_length = cumulative_lengths[-1]
    if total_length <= 0:
        return None
    anchor_strengths = compute_split_anchor_strengths(points, extrema_window=extrema_window)
    split_indices = [0]
    min_index_gap = max(4, min_points - 1)
    for ratio in target_ratios:
        target_arc = total_length * ratio
        target_idx = int(np.searchsorted(cumulative_lengths, target_arc, side="left"))
        target_idx = min(max(target_idx, split_indices[-1] + min_index_gap), len(points) - 2)
        best_idx = target_idx
        best_score = None
        for idx in range(split_indices[-1] + min_index_gap, len(points) - 1):
            if cumulative_lengths[idx] < target_arc - snap_window:
                continue
            if cumulative_lengths[idx] > target_arc + snap_window:
                break
            distance_penalty = abs(cumulative_lengths[idx] - target_arc) / max(snap_window, 1e-6)
            anchor_bonus = anchor_strengths[idx] if anchor_strengths[idx] >= snap_anchor_strength else 0.0
            score = distance_penalty - 0.4 * anchor_bonus
            if best_score is None or score < best_score:
                best_score = score
                best_idx = idx
        if best_idx <= split_indices[-1]:
            return None
        split_indices.append(best_idx)
    split_indices.append(len(points) - 1)

    segments = []
    for start_idx, end_idx in zip(split_indices[:-1], split_indices[1:]):
        fit = _fit_points_for_storage(
            points[start_idx:end_idx + 1],
            max_degree=max_degree,
            mean_error_threshold=mean_error_threshold,
            max_error_threshold=max_error_threshold,
        )
        if fit is None:
            return None
        if fit["mean_error"] > mean_error_threshold or fit["max_error"] > max_error_threshold:
            return None
        segments.append(fit)
    return segments


def harmonize_similar_smooth_paths(
    fitted_paths,
    max_degree,
    mean_error_threshold,
    max_error_threshold,
    min_points=6,
    extrema_window=5,
    min_path_length=100.0,
    max_length_ratio=1.45,
    descriptor_distance_threshold=0.09,
    snap_window=28.0,
    snap_anchor_strength=0.58,
):
    candidate_indices = []
    descriptors = {}
    lengths = {}
    for idx, path_fit in enumerate(fitted_paths):
        points = np.asarray(path_fit["original_points"], dtype=np.float64)
        plen = path_length(points)
        if plen < min_path_length or _is_closed_path(points):
            continue
        if not _is_smooth_open_path_candidate(
            points,
            max_segment_length=max(plen, 1.0),
            min_path_length=min_path_length,
            q90_turn_threshold=48.0,
            max_strong_anchor_count=5,
            strong_anchor_threshold=0.86,
            extrema_window=extrema_window,
        ):
            continue
        candidate_indices.append(idx)
        lengths[idx] = plen
        descriptors[idx] = _path_shape_descriptor(points)

    used = set()
    harmonized = list(fitted_paths)
    for idx in candidate_indices:
        if idx in used:
            continue
        best_match = None
        best_distance = None
        for other_idx in candidate_indices:
            if other_idx <= idx or other_idx in used:
                continue
            length_ratio = max(lengths[idx], lengths[other_idx]) / max(min(lengths[idx], lengths[other_idx]), 1e-6)
            if length_ratio > max_length_ratio:
                continue
            distance = float(np.linalg.norm(descriptors[idx] - descriptors[other_idx]))
            if distance > descriptor_distance_threshold:
                continue
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_match = other_idx
        if best_match is None:
            continue

        pair = [idx, best_match]
        reference_idx = max(pair, key=lambda candidate_idx: len(harmonized[candidate_idx]["segments"]))
        reference_segments = harmonized[reference_idx]["segments"]
        reference_lengths = [path_length(seg["points"]) for seg in reference_segments]
        if len(reference_lengths) <= 1:
            continue
        target_ratios = (np.cumsum(reference_lengths) / max(sum(reference_lengths), 1e-6)).tolist()[:-1]
        if not target_ratios:
            continue

        for target_idx in pair:
            if target_idx == reference_idx:
                continue
            new_segments = _fit_path_to_target_ratios(
                harmonized[target_idx]["original_points"],
                target_ratios,
                max_degree=max_degree,
                mean_error_threshold=mean_error_threshold,
                max_error_threshold=max_error_threshold,
                min_points=min_points,
                extrema_window=extrema_window,
                snap_window=snap_window,
                snap_anchor_strength=snap_anchor_strength,
            )
            if new_segments is not None:
                harmonized[target_idx] = {
                    "original_points": harmonized[target_idx]["original_points"],
                    "segments": new_segments,
                }
        used.update(pair)
    return harmonized


def _segment_chunk_with_dp(
    chunk,
    max_degree,
    mean_error_threshold,
    max_error_threshold,
    max_segment_length,
    min_points=6,
    split_anchor_weight=0.8,
    extrema_window=5,
    short_segment_penalty_weight=0.9,
    preferred_min_segment_ratio=0.45,
    candidate_length_factor=1.75,
):
    chunk = np.asarray(chunk, dtype=np.float64)
    n_points = len(chunk)
    if n_points < 2:
        return []

    min_segment_points = max(2, min_points)
    fit_cache = {}
    anchor_strengths = compute_split_anchor_strengths(chunk, extrema_window=extrema_window)
    cumulative_lengths = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(chunk, axis=0), axis=1))])
    total_length = float(cumulative_lengths[-1])
    preferred_min_length = min(
        max_segment_length * preferred_min_segment_ratio,
        total_length * 0.45,
    )
    max_candidate_length = max(max_segment_length * candidate_length_factor, preferred_min_length * 2.0)
    dp = [None] * n_points
    dp[0] = (0, 0.0, 0.0, [])

    for end_idx in range(1, n_points):
        best_state = None
        for start_idx in range(end_idx - 1, -1, -1):
            prev_state = dp[start_idx]
            if prev_state is None:
                continue
            point_count = end_idx - start_idx + 1
            if start_idx > 0 and point_count < min_segment_points:
                continue
            if start_idx == 0 and end_idx < n_points - 1 and point_count < min_segment_points:
                continue
            segment_length = float(cumulative_lengths[end_idx] - cumulative_lengths[start_idx])
            if start_idx > 0 and segment_length > max_candidate_length:
                break

            cache_key = (start_idx, end_idx)
            if cache_key not in fit_cache:
                fit_cache[cache_key] = _fit_points_for_storage(
                    chunk[start_idx:end_idx + 1],
                    max_degree=max_degree,
                    mean_error_threshold=mean_error_threshold,
                    max_error_threshold=max_error_threshold,
                )
            fit = fit_cache[cache_key]
            if fit is None:
                continue
            if fit["mean_error"] > mean_error_threshold or fit["max_error"] > max_error_threshold:
                continue

            split_penalty = prev_state[1]
            if preferred_min_length > 0.0 and segment_length < preferred_min_length:
                split_penalty += short_segment_penalty_weight * (preferred_min_length - segment_length) / preferred_min_length
            if end_idx < n_points - 1:
                split_penalty += split_anchor_weight * (1.0 - anchor_strengths[end_idx])
            error_score = prev_state[2] + fit["max_error"] + 0.25 * fit["mean_error"]
            candidate = (
                prev_state[0] + 1,
                split_penalty,
                error_score,
                prev_state[3] + [fit],
            )
            if best_state is None or candidate[:3] < best_state[:3]:
                best_state = candidate
        dp[end_idx] = best_state

    if dp[-1] is not None:
        return dp[-1][3]
    return None

def _repartition_adjacent_segments(
    combined_points,
    max_degree,
    mean_error_threshold,
    max_error_threshold,
    tiny_length_threshold,
    min_points=6,
):
    combined_points = np.asarray(combined_points, dtype=np.float64)
    n_points = len(combined_points)
    if n_points < 4:
        return None

    relaxed_mean = mean_error_threshold + 1.1
    relaxed_max = max_error_threshold + 1.1
    best_candidate = None

    start_split = max(2, min(min_points - 1, n_points - 2))
    end_split = min(n_points - 2, max(n_points - min_points, 2))
    for split_idx in range(start_split, end_split + 1):
        points_a = combined_points[:split_idx + 1]
        points_b = combined_points[split_idx:]
        length_a = path_length(points_a)
        length_b = path_length(points_b)
        if length_a <= tiny_length_threshold or length_b <= tiny_length_threshold:
            continue

        fit_a = _fit_points_for_storage(
            points_a,
            max_degree=max_degree,
            mean_error_threshold=np.inf,
            max_error_threshold=np.inf,
        )
        fit_b = _fit_points_for_storage(
            points_b,
            max_degree=max_degree,
            mean_error_threshold=np.inf,
            max_error_threshold=np.inf,
        )
        if fit_a is None or fit_b is None:
            continue

        strict_ok = (
            fit_a["mean_error"] <= mean_error_threshold and
            fit_b["mean_error"] <= mean_error_threshold and
            fit_a["max_error"] <= max_error_threshold and
            fit_b["max_error"] <= max_error_threshold
        )
        relaxed_ok = (
            fit_a["mean_error"] <= relaxed_mean and
            fit_b["mean_error"] <= relaxed_mean and
            fit_a["max_error"] <= relaxed_max and
            fit_b["max_error"] <= relaxed_max
        )
        if not strict_ok and not relaxed_ok:
            continue

        score = (
            max(fit_a["max_error"], fit_b["max_error"]) +
            0.1 * (fit_a["mean_error"] + fit_b["mean_error"]) +
            0.002 * abs(length_a - length_b)
        )
        candidate = (strict_ok, score, [fit_a, fit_b])
        if best_candidate is None or (candidate[0] and not best_candidate[0]) or (candidate[0] == best_candidate[0] and candidate[1] < best_candidate[1]):
            best_candidate = candidate

    if best_candidate is None:
        return None
    return best_candidate[2]

def cleanup_tiny_bezier_segments(
    segments,
    parent_length,
    max_degree,
    mean_error_threshold,
    max_error_threshold,
    tiny_length_threshold=3.0,
    long_path_threshold=20.0,
    min_points=6,
):
    """
    Remove pathological 1-3 pixel leftovers produced by the greedy splitter on long paths.
    """
    if parent_length < long_path_threshold or len(segments) <= 1:
        return segments

    cleaned = list(segments)
    changed = True
    while changed:
        changed = False
        for idx, segment in enumerate(cleaned):
            segment_length = path_length(segment["points"])
            if segment_length > tiny_length_threshold:
                continue

            if idx == 0 or idx == len(cleaned) - 1:
                cleaned.pop(idx)
                changed = True
                break

            # Prefer absorbing the trivial leftover into one neighbor instead of preserving
            # an extra tiny curve just to keep the local fit score high.
            candidate_merges = []
            if idx > 0:
                merged_points = np.vstack([cleaned[idx - 1]["points"], cleaned[idx]["points"][1:]])
                fit = _fit_points_for_storage(
                    merged_points,
                    max_degree=max_degree,
                    mean_error_threshold=np.inf,
                    max_error_threshold=np.inf,
                )
                if fit is not None:
                    candidate_merges.append(("prev", fit))
            if idx + 1 < len(cleaned):
                merged_points = np.vstack([cleaned[idx]["points"], cleaned[idx + 1]["points"][1:]])
                fit = _fit_points_for_storage(
                    merged_points,
                    max_degree=max_degree,
                    mean_error_threshold=np.inf,
                    max_error_threshold=np.inf,
                )
                if fit is not None:
                    candidate_merges.append(("next", fit))

            if candidate_merges:
                _, best_merge = min(
                    candidate_merges,
                    key=lambda item: (
                        item[1]["max_error"],
                        item[1]["mean_error"],
                        path_length(item[1]["points"]),
                    ),
                )
                if best_merge["max_error"] <= max_error_threshold + 2.5:
                    if candidate_merges[0][0] == "prev" and best_merge is candidate_merges[0][1]:
                        cleaned[idx - 1:idx + 1] = [best_merge]
                    elif idx + 1 < len(cleaned):
                        cleaned[idx:idx + 2] = [best_merge]
                    changed = True
                    break

            pair_start = idx - 1 if idx > 0 else idx
            pair_end = idx if idx > 0 else idx + 1
            if pair_end >= len(cleaned):
                continue

            combined_points = cleaned[pair_start]["points"]
            for merge_idx in range(pair_start + 1, pair_end + 1):
                combined_points = np.vstack([combined_points, cleaned[merge_idx]["points"][1:]])

            repartitioned = _repartition_adjacent_segments(
                combined_points,
                max_degree=max_degree,
                mean_error_threshold=mean_error_threshold,
                max_error_threshold=max_error_threshold,
                tiny_length_threshold=tiny_length_threshold,
                min_points=min_points,
            )
            if repartitioned is None:
                cleaned.pop(idx)
                changed = True
                break

            cleaned[pair_start:pair_end + 1] = repartitioned
            changed = True
            break

    return cleaned

def merge_easy_adjacent_segments(
    segments,
    max_degree,
    merge_max_error_threshold=1.6,
    merge_mean_error_threshold=0.9,
):
    if len(segments) <= 1:
        return segments

    merged = list(segments)
    changed = True
    while changed:
        changed = False
        for idx in range(len(merged) - 1):
            combined_points = np.vstack([merged[idx]["points"], merged[idx + 1]["points"][1:]])
            fit = _fit_points_for_storage(
                combined_points,
                max_degree=max_degree,
                mean_error_threshold=np.inf,
                max_error_threshold=np.inf,
            )
            if fit is None:
                continue
            if fit["max_error"] <= merge_max_error_threshold and fit["mean_error"] <= merge_mean_error_threshold:
                merged[idx:idx + 2] = [fit]
                changed = True
                break
    return merged

def fit_polyline_with_piecewise_bezier(
    points,
    max_degree=5,
    mean_error_threshold=0.6,
    max_error_threshold=2.0,
    max_segment_length=88.0,
    angle_threshold_deg=50.0,
    min_points=6,
    use_global_chunk_dp=False,
    split_anchor_weight=0.8,
    split_extrema_window=5,
    prefer_anchor_for_length_split=False,
    length_split_lookahead=32.0,
    length_split_min_strength=0.68,
    short_segment_penalty_weight=0.9,
    preferred_min_segment_ratio=0.45,
    enable_smooth_consistency=False,
    smooth_consistency_min_path_length=120.0,
    smooth_consistency_q90_turn_threshold=42.0,
    smooth_consistency_max_strong_anchors=4,
    smooth_consistency_strong_anchor_threshold=0.82,
    smooth_consistency_target_length_factor=1.0,
    smooth_consistency_snap_window=40.0,
    smooth_consistency_snap_anchor_strength=0.6,
):
    """
    Hybrid strategy:
    1. Pre-split at strong corners / long spans.
    2. On each chunk, greedily take the longest prefix that fits within the error budget.
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return []

    if enable_smooth_consistency and _is_smooth_open_path_candidate(
        points,
        max_segment_length=max_segment_length,
        min_path_length=smooth_consistency_min_path_length,
        q90_turn_threshold=smooth_consistency_q90_turn_threshold,
        max_strong_anchor_count=smooth_consistency_max_strong_anchors,
        strong_anchor_threshold=smooth_consistency_strong_anchor_threshold,
        extrema_window=split_extrema_window,
    ):
        consistent_segments = _fit_smooth_path_with_consistent_splits(
            points,
            max_degree=max_degree,
            mean_error_threshold=mean_error_threshold,
            max_error_threshold=max_error_threshold,
            max_segment_length=max_segment_length,
            min_points=min_points,
            extrema_window=split_extrema_window,
            target_length_factor=smooth_consistency_target_length_factor,
            snap_window=smooth_consistency_snap_window,
            snap_anchor_strength=smooth_consistency_snap_anchor_strength,
        )
        if consistent_segments is not None:
            return consistent_segments

    split_points = split_polyline_by_corners(
        points,
        max_segment_length=max_segment_length,
        angle_threshold_deg=angle_threshold_deg,
        min_index_gap=max(4, min_points - 1),
        min_split_arc_length=max(8.0, float(min_points)),
        prefer_anchor_for_length_split=prefer_anchor_for_length_split,
        length_split_lookahead=length_split_lookahead,
        length_split_min_strength=length_split_min_strength,
        split_extrema_window=split_extrema_window,
    )

    piecewise_segments = []
    for chunk_start, chunk_end in zip(split_points[:-1], split_points[1:]):
        if chunk_end <= chunk_start:
            continue
        chunk = points[chunk_start:chunk_end + 1]
        if use_global_chunk_dp:
            dp_segments = _segment_chunk_with_dp(
                chunk,
                max_degree=max_degree,
                mean_error_threshold=mean_error_threshold,
                max_error_threshold=max_error_threshold,
                max_segment_length=max_segment_length,
                min_points=min_points,
                split_anchor_weight=split_anchor_weight,
                extrema_window=split_extrema_window,
                short_segment_penalty_weight=short_segment_penalty_weight,
                preferred_min_segment_ratio=preferred_min_segment_ratio,
            )
            if dp_segments is not None:
                piecewise_segments.extend(dp_segments)
                continue
        start_idx = 0

        while start_idx < len(chunk) - 1:
            best_candidate = None
            fallback_candidate = None
            candidate_end = start_idx + max(2, min_points)

            while candidate_end <= len(chunk):
                sub_points = chunk[start_idx:candidate_end]
                fit = choose_best_bezier_fit(
                    sub_points,
                    max_degree=max_degree,
                    mean_error_threshold=mean_error_threshold,
                    max_error_threshold=max_error_threshold,
                )
                if fit is None:
                    break

                fit["points"] = sub_points
                fallback_candidate = fit
                if fit["mean_error"] <= mean_error_threshold and fit["max_error"] <= max_error_threshold:
                    best_candidate = fit
                    candidate_end += 1
                    continue
                break

            selected = best_candidate
            if selected is None:
                selected = fallback_candidate
                if selected is None:
                    forced_end = min(len(chunk), start_idx + max(2, min_points))
                    selected = choose_best_bezier_fit(
                        chunk[start_idx:forced_end],
                        max_degree=max_degree,
                        mean_error_threshold=np.inf,
                        max_error_threshold=np.inf,
                    )
                    selected["points"] = chunk[start_idx:forced_end]

            piecewise_segments.append(selected)
            consumed = len(selected["points"]) - 1
            if consumed <= 0:
                break
            start_idx += consumed

    return piecewise_segments

def sample_piecewise_bezier(segments, samples_per_segment=64):
    samples = []
    for segment in segments:
        control_points = np.asarray(segment["control_points"], dtype=np.float64)
        if not _control_polygon_is_stable(control_points, segment["points"]):
            continue
        control_polygon_length = path_length(control_points)
        adaptive_samples = max(samples_per_segment, int(np.ceil(control_polygon_length * 2.0)))
        adaptive_samples = min(adaptive_samples, 4096)
        t_values = np.linspace(0.0, 1.0, adaptive_samples)
        curve_points = evaluate_bezier(segment["control_points"], t_values)
        if not np.isfinite(curve_points).all():
            continue
        if samples:
            curve_points = curve_points[1:]
        samples.append(curve_points)
    if not samples:
        return np.zeros((0, 2), dtype=np.float64)
    return np.vstack(samples)

def rasterize_points(shape, points):
    raster = np.zeros(shape, dtype=np.uint8)
    if len(points) == 0:
        return raster
    rounded = np.rint(points).astype(np.int32)
    valid = (
        (rounded[:, 0] >= 0) & (rounded[:, 0] < shape[0]) &
        (rounded[:, 1] >= 0) & (rounded[:, 1] < shape[1])
    )
    rounded = rounded[valid]
    raster[rounded[:, 0], rounded[:, 1]] = 1
    if len(rounded) >= 2:
        for start, end in zip(rounded[:-1], rounded[1:]):
            rr, cc = line(start[0], start[1], end[0], end[1])
            keep = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
            raster[rr[keep], cc[keep]] = 1
    return raster

def fit_paths_with_piecewise_bezier(
    paths,
    max_degree=5,
    mean_error_threshold=0.6,
    max_error_threshold=2.0,
    max_segment_length=88.0,
    angle_threshold_deg=50.0,
    min_points=6,
    tiny_segment_length=3.0,
    min_path_length_for_bezier=6.0,
    cleanup_long_path_threshold=20.0,
    enable_tiny_cleanup=True,
    enable_easy_merge=True,
    use_global_chunk_dp=False,
    split_anchor_weight=0.8,
    split_extrema_window=5,
    prefer_anchor_for_length_split=False,
    length_split_lookahead=32.0,
    length_split_min_strength=0.68,
    short_segment_penalty_weight=0.9,
    preferred_min_segment_ratio=0.45,
    enable_smooth_consistency=False,
    smooth_consistency_min_path_length=120.0,
    smooth_consistency_q90_turn_threshold=42.0,
    smooth_consistency_max_strong_anchors=4,
    smooth_consistency_strong_anchor_threshold=0.82,
    smooth_consistency_target_length_factor=1.0,
    smooth_consistency_snap_window=40.0,
    smooth_consistency_snap_anchor_strength=0.6,
    enable_bundle_consistency=False,
    bundle_consistency_min_path_length=100.0,
    bundle_consistency_max_length_ratio=1.45,
    bundle_consistency_descriptor_threshold=0.09,
    bundle_consistency_snap_window=28.0,
    bundle_consistency_snap_anchor_strength=0.58,
):
    fitted_paths = []
    dropped_paths = []
    for path in paths:
        path = np.asarray(path, dtype=np.float64)
        if path_length(path) < min_path_length_for_bezier:
            dropped_paths.append(path)
            continue
        segments = fit_polyline_with_piecewise_bezier(
            path,
            max_degree=max_degree,
            mean_error_threshold=mean_error_threshold,
            max_error_threshold=max_error_threshold,
            max_segment_length=max_segment_length,
            angle_threshold_deg=angle_threshold_deg,
            min_points=min_points,
            use_global_chunk_dp=use_global_chunk_dp,
            split_anchor_weight=split_anchor_weight,
            split_extrema_window=split_extrema_window,
            prefer_anchor_for_length_split=prefer_anchor_for_length_split,
            length_split_lookahead=length_split_lookahead,
            length_split_min_strength=length_split_min_strength,
            short_segment_penalty_weight=short_segment_penalty_weight,
            preferred_min_segment_ratio=preferred_min_segment_ratio,
            enable_smooth_consistency=enable_smooth_consistency,
            smooth_consistency_min_path_length=smooth_consistency_min_path_length,
            smooth_consistency_q90_turn_threshold=smooth_consistency_q90_turn_threshold,
            smooth_consistency_max_strong_anchors=smooth_consistency_max_strong_anchors,
            smooth_consistency_strong_anchor_threshold=smooth_consistency_strong_anchor_threshold,
            smooth_consistency_target_length_factor=smooth_consistency_target_length_factor,
            smooth_consistency_snap_window=smooth_consistency_snap_window,
            smooth_consistency_snap_anchor_strength=smooth_consistency_snap_anchor_strength,
        )
        if enable_tiny_cleanup:
            segments = cleanup_tiny_bezier_segments(
                segments,
                parent_length=path_length(path),
                max_degree=max_degree,
                mean_error_threshold=mean_error_threshold,
                max_error_threshold=max_error_threshold,
                tiny_length_threshold=tiny_segment_length,
                long_path_threshold=cleanup_long_path_threshold,
                min_points=min_points,
            )
        if enable_easy_merge:
            segments = merge_easy_adjacent_segments(
                segments,
                max_degree=max_degree,
            )
        fitted_paths.append({
            "original_points": path,
            "segments": segments,
        })
    if enable_bundle_consistency and len(fitted_paths) > 1:
        fitted_paths = harmonize_similar_smooth_paths(
            fitted_paths,
            max_degree=max_degree,
            mean_error_threshold=mean_error_threshold,
            max_error_threshold=max_error_threshold,
            min_points=min_points,
            extrema_window=split_extrema_window,
            min_path_length=bundle_consistency_min_path_length,
            max_length_ratio=bundle_consistency_max_length_ratio,
            descriptor_distance_threshold=bundle_consistency_descriptor_threshold,
            snap_window=bundle_consistency_snap_window,
            snap_anchor_strength=bundle_consistency_snap_anchor_strength,
        )
    return fitted_paths, dropped_paths

def render_piecewise_fits(shape, fitted_paths, samples_per_segment=64):
    raster = np.zeros(shape, dtype=np.uint8)
    sampled_points = []
    for path_fit in fitted_paths:
        curve_points = sample_piecewise_bezier(path_fit["segments"], samples_per_segment=samples_per_segment)
        if len(curve_points) == 0:
            continue
        sampled_points.append(curve_points)
        raster = np.maximum(raster, rasterize_points(shape, curve_points))
    if sampled_points:
        sampled_points = np.vstack(sampled_points)
    else:
        sampled_points = np.zeros((0, 2), dtype=np.float64)
    return raster, sampled_points

def summarize_piecewise_fits(fitted_paths):
    segment_count = 0
    mean_errors = []
    max_errors = []
    degrees = Counter()

    for path_fit in fitted_paths:
        for segment in path_fit["segments"]:
            segment_count += 1
            mean_errors.append(segment["mean_error"])
            max_errors.append(segment["max_error"])
            degrees[segment["degree"]] += 1

    return {
        "path_count": len(fitted_paths),
        "segment_count": segment_count,
        "mean_segment_error": float(np.mean(mean_errors)) if mean_errors else 0.0,
        "max_segment_error": float(np.max(max_errors)) if max_errors else 0.0,
        "degree_histogram": dict(sorted(degrees.items())),
    }

def evaluate_fit(original_edge_map, fitted_raster):
    original_points = np.argwhere(original_edge_map > 0)
    fitted_points = np.argwhere(fitted_raster > 0)
    if len(original_points) == 0 or len(fitted_points) == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "chamfer": np.inf,
        }

    dilated_original = morphology.binary_dilation(original_edge_map > 0, morphology.disk(1))
    dilated_fitted = morphology.binary_dilation(fitted_raster > 0, morphology.disk(1))
    tp = int(np.logical_and(fitted_raster > 0, dilated_original).sum())
    precision = tp / max(int((fitted_raster > 0).sum()), 1)
    recall = int(np.logical_and(original_edge_map > 0, dilated_fitted).sum()) / max(int((original_edge_map > 0).sum()), 1)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    dist_matrix = distance_matrix(original_points, fitted_points)
    chamfer = float(dist_matrix.min(axis=1).mean() + dist_matrix.min(axis=0).mean()) / 2.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "chamfer": chamfer,
    }

def visualize_fit(edge_map, fitted_paths, output_path, samples_per_segment=64):
    fitted_raster, _ = render_piecewise_fits(edge_map.shape, fitted_paths, samples_per_segment=samples_per_segment)
    overlay = np.zeros((*edge_map.shape, 3), dtype=np.float32)
    overlay[..., 0] = edge_map > 0
    overlay[..., 1] = fitted_raster > 0
    overlay[..., 2] = np.logical_and(edge_map > 0, fitted_raster > 0)

    plt.figure(figsize=(10, 14))
    plt.imshow(overlay)
    plt.title("Red: original edge, Green: Bezier fit, Cyan: overlap")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

def _segment_palette():
    return [
        "#ff595e", "#ffca3a", "#8ac926", "#1982c4", "#6a4c93",
        "#f72585", "#b5179e", "#7209b7", "#4361ee", "#4cc9f0",
        "#fb5607", "#8338ec",
    ]

def visualize_colored_segments(edge_map, fitted_paths, output_path, samples_per_segment=64):
    palette = _segment_palette()
    plt.figure(figsize=(10, 14))
    plt.imshow(edge_map, cmap="gray", alpha=0.28)

    for path_idx, path_fit in enumerate(fitted_paths):
        prev_color_idx = None
        for seg_idx, segment in enumerate(path_fit["segments"]):
            curve_points = sample_piecewise_bezier([segment], samples_per_segment=samples_per_segment)
            if len(curve_points) == 0:
                continue
            color_idx = (path_idx + seg_idx) % len(palette)
            if prev_color_idx is not None and color_idx == prev_color_idx:
                color_idx = (color_idx + 1) % len(palette)
            prev_color_idx = color_idx
            color = palette[color_idx]
            plt.plot(curve_points[:, 1], curve_points[:, 0], color=color, linewidth=1.6)

    plt.gca().invert_yaxis()
    plt.axis("off")
    plt.title("Colored Bezier overlap on binary edge map")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

def run_bezier_refinement(
    image_path=None,
    edge_map_array=None,
    max_degree=5,
    mean_error_threshold=0.6,
    max_error_threshold=2.0,
    max_segment_length=88.0,
    angle_threshold_deg=50.0,
    min_points=6,
    tiny_segment_length=3.0,
    min_path_length_for_bezier=6.0,
    cleanup_long_path_threshold=20.0,
    enable_tiny_cleanup=True,
    enable_easy_merge=True,
    use_global_chunk_dp=False,
    split_anchor_weight=0.8,
    split_extrema_window=5,
    prefer_anchor_for_length_split=False,
    length_split_lookahead=32.0,
    length_split_min_strength=0.68,
    short_segment_penalty_weight=0.9,
    preferred_min_segment_ratio=0.45,
    enable_smooth_consistency=False,
    smooth_consistency_min_path_length=120.0,
    smooth_consistency_q90_turn_threshold=42.0,
    smooth_consistency_max_strong_anchors=4,
    smooth_consistency_strong_anchor_threshold=0.82,
    smooth_consistency_target_length_factor=1.0,
    smooth_consistency_snap_window=40.0,
    smooth_consistency_snap_anchor_strength=0.6,
    enable_bundle_consistency=False,
    bundle_consistency_min_path_length=100.0,
    bundle_consistency_max_length_ratio=1.45,
    bundle_consistency_descriptor_threshold=0.09,
    bundle_consistency_snap_window=28.0,
    bundle_consistency_snap_anchor_strength=0.58,
    connectivity=2,
    compute_raster=True,
    compute_summary=True,
    compute_metrics=True,
    include_debug_artifacts=True,
    output_dir=None,
):
    if edge_map_array is not None:
        edge_map = (np.asarray(edge_map_array) > 0).astype(np.uint8)
    else:
        if image_path is None:
            raise ValueError("Either image_path or edge_map_array must be provided")
        image = np.array(Image.open(image_path).convert("L"))
        edge_map = (image > 127).astype(np.uint8)
    paths, skeleton, graph, junctions, endpoints = extract_ordered_edge_paths(edge_map, connectivity=connectivity)
    fitted_paths, dropped_paths = fit_paths_with_piecewise_bezier(
        paths,
        max_degree=max_degree,
        mean_error_threshold=mean_error_threshold,
        max_error_threshold=max_error_threshold,
        max_segment_length=max_segment_length,
        angle_threshold_deg=angle_threshold_deg,
        min_points=min_points,
        tiny_segment_length=tiny_segment_length,
        min_path_length_for_bezier=min_path_length_for_bezier,
        cleanup_long_path_threshold=cleanup_long_path_threshold,
        enable_tiny_cleanup=enable_tiny_cleanup,
        enable_easy_merge=enable_easy_merge,
        use_global_chunk_dp=use_global_chunk_dp,
        split_anchor_weight=split_anchor_weight,
        split_extrema_window=split_extrema_window,
        prefer_anchor_for_length_split=prefer_anchor_for_length_split,
        length_split_lookahead=length_split_lookahead,
        length_split_min_strength=length_split_min_strength,
        short_segment_penalty_weight=short_segment_penalty_weight,
        preferred_min_segment_ratio=preferred_min_segment_ratio,
        enable_smooth_consistency=enable_smooth_consistency,
        smooth_consistency_min_path_length=smooth_consistency_min_path_length,
        smooth_consistency_q90_turn_threshold=smooth_consistency_q90_turn_threshold,
        smooth_consistency_max_strong_anchors=smooth_consistency_max_strong_anchors,
        smooth_consistency_strong_anchor_threshold=smooth_consistency_strong_anchor_threshold,
        smooth_consistency_target_length_factor=smooth_consistency_target_length_factor,
        smooth_consistency_snap_window=smooth_consistency_snap_window,
        smooth_consistency_snap_anchor_strength=smooth_consistency_snap_anchor_strength,
        enable_bundle_consistency=enable_bundle_consistency,
        bundle_consistency_min_path_length=bundle_consistency_min_path_length,
        bundle_consistency_max_length_ratio=bundle_consistency_max_length_ratio,
        bundle_consistency_descriptor_threshold=bundle_consistency_descriptor_threshold,
        bundle_consistency_snap_window=bundle_consistency_snap_window,
        bundle_consistency_snap_anchor_strength=bundle_consistency_snap_anchor_strength,
    )
    fitted_raster = None
    sampled_points = None
    if compute_raster or compute_metrics or output_dir is not None:
        fitted_raster, sampled_points = render_piecewise_fits(edge_map.shape, fitted_paths)

    summary = summarize_piecewise_fits(fitted_paths) if compute_summary else None
    metrics = evaluate_fit(edge_map, fitted_raster) if compute_metrics else None

    result = {
        "fitted_paths": fitted_paths,
        "dropped_paths": dropped_paths,
    }
    if include_debug_artifacts:
        result.update({
            "edge_map": edge_map,
            "paths": paths,
            "skeleton": skeleton,
            "graph": graph,
            "junctions": junctions,
            "endpoints": endpoints,
        })
    if fitted_raster is not None:
        result["fitted_raster"] = fitted_raster
    if sampled_points is not None:
        result["sampled_points"] = sampled_points
    if summary is not None:
        result["summary"] = summary
    if metrics is not None:
        result["metrics"] = metrics

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.splitext(os.path.basename(image_path))[0] if image_path is not None else "edge_map"
        overlay_path = os.path.join(output_dir, f"{basename}_bezier_overlay.png")
        colored_overlay_path = os.path.join(output_dir, f"{basename}_bezier_colored_overlay.png")
        fitted_path = os.path.join(output_dir, f"{basename}_bezier_fit.png")
        np.save(os.path.join(output_dir, f"{basename}_bezier_control_points.npy"), fitted_paths, allow_pickle=True)
        visualize_fit(edge_map, fitted_paths, overlay_path)
        visualize_colored_segments(edge_map, fitted_paths, colored_overlay_path)
        Image.fromarray((fitted_raster * 255).astype(np.uint8)).save(fitted_path)
        result["overlay_path"] = overlay_path
        result["colored_overlay_path"] = colored_overlay_path
        result["fitted_path"] = fitted_path

    return result

def extract_and_count_values(matrix, binary_mask):
    """
    Extracts values from the input matrix based on the binary mask and
    returns the statistics (counts) of the unique values.
    
    Parameters:
        matrix (numpy.ndarray): A 2D or 3D numpy array containing the values.
        binary_mask (numpy.ndarray): A binary mask of the same shape as the matrix.
                                     Non-zero entries in the mask indicate points of interest.
    
    Returns:
        dict: A dictionary where keys are the unique values in the masked region and 
              values are the counts of those unique values.
    """
    # Ensure the mask and matrix have the same shape
    if matrix.shape != binary_mask.shape:
        raise ValueError("The matrix and binary mask must have the same shape.")
    
    # Extract the values based on the mask
    masked_values = matrix[binary_mask > 0]
    
    # Count the occurrences of each unique value
    value_counts = dict(Counter(masked_values))
    count_values = {v: k for k, v in value_counts.items()}
    
    # Convert to a dictionary and return
    return value_counts, count_values

def decide_value_and_confidence(count_values):
    # confidence: 1 low conf, 2 medium conf, 3 high conf
    total_counts = np.sum(list(count_values.keys()))
    idx_ascending = np.argsort(list(count_values.keys()))
    max_idx = idx_ascending[-1]
    count = list(count_values.keys())[max_idx]
    value = count_values[count]

    if len(count_values) == 1:
        return value, 3
    elif count > 0.6 * total_counts:
        return value, 3
    else:
        idx_2 = idx_ascending[-2]
        count2 = list(count_values.keys())[idx_2]
        value2 = count_values[count]

        if count + count2 > 0.9 * total_counts:
            return max(value, value2), 2
        else:
            return value, 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit binary edge-map segments with piecewise Bezier curves.")
    parser.add_argument("--input", type=str, help="Path to a binary edge map.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory for debug visualizations.")
    parser.add_argument("--max-degree", type=int, default=5, help="Maximum Bezier degree for each segment.")
    parser.add_argument("--mean-error-threshold", type=float, default=0.6, help="Target mean fitting error per segment.")
    parser.add_argument("--max-error-threshold", type=float, default=2.0, help="Target max fitting error per segment.")
    parser.add_argument("--max-segment-length", type=float, default=88.0, help="Force a split when a polyline grows beyond this arclength.")
    parser.add_argument("--angle-threshold", type=float, default=50.0, help="Split near turning points above this angle in degrees.")
    parser.add_argument("--min-points", type=int, default=6, help="Minimum number of polyline samples per Bezier chunk.")
    parser.add_argument("--tiny-segment-length", type=float, default=3.0, help="Cleanup threshold for pathological short leftover segments on long paths.")
    parser.add_argument("--min-path-length", type=float, default=6.0, help="Skip fitting trivial original paths shorter than this arclength.")
    args = parser.parse_args()

    if not args.input:
        raise SystemExit("Please provide --input path/to/binary_edge_map.png")

    result = run_bezier_refinement(
        image_path=args.input,
        max_degree=args.max_degree,
        mean_error_threshold=args.mean_error_threshold,
        max_error_threshold=args.max_error_threshold,
        max_segment_length=args.max_segment_length,
        angle_threshold_deg=args.angle_threshold,
        min_points=args.min_points,
        tiny_segment_length=args.tiny_segment_length,
        min_path_length_for_bezier=args.min_path_length,
        output_dir=args.output_dir,
    )

    print("paths", result["summary"]["path_count"])
    print("segments", result["summary"]["segment_count"])
    print("dropped_paths", len(result["dropped_paths"]))
    print("degree_histogram", result["summary"]["degree_histogram"])
    print("mean_segment_error", f"{result['summary']['mean_segment_error']:.4f}")
    print("max_segment_error", f"{result['summary']['max_segment_error']:.4f}")
    print("precision", f"{result['metrics']['precision']:.4f}")
    print("recall", f"{result['metrics']['recall']:.4f}")
    print("f1", f"{result['metrics']['f1']:.4f}")
    print("chamfer", f"{result['metrics']['chamfer']:.4f}")
    if "overlay_path" in result:
        print("overlay_path", result["overlay_path"])
    if "colored_overlay_path" in result:
        print("colored_overlay_path", result["colored_overlay_path"])
    if "fitted_path" in result:
        print("fitted_path", result["fitted_path"])
