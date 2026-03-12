import numpy as np
import networkx as nx
from collections import Counter, defaultdict
from scipy.spatial import distance_matrix
from skimage.morphology import thin


def build_graph(skeleton, connectivity=2):
    """Convert a skeletonized image to a NetworkX graph."""
    graph = nx.Graph()
    rows, cols = skeleton.shape
    if connectivity == 1:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == 2:
        neighbors = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]
    else:
        raise ValueError('Connectivity must be 1 or 2')

    for y in range(rows):
        for x in range(cols):
            if skeleton[y, x]:
                graph.add_node((y, x))
                for dy, dx in neighbors:
                    ny, nx_ = y + dy, x + dx
                    if 0 <= ny < rows and 0 <= nx_ < cols and skeleton[ny, nx_]:
                        graph.add_edge((y, x), (ny, nx_))
    return graph


def merge_close_subgraphs(graph, dist_threshold=2.0):
    merged_graph = graph.copy()
    components = list(nx.connected_components(graph))
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
                merged_graph.add_edge(closest_pair[0], closest_pair[1])
    return merged_graph


def get_edge_pixel_coords(labeled_image, label):
    return np.argwhere(labeled_image == label)


def measure_edge_distance(coords1, coords2):
    if coords1.size == 0 or coords2.size == 0:
        return np.inf
    dist_mat = distance_matrix(coords1, coords2)
    d1 = dist_mat.min(axis=1).mean()
    d2 = dist_mat.min(axis=0).mean()
    return (d1 + d2) / 2.0


def average_edge(coords1, coords2):
    combined = np.vstack([coords1, coords2])
    return np.unique(combined, axis=0)


def find_start_end_for_label(labeled_image, label, junctions):
    coords = get_edge_pixel_coords(labeled_image, label)
    coords_set = {tuple(c) for c in coords}
    junction_hits = coords_set.intersection(junctions)
    if len(junction_hits) == 2:
        return tuple(sorted(junction_hits))
    return None, None


def merge_duplicate_edges_with_average(labeled_image, junctions, dist_threshold=3.0, background_label=0):
    merged_labeled = labeled_image.copy()
    labels = np.unique(merged_labeled)
    labels = labels[labels != background_label]

    edge_info = {}
    for lbl in labels:
        start_junction, end_junction = find_start_end_for_label(merged_labeled, lbl, junctions)
        if start_junction is None or end_junction is None:
            continue
        edge_info[lbl] = {
            'start': start_junction,
            'end': end_junction,
            'coords': get_edge_pixel_coords(merged_labeled, lbl),
        }

    edges_by_pair = defaultdict(list)
    for lbl, info in edge_info.items():
        edges_by_pair[(info['start'], info['end'])].append(lbl)

    used_labels = set()
    new_label_counter = merged_labeled.max() + 1

    for _, lbl_list in edges_by_pair.items():
        if len(lbl_list) < 2:
            continue
        while lbl_list:
            current_lbl = lbl_list.pop()
            if current_lbl in used_labels:
                continue
            current_coords = edge_info[current_lbl]['coords']
            to_merge = [current_lbl]
            still_unassigned = []
            for other_lbl in lbl_list:
                if other_lbl in used_labels:
                    continue
                other_coords = edge_info[other_lbl]['coords']
                if measure_edge_distance(current_coords, other_coords) < dist_threshold:
                    to_merge.append(other_lbl)
                else:
                    still_unassigned.append(other_lbl)
            lbl_list = still_unassigned

            if len(to_merge) <= 1:
                continue

            all_coords = []
            for lbl_m in to_merge:
                used_labels.add(lbl_m)
                all_coords.append(edge_info[lbl_m]['coords'])
                merged_labeled[merged_labeled == lbl_m] = background_label
            new_coords = average_edge(*all_coords[:2]) if len(all_coords) == 2 else np.unique(np.vstack(all_coords), axis=0)
            new_label = new_label_counter
            new_label_counter += 1
            for r, c in new_coords:
                merged_labeled[r, c] = new_label

    return merged_labeled


def identify_junctions_and_endpoints(graph):
    junctions = [node for node, degree in graph.degree() if degree >= 3]
    endpoints = [node for node, degree in graph.degree() if degree == 1]
    return junctions, endpoints


def trace_edges(graph, junctions, endpoints, height, width):
    labeled_image = np.zeros((height, width), dtype=np.int32)
    label = 1
    visited = set()
    start_nodes = list(set(endpoints + junctions))

    for start_node in start_nodes:
        if start_node in visited:
            continue
        stack = [(start_node, [start_node])]
        while stack:
            current, path = stack.pop()
            visited.add(current)
            for neighbor in graph.neighbors(current):
                if (neighbor in visited) and (neighbor != path[0]):
                    continue
                if (neighbor in junctions) or (neighbor in endpoints):
                    path_extended = path + [neighbor]
                    rr, cc = zip(*path_extended)
                    labeled_image[rr, cc] = label
                    label += 1
                else:
                    stack.append((neighbor, path + [neighbor]))

    cycles = [cycle for cycle in nx.cycle_basis(graph) if len(cycle) > 0]
    for cycle in cycles:
        if any(node in visited for node in cycle):
            continue
        rr, cc = zip(*cycle)
        labeled_image[rr, cc] = label
        label += 1

    return labeled_image


def split_connected_components(edge_map, connectivity=2):
    skeleton = thin(edge_map).astype(np.uint8)
    graph = build_graph(skeleton, connectivity=connectivity)
    junctions, endpoints = identify_junctions_and_endpoints(graph)
    height, width = edge_map.shape[:2]
    labeled_edges = trace_edges(graph, junctions, endpoints, height, width)
    return labeled_edges, junctions, endpoints


def extract_and_count_values(matrix, binary_mask):
    if matrix.shape != binary_mask.shape:
        raise ValueError('The matrix and binary mask must have the same shape.')
    masked_values = matrix[binary_mask > 0]
    value_counts = dict(Counter(masked_values))
    count_values = {v: k for k, v in value_counts.items()}
    return value_counts, count_values


def decide_value_and_confidence(count_values):
    total_counts = np.sum(list(count_values.keys()))
    idx_ascending = np.argsort(list(count_values.keys()))
    max_idx = idx_ascending[-1]
    count = list(count_values.keys())[max_idx]
    value = count_values[count]

    if len(count_values) == 1:
        return value, 3
    if count > 0.6 * total_counts:
        return value, 3

    idx_2 = idx_ascending[-2]
    count2 = list(count_values.keys())[idx_2]
    value2 = count_values[count]
    if count + count2 > 0.9 * total_counts:
        return max(value, value2), 2
    return value, 1
