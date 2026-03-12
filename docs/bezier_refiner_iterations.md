# Bezier Refiner Iterations

## Metric Definitions
- `short20`: number of fitted Bezier segments whose own arclength is `<= 3 px`, while the original parent polyline arclength is `>= 20 px`.
- `short40`: same definition, but the parent polyline arclength threshold is `>= 40 px`.

Interpretation:
- these are not generic short curves;
- they are specifically the suspicious `trivial curve` cases where a longer meaningful edge produced a tiny `1-3 px` Bezier segment after refinement.

## File Layout
- [original_edge_refiner.py](/Users/jiaxincheng/Desktop/expts/parametric_edge_prediction/original_edge_refiner.py): original graph/junction/edge-splitting code without Bezier fitting.
- [bezier_refiner_core.py](/Users/jiaxincheng/Desktop/expts/parametric_edge_prediction/bezier_refiner_core.py): shared Bezier fitting implementation.
- `bezier_versions/`: ablation versions with a unified `run_refiner(image_path, output_dir=None, **overrides)` interface.
- [edge_refiner.py](/Users/jiaxincheng/Desktop/expts/parametric_edge_prediction/edge_refiner.py): compatibility entrypoint for the current default version, currently `v5_anchor_consistent`.
- [ablation_api.py](/Users/jiaxincheng/Desktop/expts/parametric_edge_prediction/ablation_api.py): version loader and unified runtime API.

## Unified Interface
Every Bezier version file exports:
- `VERSION_NAME`
- `DESCRIPTION`
- `DEFAULT_CONFIG`
- `run_refiner(image_path, output_dir=None, **overrides)`

Example:
```python
from ablation_api import run_version

result = run_version(
    'v5_anchor_consistent',
    image_path='gt_rgb/test/2018_ann1.png',
    output_dir='outputs/demo_v5',
)
```

Canonical version names:
- `v1_greedy_hybrid`
- `v2_tiny_cleanup`
- `v3_trivial_pruned`
- `v4_adjacent_merge`
- `v5_anchor_consistent`

Backward-compatibility aliases remain available:
- `v1_hybrid_initial`
- `v2_dense_sampling`
- `v3_trivial_filter`
- `v4_current`
- `v5_human_split`

## Version History

### Original Baseline
- File: [original_edge_refiner.py](/Users/jiaxincheng/Desktop/expts/parametric_edge_prediction/original_edge_refiner.py)
- Observation:
  - only extracts graph/junctions/endpoints and labels edge segments;
  - no Bezier representation exists yet.
- Technique:
  - `thin -> graph -> junction/endpoints -> trace_edges`
- Use case:
  - baseline for comparing graph extraction quality before any curve fitting.

### V1: Greedy Hybrid
- File: [bezier_versions/v1_hybrid_initial.py](/Users/jiaxincheng/Desktop/expts/parametric_edge_prediction/bezier_versions/v1_hybrid_initial.py)
- Observation:
  - a single Bezier cannot fit very long or highly curved paths well;
  - greedy piecewise fitting is needed.
- Technique:
  - pre-split by corner candidates and long arclength;
  - greedy longest-prefix Bezier fitting;
  - no trivial-path filtering;
  - no adjacent-curve post-merge.
- Known failure:
  - many trivial `1-3 px` curves survive.

### V2: Tiny Cleanup
- File: [bezier_versions/v2_dense_sampling.py](/Users/jiaxincheng/Desktop/expts/parametric_edge_prediction/bezier_versions/v2_dense_sampling.py)
- Observation:
  - some errors were just rasterization gaps rather than fitting gaps;
  - some tiny leftovers appeared at the ends of longer paths.
- Technique:
  - adaptive Bezier sampling and line-connected rasterization;
  - tiny-segment cleanup on longer paths.
- Known failure:
  - trivial original paths from graph splitting still pass through into Bezier fitting.

### V3: Trivial Pruned
- File: [bezier_versions/v3_trivial_filter.py](/Users/jiaxincheng/Desktop/expts/parametric_edge_prediction/bezier_versions/v3_trivial_filter.py)
- Observation:
  - many bad short curves were not fitting artifacts;
  - the original ordered polylines were already only `1-3 px` long.
- Technique:
  - skip fitting original paths whose arclength is below `min_path_length_for_bezier`;
  - remove terminal/intermediate tiny leftovers on longer paths.
- Improvement:
  - removes a large number of trivial curves, especially on samples like `51084_ann1`.

### V4: Adjacent Merge
- File: [bezier_versions/v4_current.py](/Users/jiaxincheng/Desktop/expts/parametric_edge_prediction/bezier_versions/v4_current.py)
- Observation:
  - even after filtering trivial original paths, some neighboring curves were still unnecessarily split;
  - visible examples include adjacent segments that can be merged with low additional error.
- Technique:
  - keeps V3 filtering;
  - adds `merge_easy_adjacent_segments()` to merge neighboring curves when the merged fit stays simple enough.
- Current targeted behavior:
  - avoid trivial curves;
  - avoid obviously unnecessary curve splits;
  - keep interfaces stable for future ablation.

### V5: Anchor Consistent
- File: [bezier_versions/v5_human_split.py](/Users/jiaxincheng/Desktop/expts/parametric_edge_prediction/bezier_versions/v5_human_split.py)
- Observation:
  - `v2-v4` can still split long smooth cable-like paths at mechanically convenient positions rather than where a human would naturally place boundaries;
  - tightening the fit thresholds alone does not fix this because the main bias is earlier in the split proposal stage.
- Technique:
  - natural-anchor scoring from extrema and strong turns;
  - anchor-aware length pre-splitting so long spans prefer nearby meaningful anchors instead of hard arc-length cuts;
  - per-chunk global dynamic programming instead of pure endpoint-greedy longest-prefix fitting;
  - short-segment penalty inside the DP objective to avoid unnatural leftover fragments;
  - smooth-path consistency for long open low-curvature paths;
  - lightweight bundle consistency for similar smooth paths inside the same image.
- Intended behavior:
  - keep the trivial-curve protections from `v3-v4`;
  - make split placement more intuitive on cable-like structures;
  - reduce visually arbitrary split locations even when final F1 changes only slightly.

## Current Key Heuristics
- `min_path_length_for_bezier = 6.0`
  - trivial original paths below this are dropped before fitting.
- `tiny_segment_length = 3.0`
  - fitted segments below this are treated as suspicious tiny leftovers.
- `cleanup_long_path_threshold = 20.0`
  - tiny leftovers are only aggressively cleaned when the parent path is meaningfully long.
- `merge_easy_adjacent_segments`
  - merges adjacent curves when merged max/mean error remain low.
- natural-anchor-aware split placement
  - prefers local extrema / strong turns over mechanically uniform length cuts on long smooth spans.
- smooth/bundle consistency
  - regularizes long open low-curvature paths so similar structures receive more stable split patterns.

## Notes On Interpretation
- High `short20` or `short40` is usually a stronger signal for bad training targets than a small drop in F1.
- Some F1/chamfer regressions are acceptable if they come from intentionally removing trivial curves that should not become trainable Bezier targets.
- Visual comparisons should now use the functional version names rather than status names like `current`.
