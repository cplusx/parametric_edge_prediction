# Parametric Edge Training Strategies

This document records the strategies that are still active in the current codebase. It is intentionally limited to the live DAB mainline and no longer documents removed grouped-training mechanisms in detail.

For training entrypoints and operational run rules, see [docs/parametric_edge_experiment_log.md](./parametric_edge_experiment_log.md).

For historical ablations, see [docs/parametric_edge_ablation_log.md](./parametric_edge_ablation_log.md).

## Current Default Stack

The current active stack is:

- `DABCurveDETR`
- `ResNet50` backbone with ImageNet pretrained initialization
- `hidden_dim: 256`
- `nheads: 8`
- `6 encoder / 6 decoder`
- `dim_feedforward: 2048`
- `num_queries: 300`
- focal classification in the main configs
- one-to-one Hungarian matching
- active geometry supervision:
  - `chamfer`
- FP32 training

Code:

- [models/dab_curve_detr.py](../models/dab_curve_detr.py)
- [models/matcher.py](../models/matcher.py)
- [models/losses/matched.py](../models/losses/matched.py)
- [configs/parametric_edge/default.yaml](../configs/parametric_edge/default.yaml)

## What Was Removed

The old legacy stack is no longer part of the active codebase:

- `ParametricDETR`
- grouped matching outputs
- one-to-many supervision
- top-k grouped positives
- distinct grouped regularization

Those mechanisms are now historical only. They should not be treated as part of the current training flow, and new docs should not reference them as active behavior.

## Current Matching And Loss Path

Current mainline matching:

- one-to-one Hungarian matching
- matching costs:
  - `ce`
  - `chamfer`

Current mainline optimization:

- focal classification
- `loss_chamfer`

Inactive by default in the main configs:

- DN
- auxiliary decoder-layer supervision
- all grouped auxiliary objectives

The current mainline is deliberately narrower than the older stack. That is intentional: the repo was cleaned up to keep only the path that survived overfit debugging.

## Current Curve Supervision

Current behavior:

- predicted endpoints + control points are rendered into sampled curves
- ground-truth curves are sampled the same way
- Hungarian matching uses:
  - classification cost
  - sampled curve Chamfer cost
- optimization uses:
  - `loss_ce`
  - `loss_chamfer`
- DN stays on plain L1 curve reconstruction; it does not use Chamfer

Code:

- [models/geometry.py](../models/geometry.py)
- [models/matcher.py](../models/matcher.py)
- [models/losses/matched.py](../models/losses/matched.py)
- [models/losses/regularizers.py](../models/losses/regularizers.py)

## Curve Coordinate Mapping

Current behavior:

- External curve coordinates live in `[-0.1, 1.1]`
- Matching and loss use an internal `[0, 1]` space
- Visualization maps predictions back to external coordinates

Code:

- [models/curve_coordinates.py](../models/curve_coordinates.py)
- [models/matcher.py](../models/matcher.py)
- [models/losses/matched.py](../models/losses/matched.py)
- [callbacks/training_visualizer.py](../callbacks/training_visualizer.py)

Why it stays:

- Keeps DAB-style sigmoid refinement numerically well-behaved
- Still allows endpoints and control points to represent slightly out-of-frame geometry

## DAB Query Notes

Current query path:

- learnable curve queries
- DAB-style iterative curve refinement
- `q_modulated` currently remains a conservative placeholder (`sine_proj`), not a final structured modulation design

Code:

- [models/dab_curve_detr.py](../models/dab_curve_detr.py)

Practical interpretation:

- The DAB scaffold is in place.
- The query modulation branch is still a safe baseline, not the final task-specific design.

## Optional DN Path

DN support still exists inside `DABCurveDETR`, but it is optional and not part of the stripped default mainline unless turned back on in config.

Current behavior when enabled:

- denoising queries are prepended only during training
- inference keeps only the matching queries
- decoder self-attention is masked so matching queries do not attend to denoising queries

Code:

- [models/dab_curve_detr.py](../models/dab_curve_detr.py)
- [models/losses/regularizers.py](../models/losses/regularizers.py)

## Current Performance Notes

The heaviest remaining training-side work is still:

- Hungarian matching
- pairwise curve cost construction
- backward through the DAB encoder/decoder stack

The current native C++ helper remains useful only for no-grad matcher-side geometry kernels.

Most relevant files:

- [models/matcher.py](../models/matcher.py)
- [models/geometry.py](../models/geometry.py)
- [models/native_cost_cpp.py](../models/native_cost_cpp.py)

## 2026-03-26 Cleanup Note

The docs were rewritten to match the cleaned codebase:

- active model path is DAB-only
- old grouped auxiliary stack is no longer documented as active
- stale references to `models/parametric_detr.py`, grouped outputs, and one-to-many/top-k/distinct were removed from the main docs
- the old grouped loss-flow diagram was dropped because it no longer described the active training flow
