# Current Repo Structure

This is the current layout of the active codebase.

## Top-Level Modules

- `sam_mask_bezierization/`
  - mask-to-edge and edge-to-bezier pipeline code used after SAM2 masks are available
  - current core files:
    - `pipeline.py`
    - `mask_to_edge_methods.py`
    - `bspline.py`
- `scripts/`
  - operational scripts for preprocessing, profiling, dataset rendering, syncing, and submit helpers
- `edge_datasets/`
  - datamodules and datasets built on top of v3 bezier caches
- `models/`
  - curve DAB, endpoint DAB, matchers, geometry, curve-distance code, and loss code
- `callbacks/`
  - training visualizers for curves and endpoints
- `configs/parametric_edge/`
  - maintained training configs
- `tests/`
  - lightweight correctness tests for new utilities
- `bezierization/`
  - older bezier experimentation area; not the main preprocessing entry anymore
- `experiments/`
  - local scratch / historical experiments; not part of the maintained mainline

## Current Active Training Models

- `models/dab_curve_detr.py`
  - curve prediction model
- `models/dab_cond_curve_detr.py`
  - endpoint-conditioned curve prediction model
- `models/dab_endpoint_detr.py`
  - endpoint prediction model
- `models/curve_query_initializers.py`
  - curve-query initialization strategies
  - `model.curve_query_init_type` must now be set explicitly

Model factory:

- `models/__init__.py`

Supported `model.arch` values:

- `dab_curve_detr`
- `dab_cond_curve_detr`
- `dab_endpoint_detr`

## Current Dataset Entry Points

Datamodule selection lives in:

- `edge_datasets/__init__.py`

Current branches:

- curve branch
  - `edge_datasets/parametric_edge_dataset.py`
  - `edge_datasets/parametric_edge_datamodule.py`
- conditioned-curve branch
  - `edge_datasets/conditioned_curve_dataset.py`
  - `edge_datasets/conditioned_curve_datamodule.py`
- endpoint point-only branch
  - `edge_datasets/endpoint_dataset.py`
  - `edge_datasets/endpoint_datamodule.py`
- endpoint attach branch
  - `edge_datasets/endpoint_attach_dataset.py`
  - `edge_datasets/endpoint_attach_datamodule.py`

Shared curve/augmentation/crop preparation:

- `edge_datasets/graph_pipeline.py`

## Current Loss / Matching Entry Points

Loss selection lives in:

- `models/losses/__init__.py`

Current branches:

- curve branch
  - matcher: `models/matcher.py`
  - losses: `models/losses/composite.py`, `models/losses/matched.py`, `models/losses/regularizers.py`
- conditioned-curve branch
  - matcher: `models/matcher.py`
  - losses: `models/losses/composite.py`, `models/losses/matched.py`, `models/losses/regularizers.py`
- endpoint branch
  - matcher: `models/endpoint_matcher.py`
  - losses: `models/losses/endpoint_composite.py`, `models/losses/endpoint_matched.py`, `models/losses/endpoint_regularizers.py`

Shared geometry helpers:

- `models/geometry.py`
- `models/curve_distances.py`
- `models/curve_coordinates.py`

## Current Preprocessing Scripts

Source-edge generation from cached SAM2 masks:

- `scripts/generate_source_edges_from_sam2_batch.py`

Bezierization from source-edge PNGs:

- `scripts/generate_bezier_from_source_edges_batch.py`

Progress / sync helpers:

- `scripts/check_source_edge_progress.py`
- `scripts/check_bezierization_progress.py`
- `scripts/sync_completed_results_to_lab30.py`
- `scripts/sync_git_lab30_cluster.py`
- `scripts/rewrite_bezier_entry_cache.py`

## Current Training / Debug Scripts

Training submit helpers:

- `scripts/submit_jobs/submit_curve_dab_cluster_train.py`
- `scripts/submit_jobs/submit_curve_dab_cluster_train_line.py`
- `scripts/submit_jobs/submit_curve_dab_cluster_train_emd_edgeprob05.py`
- `scripts/submit_jobs/submit_curve_dab_cluster_train_emd_line.py`
- `scripts/submit_jobs/submit_curve_dab_lab34_edgeprob05.py`
- `scripts/submit_jobs/submit_endpoint_attach_cluster_train.py`
- `scripts/submit_jobs/submit_curve_dab_overfit8_cluster.py`

Dataset / geometry debug helpers:

- `scripts/forward_conditioned_curve_once.py`
- `scripts/render_conditioned_curve_dataset_samples.py`
- `scripts/render_v3_endpoint_dataset_samples.py`
- `scripts/render_compact_bezier_preview.py`
- `scripts/profile_v3_dataloader_setup.py`
- `scripts/profile_v3_endpoint_dataloader.py`
- `scripts/debug_endpoint_attach_opt.py`
- `scripts/visualize_curve_query_initializers.py`
- `scripts/test_curve_emd_experiments.py`

## Current Config Families

Maintained configs under `configs/parametric_edge/`:

- curve DAB, cluster, random init
  - `laion_curve_pretrain_cluster_v3_2gpu.yaml`
- curve DAB, cluster, line init
  - `laion_curve_pretrain_cluster_v3_2gpu_line.yaml`
- curve DAB, cluster, EMD variants
  - `laion_curve_pretrain_cluster_v3_2gpu_emd_edgeprob05.yaml`
  - `laion_curve_pretrain_cluster_v3_2gpu_emd_line.yaml`
- curve DAB, lab machines
  - `laion_curve_pretrain_lab30_v3_2gpu.yaml`
  - `laion_curve_pretrain_lab34_v3_edgeprob05.yaml`
- conditioned curve DAB, lab34
  - `laion_conditioned_curve_pretrain_lab34_v3_2gpu.yaml`
- endpoint DAB, point-only
  - `laion_endpoint_pretrain_lab34_v3_2gpu.yaml`
- endpoint DAB, attach branch
  - `laion_endpoint_attach_lab34_v3_2gpu.yaml`
  - `laion_endpoint_attach_cluster_v3_2gpu.yaml`

Shared defaults:

- `configs/parametric_edge/default.yaml`

Current lab34 launchers:

- `run_lab34.sh`
- `run_lab34_endpt.sh`
- `run_lab34_conditioned_curve.sh`

## What Is Not Current

These are not the maintained mainline anymore:

- stale machine-status snapshots
- old architecture diagrams that describe removed internals
- docs that mention removed training stacks as if they are current
- using source-edge directly as training input

The maintained training path now starts from v3 bezier caches.
