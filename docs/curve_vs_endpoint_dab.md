# Curve DAB vs Endpoint DAB

This document compares the current maintained curve-style and endpoint-style DAB branches.

## Shared Pieces

Both branches share:

- the same `train.py` entrypoint
- the same config merge logic
- the same datamodule factory / model factory / loss factory pattern
- binary classification logits: `edge` vs `no-object`
- the same logging and checkpoint flow in Lightning
- optional DN and auxiliary losses in the training codebase

Factories:

- `edge_datasets/__init__.py`
- `models/__init__.py`
- `models/losses/__init__.py`

## Curve DAB

Model:

- `models/dab_curve_detr.py`

Targets:

- `targets[i]['curves']`

Dataset path:

- `edge_datasets/parametric_edge_dataset.py`
- `edge_datasets/parametric_edge_datamodule.py`

Matcher:

- `models/matcher.py`

Loss path:

- `models/losses/composite.py`
- `models/losses/matched.py`
- `models/losses/regularizers.py`

Current geometry options:

- chamfer
- EMD / Sinkhorn

Special current detail:

- `model.curve_query_init_type` must be set explicitly
- supported initializers live in `models/curve_query_initializers.py`

## Conditioned Curve DAB

Model:

- `models/dab_cond_curve_detr.py`

Targets:

- `targets[i]['curves']`

Extra model inputs:

- `model_inputs['condition_points']`
- `model_inputs['condition_padding_mask']`

Dataset path:

- `edge_datasets/conditioned_curve_dataset.py`
- `edge_datasets/conditioned_curve_datamodule.py`

Matcher:

- `models/matcher.py`

Loss path:

- `models/losses/composite.py`
- `models/losses/matched.py`
- `models/losses/regularizers.py`

Current special details:

- query / geometry supervision stays the same as direct curve DAB
- every decoder layer adds endpoint-token cross-attention after image-memory cross-attention
- endpoint-token residual output is zero-initialized
- direct curve DAB checkpoints can be used as initialization through
  - `model.conditioned_curve_init_enabled`
  - `model.conditioned_curve_init_checkpoint`

## Endpoint DAB

Model:

- `models/dab_endpoint_detr.py`

Targets:

- `targets[i]['points']`

Dataset path depends on endpoint target mode.

### Point-Only Endpoint Branch

Dataset path:

- `edge_datasets/endpoint_dataset.py`
- `edge_datasets/endpoint_datamodule.py`

Matcher:

- `models/endpoint_matcher.py`

Loss path:

- `models/losses/endpoint_composite.py`
- `models/losses/endpoint_matched.py`
- `models/losses/endpoint_regularizers.py`

### Attach Endpoint Branch

Enabled by:

- `data.endpoint_target_mode: attach`
- `loss.endpoint_loss_type: attach`

Dataset path:

- `edge_datasets/endpoint_attach_dataset.py`
- `edge_datasets/endpoint_attach_datamodule.py`

Extra target fields carried by the attach branch:

- `point_degree`
- `point_is_loop_only`
- `point_curve_offsets`
- `point_curve_indices`
- `curves`

Attach-specific code:

- target construction:
  - `misc_utils/endpoint_target_utils.py`
  - `edge_datasets/graph_pipeline.py`
- attach-aware matching:
  - `models/endpoint_matcher.py`
- attach-aware matched loss:
  - `models/losses/endpoint_matched.py`
- geometry helpers:
  - `models/geometry.py`

## Main Differences

### Prediction Parameterization

Curve DAB:

- each query predicts a full curve
- output geometry is all control points

Conditioned Curve DAB:

- each query still predicts a full curve
- output geometry is all control points
- decoder also sees merged endpoint condition tokens

Endpoint DAB:

- each query predicts a single 2D endpoint
- in attach mode, curves are only GT-side supervision context, not predicted outputs

### Matching

Curve DAB:

- Hungarian matching over curve geometry cost plus edge-probability cost

Conditioned Curve DAB:

- same Hungarian matching as direct curve DAB
- current conditioning does not change matcher semantics

Endpoint DAB:

- Hungarian matching over point distance plus edge-probability cost
- attach mode can also inject distance to the GT endpoint's incident curves into matching

### Matched Geometry Loss

Curve DAB:

- geometry loss is on curves
- current options are chamfer or EMD

Conditioned Curve DAB:

- same curve-side geometry loss path as direct curve DAB
- conditioning currently changes the decoder, not the loss definition

Endpoint DAB:

- base branch uses endpoint point loss
- attach branch adds point-to-incident-curve attach distance
- loop-only and low-degree endpoints are weighted differently from high-degree endpoints

### Dataloader Semantics

Curve DAB:

- loader returns image + curve targets

Conditioned Curve DAB:

- loader returns image + curve targets
- also returns merged endpoint condition points after augment/crop
- batch pads condition points to current batch max and masks the padding

Endpoint DAB:

- loader returns image + merged endpoint targets
- attach mode additionally returns endpoint-to-curve incidence structure

## Differences Outside Loss

These are also different outside the training target itself.

### Current Visualizers

Curve DAB:

- `callbacks/training_visualizer.py`

Endpoint DAB:

- `callbacks/endpoint_visualizer.py`

### Current Debug Scripts

Curve-focused:

- `scripts/visualize_curve_query_initializers.py`
- `scripts/test_curve_emd_experiments.py`

Endpoint-focused:

- `scripts/render_v3_endpoint_dataset_samples.py`
- `scripts/debug_endpoint_attach_opt.py`
- `scripts/profile_v3_endpoint_dataloader.py`

## What Is Not Different

These are still shared.

- Lightning trainer construction in `train.py`
- optimizer / scheduler config path
- checkpointing and WandB setup
- v3 bezier-cache-based dataset source
- FP32-only training rule
