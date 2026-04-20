# Curve DAB vs Endpoint DAB

## Shared Backbone

- both use `DABResNetBackbone`
- both use `DABEncoder`
- both use the same DAB-style decoder scaffold
- both use the same class head shape: binary logits (`edge` / `no-object`)
- both support DN and auxiliary decoder losses

Code:

- `models/dab_curve_detr.py`
- `models/dab_endpoint_detr.py`

## Training Target / Loss Differences

### Curve DAB

- predicts `pred_curves`
- target is `curves`
- matcher is `HungarianCurveMatcher`
- matched losses:
  - `loss_ce`
  - `loss_chamfer`
- DN regularizer reconstructs full curves:
  - `loss_dn_curve`

Code:

- `models/matcher.py`
- `models/losses/composite.py`
- `models/losses/matched.py`
- `models/losses/regularizers.py`

### Endpoint DAB

- predicts `pred_points`
- target is `points`
- matcher is `HungarianPointMatcher`
- matched losses:
  - `loss_ce`
  - `loss_point`
- DN regularizer reconstructs points:
  - `loss_dn_point`

Code:

- `models/endpoint_matcher.py`
- `models/losses/endpoint_composite.py`
- `models/losses/endpoint_matched.py`
- `models/losses/endpoint_regularizers.py`

## Differences Outside Training Target / Loss

### Model Parameterization

- curve DAB decoder operates on full curve parameters
  - `curve_dim = (target_degree + 1) * 2`
  - reference embedding is one full curve per query
- endpoint DAB decoder operates on one 2D point per query
  - `curve_dim = 2`
  - reference embedding is one point per query

Where:

- `models/dab_curve_detr.py`
- `models/dab_endpoint_detr.py`

### FFN Config

- curve DAB uses one shared `dim_feedforward`
- endpoint DAB allows separate encoder / decoder FFN dims
  - `encoder_dim_feedforward`
  - `decoder_dim_feedforward`

Where:

- `models/dab_curve_detr.py`
- `models/dab_endpoint_detr.py`
- `configs/parametric_edge/laion_pretrain_cluster_dn_aux_ce.yaml`
- `configs/parametric_edge/laion_endpoint_pretrain_dn_aux_ce.yaml`

### Auxiliary Output Shape

- curve DAB aux outputs are `pred_curves`
- endpoint DAB aux outputs are `pred_points`
- endpoint DAB also supports `aux_last_n_layers`

Where:

- `models/dab_curve_detr.py`
- `models/dab_endpoint_detr.py`

### Visualization Callback

- curve DAB uses `ParametricEdgeVisualizer`
- endpoint DAB uses `ParametricEndpointVisualizer`
- visualization target differs:
  - curves vs points

Where:

- `callbacks/training_visualizer.py`
- `callbacks/endpoint_visualizer.py`
- `train.py`

### Dataset Output

- curve DAB dataloader returns curve targets
- endpoint DAB dataloader returns point targets
- current endpoint pipeline derives points from v3 bezier curves; it does not read source-edge

Where:

- `edge_datasets/parametric_edge_dataset.py`
- `edge_datasets/endpoint_dataset.py`
- `edge_datasets/parametric_edge_datamodule.py`
- `edge_datasets/endpoint_datamodule.py`

## What Is Not Different

- optimizer structure
- trainer entrypoint
- logger construction logic
- backbone normalization path
- general DAB query refinement pattern

## Current WandB Status

The current `lab30` endpoint run had WandB disabled for two separate reasons:

1. config disabled it:
   - `configs/parametric_edge/laion_endpoint_pretrain_lab30_v3_2gpu.yaml`
   - `logging.wandb.enabled: true` should be used

2. launcher forced it off:
   - `/home/viplab/jiaxin/parametric_edge_prediction/outputs/tmp_train_lab30_endpoint.sh`
   - remove `export WANDB_MODE=disabled`

So WandB was not failing to initialize; it had been explicitly turned off by config and by environment.
