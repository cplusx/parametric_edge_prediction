# Parametric Edge Training Strategies

This document records the training-time strategies currently used to make the DETR-style parametric edge detector trainable, why each strategy was introduced, where it comes from, and what effect it had in practice.

For experiment history, rerun commands, and ablation bookkeeping, see [docs/parametric_edge_ablation_log.md](./parametric_edge_ablation_log.md).

For a visual overview of how main queries, DN queries, and decoder refinement fit together, see [docs/query_flow_diagram.md](./query_flow_diagram.md).

## Current Default Stack

Current default configuration now uses:

- RGB image input from `edge_data/HED-BSDS/images/test`
- Parametric targets from `edge_data/HED-BSDS/gt_rgb/test`
- `DINOv2` vision transformer backbone
- FPN-like multi-scale feature pyramid built from DINOv2 intermediate features
- two-stage proposal generation from encoded memory tokens
- deformable-style multi-scale decoder cross-attention
- iterative reference-point refinement
- denoising queries enabled
- focal classification loss enabled

Code:

- [configs/parametric_edge/default.yaml](../configs/parametric_edge/default.yaml)
- [models/parametric_detr.py](../models/parametric_detr.py)
- [models/losses.py](../models/losses.py)

Why:

- The earlier `ResNet + vanilla DETR decoder` stack was sufficient for synthetic overfit diagnosis, but it was missing the backbone and query/proposal machinery that modern DETR systems rely on for stable training.

## RGB Training Input

Current implementation:

- Default training now consumes RGB images from `edge_data/HED-BSDS/images/test`.
- Edge supervision still comes from Bezierized annotations in `edge_data/HED-BSDS/gt_rgb/test`.

Why:

- Training directly on the edge map was useful for debugging regression and matching, but it is not a real detector setting.
- Formal training needs image-to-curve prediction, not edge-to-curve reconstruction.

Effect:

- The default config is now aligned with the intended downstream task.

## DINOv2 Backbone

Current implementation:

- Backbone switched from a small CNN to `DINOv2` via `timm`.
- Intermediate ViT block features are extracted and fused into a feature pyramid.
- Code:
  - [models/parametric_detr.py](../models/parametric_detr.py), `DINOv2PyramidBackbone`

Motivation:

- Strong DETR variants rely heavily on stronger pretrained backbones.
- The earlier ResNet-18-from-scratch setup was too weak to be a realistic training baseline.

Effect:

- Better semantic memory features for proposal generation and decoder attention.

References:

- DINOv2
  - https://github.com/facebookresearch/dinov2
- timm model implementations
  - https://github.com/huggingface/pytorch-image-models

## FPN-Like Multi-Scale Pyramid from ViT Features

Current implementation:

- DINOv2 intermediate features are projected to the model hidden dimension and fused into a base map.
- Additional lower-resolution levels are created by learned stride-2 downsampling blocks.

Motivation:

- ViT intermediate features have a single native patch resolution.
- Deformable DETR-style decoding needs multiple spatial scales.

Effect:

- Enables multi-scale proposal generation and multi-scale deformable cross-attention without requiring a CNN backbone.

Source:

- Internal implementation choice motivated by Deformable DETR's multi-scale memory design.

## Two-Stage Proposal Generation

Current implementation:

- Encoder memory tokens predict proposal scores and proposal reference points.
- Top-scoring proposals become decoder queries.
- Code:
  - [models/parametric_detr.py](../models/parametric_detr.py), `_build_main_queries()`

Motivation:

- Query generation should come from encoded image evidence, not only from learned slots or a uniform grid.
- This is closer to the proposal-driven query selection used by stronger DETR variants.

Effect:

- Better query initialization than pure learned queries or untrained score-only top-k.

References:

- Deformable DETR
  - https://arxiv.org/abs/2010.04159
- RT-DETR
  - https://arxiv.org/abs/2304.08069

## Deformable-Style Multi-Scale Decoder

Current implementation:

- Decoder layers use:
  - query self-attention
  - custom multi-scale deformable-style cross-attention via sparse sampling around reference points
  - FFN
- Code:
  - [models/parametric_detr.py](../models/parametric_detr.py), `MultiScaleDeformableAttention`
  - [models/parametric_detr.py](../models/parametric_detr.py), `DeformableDecoderLayer`

Motivation:

- Standard dense cross-attention is not the dominant design anymore for strong DETR training.
- Sparse sampling around reference points is a better fit for localized geometric structures like curves.

Effect:

- Brings the decoder closer to the design pattern that made Deformable DETR practical.

Reference:

- Deformable DETR
  - https://arxiv.org/abs/2010.04159

## Denoising Queries Enabled Again

Current implementation:

- DN is enabled again in the default config.
- Positive denoising curves are generated from GT curves and injected ahead of main queries.

Motivation:

- Once the base query/proposal path is stable, DN becomes useful again.
- It helps convergence and stabilizes early training of the decoder.

Effect:

- Better aligned with modern DETR training practice than the earlier no-DN debug baseline.

Reference:

- DINO
  - https://arxiv.org/abs/2203.03605

## Focal Classification Loss

Current implementation:

- Classification loss now supports focal mode and the default config uses it.
- Code:
  - [models/losses.py](../models/losses.py), `_classification_loss()`

Motivation:

- Query-heavy set prediction creates strong foreground/background imbalance.
- Focal loss is better suited than plain cross-entropy when suppressing false positives is a priority.

Effect:

- Improves score calibration pressure without changing the geometry heads.

## Coordinate Normalization

Current implementation:

- Input images are resized to a fixed square size and normalized to `[0, 1]`.
  - Code: [misc_utils/bezier_target_utils.py](../misc_utils/bezier_target_utils.py), `load_image_array()`
- Bezier control points are stored in normalized XY coordinates in `[0, 1]`, not raw pixel coordinates.
  - Code: [misc_utils/bezier_target_utils.py](../misc_utils/bezier_target_utils.py), `normalize_control_points()`
- The normalization uses per-image scale:
  - `x /= max(width - 1, 1)`
  - `y /= max(height - 1, 1)`
- Visualization denormalizes back to pixel space only for plotting.
  - Code: [misc_utils/visualization_utils.py](../misc_utils/visualization_utils.py), `draw_curve()`

Why:

- DETR-style set prediction is much easier to optimize when geometry lives in a fixed coordinate range.
- This avoids scale mismatch across images and avoids overly large regression targets.
- This also matches the design pattern of DETR-like detectors, where box centers, widths, heights, or reference points are commonly represented in normalized image coordinates.

Relation to prior work:

- DETR predicts boxes in normalized image coordinates and uses sigmoid-constrained outputs.
- DAB-DETR formulates decoder queries directly with box coordinates / reference points.
- Anchor DETR uses anchor points so each query focuses on a spatially meaningful region.

References:

- DETR: End-to-End Object Detection with Transformers
  - https://arxiv.org/abs/2005.12872
- DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR
  - https://arxiv.org/abs/2201.12329
- Anchor DETR: Query Design for Transformer-Based Object Detection
  - https://arxiv.org/abs/2109.07107

## Base Representation

Current implementation:

- Training targets use fixed `max_degree=5`, so each curve is represented by `6 x 2` control points.
- Original Bezierized outputs can have varying degree, but training re-fits each segment into degree-5 for a fixed output dimension.

Why:

- DETR heads need a fixed output shape per query.
- Degree-5 is a better approximation to the original Bezierization than cubic, while still keeping the output head simple.

Effect:

- Preserves more geometric fidelity than the earlier cubic-only training target.

## GroupNorm Backbone

Current implementation:

- Backbone BatchNorm is replaced by GroupNorm.
  - Code: [models/parametric_detr.py](../models/parametric_detr.py), `replace_batch_norm_with_group_norm()`

Motivation:

- Overfit experiments are run with batch size `1`.
- BatchNorm caused train/eval mismatch and poor single-sample memorization.

Effect:

- Train/val loss became consistent in single-sample overfit.
- Removed a major instability unrelated to the set-prediction formulation itself.

Source:

- This is not a DETR-specific invention here; it is a practical engineering fix for tiny batch training.

## No-DN Baseline for Overfit Diagnosis

Current implementation:

- Denoising queries are disabled in the key overfit configs used for diagnosis.

Motivation:

- DN improves convergence in many DETR settings, but it is an extra variable when debugging whether the model can memorize a single sample at all.
- For single-sample diagnosis, simpler is better.

Effect:

- Made it easier to isolate whether failure came from the core query/matching formulation.

Source:

- Motivation is diagnostic clarity, not a claim that DN should be removed from final large-scale training.
- Related DETR-family reference:
  - DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection
  - https://arxiv.org/abs/2203.03605

## Curvature-Aware Local Weighting

Current implementation:

- Curve curvature is computed from the source polyline.
- Optional weighting can emphasize:
  - sampled-curve loss
  - endpoint loss
- Code:
  - [misc_utils/bezier_target_utils.py](../misc_utils/bezier_target_utils.py), `polyline_curvature_score()`
  - [models/losses.py](../models/losses.py)

Motivation:

- Curvier segments were empirically harder to fit than flatter ones.

Effect:

- Global curvature weighting was too aggressive and hurt overall fit.
- The current lighter version is retained as an option, but it is not the main reason high-query training became stable.

Source:

- Internal heuristic based on observed failure modes.

## One-to-Many / Grouped Supervision

Current implementation:

- Added repeated-target Hungarian matching to allow more than one query to learn from the same GT curve.
- Code:
  - [models/matcher.py](../models/matcher.py), `repeated_hungarian_curve_matching()`
  - [models/losses.py](../models/losses.py), `_one_to_many_losses()`

Motivation:

- One-to-one matching leaves very sparse positive supervision when query count is much larger than the number of usable matches early in training.
- This is especially harmful on complex edge maps where many queries otherwise drift into `no-object` or collapse together.

Effect:

- By itself, this was not sufficient to fix high-query collapse.
- It became useful once combined with better query selection and anchored decoding.

References:

- Group DETR
  - https://arxiv.org/abs/2207.13085
- H-DETR
  - https://arxiv.org/abs/2207.13080

## Encoder Top-k Query Selection

Current implementation:

- Queries can be selected from encoder memory tokens using learned scores.
- Code:
  - [models/parametric_detr.py](../models/parametric_detr.py), `query_source='encoder_topk'`

Motivation:

- Pure learned query slots with many candidates were unstable in overfit.
- Prior DETR work often improves training by selecting stronger encoder-side proposals or query candidates.

Effect:

- Better aligned with later DETR-family designs than pure learned slots.
- Still not enough by itself; queries tended to cluster into the same local region.

References:

- Deformable DETR
  - https://arxiv.org/abs/2010.04159
- RT-DETR
  - https://arxiv.org/abs/2304.08069

## Diverse Top-k Query Gating

Current implementation:

- `encoder_diverse_topk` partitions encoder spatial coordinates into a grid and selects top-scoring queries in a round-robin way across cells.
- Code:
  - [models/parametric_detr.py](../models/parametric_detr.py), `_select_diverse_topk_indices()`

Motivation:

- Plain top-k still selected many queries from the same high-response region.
- For edge detection, this causes local collapse: many queries gather around one object or one corner of the image.

Intuition:

- Not all high scores are equally useful if they come from the same place.
- Forcing spatial diversity at query selection time gives the decoder a better distributed candidate set.

Effect:

- Prevented early-stage query clustering.
- On the large-sample overfit experiment, this changed behavior from "single local blob" to "coverage across the full image."

Source:

- Internal strategy motivated by the same spatial-reference logic used by Anchor DETR / proposal-based DETR variants.
- Not taken from a single paper verbatim.

## Uniform Grid Query Initialization

Current implementation:

- Added `encoder_uniform_grid` query sourcing.
- It builds stratified-jittered reference points over the full image and, for each reference point, selects the nearest encoder token.
- Code:
  - [models/parametric_detr.py](../models/parametric_detr.py), `_build_query_reference_points()`
  - [models/parametric_detr.py](../models/parametric_detr.py), `_select_uniform_reference_indices()`

Motivation:

- A learned or random score head is not trustworthy at `epoch 0`.
- If active queries are selected by an untrained score, they often collapse into one local region immediately.

Intuition:

- At initialization, it is better to guarantee spatial coverage first and let learning specialize later.
- Stratified jitter gives near-uniform coverage without forcing a perfectly rigid lattice.

Effect:

- The `epoch 0` prediction pattern changes from a single central cluster to roughly uniform full-image coverage.

Source:

- Internal strategy motivated by proposal sampling / anchor coverage ideas, not copied from one paper directly.

## Reference-Point Anchored Curve Decoding

Current implementation:

- Query selection returns an explicit reference point per query.
- Curve head predicts bounded offsets around that reference point instead of unconstrained global coordinates.
- Code:
  - [models/parametric_detr.py](../models/parametric_detr.py), `curve_anchor_mode='ref_point_delta'`

Motivation:

- Even with better query selection, unconstrained sigmoid curve regression let different queries drift to the same region.
- The model needed an explicit spatial starting point in output space, not just in query embedding space.

Intuition:

- This is the curve analogue of anchor/reference-point box decoding in DETR variants.
- A query should search around a meaningful local reference, not re-discover the whole image from scratch.

Effect:

- This was the key step that stopped high-query overfit from collapsing to one region.
- It made predictions spread over the image and reduced large-sample overfit loss substantially.

References:

- DAB-DETR
  - https://arxiv.org/abs/2201.12329
- Anchor DETR
  - https://arxiv.org/abs/2109.07107

## Iterative Reference-Point Refinement

Current implementation:

- Each decoder layer predicts a delta on the current reference point.
- The refined reference point is passed to the next decoder layer.
- Code:
  - [models/parametric_detr.py](../models/parametric_detr.py), `inverse_sigmoid()`
  - [models/parametric_detr.py](../models/parametric_detr.py), `self.ref_point_heads`
  - [models/parametric_detr.py](../models/parametric_detr.py), `forward()`

Motivation:

- A single-shot anchored decode still leaves the whole geometric burden on the last layer.
- DAB-style refinement lets the decoder iteratively move a query toward the right spatial support.

Effect:

- Overfit became much stronger on the large-sample case.
- The model learned to place and extend long curves more coherently instead of keeping everything as short local fragments.

References:

- DAB-DETR
  - https://arxiv.org/abs/2201.12329

## Endpoint + Interior Offsets Parameterization

Current implementation:

- The curve head no longer treats all control points equally.
- It predicts:
  - endpoint offsets around the reference point
  - interior control points as offsets around the line interpolated between the predicted endpoints
- Code:
  - [models/parametric_detr.py](../models/parametric_detr.py), `_decode_to_output()`

Motivation:

- Directly regressing all control points makes it too easy for the model to output short template-like curves.
- Many edge segments are better described as:
  - where the segment starts
  - where it ends
  - how the middle bends between them

Intuition:

- This factorization is closer to how a human would sketch a curve.
- It also biases the model toward longer, better-ordered shapes.

Effect:

- Improved early geometric structure.
- Worked particularly well together with iterative reference refinement.

Source:

- Internal parameterization choice motivated by the observed failure mode of repeated short curves.

## Top-k Positive Supervision

Current implementation:

- In addition to one-to-one matching and one-to-many repeated Hungarian, a top-k nearest-query positive objectness loss is added.
- Code:
  - [models/matcher.py](../models/matcher.py), `topk_curve_positive_indices()`
  - [models/losses.py](../models/losses.py), `_topk_positive_object_loss()`

Motivation:

- Even one-to-many matching still leaves many potentially useful nearby queries unsupervised.
- For dense curve prediction, several nearby queries can be geometrically plausible positives in early training.

Effect:

- Helped stabilize objectness learning for nearby candidate queries.
- Reduced the tendency for only a tiny subset of queries to carry all the gradient signal.

Source:

- Internal strategy inspired by the broader one-to-many supervision idea in Group DETR / H-DETR.

## Current Large-Query Overfit Takeaway

Observed on `100007_ann5.png` with `69` target curves:

- `encoder_topk + one-to-many`:
  - still collapsed locally
  - best val loss stayed around `3.21`
- `encoder_diverse_topk + one-to-many + reference-point anchored decoding`:
  - no longer collapsed to one region
  - best val loss dropped to about `1.27`

Interpretation:

- The bottleneck was not primarily network depth.
- The bottleneck was the combination of:
  - sparse positives
  - poor spatial allocation of queries
  - globally unconstrained curve regression

## Next Candidate Strategy

The next likely improvement is an anchor-aware curve length prior:

- long target curves should be easier to express from a local anchor
- current anchored decoding tends to produce spatially distributed but still too-short segments

Motivation:

- After fixing collapse, the remaining failure mode is under-extended curves rather than total spatial misallocation.
