# Parametric Edge Training Strategies

This document records the training-time strategies currently used to make the DETR-style parametric edge detector trainable, why each strategy was introduced, where it comes from, and what effect it had in practice.

For active training entrypoints, run conventions, and non-ablation experiment notes, see [docs/parametric_edge_experiment_log.md](./parametric_edge_experiment_log.md).

For ablation history and rerun bookkeeping, see [docs/parametric_edge_ablation_log.md](./parametric_edge_ablation_log.md).

For a visual overview of how main queries, DN queries, and decoder refinement fit together, see [docs/query_flow_diagram.md](./query_flow_diagram.md).
For a visual overview of how decoder outputs feed into the current loss stack, see [docs/loss_flow_diagram.md](./loss_flow_diagram.md).

## Current Default Stack

Current default configuration now uses:

- LAION synthetic pretraining as the current default training entrypoint
- RGB image input from LAION image roots with precomputed Bezier graph caches
- geometry-focused loss weights with no separate bbox regression term and no separate extent branch/loss
- `DINOv2` vision transformer backbone
- FPN-like multi-scale feature pyramid built from DINOv2 intermediate features
- two-stage proposal generation from encoded memory tokens
- grouped queries now draw from a larger proposal pool; the main group keeps the strongest proposal chunk, while auxiliary groups sample randomized non-overlapping chunks from the remaining proposal pool instead of copying the same anchors to every group
- curve matching and geometry supervision are now direction-invariant: for each predicted/target curve pair, forward and reversed target orderings are both evaluated and the lower-cost orientation is used
- deformable-style multi-scale decoder cross-attention
- iterative reference-point refinement
- denoising queries enabled
- focal classification loss enabled
- cluster-oriented 4-GPU FP32 training for 200 epochs, with a 2-GPU fallback config
- periodic validation visualization plus training-set visualization every 2000 iterations in the current LAION default
- auxiliary decoder supervision computed on a strided subset of intermediate layers (`aux_layer_stride: 2`)

Code:

- [configs/parametric_edge/default.yaml](../configs/parametric_edge/default.yaml)
- [configs/parametric_edge/bsds_formal.yaml](../configs/parametric_edge/bsds_formal.yaml)
- [models/parametric_detr.py](../models/parametric_detr.py)
- [models/losses.py](../models/losses.py)

Why:

- The earlier `ResNet + vanilla DETR decoder` stack was sufficient for synthetic overfit diagnosis, but it was missing the backbone and query/proposal machinery that modern DETR systems rely on for stable training.

## TODO

- Add a lean loss-stack ablation config that keeps only main matching + DN + one-to-many, and compare it against the current full stack with `topk_positive` and `distinct` enabled.
- Revisit whether `dynamic_class_balance` is still needed when focal loss is already active, especially if score calibration remains unstable.
- Consider reducing the runtime cost of matching/loss by moving the heaviest pairwise geometry kernels out of Python, starting with proposal-cost construction and Hungarian-side preprocessing.

## Direction-Invariant Curve Comparison

Current implementation:

- Parameterized curves are treated as direction-invariant during both matching and optimization.
- For a predicted curve `P` and target curve `T`, the system now evaluates:
  - forward ordering: `P` vs `T`
  - reversed ordering: `P` vs `reverse(T)`
- The lower-cost orientation is used.

Where it applies:

- Hungarian matching cost construction
  - control-point cost
  - sampled-point cost
- Matched geometry supervision
  - `loss_ctrl`
  - `loss_endpoint`
  - `loss_sample` / `loss_curve_dist`
  - downstream `giou` uses the same chosen orientation

Code:

- [models/geometry.py](../models/geometry.py)
- [models/matcher.py](../models/matcher.py)
- [models/losses/matched.py](../models/losses/matched.py)

Motivation:

- The same geometric edge can be parameterized in either endpoint order.
- Without orientation alignment, the model is penalized for a purely representational reversal rather than a real geometric error.

Effect:

- Matching is no longer biased against reversed-but-geometrically-correct curves.
- Optimization pressure is better aligned with the actual visual geometry of the edge set.

## Training Performance Opportunities

Current practical bottlenecks:

- Hungarian matching still depends on `scipy.optimize.linear_sum_assignment`, which requires cost matrices to be materialized on CPU.
- Matching cost construction repeatedly builds pairwise geometry terms:
  - aligned control-point L1
  - aligned sampled-point L1
  - pairwise curve Chamfer
  - pairwise gIoU
- `one-to-many` and `distinct` repeat cost-matrix-like work for grouped queries.
- Decoder loss still performs repeated Bezier sampling through `sample_bezier_curves_torch`.

Most promising acceleration targets:

1. Pairwise curve cost construction
- Files:
  - [models/matcher.py](../models/matcher.py)
  - [models/geometry.py](../models/geometry.py)
- Why:
  - These kernels are reused by main matching, aux matching, one-to-many, and top-k add-on selection.
  - This is the highest-leverage place to speed up the current training stack without changing model behavior.
- Best candidates for lower-level implementation:
  - aligned control-point cost
  - aligned sampled-point cost
  - pairwise curve Chamfer helper

2. Bezier sampling kernel
- Files:
  - [misc_utils/train_utils.py](../misc_utils/train_utils.py)
  - [models/geometry.py](../models/geometry.py)
- Why:
  - The same curve sets are sampled multiple times for matching and loss.
  - A fused lower-level implementation or cached sampling path could reduce repeated overhead.

3. Distinct / one-to-many auxiliary preprocessing
- File:
  - [models/losses/regularizers.py](../models/losses/regularizers.py)
- Why:
  - These paths currently loop over batch, group, and target dimensions in Python.
  - The logic is structurally correct, but it is an obvious candidate for vectorization or a lower-level helper if grouped training becomes the dominant runtime cost.

What is likely *not* worth doing first:

- Rewriting dataloader graph-cache reading in C/C++.
- Rewriting the full decoder in C++.
- Moving everything to a custom CUDA path before simplifying the pairwise geometry kernels.

Recommended order if more acceleration is needed:

1. Reduce repeated geometry work at the Python/Torch level where possible.
2. Move cost-kernel helpers into a native extension or fused custom op if profiling still points there.
3. Only then consider replacing the Hungarian solver path or introducing a different matching approximation.

Current native experiment status:

- A native C++ matcher-side backend has now been added for the three non-gradient pairwise geometry kernels:
  - aligned control-point pairwise cost
  - aligned sampled-point pairwise cost
  - pairwise curve Chamfer helper
- It is only used on no-grad matcher-side paths, so the differentiable training geometry losses still use the original Torch implementation.
- The backend is lazily compiled and falls back automatically if compilation is unavailable.
- Repro benchmark script:
  - [scripts/benchmark_native_matcher_costs.py](../scripts/benchmark_native_matcher_costs.py)

Observed steady-state benchmark on `llm_server`:

- `256 x 180`: about `1.38x`
- `512 x 220`: about `1.44x`
- `512 x 512`: about `1.06x`

Interpretation:

- The native backend is numerically consistent with the Torch reference up to small floating-point noise.
- The speedup is real but moderate, not transformative.
- This is still worthwhile because it accelerates exactly the no-grad matcher-side path we identified as a recurring bottleneck, without interfering with gradient-based optimization.


## Historical BSDS Training Stack

Current implementation:

- Historical BSDS training remains supported, but it is no longer the default entrypoint.
- BSDS `train` and `val` annotation splits are merged for optimization.
- BSDS `test` is used as the validation split for model selection.
- Data roots are split explicitly by split name instead of overloading a single `input_root` / `edge_glob` pair.
  - Code: [edge_datasets/parametric_edge_datamodule.py](../edge_datasets/parametric_edge_datamodule.py)
  - Code: [train.py](../train.py)

Current formal model size:

- `hidden_dim: 384`
- `nheads: 12`
- `num_encoder_layers: 4`
- `num_decoder_layers: 6`
- `dim_feedforward: 1536`
- `num_queries: 256`
- `num_sampling_points: 8`

Why:

- The earlier overfit-oriented configs were useful for diagnosis, but they were not a realistic final training recipe.
- Formal BSDS training needs split-aware data loading, a larger decoder/encoder budget, multi-GPU execution, and resumable checkpoints.

Status:

- The historical BSDS config and runtime plumbing are preserved in `configs/parametric_edge/bsds_formal.yaml`.
- The split-aware formal cache benchmark completed successfully after warming 2696 graph-cache entries.
- Validation plus test dataloader benchmarking covered 1609 samples at about 136.1 samples/sec overall, with p95 per-sample arrival about 46.6 ms.
- A short 2-GPU DDP smoke test completed successfully after switching grouped-query training to `ddp_find_unused_parameters_true`.
- Short 2-GPU parameter probes selected `batch_size: 10`, `val_batch_size: 10`, `accumulate_grad_batches: 2` as the best tested formal setting on RTX 3090 x2.
- That setting gives effective global batch size 40 and outperformed the tested `8 x 2` accumulation setting, while `12 x 2` was slower per sample.
- Enabling `channels_last` removes the observed DDP grad-stride warning for the 1x1 convolution weights and gives a small but consistent short-run throughput improvement, so it is now part of the formal config.
- Grouped training now uses a fixed `group_detr_num_groups: 3` during training and a single group during inference. The earlier dynamic active-group policy was removed after the query construction switched from duplicated groups to proposal-partitioned groups.
- The current active stack removes the separate extent branch/loss and with geometry supervision concentrated in `ctrl`, `sample`, `endpoint`, `giou`, and `curve_distance` terms plus grouped auxiliary objectives.

## RGB Training Input

Current implementation:

- Current default training consumes RGB images from LAION image roots.
- Supervision comes from precomputed Bezier graph caches built from LAION binary edge maps.

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

## Synthetic Clutter Down-Weighting

Status:

- Explored, but **not adopted into the training path**.
- The training code currently does **not** apply clutter-based per-curve down-weighting.

Motivation:

- In synthetic data, some regions produce many small, heavily overlapping loops or fragments.
- Those regions can dominate the regression signal even when the global structure is already correct.
- We want to reduce the matched regression pressure only for those local clutter regions, not suppress the whole sample.

Heuristics tried:

1. `aggressive`
- Combined overlap density, neighbor count, shortness, small area, and curvature with a strong additive score.
- Result:
  - Clearly too aggressive.
  - It down-weighted many normal structural curves, not only clutter.

2. `overlap_only`
- Used only local overlap and neighbor density.
- Result:
  - Better than `aggressive`, but still too broad.
  - It often highlighted large overlapping structural boundaries that we still want to supervise strongly.

3. `balanced`
- Requires all of the following to be present together:
  - local overlap
  - local neighbor density
  - smallness / shortness
  - mild curvature modulation
- Result:
  - Much closer to the intended behavior.
  - It leaves normal object contours mostly untouched.
  - It highlights compact, locally dense, overlapping micro-curve regions more selectively.

How we compared them:

- Comparison script:
  - [scripts/compare_clutter_weight_heuristics.py](../scripts/compare_clutter_weight_heuristics.py)
- Representative remote LAION cache samples were pulled back locally and rendered side-by-side.
- We inspected both:
  - simpler structured samples
  - highly cluttered synthetic samples

Representative findings:

- On clean samples like `batch1_000000`, `aggressive` down-weighted almost everything, while `balanced` only touched a few local overlap points.
- On medium-complexity samples like `batch2_150890`, `balanced` highlighted the small overlapping branch cluster near the left-side junctions without blanketing the whole object.
- On dense clutter samples like `batch1_024442` and `batch2_125853`, `balanced` highlighted a subset of the locally packed micro-loops, while `aggressive` and `overlap_only` highlighted far too much of the image.

Summary statistics from the comparison run:

- `batch1_000000`
  - `aggressive`: `low=51/51`
  - `overlap_only`: `low=18/51`
  - `balanced`: `low=8/51`
- `batch2_150890`
  - `aggressive`: `low=96/96`
  - `overlap_only`: `low=25/96`
  - `balanced`: `low=11/96`
- `batch1_024442`
  - `aggressive`: `low=967/967`
  - `overlap_only`: `low=392/967`
  - `balanced`: `low=64/967`
- `batch2_125853`
  - `aggressive`: `low=858/858`
  - `overlap_only`: `low=579/858`
  - `balanced`: `low=157/858`

Outcome:

- `balanced` was the most reasonable of the tried heuristics.
- However, even `balanced` still did not align consistently enough with human judgment on the highlighted clutter regions.
- In particular, some visually acceptable curves were still down-weighted, while some intuitively cluttered regions were not emphasized strongly enough.

Current decision:

- Do **not** include clutter-based match down-weighting in training for now.
- Keep the comparison script and visual analysis as reference material for future work.

Reference artifacts:

- Comparison script:
  - [scripts/compare_clutter_weight_heuristics.py](../scripts/compare_clutter_weight_heuristics.py)
- Example comparison outputs live under:
  - `outputs/remote_cache_debug_samples/clutter_strategy_compare/`

## Graph-First Data Pipeline

Current implementation:

- Edge annotations are first converted into cached graph polylines, not directly resized/raster-warped targets.
  - Code: [misc_utils/bezier_target_utils.py](../misc_utils/bezier_target_utils.py), `ensure_graph_cache()`
- Training samples load the original RGB image and cached graph, then apply synchronized geometric transforms in graph space.
  - Code: [edge_datasets/parametric_edge_dataset.py](../edge_datasets/parametric_edge_dataset.py)
  - Code: [edge_datasets/graph_pipeline.py](../edge_datasets/graph_pipeline.py), `prepare_training_sample()`
- Bezier targets are re-fit after resize / flip / affine / crop, so supervision matches the transformed image instead of an aliased raster edge map.
  - Code: [edge_datasets/graph_pipeline.py](../edge_datasets/graph_pipeline.py), `build_targets_from_polylines()`
- Persistent graph cache now lives under `edge_data/HED-BSDS/cache/graph_polyline_v1_segments_xy_v5_anchor_consistent/`.
- That directory is intentionally under `edge_data/` so cache naming stays coupled to the dataset and remains git-ignored.

Training path:

- Preserve aspect ratio while resizing up to the crop size.
- Apply optional horizontal flip and mild affine perturbation jointly to the image and graph.
- Use reflect-mode image resampling during affine warp to avoid black corners.
- Random-crop to the configured fixed training size, currently `256 x 256` in the default config.

Eval / test path:

- Preserve aspect ratio.
- Resize into the fixed canvas.
- Pad with a per-image constant color, not reflect padding.
- Shift graph coordinates by the same pad offset before target generation.

Why:

- Directly resizing binary edges aliases geometry and drifts supervision away from the real transformed structure.
- Reflect padding is acceptable as an internal affine fill strategy, but it is not a valid eval-time letterbox policy because it visually duplicates image content without duplicating GT.

Effect:

- GT curves now stay aligned with both train-time augmentations and eval-time aspect-preserved inputs.
- Eval visualizations no longer show mirrored border content with missing GT.

## Coordinate Normalization

Current implementation:

- Input images are loaded at original resolution, transformed, then converted to fixed-size tensors normalized to `[0, 1]`.
  - Code: [misc_utils/bezier_target_utils.py](../misc_utils/bezier_target_utils.py), `load_image_array_original()`
  - Code: [edge_datasets/graph_pipeline.py](../edge_datasets/graph_pipeline.py)
- Bezier control points are stored in normalized XY coordinates in `[0, 1]`, not raw pixel coordinates.
  - Code: [edge_datasets/graph_pipeline.py](../edge_datasets/graph_pipeline.py), `_normalize_xy_control_points()`
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

- Uses grouped training with `group_detr_num_groups` as the fixed number of training-time groups.
- The active group count is sampled uniformly from `[1, K_max]` after applying the per-batch cap, so sparse batches do not always force grouped training and eval/test stay strictly one-group.
- Code:
  - Current Group DETR-style one-to-many no longer uses repeated-target Hungarian matching; extra query groups are supervised with per-group one-to-one matching while decoder self-attention is masked across groups during training.
  - [models/parametric_detr.py](../models/parametric_detr.py)
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

- Top-k positive supervision is now treated as an internal add-on to the final-layer one-to-many grouped branch, not as a separate auxiliary-layer loss.
- For each auxiliary group at the final decoder layer, the grouped branch can keep the best `K` candidate queries per GT; if truncation is disabled, the grouped branch can use every candidate it finds.
- Rank-dependent weights are applied with `topk_positive_tau`, while final inference still uses only the main one-to-one group.
- Code:
  - [models/losses/regularizers.py](../models/losses/regularizers.py), `OneToManyLoss`
  - [models/losses/matched.py](../models/losses/matched.py)

Motivation:

- Grouped training can expose multiple plausible positives per GT inside the auxiliary groups, but treating all of them as equally strong positives makes score calibration noisy.
- The useful control knob is therefore whether to truncate grouped positives and how strongly to weight lower-ranked grouped matches.

Effect:

- Keeps top-k semantically aligned with the one-to-many grouped branch instead of making it look like a separate supervision axis.
- Allows `topk_positive_enabled=false` to mean “use every grouped positive,” which is the clean ablation baseline.

Source:

- Internal strategy inspired by the broader one-to-many supervision idea in Group DETR / H-DETR.

## Distinct Query Regularization

Current implementation:

- Distinct regularization only compares queries within the same group.
- It only penalizes pairs that are likely duplicates of the same GT surrogate, instead of globally penalizing any nearby confident queries.
- Code:
  - [models/losses/regularizers.py](../models/losses/regularizers.py), `DistinctQueryLoss`

Why:

- A global similarity penalty can suppress legitimate nearby edges.
- Cross-group penalties also conflict with the purpose of grouped training.

Effect:

- The loss now targets duplicate queries more narrowly and is less likely to erase dense local structure.

## Cache Warmup And Verification

Current implementation:

- Full-dataset graph-cache warmup, parallel dataloader timing, and verification visualizations are handled by:
  - [scripts/precompute_graph_cache_and_benchmark.py](../scripts/precompute_graph_cache_and_benchmark.py)
- Main visualization outputs are now saved as `.jpg`.
- The same script also supports split-aware formal configs with separate `train/val/test` roots.

Why:

- The graph-first pipeline is only trustworthy if visual overlays and warm-cache throughput are checked after structural changes.

Effect:

- Cache preparation, timing, and visual spot-checks are now reproducible with one command instead of ad hoc snippets.

Formal cache location:

- [edge_data/HED-BSDS/cache/graph_polyline_v1_segments_xy_v5_anchor_consistent](../edge_data/HED-BSDS/cache/graph_polyline_v1_segments_xy_v5_anchor_consistent)

Note:

- Cache contents are data artifacts, not source artifacts. They are intentionally excluded from Git and should be regenerated locally if missing.

## Multi-GPU Training Runtime

Current implementation:

- Training now supports `--resume-from` to continue from `last.ckpt` or any named checkpoint.
- Lightning is configured to save both the monitored best checkpoint and `last.ckpt` for resume.
- Gradient accumulation, precision, DDP strategy, and optional post-fit test evaluation are configurable from YAML.
- Validation and test metrics use synchronized distributed logging.
- Code:
  - [train.py](../train.py)
  - [pl_trainer/parametric_edge_trainer.py](../pl_trainer/parametric_edge_trainer.py)

Formal defaults currently encoded:

- `devices: 2`
- `strategy: ddp_find_unused_parameters_true`
- `precision: 16-mixed`
- `accumulate_grad_batches: 2`
- `batch_size: 10`
- `val_batch_size: 10`
- `channels_last: true`
- `save_top_k: 1` plus `save_last: true`

Why:

- The formal BSDS run is expected to be long enough that resume support is required.
- A larger effective batch is easier to reach with gradient accumulation than by only increasing per-device batch size.
- Dynamic grouped-query training activates only a subset of query groups on some steps, so DDP needs unused-parameter detection enabled for the full all-features setup.
- The current model is compute-bound enough that data-loader changes have less impact than layout fixes; moving the conv path to `channels_last` is a low-risk throughput optimization.

Status:

- Runtime support is in place.
- The short formal search run chose `batch_size: 10`, `val_batch_size: 10`, `accumulate_grad_batches: 2` for the first production launch.
- The temporary 50-epoch search overrides have been removed after folding the selected setting into the default config.

Measured on 2026-03-14 with `configs/parametric_edge/default.yaml`:

- Full graph-cache precompute over `1063` images finished successfully.
- Using `12` cache workers, cache generation took `2862.23s` total, about `0.37 samples/s` end to end.
- After cache prep, a `4`-worker, `batch_size=1` DataLoader benchmark over the full dataset took `14.99s` total, about `70.94 samples/s`.
- Warm-cache per-sample arrival statistics were:
  - mean `14.10 ms`
  - median `4.43 ms`
  - p95 `56.53 ms`
  - max `125.93 ms`
- Verification artifacts were written to:
  - `outputs/parametric_edge_training/data_debug/cache_benchmark_v2/summary.json`
  - `outputs/parametric_edge_training/data_debug/cache_benchmark_v2/per_image_read_times.csv`
  - `outputs/parametric_edge_training/data_debug/cache_benchmark_v2/eval_samples_01.jpg`
  - `outputs/parametric_edge_training/data_debug/cache_benchmark_v2/eval_samples_02.jpg`
  - `outputs/parametric_edge_training/data_debug/cache_benchmark_v2/eval_samples_03.jpg`
  - `outputs/parametric_edge_training/data_debug/cache_benchmark_v2/train_samples_01.jpg`
  - `outputs/parametric_edge_training/data_debug/cache_benchmark_v2/train_samples_02.jpg`
  - `outputs/parametric_edge_training/data_debug/cache_benchmark_v2/train_samples_03.jpg`

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
