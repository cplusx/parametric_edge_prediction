# Parametric Edge Ablation Log

This document records the ablations that have already been run in this repository, where their artifacts live, and how to rerun or extend them.

This file should be updated whenever a new ablation is added, rerun, or materially reinterpreted.

## Update Policy

When adding a new ablation, update this file in the same change with:

- the purpose of the ablation
- the config file or script used to launch it
- the exact command used to rerun it
- the output directory
- the summary or report file that records its result
- the current conclusion, if one exists

If an ablation changes an existing default setting, also note which config was updated.

## Environment

The commands below assume the repo root is the working directory:

```bash
cd /home/viplab/jiaxin/parametric_edge_prediction
conda activate diffusers
```

Core training entrypoint:

```bash
python train.py --config <config.yaml>
python train.py --config <base_config.yaml> --override-config <override.yaml>
```

Config loading behavior:

- If `--config` points to a non-`default.yaml` file and no override is provided, `configs/parametric_edge/default.yaml` is merged first.
- If `--override-config` is used, the override is merged on top of the config loaded by `--config`.
- For the add-back suite below, use `overfit_diverse16_2000_memorization_base.yaml` as the base config and the individual add-back file as the override.

## Output Conventions

Main ablation artifacts are currently written to:

- `outputs/parametric_edge_training/current_sweep_analysis/`
- `outputs/parametric_edge_training/current_sweep_analysis/matching_synthetic/`

Training runs themselves write to experiment-specific directories under:

- `outputs/parametric_edge_training/`

Dataset-side cache artifacts are stored separately under:

- `edge_data/HED-BSDS/cache/graph_polyline_v1_segments_xy_v5_anchor_consistent/`

Those cache files are not source-controlled and should not be expected to appear in GitHub commits.

## Formal Training Plan

Purpose:

- Move from overfit diagnostics to a proper BSDS train/val/test training run.

Config:

- [configs/parametric_edge/default.yaml](../configs/parametric_edge/default.yaml)

Current intended command:

```bash
python train.py \
  --config configs/parametric_edge/default.yaml
```

Current runtime design:

- Train on BSDS `train`
- Train on merged BSDS `train + val`
- Validate on BSDS `test`
- Use 2 GPUs with DDP and mixed precision
- Keep both the best monitored checkpoint and `last.ckpt`
- Use `ddp_find_unused_parameters_true` because grouped-query training activates only a subset of groups on some steps.
- Use `batch_size: 10`, `val_batch_size: 10`, `accumulate_grad_batches: 2`, giving effective global batch size 40.
- Use `channels_last: true` because it removes the observed DDP grad-stride warning and slightly improves short-run throughput.
- Use the selected geometry-focused loss setting with `bbox_weight: 0.0` and `extent_weight: 1.0`.
- Record training-set visualizations every 500 iterations in addition to periodic validation visualizations.

Status:

- Config and runtime support are checked in.
- Split-aware cache warmup and benchmark now complete successfully.
- Quick 2-GPU parameter probes under the formal config found `batch_size: 10`, `accumulate_grad_batches: 2` to be the best tested setting among the checked `8 x 2`, `10 x 2`, and `12 x 2` per-GPU batch candidates.
- A short A/B benchmark showed that enabling `channels_last` removes the 1x1-conv grad-stride warning seen under DDP and marginally improves wall time on the same 8-batch micro-run.
- The grouped auxiliary losses are now forced to stay active during training by using the maximum valid group count instead of random `1..K` sampling; this removes the previous step-to-step zero spikes in `loss_om_*` and `loss_topk_pos_*`.
- The temporary 50-epoch search configs and search-run artifacts were removed after selecting the final default recipe.
- The first full formal run should use the default config directly, since the selected setting is now folded into [configs/parametric_edge/default.yaml](../configs/parametric_edge/default.yaml).

Important reset on 2026-03-14:

- Older overfit outputs were generated before the current graph-first data pipeline, `loss_main`-based comparison policy, and the eval-padding / XY normalization fixes stabilized.
- Those stale overfit artifacts should not be used for current conclusions and are cleared from `outputs/parametric_edge_training/` before rerunning the suite.
- If a section below mentions an older overfit directory that no longer exists, treat it as historical context only and rerun the command instead of relying on the removed artifact.

## Ablation 1: Memorization Add-Back Suite

Purpose:

- Start from a memorization-oriented overfit setting on the 16-sample diverse subset.
- Add back one training technique at a time to measure which one most strongly hurts pure overfitting.

Base config:

- [configs/parametric_edge/overfit_diverse16_2000_memorization_base.yaml](../configs/parametric_edge/overfit_diverse16_2000_memorization_base.yaml)

Add-back overrides:

- [configs/parametric_edge/overfit_diverse16_2000_addback_aux.yaml](../configs/parametric_edge/overfit_diverse16_2000_addback_aux.yaml)
- [configs/parametric_edge/overfit_diverse16_2000_addback_dn.yaml](../configs/parametric_edge/overfit_diverse16_2000_addback_dn.yaml)
- [configs/parametric_edge/overfit_diverse16_2000_addback_dn_curve_anchor_fix.yaml](../configs/parametric_edge/overfit_diverse16_2000_addback_dn_curve_anchor_fix.yaml)
- [configs/parametric_edge/overfit_diverse16_2000_addback_dn_curve_anchor_fix_legacy_compare.yaml](../configs/parametric_edge/overfit_diverse16_2000_addback_dn_curve_anchor_fix_legacy_compare.yaml)
- [configs/parametric_edge/overfit_diverse16_2000_addback_dn_proposal_curve_legacy_compare.yaml](../configs/parametric_edge/overfit_diverse16_2000_addback_dn_proposal_curve_legacy_compare.yaml)
- [configs/parametric_edge/overfit_diverse16_2000_addback_onetomany.yaml](../configs/parametric_edge/overfit_diverse16_2000_addback_onetomany.yaml)
- [configs/parametric_edge/overfit_diverse16_2000_addback_topk.yaml](../configs/parametric_edge/overfit_diverse16_2000_addback_topk.yaml)
- [configs/parametric_edge/overfit_diverse16_2000_addback_distinct.yaml](../configs/parametric_edge/overfit_diverse16_2000_addback_distinct.yaml)

Recommended run commands:

```bash
python train.py \
  --config configs/parametric_edge/overfit_diverse16_2000_memorization_base.yaml

python train.py \
  --config configs/parametric_edge/overfit_diverse16_2000_memorization_base.yaml \
  --override-config configs/parametric_edge/overfit_diverse16_2000_addback_aux.yaml

python train.py \
  --config configs/parametric_edge/overfit_diverse16_2000_memorization_base.yaml \
  --override-config configs/parametric_edge/overfit_diverse16_2000_addback_dn.yaml

python train.py \
  --config configs/parametric_edge/overfit_diverse16_2000_memorization_base.yaml \
  --override-config configs/parametric_edge/overfit_diverse16_2000_addback_dn_curve_anchor_fix.yaml

python train.py \
  --config configs/parametric_edge/overfit_diverse16_2000_memorization_base.yaml \
  --override-config configs/parametric_edge/overfit_diverse16_2000_addback_dn_curve_anchor_fix_legacy_compare.yaml

python train.py \
  --config configs/parametric_edge/overfit_diverse16_2000_memorization_base.yaml \
  --override-config configs/parametric_edge/overfit_diverse16_2000_addback_dn_proposal_curve_legacy_compare.yaml

python train.py \
  --config configs/parametric_edge/overfit_diverse16_2000_memorization_base.yaml \
  --override-config configs/parametric_edge/overfit_diverse16_2000_addback_onetomany.yaml

python train.py \
  --config configs/parametric_edge/overfit_diverse16_2000_memorization_base.yaml \
  --override-config configs/parametric_edge/overfit_diverse16_2000_addback_topk.yaml

python train.py \
  --config configs/parametric_edge/overfit_diverse16_2000_memorization_base.yaml \
  --override-config configs/parametric_edge/overfit_diverse16_2000_addback_distinct.yaml
```

Rerun note after DN/query refactor on 2026-03-14:

- Purpose:
  - Re-test the DN add-back setting after replacing the old DN geometry MLP path with direct noisy curve anchors and a derived 2D deformable-attention reference.
- Config:
  - [configs/parametric_edge/overfit_diverse16_2000_addback_dn_curve_anchor_fix.yaml](../configs/parametric_edge/overfit_diverse16_2000_addback_dn_curve_anchor_fix.yaml)
- Output directory:
  - [outputs/parametric_edge_training/overfit_diverse16_2000_addback_dn_curve_anchor_fix](../outputs/parametric_edge_training/overfit_diverse16_2000_addback_dn_curve_anchor_fix)
- Command:
```bash
python train.py \
  --config configs/parametric_edge/overfit_diverse16_2000_memorization_base.yaml \
  --override-config configs/parametric_edge/overfit_diverse16_2000_addback_dn_curve_anchor_fix.yaml
```
- Current conclusion:
  - This refactor regressed early overfit behavior badly.
  - The rerun was stopped at epoch 53 after reaching `train/loss=1.2933`, `val/loss=1.1601`.
  - The historical successful DN run at the same epoch had `val/loss=0.3687`, so the new curve-anchor implementation is substantially worse in its current form and should not replace the previous DN path yet.

Stricter historical DN-only comparison on 2026-03-14:

- Clarification:
  - The historical completed run [outputs/parametric_edge_training/overfit_diverse16_2000_addback_dn/csv_logs/version_1](../outputs/parametric_edge_training/overfit_diverse16_2000_addback_dn/csv_logs/version_1) is indeed the old `memorization + DN only` run, not a no-DN baseline.
  - Its saved [hparams.yaml](../outputs/parametric_edge_training/overfit_diverse16_2000_addback_dn/csv_logs/version_1/hparams.yaml) shows `dn_num_groups: 4`, `dn_weight: 1.0`, `dn_curve_weight: 5.0`, with the other memorization regularizers still disabled.
- Purpose:
  - Rerun the new curve-anchor/DN implementation with loss weights adjusted to match the historical completed DN-only run as closely as the current code permits.
- Config:
  - [configs/parametric_edge/overfit_diverse16_2000_addback_dn_curve_anchor_fix_legacy_compare.yaml](../configs/parametric_edge/overfit_diverse16_2000_addback_dn_curve_anchor_fix_legacy_compare.yaml)
- Output directory:
  - [outputs/parametric_edge_training/overfit_diverse16_2000_addback_dn_curve_anchor_fix_legacy_compare](../outputs/parametric_edge_training/overfit_diverse16_2000_addback_dn_curve_anchor_fix_legacy_compare)
- Current conclusion:
  - This stricter apples-to-apples rerun is still much worse than the old completed DN-only run.
  - It was stopped during epoch 35 after logging `val/loss=1.3089` at epoch 31 and best-so-far `val/loss=1.2890` at epoch 22.
  - The old completed DN-only run had `val/loss=0.3697` at epoch 31 and best-so-far `0.3584` by epoch 29.
  - So the regression is not explained just by changed default matcher weights; it persists even under a much closer historical comparison.

Proposal-curve main-query rerun on 2026-03-14:

- Purpose:
  - Replace the bad “single 2D point copied across all control points” main-query initialization with encoder proposals that directly predict full endpoint/control-point anchors, while deriving the deformable 2D reference from the curve anchor through an MLP.
- Config:
  - [configs/parametric_edge/overfit_diverse16_2000_addback_dn_proposal_curve_legacy_compare.yaml](../configs/parametric_edge/overfit_diverse16_2000_addback_dn_proposal_curve_legacy_compare.yaml)
- Output directory:
  - [outputs/parametric_edge_training/overfit_diverse16_2000_addback_dn_proposal_curve_legacy_compare](../outputs/parametric_edge_training/overfit_diverse16_2000_addback_dn_proposal_curve_legacy_compare)
- Current conclusion:
  - This is better aligned with the intended geometry design than the point-copy version, but it still does not recover the old DN-only memorization behavior.
  - The run was stopped during epoch 45; by epoch 41 it had `val/loss=1.2140`, with best-so-far `val/loss=1.2140`.
  - The historical old DN-only run at epoch 41 had `val/loss=0.3698`, with best-so-far `0.3584` by epoch 29.
  - So moving main proposals to full curve anchors alone is not sufficient; the new query geometry path is still materially underperforming the old DN implementation.

Outputs:

- Summary script: [scripts/summarize_addback_ablation.py](../scripts/summarize_addback_ablation.py)
- Current summary artifacts are intentionally regenerated only after fresh reruns under the current data pipeline.

How to regenerate the summary after runs finish:

```bash
python scripts/summarize_addback_ablation.py
```

Current conclusion:

- Stale pre-reset outputs were removed.
- This suite needs to be rerun before drawing any updated ranking across `aux`, `dn`, `one_to_many`, `topk`, and `distinct`.

## Ablation 2: All Functions Except DN

Purpose:

- Re-enable the non-DN regularizers together to measure their combined overfit cost without denoising.

Config:

- [configs/parametric_edge/overfit_diverse16_2000_all_except_dn.yaml](../configs/parametric_edge/overfit_diverse16_2000_all_except_dn.yaml)

Recommended run command:

```bash
python train.py \
  --config configs/parametric_edge/overfit_diverse16_2000_memorization_base.yaml \
  --override-config configs/parametric_edge/overfit_diverse16_2000_all_except_dn.yaml
```

Monitoring script:

- [scripts/monitor_all_except_dn.py](../scripts/monitor_all_except_dn.py)

One-shot status refresh:

```bash
python scripts/monitor_all_except_dn.py --once
```

Continuous monitoring:

```bash
python scripts/monitor_all_except_dn.py --interval-seconds 300
```

Outputs:

- Status artifacts should be regenerated after the suite is rerun under the current pipeline.

Current conclusion:

- Stale pre-reset outputs were removed.
- Rerun required before interpreting the combined non-DN stack again.

## Ablation 3: Synthetic Matching-Cost Visualization

Purpose:

- Inspect whether the current matching cost is geometrically sensible.
- Generate a target curve `A` near the image center.
- Generate many synthetic query curves across the image.
- Annotate each query curve with its per-term matching costs and total cost.
- Compare the matcher's ranking against a denser curve-distance reference.

Script:

- [scripts/visualize_matching_cost_synthetic.py](../scripts/visualize_matching_cost_synthetic.py)

Base command using current config weights:

```bash
python scripts/visualize_matching_cost_synthetic.py
```

Example command for a named comparison run:

```bash
python scripts/visualize_matching_cost_synthetic.py \
  --control-cost 1.5 \
  --box-cost 0.5 \
  --giou-cost 1.0 \
  --curve-distance-cost 6.0 \
  --name-suffix _selected
```

Outputs:

- Directory: [outputs/parametric_edge_training/current_sweep_analysis/matching_synthetic](../outputs/parametric_edge_training/current_sweep_analysis/matching_synthetic)
- Per-run summary markdown: `*_matching_summary.md`
- Per-run machine-readable records: `*_matching_records.json`, `*_matching_records.csv`
- Per-run figures: `*_matching_visualization.png`

Named synthetic runs already generated:

- `*_geom_only`: current geometry-only matcher after removing class cost
- `*_balanced`: candidate with moderate curve upweighting
- `*_curve_heavy`: more aggressive curve-driven matcher
- `*_selected`: the current chosen default matcher weights

Current conclusion:

- Matching should not include a class cost in this edge-only setup.
- The old matcher over-weighted control-point L1 and under-weighted curve-distance cost.
- The current selected matcher weights are:

```text
control_cost = 1.5
box_cost = 0.5
giou_cost = 1.0
curve_distance_cost = 6.0
```

These selected weights are now reflected in:

- [configs/parametric_edge/default.yaml](../configs/parametric_edge/default.yaml)
- [configs/parametric_edge/overfit_diverse16_2000_memorization_base.yaml](../configs/parametric_edge/overfit_diverse16_2000_memorization_base.yaml)

Reference summaries for the chosen setting:

- [outputs/parametric_edge_training/current_sweep_analysis/matching_synthetic/constant_selected_matching_summary.md](../outputs/parametric_edge_training/current_sweep_analysis/matching_synthetic/constant_selected_matching_summary.md)
- [outputs/parametric_edge_training/current_sweep_analysis/matching_synthetic/random_selected_matching_summary.md](../outputs/parametric_edge_training/current_sweep_analysis/matching_synthetic/random_selected_matching_summary.md)

## Suggested Format for Future Ablations

For every future ablation, add a new section with this template:

````md
## Ablation N: <name>

Purpose:

- <what this ablation is testing>

Config or script:

- <config path or script path>

Run command:

```bash
<exact command>
```

Outputs:

- <summary file>
- <report file>
- <output directory>

Current conclusion:

- <short takeaway>
````

When the ablation is still running, replace the conclusion with `Status: running` and update it after completion.