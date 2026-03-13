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

## Ablation 1: Memorization Add-Back Suite

Purpose:

- Start from a memorization-oriented overfit setting on the 16-sample diverse subset.
- Add back one training technique at a time to measure which one most strongly hurts pure overfitting.

Base config:

- [configs/parametric_edge/overfit_diverse16_2000_memorization_base.yaml](../configs/parametric_edge/overfit_diverse16_2000_memorization_base.yaml)

Add-back overrides:

- [configs/parametric_edge/overfit_diverse16_2000_addback_aux.yaml](../configs/parametric_edge/overfit_diverse16_2000_addback_aux.yaml)
- [configs/parametric_edge/overfit_diverse16_2000_addback_dn.yaml](../configs/parametric_edge/overfit_diverse16_2000_addback_dn.yaml)
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
  --override-config configs/parametric_edge/overfit_diverse16_2000_addback_onetomany.yaml

python train.py \
  --config configs/parametric_edge/overfit_diverse16_2000_memorization_base.yaml \
  --override-config configs/parametric_edge/overfit_diverse16_2000_addback_topk.yaml

python train.py \
  --config configs/parametric_edge/overfit_diverse16_2000_memorization_base.yaml \
  --override-config configs/parametric_edge/overfit_diverse16_2000_addback_distinct.yaml
```

Outputs:

- Summary markdown: [outputs/parametric_edge_training/current_sweep_analysis/addback_ablation_summary.md](../outputs/parametric_edge_training/current_sweep_analysis/addback_ablation_summary.md)
- Summary json: [outputs/parametric_edge_training/current_sweep_analysis/addback_ablation_summary.json](../outputs/parametric_edge_training/current_sweep_analysis/addback_ablation_summary.json)
- Report: [outputs/parametric_edge_training/current_sweep_analysis/addback_ablation_report.md](../outputs/parametric_edge_training/current_sweep_analysis/addback_ablation_report.md)
- Summary script: [scripts/summarize_addback_ablation.py](../scripts/summarize_addback_ablation.py)

How to regenerate the summary after runs finish:

```bash
python scripts/summarize_addback_ablation.py
```

Current conclusion:

1. `dn` is the strongest anti-memorization technique in this setup.
2. `aux` is the second most harmful add-back.
3. `one_to_many` has a moderate effect.
4. `topk` has a small effect.
5. `distinct` is nearly free in the overfit regime.

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

- Status markdown: [outputs/parametric_edge_training/current_sweep_analysis/all_except_dn_status.md](../outputs/parametric_edge_training/current_sweep_analysis/all_except_dn_status.md)
- Status json: [outputs/parametric_edge_training/current_sweep_analysis/all_except_dn_status.json](../outputs/parametric_edge_training/current_sweep_analysis/all_except_dn_status.json)
- Final report section appended into: [outputs/parametric_edge_training/current_sweep_analysis/addback_ablation_report.md](../outputs/parametric_edge_training/current_sweep_analysis/addback_ablation_report.md)

Current conclusion:

- The combined non-DN regularization stack degrades memorization noticeably, but still far less than DN itself.

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