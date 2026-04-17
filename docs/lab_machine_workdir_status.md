# Lab Machine Workdir Status

Current operational snapshot for the active preprocessing pipeline:

`images -> source_edge -> bezier`

This file is a current-state operations note. It is not a historical experiment log.

## Connectivity Rules

- Local machine can connect to lab machines.
- Any lab machine can connect to cluster.
- Cluster cannot ssh to lab machines.
- Therefore, cluster-to-lab transfers must go through a lab-machine relay.

## Code And Conda Env Mapping

### `scripts/generate_source_edges_from_sam2_batch.py`

- `lab30`: `sam2`
- `lab34`: `diffusers`
- `cluster`: `sam2`

### `scripts/generate_bezier_from_source_edges_batch.py`

- `lab30`: `sam2`
- `lab34`: `diffusers`
- `lab21`: prefer existing env in this order:
  - `diffusers`
  - `sam2`
  - otherwise create a dedicated CPU-only env for bezierization

### Dataloader / debug / training code

- `lab30`: `diffusers`
- `lab34`: `diffusers`
- `cluster`: use the training env for the job, not `sam2`

Rule of thumb:

- `sam2` only for source-edge generation
- `diffusers` for dataloader debugging, dataset visualization, and training-side code

## Machine Snapshot

| Machine | Connectivity | Repo / Workdir | Data Roots | Active Batches | Current Stage | Output Roots | Launchers / Logs | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `lab30` | local -> `lab30`; `lab30` -> cluster | `/home/viplab/jiaxin/parametric_edge_prediction` | `/home/devdata/laion/edge_detection` | `batch7`, `batch8` | `batch7`: source_edge essentially complete, bezier running; `batch8`: source_edge running | source_edge: `/home/devdata/laion/edge_detection/laion_edge_v3_source_edge`; bezier: `/home/devdata/laion/edge_detection/laion_edge_v3_bezier` | source_edge logs: `/home/viplab/jiaxin/parametric_edge_prediction/outputs/sam2_source_edge_batch7_full_logs`, `/home/viplab/jiaxin/parametric_edge_prediction/outputs/sam2_source_edge_batch8_full_logs`; bezier logs: `/home/viplab/jiaxin/parametric_edge_prediction/outputs/bezier_from_source_edge_batch7_logs` | Current counts observed locally: `batch7 source_edge=29722`, `batch8 source_edge=19189`, `batch7 bezier=18254` |
| `lab34` | local -> `lab34`; `lab34` -> cluster | `/data/jiaxin/parametric_edge_prediction` | `/data/jiaxin/laion/edge_detection` | `batch13`, `batch12` | `batch13`: source_edge running, bezier running; `batch12`: images copied, delayed source_edge launcher armed | source_edge: `/data/jiaxin/laion/edge_detection/laion_edge_v3_source_edge`; bezier: `/data/jiaxin/laion/edge_detection/laion_edge_v3_bezier` | source_edge logs: `/data/jiaxin/parametric_edge_prediction/outputs/sam2_source_edge_batch13_full_logs`, `/data/jiaxin/parametric_edge_prediction/outputs/sam2_source_edge_batch12_delayed_logs`; bezier logs: `/data/jiaxin/parametric_edge_prediction/outputs/bezier_from_source_edge_batch13_logs` | Current counts observed: `batch13 source_edge=23112`, `batch13 bezier=94`, `batch12 images=29344` |
| `lab21` | local -> `lab21`; `lab21` -> cluster; cluster cannot initiate to `lab21` | planned root: `/media/jiaxin/parametric_edge_prediction` | planned root: `/media/jiaxin/laion/edge_detection` | target next batch: `batch4` | target stage: `source_edge -> bezier` | planned source_edge root: `/media/jiaxin/laion/edge_detection/laion_edge_v3_source_edge`; planned bezier root: `/media/jiaxin/laion/edge_detection/laion_edge_v3_bezier` | planned logs: `/media/jiaxin/parametric_edge_prediction/outputs/bezier_from_source_edge_batch4_logs` | `lab21` is CPU-only (`56` CPU, `62GiB` RAM observed earlier). At the time of this snapshot, `lab21` was temporarily unreachable over ssh, so batch4 handoff was not yet completed. |
| `cluster` | reachable from lab machines; cannot ssh to lab machines | `/home/user/yc47434/parametric_edge_prediction` | `/home/user/yc47434/laion/edge_detection` | `batch1`, `batch2`, `batch4`, `batch5` | `batch4`: completed; `batch1`: active / near complete; `batch2`: active; `batch5`: queued / waiting | source_edge: `/home/user/yc47434/laion/edge_detection/laion_edge_v3_source_edge` | job logs under `/home/user/yc47434/cluster_runs/parametric_edge_prediction` | Current batch4 completed source_edge count observed: `29668` |

## Stage Legend

- `images`: raw image files are present, no stage output yet
- `source_edge`: SAM2 + mask-to-edge stage is producing or has produced `source_edge`
- `bezier`: `source_edge -> bezier` CPU stage is producing compact bezier caches

## `lab21` CPU Bezier Worker Plan

When `lab21` is reachable again, the next handoff is:

1. Create and use:
   - `/media/jiaxin/parametric_edge_prediction`
   - `/media/jiaxin/laion/edge_detection/laion_edge_v3_source_edge/batch4`
   - `/media/jiaxin/laion/edge_detection/laion_edge_v3_bezier/batch4`
   - `/media/jiaxin/parametric_edge_prediction/outputs/bezier_from_source_edge_batch4_logs`
2. Sync the bezierization code needed by:
   - `sam_mask_bezierization/`
   - `scripts/generate_bezier_from_source_edges_batch.py`
   - required repo-local bezierization imports
3. Relay completed cluster `batch4` source_edge via:
   - cluster -> `lab30` -> `lab21`
4. Start bezierization on `lab21` with `24` workers, CPU only.

CPU runtime defaults for `lab21`:

- `OMP_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
- `OPENBLAS_NUM_THREADS=1`
- `NUMEXPR_NUM_THREADS=1`
