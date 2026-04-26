# Parametric Edge Docs

This directory only keeps the current, maintained documentation for the active code paths.

## Read This First

- [current_repo_structure.md](./current_repo_structure.md)
  - what each top-level module is for
  - which scripts are active
  - where preprocessing code vs training code lives
- [training_and_submission.md](./training_and_submission.md)
  - config inheritance
  - active training configs
  - submit scripts
  - current runtime defaults
- [curve_vs_endpoint_dab.md](./curve_vs_endpoint_dab.md)
  - current difference between curve DAB and endpoint DAB
  - includes the attach branch and conditioned-curve branch
- [lab_cluster_practical_manual.md](./lab_cluster_practical_manual.md)
  - machine roots
  - env mapping
  - connectivity rules
  - sync / submit workflow

## Current End-to-End Data Flow

```text
images
  -> SAM2 masks
  -> source_edge PNGs
  -> bezier NPZs
  -> entry cache TXT
  -> dataloader
  -> DAB training
```

## Current Training Branches

- direct curve DAB
- endpoint DAB
- endpoint-attach DAB
- endpoint-conditioned curve DAB

## Scope Rule

The old docs for historical batch status, removed training stacks, and outdated structure diagrams were intentionally removed.

If something changes in code structure, configs, or launch flow, update the matching doc in the same change.
