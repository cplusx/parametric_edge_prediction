# Bezierization

This directory contains the Bezier-curve conversion pipeline used to turn binary edge maps into curve-based training targets.

## Main entrypoints
- `edge_refiner.py`: default single-image Bezierization entrypoint.
- `ablation_api.py`: unified version-loading API.
- `bezier_refiner_core.py`: shared curve fitting, segmentation, cleanup, and visualization utilities.

## Subdirectories
- `bezier_versions/`: versioned Bezierization variants for ablation.
- `bezierize_ablation/`: evaluation and reporting scripts.
- `bezierize_animation/`: comparison-video generation scripts.
- `docs/`: iteration notes and method history.
- `scripts/`: batch helpers.

## Compatibility
Thin wrappers remain at the repository root so old commands still run, but new code should import from `bezierization.*`.
