from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.segmentation import find_boundaries

from sam_mask_bezierization.mask_to_edge_methods import (
    masks_to_label_image,
    polish_masks,
    suppress_near_duplicate_masks,
)
from sam_mask_bezierization.pipeline import (
    apply_keep95,
    build_mask_generator,
    detect_small_bubbles,
    generate_masks,
    prune_tiny_edge_cc,
    repair_global_band_thin,
)


VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-name", type=str, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--worker-role", choices=("front", "back"), required=True)
    parser.add_argument("--points-per-side", type=int, default=64)
    parser.add_argument("--stability-score-thresh", type=float, default=0.85)
    parser.add_argument("--crop-n-layers", type=int, default=1)
    parser.add_argument("--keep-ratio", type=float, default=0.95)
    parser.add_argument("--bubble-area-limit", type=int, default=20)
    parser.add_argument("--bubble-area-ratio", type=float, default=0.0008)
    parser.add_argument("--tiny-cc-limit", type=int, default=15)
    parser.add_argument("--log-every", type=int, default=5)
    return parser.parse_args()


def iter_images(images_dir: Path, limit: int, worker_role: str) -> list[Path]:
    images = sorted(
        [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS],
        key=lambda p: p.stem,
    )
    if limit > 0:
        images = images[:limit]
    if worker_role == "back":
        images = list(reversed(images))
    return images


def save_binary_png(path: Path, edge: np.ndarray) -> None:
    tmp_path = path.with_name(f"{path.stem}.{os.getpid()}.tmp{path.suffix}")
    Image.fromarray((edge.astype(np.uint8) * 255)).save(tmp_path, format="PNG")
    tmp_path.replace(path)


def build_source_edge(
    image_path: Path,
    mask_generator,
    keep_ratio: float,
    bubble_area_limit: int,
    bubble_area_ratio: float,
    tiny_cc_limit: int,
) -> np.ndarray:
    image_np = np.asarray(Image.open(image_path).convert("RGB"))
    raw_masks = generate_masks(mask_generator, image_np)
    polished_masks = suppress_near_duplicate_masks(polish_masks(raw_masks, gap_threshold=2))
    kept_masks, _ = apply_keep95(polished_masks, keep_ratio=keep_ratio)
    label_image = masks_to_label_image(kept_masks)
    raw_edge = find_boundaries(label_image, mode="inner", background=0).astype(bool)
    del bubble_area_limit
    bubble_limit = max(1, int(raw_edge.shape[0] * raw_edge.shape[1] * bubble_area_ratio))
    selected_bubbles = detect_small_bubbles(raw_edge, max_bubble_area=bubble_limit)
    source_edge = repair_global_band_thin(raw_edge, selected_bubbles, band_radius=1)
    source_edge, _, _ = prune_tiny_edge_cc(source_edge, pixel_limit=tiny_cc_limit)
    return source_edge


def main() -> int:
    args = parse_args()

    output_batch_dir = args.output_root / args.batch_name
    output_batch_dir.mkdir(parents=True, exist_ok=True)

    images = iter_images(args.images_dir, args.limit, args.worker_role)
    if not images:
        print("No images found.", flush=True)
        return 1

    print(
        f"worker_role={args.worker_role} batch={args.batch_name} "
        f"images={len(images)} output={output_batch_dir}",
        flush=True,
    )
    print(f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)

    mask_generator = build_mask_generator(
        points_per_side=args.points_per_side,
        stability_score_thresh=args.stability_score_thresh,
        crop_n_layers=args.crop_n_layers,
    )

    processed = 0
    skipped_existing = 0
    failed = 0
    start_time = time.time()

    for idx, image_path in enumerate(images, start=1):
        sample_id = image_path.stem
        output_path = output_batch_dir / f"{sample_id}.png"

        if output_path.exists():
            skipped_existing += 1
            continue

        try:
            if output_path.exists():
                skipped_existing += 1
                continue
            edge = build_source_edge(
                image_path=image_path,
                mask_generator=mask_generator,
                keep_ratio=args.keep_ratio,
                bubble_area_limit=args.bubble_area_limit,
                bubble_area_ratio=args.bubble_area_ratio,
                tiny_cc_limit=args.tiny_cc_limit,
            )
            save_binary_png(output_path, edge)
            processed += 1
        except Exception as exc:  # pragma: no cover - operational path
            failed += 1
            print(f"[ERROR] {sample_id}: {exc}", file=sys.stderr, flush=True)

        if idx % max(args.log_every, 1) == 0:
            elapsed = time.time() - start_time
            print(
                f"[{idx}/{len(images)}] processed={processed} "
                f"skipped_existing={skipped_existing} "
                f"failed={failed} elapsed_sec={elapsed:.1f}",
                flush=True,
            )

    elapsed = time.time() - start_time
    print(
        f"DONE processed={processed} skipped_existing={skipped_existing} "
        f"failed={failed} elapsed_sec={elapsed:.1f}",
        flush=True,
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
