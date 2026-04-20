from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite a v3 bezier entry cache to match local image/bezier roots.")
    parser.add_argument("--input-cache", required=True)
    parser.add_argument("--output-cache", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--bezier-root", required=True)
    args = parser.parse_args()

    input_cache = Path(args.input_cache)
    output_cache = Path(args.output_cache)
    image_root = Path(args.image_root)
    bezier_root = Path(args.bezier_root)

    output_cache.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output_cache.with_suffix(output_cache.suffix + ".tmp") if output_cache == input_cache else output_cache
    count = 0
    with input_cache.open("r", encoding="utf-8") as src, temp_output.open("w", encoding="utf-8") as dst:
        for raw_line in src:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            batch_name, image_id, *_rest = line.split("\t")
            image_path = image_root / batch_name / "images" / f"{image_id}.jpg"
            if not image_path.exists():
                alt = image_root / batch_name / "images" / f"{image_id}.png"
                if alt.exists():
                    image_path = alt
            bezier_path = bezier_root / batch_name / f"{image_id}.npz"
            dst.write(f"{batch_name}\t{image_id}\t{image_path}\t{bezier_path}\n")
            count += 1
    if temp_output != output_cache:
        temp_output.replace(output_cache)
    print(f"rewrote {count} entries -> {output_cache}")


if __name__ == "__main__":
    main()
