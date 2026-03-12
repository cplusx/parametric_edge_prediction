import argparse
import json
import os
import subprocess
import sys


SELECTED_SAMPLES = [
    {
        'image_path': 'gt_rgb/test/51084_ann1.png',
        'reason': 'Strong trivial-path filtering and the clearest adjacent-merge reduction.',
        'metrics': {'dropped_paths': 192, 'cleanup_delta': 2, 'merge_delta': 5, 'final_segments': 96},
    },
    {
        'image_path': 'gt_rgb/test/207038_ann4.png',
        'reason': 'Large path drop and visible cleanup plus merge changes on a dense structure.',
        'metrics': {'dropped_paths': 327, 'cleanup_delta': 8, 'merge_delta': 3, 'final_segments': 172},
    },
    {
        'image_path': 'gt_rgb/test/48025_ann3.png',
        'reason': 'Strong path filtering with a visible adjacent-merge step on long boundaries.',
        'metrics': {'dropped_paths': 318, 'cleanup_delta': 1, 'merge_delta': 3, 'final_segments': 111},
    },
    {
        'image_path': 'gt_rgb/test/97010_ann4.png',
        'reason': 'Balanced example where all stages are present without an overly crowded scene.',
        'metrics': {'dropped_paths': 142, 'cleanup_delta': 1, 'merge_delta': 2, 'final_segments': 57},
    },
    {
        'image_path': 'gt_rgb/test/225022_ann5.png',
        'reason': 'All stages trigger and the later merge is still readable at moderate complexity.',
        'metrics': {'dropped_paths': 92, 'cleanup_delta': 2, 'merge_delta': 2, 'final_segments': 75},
    },
]


def main():
    parser = argparse.ArgumentParser(description='Generate representative Bezier comparison videos.')
    parser.add_argument('--output-dir', default='bezierize_animation/outputs/representative_videos')
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--fit-frames', type=int, default=36)
    parser.add_argument('--hold-frames', type=int, default=18)
    parser.add_argument('--fade-frames', type=int, default=12)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    script_path = os.path.join(os.path.dirname(__file__), 'make_comparison_animation.py')
    manifest = []
    for sample in SELECTED_SAMPLES:
        image_path = sample['image_path']
        stem = os.path.splitext(os.path.basename(image_path))[0]
        sample_output_dir = os.path.join(args.output_dir, stem)
        os.makedirs(sample_output_dir, exist_ok=True)
        cmd = [
            sys.executable,
            script_path,
            '--input',
            image_path,
            '--output-dir',
            sample_output_dir,
            '--fps',
            str(args.fps),
            '--fit-frames',
            str(args.fit_frames),
            '--hold-frames',
            str(args.hold_frames),
            '--fade-frames',
            str(args.fade_frames),
        ]
        subprocess.run(cmd, check=True)
        manifest.append(
            {
                'image_path': image_path,
                'output_dir': sample_output_dir,
                'reason': sample['reason'],
                'metrics': sample['metrics'],
            }
        )

    manifest_path = os.path.join(args.output_dir, 'selection_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    print(manifest_path)


if __name__ == '__main__':
    main()
