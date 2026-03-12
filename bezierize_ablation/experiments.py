EXPERIMENTS = [
    {
        'name': 'version_v1_greedy_hybrid',
        'kind': 'version',
        'version': 'v1_greedy_hybrid',
        'overrides': {},
        'notes': 'Initial hybrid piecewise fitting baseline.',
    },
    {
        'name': 'version_v2_tiny_cleanup',
        'kind': 'version',
        'version': 'v2_tiny_cleanup',
        'overrides': {},
        'notes': 'Dense rasterization and early tiny-cleanup variant.',
    },
    {
        'name': 'version_v3_trivial_pruned',
        'kind': 'version',
        'version': 'v3_trivial_pruned',
        'overrides': {},
        'notes': 'Adds trivial original-path filtering.',
    },
    {
        'name': 'version_v4_adjacent_merge',
        'kind': 'version',
        'version': 'v4_adjacent_merge',
        'overrides': {},
        'notes': 'Trivial filtering, tiny cleanup, and easy adjacent merge.',
    },
    {
        'name': 'version_v5_anchor_consistent',
        'kind': 'version',
        'version': 'v5_anchor_consistent',
        'overrides': {},
        'notes': 'Natural-anchor-aware split placement with smoother, more consistent long-path segmentation.',
    },
    {
        'name': 'hp_v4_minpath4',
        'kind': 'hyperparam',
        'version': 'v4_adjacent_merge',
        'overrides': {
            'min_path_length_for_bezier': 4.0,
        },
        'notes': 'Looser trivial-path filtering; keeps more short original paths.',
    },
    {
        'name': 'hp_v4_minpath8',
        'kind': 'hyperparam',
        'version': 'v4_adjacent_merge',
        'overrides': {
            'min_path_length_for_bezier': 8.0,
        },
        'notes': 'More aggressive trivial-path filtering.',
    },
    {
        'name': 'hp_v4_loose_fit',
        'kind': 'hyperparam',
        'version': 'v4_adjacent_merge',
        'overrides': {
            'mean_error_threshold': 0.75,
            'max_error_threshold': 2.5,
        },
        'notes': 'Looser fit tolerances; favors fewer curves.',
    },
    {
        'name': 'hp_v4_strict_fit',
        'kind': 'hyperparam',
        'version': 'v4_adjacent_merge',
        'overrides': {
            'mean_error_threshold': 0.5,
            'max_error_threshold': 1.5,
        },
        'notes': 'Stricter fit tolerances; favors tighter local approximation.',
    },
]
