from bezierization.bezier_versions.v1_hybrid_initial import VERSION_NAME as V1_NAME
from bezierization.bezier_versions.v2_dense_sampling import VERSION_NAME as V2_NAME
from bezierization.bezier_versions.v3_trivial_filter import VERSION_NAME as V3_NAME
from bezierization.bezier_versions.v4_current import VERSION_NAME as V4_NAME
from bezierization.bezier_versions.v5_human_split import VERSION_NAME as V5_NAME

VERSION_REGISTRY = {
    V1_NAME: 'bezierization.bezier_versions.v1_hybrid_initial',
    V2_NAME: 'bezierization.bezier_versions.v2_dense_sampling',
    V3_NAME: 'bezierization.bezier_versions.v3_trivial_filter',
    V4_NAME: 'bezierization.bezier_versions.v4_current',
    V5_NAME: 'bezierization.bezier_versions.v5_human_split',
    'v1_hybrid_initial': 'bezierization.bezier_versions.v1_hybrid_initial',
    'v2_dense_sampling': 'bezierization.bezier_versions.v2_dense_sampling',
    'v3_trivial_filter': 'bezierization.bezier_versions.v3_trivial_filter',
    'v4_current': 'bezierization.bezier_versions.v4_current',
    'v5_human_split': 'bezierization.bezier_versions.v5_human_split',
}

CANONICAL_VERSIONS = [
    V1_NAME,
    V2_NAME,
    V3_NAME,
    V4_NAME,
    V5_NAME,
]
