"""
DN-MI: Dimensionality Normalization for Mutual Information

A package for cross-dimensionality comparability of mutual information
using the CDF transform (probability integral transform).
"""

__version__ = "0.1.0"

# Import main API
from .estimators import (
    plugin_mi_2d,
    plugin_mi_3d,
    chi2_df,
    g_statistic,
)

from .transforms import (
    basharin_bias,
    correct_basharin,
    mi_to_z,
    mi_to_z_cdf,
    sigma0_prediction,
)

from .data import (
    partition_into_k,
    coarsen_partition,
    build_3d_table,
    generate_null,
    generate_2x2_with_mi,
)

__all__ = [
    # Estimators
    "plugin_mi_2d",
    "plugin_mi_3d",
    "chi2_df",
    "g_statistic",
    # Transforms
    "basharin_bias",
    "correct_basharin",
    "mi_to_z",
    "mi_to_z_cdf",
    "sigma0_prediction",
    # Data utilities
    "partition_into_k",
    "coarsen_partition",
    "build_3d_table",
    "generate_null",
    "generate_2x2_with_mi",
]
