"""
Tests for DN-MI estimators and transforms.
"""

import numpy as np
import pytest
from scipy import stats

from dn_mi import (
    plugin_mi_2d,
    plugin_mi_3d,
    chi2_df,
    g_statistic,
    correct_basharin,
    mi_to_z,
    mi_to_z_cdf,
)


def test_plugin_mi_2d_independent():
    """Test that MI is near zero for independent variables."""
    # Perfect independence: uniform distribution
    table = np.ones((3, 2)) * 100
    mi = plugin_mi_2d(table)
    assert mi < 1e-10, "MI should be near zero for independent variables"


def test_plugin_mi_2d_perfect_dependence():
    """Test that MI is positive for dependent variables."""
    # Strong association
    table = np.array([[100, 0], [0, 100]])
    mi = plugin_mi_2d(table)
    assert mi > 0.5, "MI should be large for perfectly dependent variables"


def test_plugin_mi_3d_zero_for_null():
    """Test that average MI across strata is near zero under null."""
    # Independent in each stratum
    table = np.ones((3, 2, 5)) * 10
    mi = plugin_mi_3d(table)
    assert mi < 1e-10, "CMI should be near zero when independent in all strata"


def test_chi2_df():
    """Test degrees of freedom calculation."""
    assert chi2_df(3, 2, 1) == 2
    assert chi2_df(3, 2, 10) == 20
    assert chi2_df(4, 3, 5) == 30


def test_g_statistic_zero_mi():
    """Test G-statistic is zero when MI is zero."""
    G = g_statistic(0.0, 1000)
    assert G == 0.0


def test_g_statistic_positive():
    """Test G-statistic is positive for positive MI."""
    G = g_statistic(0.1, 1000)
    assert G > 0


def test_basharin_correction_reduces_bias():
    """Test that Basharin correction reduces mean bias under null."""
    # Generate multiple null samples to test mean bias reduction
    rng = np.random.default_rng(42)
    N = 1000
    k_x, k_y, k_z = 3, 2, 10
    n_simulations = 100

    mi_raw_values = []
    mi_corrected_values = []

    for _ in range(n_simulations):
        # Generate null data
        table = rng.multinomial(N, np.ones(k_x * k_y * k_z) / (k_x * k_y * k_z))
        table = table.reshape(k_x, k_y, k_z)

        mi_raw = plugin_mi_3d(table)
        mi_corrected = correct_basharin(mi_raw, k_x, k_y, k_z, N)

        mi_raw_values.append(mi_raw)
        mi_corrected_values.append(mi_corrected)

    # Mean of corrected values should be closer to zero than raw
    mean_raw = np.mean(mi_raw_values)
    mean_corrected = np.mean(mi_corrected_values)

    assert abs(mean_corrected) < abs(mean_raw), \
        f"Corrected mean ({mean_corrected:.6f}) should be closer to 0 than raw ({mean_raw:.6f})"


def test_mi_to_z_standardizes():
    """Test that DN-basic produces reasonable z-scores."""
    # Generate some null data
    rng = np.random.default_rng(42)
    N = 10000
    k_x, k_y, k_z = 3, 2, 10

    # Uniform table (approximate null)
    table = rng.multinomial(N, np.ones(k_x * k_y * k_z) / (k_x * k_y * k_z))
    table = table.reshape(k_x, k_y, k_z)

    mi = plugin_mi_3d(table)
    z = mi_to_z(mi, k_x, k_y, k_z, N)

    # Should be within reasonable range for null
    assert -5 < z < 5, "z-score should be in reasonable range under null"


def test_mi_to_z_cdf_null_distribution():
    """Test that CDF transform produces ~N(0,1) under null."""
    rng = np.random.default_rng(42)
    N = 10000
    k_x, k_y, k_z = 3, 2, 10
    n_simulations = 100

    z_values = []
    for _ in range(n_simulations):
        # Generate null data
        table = rng.multinomial(N, np.ones(k_x * k_y * k_z) / (k_x * k_y * k_z))
        table = table.reshape(k_x, k_y, k_z)

        mi = plugin_mi_3d(table)
        z = mi_to_z_cdf(mi, k_x, k_y, k_z, N)
        z_values.append(z)

    z_values = np.array(z_values)

    # Test that distribution looks approximately N(0,1)
    mean = np.mean(z_values)
    std = np.std(z_values)

    assert abs(mean) < 0.3, f"Mean should be near 0, got {mean}"
    assert abs(std - 1) < 0.3, f"Std should be near 1, got {std}"

    # KS test for normality
    ks_stat, ks_pval = stats.kstest(z_values, 'norm')
    assert ks_pval > 0.01, f"Distribution should pass normality test (p={ks_pval})"


def test_mi_to_z_cdf_monotonic():
    """Test that CDF transform is monotonic in MI."""
    N = 1000
    k_x, k_y, k_z = 3, 2, 10

    mi_values = [0.001, 0.01, 0.05, 0.1]
    z_values = [mi_to_z_cdf(mi, k_x, k_y, k_z, N) for mi in mi_values]

    # Should be strictly increasing
    for i in range(len(z_values) - 1):
        assert z_values[i] < z_values[i+1], "CDF transform should be monotonic"


def test_edge_cases():
    """Test edge cases."""
    # Empty table
    assert plugin_mi_2d(np.zeros((3, 2))) == 0.0
    assert plugin_mi_3d(np.zeros((3, 2, 5))) == 0.0

    # Very small MI
    mi = 1e-10
    z = mi_to_z_cdf(mi, 3, 2, 10, 1000)
    assert np.isfinite(z), "Should handle very small MI values"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
