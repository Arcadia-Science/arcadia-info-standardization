"""
Utility functions for generating synthetic data and contingency tables.
"""

import numpy as np


def partition_into_k(N, k_z, rng):
    """
    Generate random partition assignment into k_z groups.

    Parameters
    ----------
    N : int
        Sample size
    k_z : int
        Number of partitions
    rng : numpy.random.Generator
        Random number generator

    Returns
    -------
    p : ndarray, shape (N,)
        Partition assignments [0, k_z)
    """
    p = np.repeat(np.arange(k_z), N // k_z + 1)[:N]
    rng.shuffle(p)
    return p


def coarsen_partition(p_max, k_z_max, k_z):
    """
    Coarsen a fine partition to fewer groups.

    Parameters
    ----------
    p_max : ndarray
        Fine partition assignments [0, k_z_max)
    k_z_max : int
        Number of groups in fine partition
    k_z : int
        Target number of groups (k_z ≤ k_z_max)

    Returns
    -------
    p : ndarray
        Coarsened partition assignments [0, k_z)
    """
    return (p_max * k_z // k_z_max).astype(int)


def build_3d_table(g, d, p, k_x, k_y, k_z):
    """
    Build 3D contingency table from categorical assignments.

    Parameters
    ----------
    g : ndarray, shape (N,)
        Categories for variable X (values in [0, k_x))
    d : ndarray, shape (N,)
        Categories for variable Y (values in [0, k_y))
    p : ndarray, shape (N,)
        Partition/stratum assignments (values in [0, k_z))
    k_x : int
        Number of categories in X
    k_y : int
        Number of categories in Y
    k_z : int
        Number of strata

    Returns
    -------
    table : ndarray, shape (k_x, k_y, k_z)
        3D contingency table with counts
    """
    table = np.zeros((k_x, k_y, k_z), dtype=float)
    for i in range(len(g)):
        table[g[i], d[i], p[i]] += 1
    return table


def generate_null(N, k_x, k_z_max, strat_effect=0.0, noise=0.0, rng=None):
    """
    Generate null data where X and Y are independent given partition Z.

    Parameters
    ----------
    N : int
        Sample size
    k_x : int
        Number of categories for X (genotype)
    k_z_max : int
        Number of partitions
    strat_effect : float, optional
        Strength of stratification effect on allele frequency (default: 0.0)
    noise : float, optional
        Random noise added to allele frequencies (default: 0.0)
    rng : numpy.random.Generator, optional
        Random number generator (default: creates new one)

    Returns
    -------
    g : ndarray, shape (N,)
        Genotype categories [0, k_x)
    d : ndarray, shape (N,)
        Disease status (binary: 0 or 1)
    p : ndarray, shape (N,)
        Partition assignments [0, k_z_max)

    Notes
    -----
    Simulates genetic association study with population structure:
    - g (genotype): depends on partition (population structure)
    - d (disease): independent of g given p (null hypothesis)
    - p (partition): random stratification

    When strat_effect > 0, allele frequencies vary across partitions.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Random partition
    p = partition_into_k(N, k_z_max, rng)

    # Allele frequencies vary by partition if strat_effect > 0
    p_center = (k_z_max - 1) / 2.0
    base_af = np.clip(
        0.30 + strat_effect * (np.arange(k_z_max) - p_center) / max(p_center, 1),
        0.05, 0.95
    )

    if noise > 0:
        base_af = np.clip(base_af + rng.normal(0, noise, k_z_max), 0.05, 0.95)

    af = base_af[p]

    # Generate genotypes (Hardy-Weinberg equilibrium)
    u = rng.uniform(size=N)
    g = np.zeros(N, dtype=int)
    g[u < af ** 2] = 0  # AA
    g[(u >= af ** 2) & (u < af ** 2 + 2 * af * (1 - af))] = 1  # Aa
    g[u >= af ** 2 + 2 * af * (1 - af)] = 2  # aa

    # Ensure we don't exceed k_x categories
    g = np.clip(g, 0, k_x - 1)

    # Generate disease status (independent of genotype given partition)
    # Fixed disease prevalence
    d = rng.integers(0, 2, size=N)

    return g, d, p


def generate_2x2_with_mi(N, true_mi_bits, rng=None):
    """
    Generate a 2x2 contingency table with specified mutual information.

    Uses iterative sampling to achieve target MI under multinomial model.

    Parameters
    ----------
    N : int
        Sample size
    true_mi_bits : float
        Target mutual information in bits
    rng : numpy.random.Generator, optional
        Random number generator

    Returns
    -------
    table : ndarray, shape (2, 2)
        2x2 contingency table

    Notes
    -----
    Generates table with positive association. For true_mi_bits=0,
    generates independent variables.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Compute 2x2 cell probabilities for target MI
    # Using simple parameterization: vary p(1,1) to control MI
    if true_mi_bits <= 0:
        # Independent: uniform marginals
        probs = np.array([[0.25, 0.25], [0.25, 0.25]])
    else:
        # Binary search for joint probability that yields target MI
        from .estimators import plugin_mi_2d

        def mi_for_p11(p11):
            p10 = 0.5 - p11
            p01 = 0.5 - p11
            p00 = p11
            probs_trial = np.array([[p00, p01], [p10, p11]])
            # Normalize
            probs_trial = probs_trial / probs_trial.sum()
            # Compute MI on large sample to approximate true MI
            table_trial = (probs_trial * 100000).astype(int)
            return plugin_mi_2d(table_trial)

        # Binary search
        lo, hi = 0.26, 0.5
        for _ in range(20):
            mid = (lo + hi) / 2
            mi_mid = mi_for_p11(mid)
            if mi_mid < true_mi_bits:
                lo = mid
            else:
                hi = mid

        p11 = (lo + hi) / 2
        p10 = 0.5 - p11
        p01 = 0.5 - p11
        p00 = p11
        probs = np.array([[p00, p01], [p10, p11]])
        probs = probs / probs.sum()

    # Sample from multinomial
    probs_flat = probs.flatten()
    counts = rng.multinomial(N, probs_flat)
    table = counts.reshape(2, 2)

    return table
