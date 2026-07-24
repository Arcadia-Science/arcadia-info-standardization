"""
Dimensionality normalization transforms for mutual information.
"""

import numpy as np
from scipy import stats
from .estimators import chi2_df, g_statistic


def basharin_bias(k_x, k_y, k_z, N):
    """
    Expected value of plugin MI under independence.

    E[MI_plugin | H₀] = df / (2N ln 2)

    Parameters
    ----------
    k_x : int
        Number of categories in X
    k_y : int
        Number of categories in Y
    k_z : int
        Number of strata
    N : int
        Sample size

    Returns
    -------
    bias : float
        Expected MI in bits under null

    References
    ----------
    Basharin, G. P. (1959). On a statistical estimate for the entropy
    of a sequence of independent random variables. Theory of Probability
    & Its Applications, 4(3), 333-336.
    """
    df = chi2_df(k_x, k_y, k_z)
    return df / (2 * N * np.log(2))


def correct_basharin(mi, k_x, k_y, k_z, N):
    """
    Basharin correction: subtract mean bias.

    MI_corrected = MI_plugin - E[MI_plugin | H₀]

    Parameters
    ----------
    mi : float
        Plugin MI estimate in bits
    k_x : int
        Number of categories in X
    k_y : int
        Number of categories in Y
    k_z : int
        Number of strata
    N : int
        Sample size

    Returns
    -------
    mi_corrected : float
        Bias-corrected MI in bits

    Notes
    -----
    Corrects mean but NOT variance or higher moments.
    Not recommended for cross-K comparability.
    """
    return mi - basharin_bias(k_x, k_y, k_z, N)


def mi_to_z(mi, k_x, k_y, k_z, N):
    """
    DN-basic: standardized z-score from MI.

    z = (G - df) / √(2·df)

    where G = 2N ln(2) MI and df = (k_x-1)(k_y-1)k_z

    Parameters
    ----------
    mi : float
        Mutual information in bits
    k_x : int
        Number of categories in X
    k_y : int
        Number of categories in Y
    k_z : int
        Number of strata
    N : int
        Sample size

    Returns
    -------
    z : float
        Standardized z-score

    Notes
    -----
    Corrects mean and variance but NOT skewness or kurtosis.
    For small df, the chi-squared distribution is right-skewed,
    so this transformation is approximate.

    For exact cross-K comparability, use mi_to_z_cdf() instead.
    """
    G = g_statistic(mi, N)
    df = chi2_df(k_x, k_y, k_z)
    return (G - df) / np.sqrt(2.0 * df)


def mi_to_z_cdf(mi, k_x, k_y, k_z, N):
    """
    CDF transform: probability integral transform to N(0,1).

    z = Φ⁻¹(F_χ²(df)(G))

    where:
    - G = 2N ln(2) MI (G-statistic)
    - F_χ²(df) is the chi-squared CDF with df degrees of freedom
    - Φ⁻¹ is the inverse normal CDF

    This is the **recommended** transform for:
    - Cross-K comparability of MI values
    - Interaction detection in stratified analyses
    - Null hypothesis testing

    Parameters
    ----------
    mi : float or array_like
        Mutual information in bits
    k_x : int
        Number of categories in X
    k_y : int
        Number of categories in Y
    k_z : int
        Number of strata
    N : int or array_like
        Sample size (total count)

    Returns
    -------
    z : float or ndarray
        Z-score following N(0,1) under null hypothesis

    Notes
    -----
    Under the null hypothesis of independence (MI = 0), z ~ N(0,1) exactly,
    correcting ALL moments (mean, variance, skewness, kurtosis, etc.).

    The transformation is monotonic, so larger MI values always yield
    larger z values, but the scale is now comparable across different
    k_x, k_y, k_z combinations.

    Numerical stability: p-values are clipped to [1e-15, 1-1e-15] to avoid
    infinities at extreme tail probabilities.

    Examples
    --------
    >>> from dn_mi import plugin_mi_3d, mi_to_z_cdf
    >>> import numpy as np
    >>>
    >>> # Generate null data (independent X, Y given Z)
    >>> table = np.random.randint(0, 50, size=(3, 2, 10))
    >>> mi = plugin_mi_3d(table)
    >>> z = mi_to_z_cdf(mi, k_x=3, k_y=2, k_z=10, N=table.sum())
    >>>
    >>> # z should be ~N(0,1) under null
    >>> print(f"z = {z:.3f}")

    References
    ----------
    The probability integral transform (PIT) or inverse transform sampling
    is a fundamental result in probability theory. This implementation
    applies it to the chi-squared null distribution of the G-statistic.
    """
    G = g_statistic(mi, N)
    df = chi2_df(k_x, k_y, k_z)

    # Transform G to uniform via chi-squared CDF
    p = stats.chi2.cdf(G, df)

    # Handle edge cases for numerical stability
    p = np.clip(p, 1e-15, 1 - 1e-15)

    # Transform uniform to N(0,1) via inverse normal CDF
    z = stats.norm.ppf(p)

    return z


def sigma0_prediction(k_x, k_y):
    """
    Predicted σ₀ from chi-squared variance.

    σ₀ = √(2(k_x-1)(k_y-1) / (k_x·k_y·(2ln2)²))

    This is the theoretical standard deviation of MI under the null
    for a single partition (k_z=1), scaled by √C where C = k_x·k_y.

    Parameters
    ----------
    k_x : int
        Number of categories in X
    k_y : int
        Number of categories in Y

    Returns
    -------
    sigma0 : float
        Predicted σ₀ parameter

    Notes
    -----
    For k_x=3, k_y=2: σ₀ ≈ 0.588
    """
    return np.sqrt(2.0 * (k_x - 1) * (k_y - 1) /
                   (k_x * k_y * (2.0 * np.log(2)) ** 2))
