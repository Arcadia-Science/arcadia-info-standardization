"""
Plugin mutual information estimators for discrete contingency tables.
"""

import numpy as np


def plugin_mi_2d(table):
    """
    Plugin MI estimator for a 2D contingency table.

    Computes I(X; Y) = Σ p(x,y) log₂(p(x,y) / (p(x)p(y)))

    Parameters
    ----------
    table : ndarray, shape (k_x, k_y)
        2D contingency table with counts

    Returns
    -------
    mi : float
        Mutual information in bits (≥ 0)

    Examples
    --------
    >>> table = np.array([[10, 20], [30, 40]])
    >>> mi = plugin_mi_2d(table)
    """
    table = table.astype(float)
    n = table.sum()
    if n == 0:
        return 0.0

    p_xy = table / n
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)

    mi = 0.0
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))

    return max(mi, 0.0)


def plugin_mi_3d(table):
    """
    Plugin conditional MI estimator for a 3D contingency table.

    Computes CMI: I(X; Y | Z) = Σ p(x,y,z) log₂(p(x,y|z) / (p(x|z)p(y|z)))

    Parameters
    ----------
    table : ndarray, shape (k_x, k_y, k_z)
        3D contingency table with counts

    Returns
    -------
    cmi : float
        Conditional mutual information in bits (≥ 0)

    Notes
    -----
    Computes MI within each stratum k and averages weighted by p(z=k).

    Examples
    --------
    >>> table = np.random.randint(0, 100, size=(3, 2, 5))
    >>> cmi = plugin_mi_3d(table)
    """
    n = table.sum()
    if n == 0:
        return 0.0

    cmi = 0.0
    for k in range(table.shape[2]):
        slice_k = table[:, :, k]
        p_k = slice_k.sum() / n
        if p_k > 0:
            mi_k = plugin_mi_2d(slice_k)
            cmi += p_k * mi_k

    return max(cmi, 0.0)


def chi2_df(k_x, k_y, k_z=1):
    """
    Degrees of freedom for the chi-squared null of plugin MI/CMI.

    df = (k_x - 1)(k_y - 1)k_z

    Parameters
    ----------
    k_x : int
        Number of categories in X
    k_y : int
        Number of categories in Y
    k_z : int, optional
        Number of strata (default=1 for unconditional MI)

    Returns
    -------
    df : int
        Degrees of freedom

    Examples
    --------
    >>> chi2_df(3, 2, 10)
    20
    """
    return (k_x - 1) * (k_y - 1) * k_z


def g_statistic(mi, N):
    """
    G-statistic (likelihood ratio test statistic) from plugin MI.

    G = 2N ln(2) MI

    Under the null hypothesis of independence, G ~ χ²(df).

    Parameters
    ----------
    mi : float
        Mutual information in bits
    N : int
        Sample size (total count)

    Returns
    -------
    G : float
        G-statistic

    Examples
    --------
    >>> mi = 0.1
    >>> N = 1000
    >>> G = g_statistic(mi, N)
    >>> print(f"G = {G:.2f}")
    """
    return 2 * N * np.log(2) * mi
