"""
figures.py
==========
Standalone script to generate publication figures for:
  "Dimensionality Normalization: A Unified Chi-Squared Framework for
   Mutual Information, CMI, Entropy, and KL Divergence"

Generates 8 figures in PNG and SVG format:
  Figure 1: experiment_1 output (correction progression)
  Figure 2: experiment_2 output (G-statistic validation)
  Figure 3: experiment_16 output (cross-dataset comparability)
  Figure 4: custom combined figure (exp4 right + exp5 left)
  Figure 5: experiment_23 output (normality across N and k_z)
  Figure 6: experiment_6_7 output (validity regime)
  Figure 7: experiment_6_5 output (tail calibration)
  Figure 8: experiment_20 output (DN vs permutation)

Usage
-----
  python figures.py --save-dir publication_figs/
  python figures.py --experiments 1 2 3 --save-dir figs/
  python figures.py --fast
"""

import argparse
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import stats

from dn_mi import (
    __version__ as dn_mi_version,
    build_3d_table,
    chi2_df,
    coarsen_partition,
    correct_basharin,
    g_statistic,
    mi_to_z,
    mi_to_z_cdf,
    partition_into_k,
    plugin_mi_2d,
    plugin_mi_3d,
    sigma0_prediction,
)


# =============================================================================
# CONSTANTS AND DEFAULTS
# =============================================================================

# Arcadia color palette (from matplotlibrc)
ARCADIA_COLORS = ['#5088C5', '#F28360', '#3B9886', '#F7B846', '#7A77AB',
                  '#F898AE', '#97CD78', '#73B5E3', '#FFB984', '#BAB0A8',
                  '#C85152', '#8A99AD']

# Consistent k_z color mapping (gradient from blue to orange)
# Using a gradient across the Arcadia palette for k_z values
K_Z_COLORS = {
    6: '#5088C5',    # Blue
    10: '#73B5E3',   # Light blue
    30: '#3B9886',   # Teal
    50: '#F7B846',   # Yellow
    70: '#F28360',   # Orange
    100: '#C85152',  # Red
    # Additional values that might appear
    2: '#5088C5',
    3: '#5fa0d4',
    5: '#6fb0e0',
    15: '#3B9886',
    20: '#3B9886',
    200: '#8A3F3F'
}

SIGMA_0 = 0.588  # DN parameter: sqrt(2(kx-1)(ky-1) / (kx*ky*(2ln2)^2)) for kx=3,ky=2

DEFAULTS = dict(
    N=100_000,
    k_x=3,
    k_y=2,
    k_z_baseline=10,
    k_z_values=(10, 30, 50, 70, 100),
    strat_effects=(0.0, 0.0),
    within_strat_noise=0.0,
    n_variants=500,
    n_perm_variants=50,
    n_perms=20,
    seed=42,
)

FAST = dict(n_variants=100, n_perm_variants=20, n_perms=10)


def chi2_quantiles_dn(probs, df):
    """Quantiles of standardized chi-squared: (chi2(df)-df)/sqrt(2*df)."""
    q = stats.chi2.ppf(probs, df)
    return (q - df) / np.sqrt(2.0 * df)


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_figure_null(N, k_x, k_z_max, strat_effect, noise, rng):
    """
    Null variant: G and D independent given partition.
    """
    p = partition_into_k(N, k_z_max, rng)
    p_center = (k_z_max - 1) / 2.0
    base_af = np.clip(
        0.30 + strat_effect * (np.arange(k_z_max) - p_center) / max(p_center, 1),
        0.05, 0.95)
    if noise > 0:
        base_af = np.clip(base_af + rng.normal(0, noise, k_z_max), 0.05, 0.95)
    af = base_af[p]
    g = (rng.uniform(size=N) < af).astype(int) + \
        (rng.uniform(size=N) < af).astype(int)
    d = rng.integers(0, 2, size=N)
    return g, d, p


def compute_2x2_probs_with_mi(true_mi_bits, rng):
    """
    Compute 2x2 joint probability distribution with target MI.
    """
    from scipy.optimize import brentq

    if true_mi_bits <= 0:
        # Independent variables
        p_x = rng.uniform(0.3, 0.7)
        p_y = rng.uniform(0.3, 0.7)
        p = np.array([[p_x * p_y, p_x * (1 - p_y)],
                      [(1 - p_x) * p_y, (1 - p_x) * (1 - p_y)]])
    else:
        # Choose marginals
        p_x = rng.uniform(0.3, 0.7)
        p_y = rng.uniform(0.3, 0.7)

        p_11_min = max(0, p_x + p_y - 1) + 1e-6
        p_11_max = min(p_x, p_y) - 1e-6

        def mi_from_p11(p_11):
            p_joint = np.array([
                [p_11, p_x - p_11],
                [p_y - p_11, 1 - p_x - p_y + p_11]
            ])
            mi_val = 0.0
            p_x_arr = p_joint.sum(axis=1)
            p_y_arr = p_joint.sum(axis=0)
            for i in range(2):
                for j in range(2):
                    if p_joint[i, j] > 0:
                        mi_val += p_joint[i, j] * np.log2(
                            p_joint[i, j] / (p_x_arr[i] * p_y_arr[j]))
            return mi_val

        p_11_indep = p_x * p_y
        mi_max = mi_from_p11(p_11_max)

        if true_mi_bits >= mi_max:
            p_11 = p_11_max
        else:
            try:
                p_11 = brentq(lambda x: mi_from_p11(x) - true_mi_bits,
                             p_11_indep, p_11_max)
            except ValueError:
                frac = true_mi_bits / mi_max
                p_11 = p_11_indep + frac * (p_11_max - p_11_indep)

        p = np.array([
            [p_11, p_x - p_11],
            [p_y - p_11, 1 - p_x - p_y + p_11]
        ])
        p = np.clip(p, 1e-6, 1)
        p = p / p.sum()

    return p


def generate_figure_2x2_with_mi(N, true_mi_bits, rng):
    """
    Generate a 2x2 contingency table with exact MI.
    """
    p = compute_2x2_probs_with_mi(true_mi_bits, rng)
    counts = rng.multinomial(N, p.ravel()).reshape(2, 2)
    return counts


def permutation_test(x, y, k_x, k_y, n_perms, rng):
    """
    Permutation test for MI significance.
    """
    # Observed MI
    table_obs = np.zeros((k_x, k_y))
    for i in range(len(x)):
        table_obs[x[i], y[i]] += 1
    mi_obs = plugin_mi_2d(table_obs)

    # Permutation distribution
    mi_null = []
    y_perm = y.copy()
    for _ in range(n_perms):
        rng.shuffle(y_perm)
        table_perm = np.zeros((k_x, k_y))
        for i in range(len(x)):
            table_perm[x[i], y_perm[i]] += 1
        mi_null.append(plugin_mi_2d(table_perm))

    # Two-sided p-value
    p_value = np.mean(np.abs(mi_null) >= np.abs(mi_obs))
    return p_value


# =============================================================================
# UTILITIES
# =============================================================================

def arcadia_colors(n=None):
    """
    Get Arcadia color palette.
    """
    if n is None:
        return ARCADIA_COLORS
    elif n == 1:
        return ARCADIA_COLORS[0]
    else:
        # Cycle through colors if more requested than available
        return [ARCADIA_COLORS[i % len(ARCADIA_COLORS)] for i in range(n)]


def get_kz_colors(k_z_values):
    """
    Get consistent colors for k_z values using the K_Z_COLORS mapping.
    Falls back to gradient interpolation if value not in mapping.
    """
    colors = []
    for k_z in k_z_values:
        if k_z in K_Z_COLORS:
            colors.append(K_Z_COLORS[k_z])
        else:
            # Fallback: use arcadia_colors
            idx = len(colors) % len(ARCADIA_COLORS)
            colors.append(ARCADIA_COLORS[idx])
    return colors


def arcadia_gradient_cmap(gradient='sunset', name=None):
    """
    Create an Arcadia gradient colormap.

    Official Arcadia gradients from arcadia-pycolor:
    https://github.com/Arcadia-Science/arcadia-pycolor

    Parameters
    ----------
    gradient : str
        Name of gradient: 'magma', 'viridis', 'verde', 'sunset', 'wine', 'lisafrank',
        'reds', 'oranges', 'greens', 'sages', 'blues', 'purples',
        'orange_sage', 'red_blue', 'purple_green'
    name : str, optional
        Custom name for the colormap

    Returns
    -------
    LinearSegmentedColormap
    """
    from matplotlib.colors import LinearSegmentedColormap

    # Arcadia color definitions
    ARCADIA_COLORS = {
        # Perceptually uniform gradients
        'magma': ['#341E60', '#54448C', '#A96789', '#E9A482', '#F5DFB2'],  # concord → tanzanite → heather → tumbleweed → wheat
        'viridis': ['#282A49', '#5088C5', '#97CD78', '#FFFDBD'],  # space → aegean → lime → butter
        'verde': ['#09473E', '#4E7F72', '#FFCC7B', '#FFE3D4'],  # depths → shire → topaz → putty
        'sunset': ['#4D2500', '#A85E28', '#E9A482', '#FFCC7B', '#FFE3D4'],  # soil → umber → tumbleweed → topaz → putty
        'wine': ['#52180A', '#C85152', '#FFB883', '#F8F4F1'],  # redwood → dragon → tangerine → dawn
        'lisafrank': ['#09473E', '#5088C5', '#BABEE0', '#F4CAE3'],  # depths → aegean → wish → blossom

        # Monocolor gradients
        'reds': ['#9E3F41', '#C85152', '#FFF3F4'],  # cinnabar → dragon → blush
        'oranges': ['#964222', '#FFB883', '#F8F4F1'],  # terracotta → tangerine → dawn
        'greens': ['#47784A', '#97CD78', '#F7FBEF'],  # fern → lime → lichen
        'sages': ['#2A6B5E', '#B5BEA4', '#F7FBEF'],  # asparagus → sage → lichen
        'blues': ['#2B66A2', '#5088C5', '#F4FBFE'],  # lapis → aegean → zephyr
        'purples': ['#6862AB', '#7A77AB', '#FCF7FF'],  # lilac → aster → ghost

        # Bicolor gradients
        'orange_sage': ['#964222', '#FFB883', '#F8F4F1', '#B5BEA4', '#2A6B5E'],  # oranges + reversed sages
        'red_blue': ['#9E3F41', '#C85152', '#FFF3F4', '#5088C5', '#2B66A2'],  # reds + reversed blues
        'purple_green': ['#6862AB', '#7A77AB', '#FCF7FF', '#97CD78', '#47784A'],  # purples + reversed greens
    }

    if gradient not in ARCADIA_COLORS:
        raise ValueError(f"Unknown gradient '{gradient}'. Choose from: {', '.join(ARCADIA_COLORS.keys())}")

    colors = ARCADIA_COLORS[gradient]
    cmap_name = name if name else f'arcadia_{gradient}'
    return LinearSegmentedColormap.from_list(cmap_name, colors)


def set_arcadia_style(ax):
    """
    Apply Arcadia styling to axis (remove top/right spines).
    Set tick labels to use monospace font for numbers.
    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set tick labels to use monospace font (for numbers)
    for label in ax.get_xticklabels():
        label.set_fontfamily('Atkinson Hyperlegible Mono')
    for label in ax.get_yticklabels():
        label.set_fontfamily('Atkinson Hyperlegible Mono')


def save_figure(fig, save_path):
    """Save a figure in PNG and SVG formats."""
    png_path = save_path if save_path.endswith('.png') else f"{save_path}.png"
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {png_path}")
    svg_path = png_path.rsplit('.', 1)[0] + '.svg'
    fig.savefig(svg_path, bbox_inches='tight')
    print(f"  Saved: {svg_path}")


def save_or_show(fig, save_path):
    """Save a figure or display it."""
    if save_path:
        save_figure(fig, save_path)
    else:
        plt.show()
    plt.close(fig)


def save_results(save_path, **results):
    if save_path:
        result_path = os.path.splitext(save_path)[0] + '.npz'
        np.savez_compressed(result_path, **results)
        print(f"  Saved: {result_path}")


def qq_scatter(ax, x, y, color='steelblue', s=20, alpha=0.5):
    """Scatter QQ plot with y=x reference line and equal axes."""
    ax.scatter(x, y, s=s, alpha=alpha, color=color, zorder=3)
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], '--', lw=1.5, color='#4A4A4A', zorder=2)
    # Set equal axes for proper visual calibration
    ax.set_xlim([lo, hi])
    ax.set_ylim([lo, hi])
    ax.set_aspect('equal', adjustable='box')


def bulk_slope(x, y, lo=0.1, hi=0.9):
    """OLS slope in central bulk of sorted arrays."""
    n = len(x)
    l, h = int(lo * n), int(hi * n)
    if h > l + 2:
        return stats.linregress(x[l:h], y[l:h])[0]
    return np.nan


# =============================================================================
# EXPERIMENT FUNCTIONS
# =============================================================================

def experiment_1(N=100_000, k_x=3, k_y=2, k_z_baseline=10,
                 k_z_values=(10, 30, 50, 70, 100),
                 strat_effect=0.0, noise=0.0,
                 n_variants=500, seed=42, save_path=None):
    """
    E1: LEAD FIGURE - CDF transform essential for cross-K comparability.
    """
    print("  E1: Inter-K QQ with higher-moment correction validation...")
    rng = np.random.default_rng(seed)
    k_z_max = max(k_z_values)

    methods = {
        'raw': {},
        'basharin': {},
        'dn_basic': {},
        'cdf': {}
    }

    for method in methods:
        methods[method] = {
            'mi': {k: [] for k in k_z_values},
            'cmi': {k: [] for k in k_z_values}
        }

    for _ in range(n_variants):
        g, d, p_max = generate_figure_null(
            N, k_x, k_z_max, strat_effect, noise, rng)
        p_base = coarsen_partition(p_max, k_z_max, k_z_baseline)
        tbl_base = build_3d_table(g, d, p_base, k_x, k_y, k_z_baseline)
        mi_raw_base = plugin_mi_3d(tbl_base)

        # Baseline MI with four methods
        mi_bash_base = correct_basharin(mi_raw_base, k_x, k_y, k_z_baseline, N)
        mi_dn_base = mi_to_z(mi_raw_base, k_x, k_y, k_z_baseline, N)
        mi_cdf_base = mi_to_z_cdf(mi_raw_base, k_x, k_y, k_z_baseline, N)

        for k_z in k_z_values:
            p_k = coarsen_partition(p_max, k_z_max, k_z)
            tbl_k = build_3d_table(g, d, p_k, k_x, k_y, k_z)
            cmi_raw = plugin_mi_3d(tbl_k)

            # Four methods
            cmi_bash = correct_basharin(cmi_raw, k_x, k_y, k_z, N)
            cmi_dn = mi_to_z(cmi_raw, k_x, k_y, k_z, N)
            cmi_cdf = mi_to_z_cdf(cmi_raw, k_x, k_y, k_z, N)

            methods['raw']['mi'][k_z].append(mi_raw_base)
            methods['raw']['cmi'][k_z].append(cmi_raw)
            methods['basharin']['mi'][k_z].append(mi_bash_base)
            methods['basharin']['cmi'][k_z].append(cmi_bash)
            methods['dn_basic']['mi'][k_z].append(mi_dn_base)
            methods['dn_basic']['cmi'][k_z].append(cmi_dn)
            methods['cdf']['mi'][k_z].append(mi_cdf_base)
            methods['cdf']['cmi'][k_z].append(cmi_cdf)

    # Four-panel figure (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    k_colors = get_kz_colors(k_z_values)

    panels = [
        ('raw', 'Raw plugin (no correction)',
         f'$MI$ (raw), $k_z$ = {k_z_baseline}', '$CMI$ (raw)'),
        ('basharin', 'Basharin (mean correction only)',
         f'$MI$ (Basharin), $k_z$ = {k_z_baseline}', '$CMI$ (Basharin)'),
        ('dn_basic', 'DN-basic: $z^{{DN}}$ (mean + variance)',
         f'$z_{{MI}}^{{DN}}$ $k_z={k_z_baseline}$', '$z_{{CMI}}^{{DN}}$'),
        ('cdf', 'CDF transform: $z^{{CDF}}$ (all moments)',
         f'$z_{{MI}}^{{CDF}}$ $k_z={k_z_baseline}$', '$z_{{CMI}}^{{CDF}}$')
    ]

    # Store handles and labels for creating a single figure legend
    legend_handles = []
    legend_labels = []

    # Panel labels
    panel_labels = ['A', 'B', 'C', 'D']

    for idx, (method, title, xlabel, ylabel) in enumerate(panels):
        ax = axes[idx]

        # Add panel label
        ax.text(-0.15, 1.05, panel_labels[idx], transform=ax.transAxes, fontsize=18,
                fontweight='bold', va='top', ha='right', color='#484B50')

        mi_d = methods[method]['mi']
        cmi_d = methods[method]['cmi']

        for i, k_z in enumerate(k_z_values):
            xs = np.sort(mi_d[k_z])
            ys = np.sort(cmi_d[k_z])
            scatter = ax.scatter(xs, ys, s=20, alpha=0.4, color=k_colors[i])

            # Collect legend handles only from first panel
            if idx == 0:
                legend_handles.append(scatter)
                legend_labels.append(f'$k_z$={k_z}')

        # Diagonal reference (using dark gray instead of red)
        lo = min(min(np.min(mi_d[k]) for k in k_z_values),
                min(np.min(cmi_d[k]) for k in k_z_values))
        hi = max(max(np.max(mi_d[k]) for k in k_z_values),
                max(np.max(cmi_d[k]) for k in k_z_values))
        line = ax.plot([lo, hi], [lo, hi], '--', lw=1.5, color='#4A4A4A')[0]

        # Add y=x line to legend handles only from first panel
        if idx == 0:
            legend_handles.append(line)
            legend_labels.append('$y=x$')

        # Set equal axes for proper visual calibration
        ax.set_xlim([lo, hi])
        ax.set_ylim([lo, hi])
        ax.set_aspect('equal', adjustable='box')

        ax.set_xlabel(xlabel, fontsize=15)
        ax.set_ylabel(ylabel, fontsize=15)
        set_arcadia_style(ax)
        ax.tick_params(labelsize=14.5)

    # For top two panels (idx 0, 1), show only every other x-tick label to prevent overlap
    for idx in [0, 1]:
        ax = axes[idx]
        xticks = ax.get_xticks()
        xticklabels = ax.get_xticklabels()
        # Keep every other label, replace others with empty string
        new_labels = [label.get_text() if i % 2 == 0 else ''
                     for i, label in enumerate(xticklabels)]
        ax.set_xticklabels(new_labels)

    # Create a columnar legend in the lower right quadrant of the upper right panel
    axes[1].legend(legend_handles, legend_labels, loc='lower right',
                   ncol=1, fontsize=15, frameon=False)

    plt.tight_layout()

    archived = {
        'N': N, 'k_x': k_x, 'k_y': k_y, 'k_z_baseline': k_z_baseline,
        'k_z_values': k_z_values, 'n_variants': n_variants, 'seed': seed,
        'strat_effect': strat_effect, 'noise': noise,
    }
    archived.update({
        f'{method}_{measure}_kz_{k_z}': np.asarray(values)
        for method, method_results in methods.items()
        for measure, by_kz in method_results.items()
        for k_z, values in by_kz.items()
    })
    save_results(save_path, **archived)

    save_or_show(fig, save_path)


def experiment_2(N=100_000, k_x=3, k_y=2,
                 k_z_values=(6, 10, 30, 100),
                 strat_effect=0.0, noise=0.0,
                 n_variants=1000, seed=42, save_path=None, show_legend=True,
                 no_legend_save_path=None):
    """
    E2: Foundation experiment. Shows raw plugin MI and its relationship to G-statistic.
    """
    print("  E2: Raw G-statistic vs chi2(df)...")
    rng = np.random.default_rng(seed)
    k_z_max = max(k_z_values)

    G_by_kz = {k_z: [] for k_z in k_z_values}
    MI_by_kz = {k_z: [] for k_z in k_z_values}

    for _ in range(n_variants):
        g, d, p_max = generate_figure_null(
            N, k_x, k_z_max, strat_effect, noise, rng)
        for k_z in k_z_values:
            p_k = coarsen_partition(p_max, k_z_max, k_z)
            tbl = build_3d_table(g, d, p_k, k_x, k_y, k_z)
            mi = plugin_mi_3d(tbl)
            G_by_kz[k_z].append(g_statistic(mi, N))
            MI_by_kz[k_z].append(mi)

    for k_z in k_z_values:
        G_by_kz[k_z] = np.array(G_by_kz[k_z])
        MI_by_kz[k_z] = np.array(MI_by_kz[k_z])

    # New layout: side-by-side QQ and histogram for each k_z
    n_plots = len(k_z_values)
    fig, axes = plt.subplots(n_plots, 2, figsize=(14, 3.5 * n_plots))
    if n_plots == 1:
        axes = axes.reshape(1, -1)
    k_colors = get_kz_colors(k_z_values)

    # Plot side by side: QQ on left, histogram on right
    for idx, k_z in enumerate(k_z_values):
        df = chi2_df(k_x, k_y, k_z)
        G = G_by_kz[k_z]
        MI = MI_by_kz[k_z]
        n = len(MI)
        probs = (np.arange(1, n + 1) - 0.5) / n

        scaling_factor = 2 * N * np.log(2)
        theo_mi_mean = df / scaling_factor
        theo_mi_std = np.sqrt(2 * df) / scaling_factor

        # Left: QQ plot
        ax_qq = axes[idx, 0]
        theo_chi2 = stats.chi2.ppf(probs, df)
        theo_mi = theo_chi2 / scaling_factor
        qq_scatter(ax_qq, theo_mi, np.sort(MI), color=k_colors[idx])

        # Only show x-axis label on bottom row
        if idx == n_plots - 1:
            ax_qq.set_xlabel(f"Theoretical $MI$ quantiles", fontsize=15)
        else:
            ax_qq.set_xlabel('')
        ax_qq.set_ylabel(f"Raw $MI$ quantiles, $k_z$ = {k_z}", fontsize=15)
        lo = min(theo_mi.min(), np.sort(MI).min())
        hi = max(theo_mi.max(), np.sort(MI).max())
        ax_qq.set_xlim([lo, hi])
        ax_qq.set_ylim([lo, hi])
        ax_qq.set_aspect('equal', adjustable='box')
        set_arcadia_style(ax_qq)
        ax_qq.tick_params(labelsize=14.5)

        # Right: Distribution plot
        ax_dist = axes[idx, 1]
        mi_range = np.linspace(0, np.percentile(MI, 99.5), 300)
        ax_dist.hist(MI, bins=40, density=True, alpha=0.6, color=k_colors[idx],
                     label='Empirical $MI$')
        mi_theo_pdf = scaling_factor * stats.chi2.pdf(mi_range * scaling_factor, df)
        ax_dist.plot(mi_range, mi_theo_pdf, '-', lw=2,
                     color='#4A4A4A', label='Theoretical $MI$')
        # Only show x-axis label on bottom row
        if idx == n_plots - 1:
            ax_dist.set_xlabel("Raw plugin $MI$ (bits)", fontsize=15)
        else:
            ax_dist.set_xlabel('')
        ax_dist.set_ylabel(f"Density, $k_z$ = {k_z}", fontsize=15)
        if show_legend and idx == 0:  # Only show legend on first row
            ax_dist.legend(fontsize=15, frameon=False)
        set_arcadia_style(ax_dist)
        ax_dist.tick_params(labelsize=14.5)

        # For bottom right panel, show only every other x-tick label to prevent overlap
        if idx == n_plots - 1:
            xticks = ax_dist.get_xticks()
            xticklabels = [label.get_text() for label in ax_dist.get_xticklabels()]
            new_labels = [xticklabels[i] if i % 2 == 0 else '' for i in range(len(xticklabels))]
            ax_dist.set_xticklabels(new_labels)

    plt.tight_layout()

    archived = {
        'N': N, 'k_x': k_x, 'k_y': k_y, 'k_z_values': k_z_values,
        'n_variants': n_variants, 'seed': seed,
        'strat_effect': strat_effect, 'noise': noise,
    }
    archived.update({f'G_kz_{k_z}': values
                     for k_z, values in G_by_kz.items()})
    archived.update({f'MI_kz_{k_z}': values
                     for k_z, values in MI_by_kz.items()})
    save_results(save_path, **archived)

    if save_path and no_legend_save_path:
        save_figure(fig, save_path)
        legend = axes[0, 1].get_legend()
        if legend:
            legend.remove()
        plt.tight_layout()
        save_figure(fig, no_legend_save_path)
        plt.close(fig)
    else:
        save_or_show(fig, save_path)


def experiment_4(k_x_values=(2, 3, 4, 5), k_y_values=(2, 3, 4, 5),
                 k_z_values=(6, 10, 20, 50, 100, 200),
                 N=100_000, n_variants=300, seed=42,
                 strat_effect=0.0, noise=0.0, save_path=None):
    """
    E4: Validates sigma_0 consistency check.
    """
    print("  E4: sigma_0 consistency check...")
    rng = np.random.default_rng(seed)

    # Panel 1: theoretical heatmap
    mat = np.array([[sigma0_prediction(kx, ky) for ky in k_y_values]
                    for kx in k_x_values])

    # Panel 2: empirical for (k_x=3, k_y=2)
    k_x, k_y = 3, 2
    k_z_max = max(k_z_values)
    chi2_pred = sigma0_prediction(k_x, k_y)
    n_groups = 10
    group_size = max(n_variants // n_groups, 5)
    n_total = n_groups * group_size

    emp_sigma0_by_kz = {}

    for k_z in k_z_values:
        C = k_x * k_y * k_z
        rng2 = np.random.default_rng(seed + k_z)
        all_mi_b = []
        for _ in range(n_total):
            g, d, p_max = generate_figure_null(
                N, k_x, k_z_max, strat_effect, noise, rng2)
            p_k = coarsen_partition(p_max, k_z_max, k_z)
            tbl = build_3d_table(g, d, p_k, k_x, k_y, k_z)
            mi_r = plugin_mi_3d(tbl)
            mi_b = correct_basharin(mi_r, k_x, k_y, k_z, N)
            all_mi_b.append(mi_b)
        all_mi_b = np.array(all_mi_b)
        group_estimates = []
        for g_idx in range(n_groups):
            group = all_mi_b[g_idx * group_size:(g_idx + 1) * group_size]
            group_estimates.append(np.std(group) * N / np.sqrt(C))
        emp_sigma0_by_kz[k_z] = np.array(group_estimates)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: heatmap
    ax = axes[0]
    im = ax.imshow(mat, aspect='auto', cmap='viridis', origin='lower',
                   extent=[k_y_values[0] - 0.5, k_y_values[-1] + 0.5,
                            k_x_values[0] - 0.5, k_x_values[-1] + 0.5])
    cbar = plt.colorbar(im, ax=ax, label='$\\sigma_0$ ($\\chi^2$ prediction)')
    cbar.ax.tick_params(labelsize=14.5)
    for label in cbar.ax.get_yticklabels():
        label.set_fontfamily('Atkinson Hyperlegible Mono')
    for i, kx in enumerate(k_x_values):
        for j, ky in enumerate(k_y_values):
            ax.text(ky, kx, f"{mat[i, j]:.3f}", ha='center', va='center',
                    fontsize=14,
                    color='white' if mat[i, j] < mat.mean() else 'black')
    ax.add_patch(plt.Rectangle((2 - 0.5, 3 - 0.5), 1, 1, fill=False,
                                edgecolor=arcadia_colors()[1], lw=2))
    ax.set_xlabel("$k_y$", fontsize=15)
    ax.set_ylabel("$k_x$", fontsize=15)
    ax.tick_params(labelsize=14.5)
    set_arcadia_style(ax)

    # Panel 2: violin of sigma_0 estimates
    ax = axes[1]
    violin_data = [emp_sigma0_by_kz[k_z] for k_z in k_z_values]
    k_colors_p2 = arcadia_colors(len(k_z_values))
    parts = ax.violinplot(violin_data, positions=range(len(k_z_values)),
                          showmedians=True, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(k_colors_p2[i])
        pc.set_alpha(0.7)
    ax.axhline(chi2_pred, color=arcadia_colors()[1], lw=2, ls='--',
               label=f'$\\chi^2$ prediction = {chi2_pred:.4f}')
    ax.axhline(SIGMA_0, color=arcadia_colors()[2], lw=2, ls=':',
               label=f'Empirical $\\sigma_0$ = {SIGMA_0}')
    ax.set_xticks(range(len(k_z_values)))
    ax.set_xticklabels([str(k) for k in k_z_values])
    ax.set_xlabel("$k_z$", fontsize=15)
    ax.set_ylabel("$\\sigma_0$ estimate (per subgroup)", fontsize=15)
    ax.legend(fontsize=15, frameon=False)
    ax.tick_params(labelsize=14.5)
    set_arcadia_style(ax)

    plt.tight_layout()
    archived = {
        'N': N, 'k_x_values': k_x_values, 'k_y_values': k_y_values,
        'k_z_values': k_z_values, 'n_variants': n_variants, 'seed': seed,
        'strat_effect': strat_effect, 'noise': noise, 'sigma0_prediction': mat,
    }
    archived.update({f'sigma0_kz_{k_z}': values
                     for k_z, values in emp_sigma0_by_kz.items()})
    save_results(save_path, **archived)
    save_or_show(fig, save_path)


def experiment_5(k_x=3, k_y=2,
                 k_z_values=(2, 3, 5, 6, 10, 15, 20, 30, 50, 100),
                 N=10_000, n_variants=2000, seed=42, save_path=None):
    """
    E5: Validates Proposition 4 — df_CMI = k_z * df_MI.
    """
    print("  E5: df_CMI = k_z * df_MI validation...")
    df_mi = chi2_df(k_x, k_y, 1)
    rng = np.random.default_rng(seed)
    k_z_max = max(k_z_values)
    C = k_x * k_y * k_z_max

    results = {}
    for k_z in k_z_values:
        df_pred = df_mi * k_z
        G_vals = []
        for _ in range(n_variants):
            counts = rng.multinomial(N, np.ones(C) / C)
            tbl = counts.reshape(k_x, k_y, k_z_max)
            # Coarsen to target k_z
            tbl_k = np.zeros((k_x, k_y, k_z), dtype=float)
            for ki in range(k_z_max):
                tbl_k[:, :, ki * k_z // k_z_max] += tbl[:, :, ki]
            mi = plugin_mi_3d(tbl_k)
            G_vals.append(g_statistic(mi, N))
        G = np.array(G_vals)
        df_mean = G.mean()
        df_var = G.var() / 2.0
        df_mle, _, _ = stats.chi2.fit(G, floc=0, fscale=1)
        results[k_z] = dict(df_pred=df_pred, df_mean=df_mean,
                             df_var=df_var, df_mle=df_mle)

    k_z_arr = np.array(k_z_values)
    df_pred = np.array([results[k]['df_pred'] for k in k_z_values])
    df_mean = np.array([results[k]['df_mean'] for k in k_z_values])
    df_var = np.array([results[k]['df_var'] for k in k_z_values])
    df_mle = np.array([results[k]['df_mle'] for k in k_z_values])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: predicted vs empirical
    ax = axes[0]
    ax.plot(df_pred, df_pred, '--', color='#4A4A4A',
            label='$y=x$ (perfect)', zorder=1, lw=1.5)
    ax.scatter(df_pred, df_mean, s=60, marker='o', label='From mean',
               color=arcadia_colors()[0], zorder=3)
    ax.scatter(df_pred, df_var, s=60, marker='s', label='From variance',
               color=arcadia_colors()[1], zorder=3)
    ax.scatter(df_pred, df_mle, s=60, marker='^', label='MLE fit',
               color=arcadia_colors()[2], zorder=3)
    ax.set_xlabel("Predicted $df = (k_x-1)(k_y-1) \\cdot k_z$", fontsize=15)
    ax.set_ylabel("Empirical $df$", fontsize=15)
    ax.legend(fontsize=15, frameon=False)
    ax.tick_params(labelsize=14.5)
    set_arcadia_style(ax)

    # Panel 2: relative error
    ax = axes[1]
    ax.plot(k_z_arr, (df_mean - df_pred) / df_pred * 100, 'o-', ms=6,
            label='From mean', color=arcadia_colors()[0], lw=1.5)
    ax.plot(k_z_arr, (df_var - df_pred) / df_pred * 100, 's-', ms=6,
            label='From variance', color=arcadia_colors()[1], lw=1.5)
    ax.plot(k_z_arr, (df_mle - df_pred) / df_pred * 100, '^-', ms=6,
            label='MLE', color=arcadia_colors()[2], lw=1.5)
    ax.axhline(0, color='#4A4A4A', ls='--', lw=1.5, label='Zero error')
    ax.fill_between(k_z_arr, -1, 1, alpha=0.15, color=arcadia_colors()[2],
                    label='±1% tolerance')
    ax.set_xlabel("$k_z$", fontsize=15)
    ax.set_ylabel("Relative error (%)", fontsize=15)
    ax.legend(fontsize=15, frameon=False)
    ax.tick_params(labelsize=14.5)
    set_arcadia_style(ax)
    ax.set_ylim([-5, 5])

    plt.tight_layout()
    save_results(
        save_path, N=N, k_x=k_x, k_y=k_y, k_z_values=k_z_arr,
        n_variants=n_variants, seed=seed, df_pred=df_pred,
        df_mean=df_mean, df_var=df_var, df_mle=df_mle)
    save_or_show(fig, save_path)


def experiment_6_5(N=100_000, k_x=3, k_y=2,
                   k_z_values=(6, 10, 30, 100),
                   strat_effect=0.0, noise=0.0,
                   n_variants=10_000,
                   seed=42, save_path=None):
    """
    E6.5: Tail calibration showing CDF correction enables valid p-values.
    """
    print("  E6.5: Tail calibration (all methods vs N(0,1))...")
    rng = np.random.default_rng(seed)
    k_z_max = max(k_z_values)

    methods = ['raw', 'dn_basic', 'cdf']
    results = {m: {k: [] for k in k_z_values} for m in methods}

    print(f"    Generating {n_variants} null variants for tail calibration...")
    for i in range(n_variants):
        if i % 1000 == 0:
            print(f"      {i}/{n_variants}...")
        g, d, p_max = generate_figure_null(
            N, k_x, k_z_max, strat_effect, noise, rng)

        for k_z in k_z_values:
            p_k = coarsen_partition(p_max, k_z_max, k_z)
            tbl = build_3d_table(g, d, p_k, k_x, k_y, k_z)
            mi = plugin_mi_3d(tbl)

            z_dn = mi_to_z(mi, k_x, k_y, k_z, N)
            z_cdf = mi_to_z_cdf(mi, k_x, k_y, k_z, N)

            results['raw'][k_z].append(mi)
            results['dn_basic'][k_z].append(z_dn)
            results['cdf'][k_z].append(z_cdf)

    # Convert to arrays
    for m in methods:
        for k_z in k_z_values:
            results[m][k_z] = np.array(results[m][k_z])

    # Create QQ plots
    method_labels = {
        'raw': 'Raw MI (uncorrected)',
        'dn_basic': 'DN ($z_{{MI}}^{{DN}}$)',
        'cdf': 'CDF transform ($z_{{MI}}^{{CDF}}$)'
    }

    n_methods = len(methods)
    n_kz = len(k_z_values)

    # New layout: 3 columns (methods) x 4 rows (k_z values)
    fig, axes = plt.subplots(n_kz, n_methods, figsize=(4*n_methods, 3.5*n_kz))
    if n_methods == 1:
        axes = axes.reshape(-1, 1)
    if n_kz == 1:
        axes = axes.reshape(1, -1)

    colors = get_kz_colors(k_z_values)

    # New layout: rows are k_z, columns are methods
    for row, k_z in enumerate(k_z_values):
        for col, method in enumerate(methods):
            ax = axes[row, col]
            vals = results[method][k_z]

            # QQ plot: empirical quantiles vs theoretical N(0,1) quantiles
            sorted_vals = np.sort(vals)
            n = len(sorted_vals)
            probs = (np.arange(1, n + 1) - 0.5) / n
            theo_quantiles = stats.norm.ppf(probs)

            # Plot empirical quantiles against theoretical N(0,1) quantiles
            ax.plot(theo_quantiles, sorted_vals, 'o', ms=5, alpha=0.3,
                   color=colors[row])

            # Only draw y=x line for z-score methods (DN and CDF), not raw MI
            if method != 'raw':
                ax.plot([theo_quantiles.min(), theo_quantiles.max()],
                       [theo_quantiles.min(), theo_quantiles.max()],
                       '--', lw=1.5, color='#4A4A4A')

            # Highlight extreme tail (>99%) with consistent color (orange instead of red)
            tail_idx = int(0.99 * n)
            ax.plot(theo_quantiles[tail_idx:], sorted_vals[tail_idx:],
                   'o', ms=8, color='#F28360', alpha=0.7)

            # Set y-axis limits appropriately for raw MI (different scale than z-scores)
            if method == 'raw':
                # Set y-axis to show raw MI values, not extended to N(0,1) range
                y_min, y_max = sorted_vals.min(), sorted_vals.max()
                y_range = y_max - y_min
                ax.set_ylim([y_min - 0.1*y_range, y_max + 0.1*y_range])

            # Only show x-axis label on bottom row
            if row == n_kz - 1:
                ax.set_xlabel('Theoretical $N(0,1)$ quantiles', fontsize=15)
            else:
                ax.set_xlabel('')

            # Only show y-axis label on left column
            if col == 0:
                if method == 'raw':
                    ax.set_ylabel(f'Raw $MI$ ($k_z$={k_z})', fontsize=15)
                else:
                    ax.set_ylabel(f'Empirical $z$ ($k_z$={k_z})', fontsize=15)
            else:
                ax.set_ylabel('')

            # Column headers on top row
            if row == 0:
                ax.text(0.5, 1.05, method_labels[method],
                       transform=ax.transAxes,
                       va='bottom', ha='center', fontsize=15, fontweight='bold')

            ax.tick_params(labelsize=14.5)
            set_arcadia_style(ax)

            # Only use equal aspect for z-score methods, not raw MI
            if method != 'raw':
                ax.set_aspect('equal', adjustable='datalim')

    plt.tight_layout()
    archived = {
        'N': N, 'k_x': k_x, 'k_y': k_y, 'k_z_values': k_z_values,
        'n_variants': n_variants, 'seed': seed,
        'strat_effect': strat_effect, 'noise': noise,
    }
    archived.update({
        f'{method}_kz_{k_z}': values
        for method, by_kz in results.items()
        for k_z, values in by_kz.items()
    })
    save_results(save_path, **archived)
    save_or_show(fig, save_path)


def experiment_6_7(k_x=3, k_y=2, k_z_values=(6, 10, 30, 50, 70),
                   N_values=(100, 500, 1000, 5000, 20_000, 100_000),
                   n_variants=500, seed=42, save_path=None):
    """
    E6.7: Validity regime - minimum sample size for chi-squared approximation.
    """
    print("  E6.7: Validity regime (sparse cell characterization)...")
    rng = np.random.default_rng(seed)

    ks_stats = np.zeros((len(N_values), len(k_z_values)))
    ks_pvalues = np.zeros((len(N_values), len(k_z_values)))
    expected_counts = np.zeros((len(N_values), len(k_z_values)))

    for i, N in enumerate(N_values):
        print(f"    N={N}...")
        for j, k_z in enumerate(k_z_values):
            n_cells = k_x * k_y * k_z
            expected_counts[i, j] = N / n_cells

            z_values = []
            for _ in range(n_variants):
                g = rng.integers(0, k_x, size=N)
                d = rng.integers(0, k_y, size=N)
                p = rng.integers(0, k_z, size=N)
                tbl = build_3d_table(g, d, p, k_x, k_y, k_z)
                mi = plugin_mi_3d(tbl)
                z = mi_to_z_cdf(mi, k_x, k_y, k_z, N)
                z_values.append(z)

            z_values = np.array(z_values)
            ks_stat, ks_pvalue = stats.kstest(z_values, 'norm')
            ks_stats[i, j] = ks_stat
            ks_pvalues[i, j] = ks_pvalue

    # Heatmap
    from matplotlib.colors import ListedColormap
    import matplotlib.gridspec as gridspec

    binary_cmap = ListedColormap(['#F28360', '#5088C5'])
    # Use 'viridis' gradient reversed (higher values darker, lower values lighter)
    continuous_cmap = arcadia_gradient_cmap('viridis').reversed()

    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 0.05], wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cax2 = fig.add_subplot(gs[0, 2])

    # Panel 1: KS test
    ax1.text(-0.15, 1.05, 'A', transform=ax1.transAxes, fontsize=18,
             fontweight='bold', va='top', ha='right', color='#484B50')
    ks_binary = (ks_pvalues > 0.05).astype(int)
    im1 = ax1.imshow(ks_binary, aspect='auto', cmap=binary_cmap,
                     vmin=0, vmax=1)
    ax1.set_xticks(range(len(k_z_values)))
    ax1.set_xticklabels(k_z_values)
    ax1.set_yticks(range(len(N_values)))
    ax1.set_yticklabels(N_values)
    ax1.set_xlabel('$k_z$', fontsize=15)
    ax1.set_ylabel('$N$', fontsize=15)
    ax1.tick_params(labelsize=14.5)
    set_arcadia_style(ax1)

    for i in range(len(N_values)):
        for j in range(len(k_z_values)):
            color = 'white' if ks_pvalues[i, j] > 0.05 else 'black'
            text = ax1.text(j, i, f'{ks_stats[i, j]:.3f}',
                           ha="center", va="center", color=color, fontsize=14)

    # Panel 2: Expected counts
    ax2.text(-0.15, 1.05, 'B', transform=ax2.transAxes, fontsize=18,
             fontweight='bold', va='top', ha='right', color='#484B50')
    im2 = ax2.imshow(expected_counts, aspect='auto', cmap=continuous_cmap,
                     norm=plt.matplotlib.colors.LogNorm())
    ax2.set_xticks(range(len(k_z_values)))
    ax2.set_xticklabels(k_z_values)
    ax2.set_yticks(range(len(N_values)))
    ax2.set_yticklabels(N_values)
    ax2.set_xlabel('$k_z$', fontsize=15)
    ax2.set_ylabel('')
    ax2.tick_params(labelsize=14.5)
    set_arcadia_style(ax2)

    cbar2 = plt.colorbar(im2, cax=cax2, label='$E$[count/cell]')
    cbar2.ax.tick_params(labelsize=14.5)
    for label in cbar2.ax.get_yticklabels():
        label.set_fontfamily('Atkinson Hyperlegible Mono')

    for i in range(len(N_values)):
        for j in range(len(k_z_values)):
            text = ax2.text(j, i, f'{expected_counts[i, j]:.1f}',
                           ha="center", va="center",
                           color="white" if expected_counts[i, j] < 10 else "black",
                           fontsize=14)

    plt.tight_layout()
    save_results(
        save_path, k_x=k_x, k_y=k_y, N_values=N_values,
        k_z_values=k_z_values, n_variants=n_variants, seed=seed,
        ks_statistics=ks_stats, ks_pvalues=ks_pvalues,
        expected_counts=expected_counts)
    save_or_show(fig, save_path)


def experiment_16(k_x=3, k_y=2, k_z=6,
                  N_values=(1_000, 5_000, 20_000, 100_000),
                  strat_effect=0.0, noise=0.0,
                  n_variants=1000, seed=42, save_path=None, show_legend=True):
    """
    E16: Raw plugin MI distribution across different sample sizes N.
    """
    print("  E16: Raw MI distribution across sample sizes N...")
    rng = np.random.default_rng(seed)
    k_z_max = k_z
    df = chi2_df(k_x, k_y, k_z)

    G_by_N = {N: [] for N in N_values}
    MI_by_N = {N: [] for N in N_values}

    for N in N_values:
        print(f"    N={N:,}...")
        rng_N = np.random.default_rng(seed + N)
        for _ in range(n_variants):
            g, d, p_max = generate_figure_null(
                N, k_x, k_z_max, strat_effect, noise, rng_N)
            p_k = coarsen_partition(p_max, k_z_max, k_z)
            tbl = build_3d_table(g, d, p_k, k_x, k_y, k_z)
            mi = plugin_mi_3d(tbl)
            G_by_N[N].append(g_statistic(mi, N))
            MI_by_N[N].append(mi)

    for N in N_values:
        G_by_N[N] = np.array(G_by_N[N])
        MI_by_N[N] = np.array(MI_by_N[N])

    # New layout: side-by-side QQ and histogram for each N
    n_plots = len(N_values)
    fig, axes = plt.subplots(n_plots, 2, figsize=(14, 3.5 * n_plots))
    if n_plots == 1:
        axes = axes.reshape(1, -1)
    # Use different colors for N values (not k_z colors)
    # aster (purple), rose (pink), lime (green), tangerine (orange)
    N_colors = ['#7A77AB', '#F898AE', '#97CD78', '#FFB883']

    # Plot side by side: QQ on left, histogram on right
    for idx, N in enumerate(N_values):
        G = G_by_N[N]
        MI = MI_by_N[N]
        n = len(MI)
        probs = (np.arange(1, n + 1) - 0.5) / n

        scaling_factor = 2 * N * np.log(2)
        theo_mi_mean = df / scaling_factor
        theo_mi_std = np.sqrt(2 * df) / scaling_factor

        # Left: QQ plot
        ax_qq = axes[idx, 0]
        theo_chi2 = stats.chi2.ppf(probs, df)
        theo_mi = theo_chi2 / scaling_factor
        qq_scatter(ax_qq, theo_mi, np.sort(MI), color=N_colors[idx])

        # Only show x-axis label on bottom row
        if idx == n_plots - 1:
            ax_qq.set_xlabel(f"Theoretical $MI$ quantiles", fontsize=15)
        else:
            ax_qq.set_xlabel('')
        ax_qq.set_ylabel(f"Raw $MI$ quantiles, $N$ = {N:,}", fontsize=15)
        lo = min(theo_mi.min(), np.sort(MI).min())
        hi = max(theo_mi.max(), np.sort(MI).max())
        ax_qq.set_xlim([lo, hi])
        ax_qq.set_ylim([lo, hi])
        ax_qq.set_aspect('equal', adjustable='box')
        set_arcadia_style(ax_qq)
        ax_qq.tick_params(labelsize=14.5)

        # Right: Distribution plot
        ax_dist = axes[idx, 1]
        mi_range = np.linspace(0, np.percentile(MI, 99.5), 300)
        ax_dist.hist(MI, bins=40, density=True, alpha=0.6, color=N_colors[idx],
                     label='Empirical $MI$')
        mi_theo_pdf = scaling_factor * stats.chi2.pdf(mi_range * scaling_factor, df)
        ax_dist.plot(mi_range, mi_theo_pdf, '-', lw=2,
                     color='#4A4A4A', label='Theoretical $MI$')
        # Only show x-axis label on bottom row
        if idx == n_plots - 1:
            ax_dist.set_xlabel("Raw plugin $MI$ (bits)", fontsize=15)
        else:
            ax_dist.set_xlabel('')
        ax_dist.set_ylabel(f"Density, $N$ = {N:,}", fontsize=15)
        if show_legend and idx == 0:  # Only show legend on first row
            ax_dist.legend(fontsize=15, frameon=False)
        set_arcadia_style(ax_dist)
        ax_dist.tick_params(labelsize=14.5)

        # For bottom right panel, show only every other x-tick label to prevent overlap
        if idx == n_plots - 1:
            xticks = ax_dist.get_xticks()
            xticklabels = [label.get_text() for label in ax_dist.get_xticklabels()]
            new_labels = [xticklabels[i] if i % 2 == 0 else '' for i in range(len(xticklabels))]
            ax_dist.set_xticklabels(new_labels)

    plt.tight_layout()

    archived = {
        'k_x': k_x, 'k_y': k_y, 'k_z': k_z, 'N_values': N_values,
        'n_variants': n_variants, 'seed': seed,
        'strat_effect': strat_effect, 'noise': noise,
    }
    archived.update({f'G_N_{N}': values for N, values in G_by_N.items()})
    archived.update({f'MI_N_{N}': values for N, values in MI_by_N.items()})
    save_results(save_path, **archived)

    save_or_show(fig, save_path)


def experiment_20(N_values=(1000, 2000, 5000, 20000),
                  true_mi_values=(0.001, 0.003, 0.005, 0.01, 0.02),
                  k_x=2, k_y=2, n_reps=100, n_perms=200,
                  alpha=0.05, seed=42, save_path=None):
    """
    E20: DN vs Permutation Power Comparison - 3 panel figure.
    """
    print("  E20: DN vs Permutation Power Comparison...")
    import time as time_module

    results_dn = {N: {mi: {'power': 0, 'pvals': [], 'time': 0}
                      for mi in true_mi_values} for N in N_values}
    results_perm = {N: {mi: {'power': 0, 'pvals': [], 'time': 0}
                        for mi in true_mi_values} for N in N_values}

    for N in N_values:
        for true_mi in true_mi_values:
            rng = np.random.default_rng(seed + N)
            rej_dn = 0
            rej_perm = 0

            for _ in range(n_reps):
                table = generate_figure_2x2_with_mi(N, true_mi, rng)

                # DN test (using chi-squared on G)
                t0 = time_module.time()
                mi = plugin_mi_2d(table)
                g = g_statistic(mi, N)
                df = chi2_df(k_x, k_y, 1)
                p_dn = 1 - stats.chi2.cdf(g, df)
                t_dn = time_module.time() - t0
                results_dn[N][true_mi]['pvals'].append(p_dn)
                results_dn[N][true_mi]['time'] += t_dn
                if p_dn < alpha:
                    rej_dn += 1

                # Permutation test
                t0 = time_module.time()
                x = []
                y = []
                for i in range(table.shape[0]):
                    for j in range(table.shape[1]):
                        x.extend([i] * int(table[i, j]))
                        y.extend([j] * int(table[i, j]))
                x = np.array(x)
                y = np.array(y)
                p_perm = permutation_test(x, y, k_x, k_y, n_perms, rng)
                t_perm = time_module.time() - t0
                results_perm[N][true_mi]['pvals'].append(p_perm)
                results_perm[N][true_mi]['time'] += t_perm
                if p_perm < alpha:
                    rej_perm += 1

            results_dn[N][true_mi]['power'] = rej_dn / n_reps
            results_perm[N][true_mi]['power'] = rej_perm / n_reps

    # 3 panels arranged horizontally (1 row, 3 columns), wide figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = arcadia_colors(len(N_values))

    # Panel 1: Power agreement
    ax = axes[0]
    ax.text(-0.15, 1.05, 'A', transform=ax.transAxes, fontsize=18,
            fontweight='bold', va='top', ha='right', color='#484B50')
    for i, N in enumerate(N_values):
        powers_dn = [results_dn[N][mi]['power'] for mi in true_mi_values]
        powers_perm = [results_perm[N][mi]['power'] for mi in true_mi_values]
        ax.scatter(powers_dn, powers_perm, s=120, color=colors[i],
                   label=f'$N$={N:,}', alpha=0.7)
    ax.plot([0, 1], [0, 1], '--', lw=1.5, color='#4A4A4A')
    ax.set_xlabel("$\\chi^2$ Power", fontsize=15)
    ax.set_ylabel("Permutation Power", fontsize=15)
    ax.legend(fontsize=15, frameon=False, loc='upper left')
    ax.tick_params(labelsize=14.5)
    set_arcadia_style(ax)

    # Panel 2: P-value correlation
    ax = axes[1]
    ax.text(-0.15, 1.05, 'B', transform=ax.transAxes, fontsize=18,
            fontweight='bold', va='top', ha='right', color='#484B50')
    eps = 1e-10

    for i, N in enumerate(N_values):
        for mi in true_mi_values:
            pvals_dn = np.array(results_dn[N][mi]['pvals'])
            pvals_perm = np.array(results_perm[N][mi]['pvals'])
            pvals_dn_plot = np.maximum(pvals_dn, eps)
            pvals_perm_plot = np.maximum(pvals_perm, eps)
            ax.scatter(pvals_dn_plot, pvals_perm_plot, s=40, color=colors[i],
                       alpha=0.3)

    ax.plot([eps, 1], [eps, 1], '--', lw=1.5, color='#4A4A4A')

    # Show permutation resolution limit (using orange instead of red)
    perm_resolution = 1.0 / n_perms
    ax.axhline(perm_resolution, color='#F28360', ls=':', lw=2)

    ax.set_xlabel("$\\chi^2$ P-value", fontsize=15)
    ax.set_ylabel("Permutation P-value", fontsize=15)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([eps, 1])
    ax.set_ylim([eps, 1])
    ax.tick_params(labelsize=14.5)
    set_arcadia_style(ax)

    # Panel 3: Agreement heatmap with reduced dynamic range
    ax = axes[2]
    ax.text(-0.15, 1.05, 'C', transform=ax.transAxes, fontsize=18,
            fontweight='bold', va='top', ha='right', color='#484B50')
    agreement = np.zeros((len(N_values), len(true_mi_values)))
    for i, N in enumerate(N_values):
        for j, mi in enumerate(true_mi_values):
            p_dn = results_dn[N][mi]['power']
            p_perm = results_perm[N][mi]['power']
            agreement[i, j] = 1 - abs(p_dn - p_perm)

    # Use Arcadia gradient reversed (higher values darker, lower values lighter) and reduced dynamic range (0.8-1.0)
    im = ax.imshow(agreement, aspect='equal', cmap=arcadia_gradient_cmap('viridis').reversed(),
                   vmin=0.8, vmax=1.0)
    ax.set_xticks(range(len(true_mi_values)))
    ax.set_xticklabels([f'{mi:.3f}' for mi in true_mi_values])
    ax.set_yticks(range(len(N_values)))
    ax.set_yticklabels([f'{N:,}' for N in N_values])
    ax.set_xlabel("True $MI$ (bits)", fontsize=15)
    ax.set_ylabel("Sample Size ($N$)", fontsize=15)
    ax.tick_params(labelsize=14.5)
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=14.5)
    for label in cbar.ax.get_yticklabels():
        label.set_fontfamily('Atkinson Hyperlegible Mono')
    set_arcadia_style(ax)

    plt.tight_layout()
    archived = {
        'N_values': N_values, 'true_mi_values': true_mi_values,
        'k_x': k_x, 'k_y': k_y, 'n_reps': n_reps, 'n_perms': n_perms,
        'alpha': alpha, 'seed': seed,
    }
    for N in N_values:
        for true_mi in true_mi_values:
            key = f'N_{N}_mi_{true_mi:g}'.replace('.', 'p')
            archived[f'dn_pvalues_{key}'] = results_dn[N][true_mi]['pvals']
            archived[f'perm_pvalues_{key}'] = results_perm[N][true_mi]['pvals']
            archived[f'dn_power_{key}'] = results_dn[N][true_mi]['power']
            archived[f'perm_power_{key}'] = results_perm[N][true_mi]['power']
    save_results(save_path, **archived)
    save_or_show(fig, save_path)


def experiment_23(N_values=(5_000, 20_000, 100_000),
                  k_z_values=(6, 10, 30, 100),
                  k_x=3, k_y=2,
                  strat_effect=0.0, noise=0.0,
                  n_variants=1000, seed=42, save_path=None):
    """
    E23: Demonstrates that mi_z_cdf produces N(0,1) distributions.
    """
    print("  E23: CDF-corrected MI normality across N and k_z...")
    rng = np.random.default_rng(seed)

    results = {}

    for N in N_values:
        for k_z in k_z_values:
            print(f"    Generating data for N={N:>6,}, k_z={k_z:>3}...")
            df = chi2_df(k_x, k_y, k_z)
            z_values = []

            for _ in range(n_variants):
                g, d, p = generate_figure_null(
                    N, k_x, k_z, strat_effect, noise, rng)
                tbl = build_3d_table(g, d, p, k_x, k_y, k_z)
                mi = plugin_mi_3d(tbl)
                z = mi_to_z_cdf(mi, k_x, k_y, k_z, N)
                z_values.append(z)

            z_values = np.array(z_values)

            mean = np.mean(z_values)
            std = np.std(z_values)
            skewness = stats.skew(z_values)
            kurtosis = stats.kurtosis(z_values)

            ks_stat, ks_pvalue = stats.kstest(z_values, 'norm')

            if ks_pvalue > 0.05:
                normality = "Normal"
            elif ks_pvalue > 0.01:
                normality = "Marginal"
            else:
                normality = "Reject"

            results[(N, k_z)] = {
                'z_values': z_values,
                'mean': mean,
                'std': std,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'ks_stat': ks_stat,
                'ks_p': ks_pvalue,
                'normality': normality,
                'df': df
            }

    # Create figure with QQ plots grid (3 columns x 4 rows)
    # Columns = N values (5k, 20k, 100k), Rows = k_z values (6, 10, 30, 100)
    n_rows = len(k_z_values)
    n_cols = len(N_values)

    fig1, axes = plt.subplots(n_rows, n_cols, figsize=(12, 14))

    for row_idx, k_z in enumerate(k_z_values):
        for col_idx, N in enumerate(N_values):
            ax = axes[row_idx, col_idx]
            r = results[(N, k_z)]
            z_values = r['z_values']

            (osm, osr), (slope, intercept, r_val) = stats.probplot(z_values, dist='norm')

            # Use consistent k_z colors
            k_z_color = K_Z_COLORS.get(k_z, arcadia_colors()[0])
            ax.plot(osm, osr, 'o', color=k_z_color, ms=6, alpha=0.6)
            ax.plot(osm, osm, '--', color='#4A4A4A', lw=1.5)

            lims = [min(osm.min(), osr.min()), max(osm.max(), osr.max())]
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_aspect('equal')

            # Only show x-axis label on bottom row with N label
            if row_idx == n_rows - 1:
                ax.set_xlabel(f'Theoretical quantiles, $\\mathcal{{N}}(0,1)$\n$N$ = {N:,}', fontsize=15)
            else:
                ax.set_xlabel('')

            # Only show y-axis label on leftmost column with k_z label
            if col_idx == 0:
                ax.set_ylabel(f'Empirical quantiles\n$k_z$ = {k_z}', fontsize=15)
            else:
                ax.set_ylabel('')

            ax.tick_params(labelsize=14.5)
            set_arcadia_style(ax)

    plt.tight_layout()

    archived = {
        'N_values': N_values, 'k_z_values': k_z_values, 'k_x': k_x,
        'k_y': k_y, 'n_variants': n_variants, 'seed': seed,
        'strat_effect': strat_effect, 'noise': noise,
    }
    for (N, k_z), result in results.items():
        prefix = f'N_{N}_kz_{k_z}'
        archived[f'z_values_{prefix}'] = result['z_values']
        for name in ('mean', 'std', 'skewness', 'kurtosis', 'ks_stat',
                     'ks_p', 'df'):
            archived[f'{name}_{prefix}'] = result[name]
    save_results(save_path, **archived)

    save_or_show(fig1, save_path)


# =============================================================================
# FIGURE 4: CUSTOM COMBINED FIGURE
# =============================================================================

def figure_4(n_variants=300, n_variants_exp5=2000, save_path=None):
    """
    Custom Figure 4: Combines exp4 right panel (left) + exp5 left panel (right).
    """
    print("  Figure 4: Custom combined figure (σ₀ consistency + df structure)...")

    # Parameters
    k_z_values = (6, 10, 20, 50, 100, 200)
    N = 100_000
    seed = 42
    strat_effect = 0.0
    noise = 0.0

    # Generate data for right panel (from exp4)
    rng = np.random.default_rng(seed)
    k_x, k_y = 3, 2
    k_z_max = max(k_z_values)
    chi2_pred = sigma0_prediction(k_x, k_y)
    n_groups = 10
    group_size = max(n_variants // n_groups, 5)
    n_total = n_groups * group_size

    emp_sigma0_by_kz = {}
    for k_z in k_z_values:
        C = k_x * k_y * k_z
        rng2 = np.random.default_rng(seed + k_z)
        all_mi_b = []
        for _ in range(n_total):
            g, d, p_max = generate_figure_null(
                N, k_x, k_z_max, strat_effect, noise, rng2)
            p_k = coarsen_partition(p_max, k_z_max, k_z)
            tbl = build_3d_table(g, d, p_k, k_x, k_y, k_z)
            mi_r = plugin_mi_3d(tbl)
            mi_b = correct_basharin(mi_r, k_x, k_y, k_z, N)
            all_mi_b.append(mi_b)
        all_mi_b = np.array(all_mi_b)
        group_estimates = []
        for g_idx in range(n_groups):
            group = all_mi_b[g_idx * group_size:(g_idx + 1) * group_size]
            group_estimates.append(np.std(group) * N / np.sqrt(C))
        emp_sigma0_by_kz[k_z] = np.array(group_estimates)

    # Generate data for left panel (from exp5)
    k_z_values_exp5 = (2, 3, 5, 6, 10, 15, 20, 30, 50, 100)
    N_exp5 = 10_000
    df_mi = chi2_df(k_x, k_y, 1)
    k_z_max_exp5 = max(k_z_values_exp5)
    C_exp5 = k_x * k_y * k_z_max_exp5

    results = {}
    for k_z in k_z_values_exp5:
        df_pred = df_mi * k_z
        G_vals = []
        for _ in range(n_variants_exp5):
            counts = rng.multinomial(N_exp5, np.ones(C_exp5) / C_exp5)
            tbl = counts.reshape(k_x, k_y, k_z_max_exp5)
            tbl_k = np.zeros((k_x, k_y, k_z), dtype=float)
            for ki in range(k_z_max_exp5):
                tbl_k[:, :, ki * k_z // k_z_max_exp5] += tbl[:, :, ki]
            mi = plugin_mi_3d(tbl_k)
            G_vals.append(g_statistic(mi, N_exp5))
        G = np.array(G_vals)
        df_mean = G.mean()
        df_var = G.var() / 2.0
        df_mle, _, _ = stats.chi2.fit(G, floc=0, fscale=1)
        results[k_z] = dict(df_pred=df_pred, df_mean=df_mean,
                             df_var=df_var, df_mle=df_mle)

    k_z_arr = np.array(k_z_values_exp5)
    df_pred = np.array([results[k]['df_pred'] for k in k_z_values_exp5])
    df_mean = np.array([results[k]['df_mean'] for k in k_z_values_exp5])
    df_var = np.array([results[k]['df_var'] for k in k_z_values_exp5])
    df_mle = np.array([results[k]['df_mle'] for k in k_z_values_exp5])

    # Create figure with 2 panels
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: sigma_0 scatter (from exp4)
    ax = axes[0]
    ax.text(-0.15, 1.05, 'A', transform=ax.transAxes, fontsize=18,
            fontweight='bold', va='top', ha='right', color='#484B50')
    violin_data = [emp_sigma0_by_kz[k_z] for k_z in k_z_values]
    k_colors_p2 = get_kz_colors(k_z_values)
    parts = ax.violinplot(violin_data, positions=range(len(k_z_values)),
                          showmedians=True, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(k_colors_p2[i])
        pc.set_alpha(0.7)
    ax.axhline(chi2_pred, color=arcadia_colors()[1], lw=2, ls='--')
    ax.axhline(SIGMA_0, color=arcadia_colors()[2], lw=2, ls=':')
    ax.set_xticks(range(len(k_z_values)))
    ax.set_xticklabels([str(k) for k in k_z_values])
    ax.set_xlabel("$k_z$", fontsize=15)
    ax.set_ylabel("$\\sigma_0$ estimate (per subgroup)", fontsize=15)
    ax.tick_params(labelsize=14.5)
    set_arcadia_style(ax)

    # Right panel: df_CMI relationship (from exp5)
    ax = axes[1]
    ax.text(-0.15, 1.05, 'B', transform=ax.transAxes, fontsize=18,
            fontweight='bold', va='top', ha='right', color='#484B50')
    ax.plot(df_pred, df_pred, '--', color=arcadia_colors()[10],
            zorder=1, lw=1.5)
    # Plot in order: squares, circles, triangles for legibility
    ax.scatter(df_pred, df_var, s=60, marker='s',
               color=arcadia_colors()[1], zorder=3, label='From variance')
    ax.scatter(df_pred, df_mean, s=60, marker='o',
               color=arcadia_colors()[0], zorder=4, label='From mean')
    ax.scatter(df_pred, df_mle, s=60, marker='^',
               color=arcadia_colors()[2], zorder=5, label='From MLE')

    # Calculate regression statistics for each estimator
    from scipy import stats as scipy_stats

    # Function to calculate statistics
    def calc_stats(y_true, y_pred):
        # Linear regression
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(y_true, y_pred)
        r_squared = r_value ** 2
        # Deviation from y=x line
        deviations = y_pred - y_true
        mae = np.mean(np.abs(deviations))
        max_dev = np.max(np.abs(deviations))
        return slope, r_squared, mae, max_dev

    slope_mean, r2_mean, mae_mean, max_mean = calc_stats(df_pred, df_mean)
    slope_var, r2_var, mae_var, max_var = calc_stats(df_pred, df_var)
    slope_mle, r2_mle, mae_mle, max_mle = calc_stats(df_pred, df_mle)

    # Print statistics to command line
    print("  Figure 4 Panel 2 Statistics:")
    print(f"    From mean:     slope={slope_mean:.4f}, R²={r2_mean:.5f}, MAE={mae_mean:.2f}, max_dev={max_mean:.2f}")
    print(f"    From variance: slope={slope_var:.4f}, R²={r2_var:.5f}, MAE={mae_var:.2f}, max_dev={max_var:.2f}")
    print(f"    From MLE:      slope={slope_mle:.4f}, R²={r2_mle:.5f}, MAE={mae_mle:.2f}, max_dev={max_mle:.2f}")

    ax.set_xlabel("Predicted $df = (k_x-1)(k_y-1) \\cdot k_z$", fontsize=15)
    ax.set_ylabel("Empirical $df$", fontsize=15)
    ax.tick_params(labelsize=14.5)
    ax.legend(fontsize=15, frameon=False, loc='upper left')
    set_arcadia_style(ax)

    plt.tight_layout()
    archived = {
        'N': N, 'N_exp5': N_exp5, 'k_z_values': k_z_values,
        'k_z_values_exp5': k_z_values_exp5, 'n_variants': n_variants,
        'n_variants_exp5': n_variants_exp5, 'seed': seed,
        'strat_effect': strat_effect, 'noise': noise,
        'df_pred': df_pred, 'df_mean': df_mean, 'df_var': df_var,
        'df_mle': df_mle,
    }
    archived.update({f'sigma0_kz_{k_z}': values
                     for k_z, values in emp_sigma0_by_kz.items()})
    save_results(save_path, **archived)
    save_or_show(fig, save_path)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def save_run_metadata(save_dir, args, effective_parameters):
    commit = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
        check=False).stdout.strip() or None
    dirty = bool(subprocess.run(
        ['git', 'status', '--porcelain'], capture_output=True, text=True,
        check=False).stdout.strip())
    metadata = {
        'arguments': vars(args),
        'effective_parameters': effective_parameters,
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'git_commit': commit,
        'git_dirty': dirty,
        'platform': platform.platform(),
        'python': platform.python_version(),
        'versions': {
            'dn_mi': dn_mi_version,
            'matplotlib': mpl.__version__,
            'numpy': np.__version__,
            'scipy': scipy.__version__,
        },
    }
    metadata_path = os.path.join(save_dir, 'run_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(f"Saved run metadata to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate publication figures for DN-MI paper')
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Directory to save figures (creates if needed)')
    parser.add_argument('--experiments', type=int, nargs='+',
                       default=[1, 2, 3, 4, 5, 6, 7, 8],
                       help='Which figures to generate (1-8)')
    parser.add_argument('--fast', action='store_true',
                       help='Use reduced parameters for faster execution')

    args = parser.parse_args()

    # Create save directory if specified
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"Saving figures to: {args.save_dir}/")

    # Apply fast mode if requested
    if args.fast:
        print("Fast mode enabled (reduced parameters)")
        n_variants = FAST['n_variants']
        n_variants_tail = 1000  # Reduced from 10000
    else:
        n_variants = DEFAULTS['n_variants']
        n_variants_tail = 10_000

    if args.save_dir:
        save_run_metadata(args.save_dir, args, {
            'figure_4_n_variants': FAST['n_variants'] if args.fast else 300,
            'figure_4_n_variants_exp5': (
                FAST['n_variants'] if args.fast else 2000),
            'n_variants': n_variants,
            'n_variants_tail': n_variants_tail,
            'n_perms': FAST['n_perms'] * 10 if args.fast else 200,
            'n_reps': FAST['n_perm_variants'] if args.fast else 100,
            'seed': DEFAULTS['seed'],
        })

    # Helper functions to generate figures with and without legends
    def generate_fig2():
        experiment_2(
            n_variants=n_variants,
            save_path=os.path.join(args.save_dir, 'figure_2.png') if args.save_dir else None,
            show_legend=True,
            no_legend_save_path=(
                os.path.join(args.save_dir, 'figure_2_nolegend.png')
                if args.save_dir else None)
        )

    def generate_fig3():
        experiment_16(
            n_variants=n_variants,
            save_path=os.path.join(args.save_dir, 'figure_3.png') if args.save_dir else None,
            show_legend=True
        )
        if args.save_dir:
            experiment_16(
                n_variants=n_variants,
                save_path=os.path.join(args.save_dir, 'figure_3_nolegend.png'),
                show_legend=False
            )

    # Figure mapping
    experiments = {
        1: ('figure_1', lambda: experiment_1(
            n_variants=n_variants,
            save_path=os.path.join(args.save_dir, 'figure_1.png') if args.save_dir else None
        )),
        2: ('figure_2', generate_fig2),
        3: ('figure_3', generate_fig3),
        4: ('figure_4', lambda: figure_4(
            n_variants=FAST['n_variants'] if args.fast else 300,
            n_variants_exp5=FAST['n_variants'] if args.fast else 2000,
            save_path=os.path.join(args.save_dir, 'figure_4.png') if args.save_dir else None
        )),
        5: ('figure_5', lambda: experiment_23(
            n_variants=n_variants,
            save_path=os.path.join(args.save_dir, 'figure_5.png') if args.save_dir else None
        )),
        6: ('figure_6', lambda: experiment_6_7(
            n_variants=n_variants,
            save_path=os.path.join(args.save_dir, 'figure_6.png') if args.save_dir else None
        )),
        7: ('figure_7', lambda: experiment_6_5(
            n_variants=n_variants_tail,
            save_path=os.path.join(args.save_dir, 'figure_7.png') if args.save_dir else None
        )),
        8: ('figure_8', lambda: experiment_20(
            n_reps=FAST['n_perm_variants'] if args.fast else 100,
            n_perms=FAST['n_perms']*10 if args.fast else 200,
            save_path=os.path.join(args.save_dir, 'figure_8.png') if args.save_dir else None
        )),
    }

    # Generate requested figures
    print(f"\nGenerating {len(args.experiments)} figures...")
    print("="*70)

    for fig_num in sorted(args.experiments):
        if fig_num in experiments:
            fig_name, fig_func = experiments[fig_num]
            print(f"\n{fig_name.upper().replace('_', ' ')}")
            print("-"*70)
            start_time = time.time()
            fig_func()
            elapsed = time.time() - start_time
            print(f"  Completed in {elapsed:.1f}s")
        else:
            print(f"\nWarning: Figure {fig_num} not found (valid: 1-8)")

    print("\n" + "="*70)
    print("All figures generated successfully!")
    if args.save_dir:
        print(f"Figures saved to: {args.save_dir}/")
        print("Format: PNG (300 DPI) and SVG (vector)")


if __name__ == '__main__':
    main()
