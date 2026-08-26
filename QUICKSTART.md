# Quick Start Guide

## Installation

```bash
# Clone the repository
git clone https://github.com/Arcadia-Science/arcadia-info-standardization.git
cd arcadia-info-standardization

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

## Basic Usage

```python
import numpy as np
from dn_mi import plugin_mi_3d, mi_to_z_cdf

# Your 3D contingency table (k_x, k_y, k_z)
table = np.random.randint(10, 100, size=(3, 2, 10))

# Calculate plugin MI
mi = plugin_mi_3d(table)
print(f"MI: {mi:.6f} bits")

# Transform to z-score for cross-K comparability
k_x, k_y, k_z = table.shape
N = int(table.sum())
z = mi_to_z_cdf(mi, k_x, k_y, k_z, N)
print(f"z-score: {z:.3f}")
```

## Reproducing Paper Figures

```bash
# Quick test (reduced iterations)
python figures_pub.py --fast --save-dir test_figures/

# Full quality figures
python figures_pub.py --save-dir publication_figures/

# Specific figures only
python figures_pub.py --experiments 1 2 3 --save-dir figures/
```

Each figure is accompanied by an NPZ results file.
The output directory also contains `run_metadata.json` with the parameters, commit, and software versions.

## Running Tests

```bash
pytest tests/
```

## Example Notebooks

See `examples/apply_dn_to_your_data.ipynb` for detailed usage examples.

## Key Functions

### Estimators
- `plugin_mi_2d(table)` - MI for 2D tables
- `plugin_mi_3d(table)` - Conditional MI for 3D tables

### Transforms (Corrections)
- `correct_basharin(mi, k_x, k_y, k_z, N)` - Bias correction
- `mi_to_z(mi, k_x, k_y, k_z, N)` - DN-basic standardization
- `mi_to_z_cdf(mi, k_x, k_y, k_z, N)` - **Recommended**: CDF transform

### Why use the CDF transform?

The CDF transform (`mi_to_z_cdf`) is the **recommended** method because it:
1. Corrects ALL moments (not just mean and variance)
2. Produces exact N(0,1) distribution under null
3. Enables cross-K comparability
4. Works for any k_x, k_y, k_z combination

## Common Use Cases

### 1. Test for association in stratified data

```python
from dn_mi import plugin_mi_3d, mi_to_z_cdf
from scipy import stats

# Your data
mi = plugin_mi_3d(table)
z = mi_to_z_cdf(mi, k_x, k_y, k_z, N)

# Upper-tail test for association
p_value = stats.norm.sf(z)
print(f"p-value: {p_value:.4e}")
```

### 2. Compare MI across different stratification levels

```python
# Tables with different k_z
table_5 = ...   # k_z = 5
table_10 = ...  # k_z = 10
table_20 = ...  # k_z = 20

# Transform all to z-scores for comparison
z_5 = mi_to_z_cdf(plugin_mi_3d(table_5), 3, 2, 5, N)
z_10 = mi_to_z_cdf(plugin_mi_3d(table_10), 3, 2, 10, N)
z_20 = mi_to_z_cdf(plugin_mi_3d(table_20), 3, 2, 20, N)

# Now directly comparable!
```

## Next Steps

- Read the full documentation in README.md
- Explore `examples/apply_dn_to_your_data.ipynb`
- Check out the paper for theoretical details
- Run tests: `pytest tests/ -v`
