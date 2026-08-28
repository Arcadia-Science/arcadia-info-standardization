# Arcadia information theoretic standardization and parametric statistics

A Python package for cross-dimensionality comparability of mutual information (MI) and conditional mutual information (CMI) using the CDF transform.

This repository is associated with the pub: **A likelihood-ratio framework for inference with discrete information-theoretic measures**
doi: 10.57844/arcadia-y5aa-6u5u

## Overview

When comparing mutual information across different contingency table dimensions (K), the raw plugin MI estimator suffers from dimensionality-dependent bias and variance. This package implements:

- **Plugin MI estimators** for 2D and 3D contingency tables
- **Basharin correction** (mean-only bias correction)
- **DN-basic** (mean + variance correction via standardization)
- **CDF transform** (probability integral transform correcting all moments)

The CDF transform enables:
- Cross-K comparability of MI values
- Reliable interaction detection in stratified analyses
- Proper null hypothesis testing under chi-squared approximation

## Installation

### Quick Install (Users)

```bash
pip install git+https://github.com/Arcadia-Science/arcadia-info-standardization.git
```

### Development Install

```bash
git clone https://github.com/Arcadia-Science/arcadia-info-standardization.git
cd arcadia-info-standardization
pip install -e ".[dev]"
```

**For detailed setup instructions, virtual environments, and troubleshooting, see [SETUP.md](SETUP.md).**

## Quick Start

```python
import numpy as np
from dn_mi import plugin_mi_3d, mi_to_z_cdf

# Your 3D contingency table: shape (k_x, k_y, k_z)
table = np.array([...])  # Your data here

# Calculate plugin MI
mi = plugin_mi_3d(table)

# Transform to z-score via CDF for cross-K comparability
N = table.sum()  # Sample size
k_x, k_y, k_z = table.shape
z = mi_to_z_cdf(mi, k_x, k_y, k_z, N)

# z follows N(0,1) under null hypothesis
# Comparable across different k_z values!
```

## Key Functions

### Estimators
- `plugin_mi_2d(table)` - Plugin MI for 2D contingency tables
- `plugin_mi_3d(table)` - Plugin MI for 3D contingency tables (CMI)

### Transformations
- `correct_basharin(mi, k_x, k_y, k_z, N)` - Basharin bias correction
- `mi_to_z(mi, k_x, k_y, k_z, N)` - DN-basic standardization
- `mi_to_z_cdf(mi, k_x, k_y, k_z, N)` - **Recommended**: CDF transform

### Utilities
- `chi2_df(k_x, k_y, k_z)` - Degrees of freedom
- `g_statistic(mi, N)` - G-statistic from MI

## Reproducing Publication Figures

The repository includes a standalone script to reproduce all figures from the paper:

```bash
# Fast mode (reduced iterations for testing)
python figures_pub.py --fast --save-dir figures/

# Full quality (may take several minutes)
python figures_pub.py --save-dir figures/
```

This generates 8 publication-quality figures demonstrating the CDF transform's properties.
Each figure also includes a compressed NPZ file containing its numerical results.
The output directory also includes a JSON manifest with run parameters and software versions.

## Citation

If you use this package in your research, please cite:

```
[Citation information will be added upon publication]
```

## Related Work

For continuous data and GMM-based mutual information estimation, see:
- [arcadia-gmm-infotheory](https://github.com/Arcadia-Science/arcadia-gmm-infotheory)

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please open an issue or pull request.
