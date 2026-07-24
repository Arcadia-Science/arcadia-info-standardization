# Environment Setup Guide

Complete instructions for setting up a Python environment to use DN-MI.

## Prerequisites

- **Python 3.8 or higher** (recommended: Python 3.10+)
- **pip** (Python package installer)
- **git** (for cloning the repository)

Check your Python version:
```bash
python3 --version
```

## Option 1: Quick Setup (User Installation)

For users who just want to use the package:

```bash
# Install directly from GitHub
pip install git+https://github.com/Arcadia-Science/arcadia-info-standardization.git

# Verify installation
python3 -c "from dn_mi import plugin_mi_3d, mi_to_z_cdf; print('DN-MI installed successfully!')"
```

## Option 2: Development Setup (Recommended for Contributors)

For developers who want to modify the code or reproduce figures:

### Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/Arcadia-Science/arcadia-info-standardization.git
cd arcadia-info-standardization
```

### Step 2: Create a Virtual Environment

**Using venv (built-in to Python):**

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# You should see (venv) in your terminal prompt
```

**Using conda (alternative):**

```bash
# Create conda environment
conda create -n dn-mi python=3.10
conda activate dn-mi
```

### Step 3: Install the Package

**For development (editable install):**

```bash
# Install in editable mode with development dependencies
pip install -e ".[dev]"
```

This installs:
- Core dependencies: `numpy`, `scipy`
- Dev dependencies: `pytest`, `matplotlib`, `jupyter`

**For basic usage only:**

```bash
# Install in editable mode (no dev dependencies)
pip install -e .
```

### Step 4: Verify Installation

```bash
# Run tests to verify everything works
pytest tests/ -v

# Try importing the package
python3 -c "from dn_mi import plugin_mi_3d, mi_to_z_cdf, __version__; print(f'DN-MI v{__version__} installed!')"
```

## Option 3: Figure Reproduction Only

If you only want to reproduce the publication figures (no package installation):

### Step 1: Clone and Enter Directory

```bash
git clone https://github.com/Arcadia-Science/arcadia-info-standardization.git
cd arcadia-info-standardization
```

### Step 2: Install Requirements

```bash
# Install only what's needed for figures
pip install numpy scipy matplotlib
```

### Step 3: Run Figure Script

```bash
# Quick test (fast mode)
python figures_pub.py --fast --save-dir test_figures/

# Full quality figures (takes longer)
python figures_pub.py --save-dir publication_figures/
```

## Working with Jupyter Notebooks

To use the example notebooks:

```bash
# Make sure you have jupyter installed
pip install jupyter

# Start Jupyter
jupyter notebook

# Navigate to examples/apply_dn_to_your_data.ipynb
```

**Or use JupyterLab:**

```bash
pip install jupyterlab
jupyter lab
```

## Package Dependencies

### Core Dependencies (required)
- `numpy >= 1.20` - Numerical computing
- `scipy >= 1.7` - Statistical functions (chi-squared, normal distributions)

### Development Dependencies (optional)
- `pytest >= 7.0` - Running tests
- `matplotlib >= 3.5` - Plotting (for figures_pub.py)
- `jupyter >= 1.0` - Interactive notebooks

### Font Dependencies (for figures)
- `AtkinsonHyperlegibleNext` - Main font for text
- `AtkinsonHyperlegibleMono` - Monospace font for numbers

Install fonts (Linux):
```bash
# Download fonts (example)
wget https://github.com/googlefonts/atkinson-hyperlegible/archive/refs/heads/main.zip
unzip main.zip
sudo cp atkinson-hyperlegible-main/fonts/otf/*.otf /usr/local/share/fonts/
fc-cache -fv
```

## Troubleshooting

### "Module 'dn_mi' not found"

**Solution:** Make sure you're in the correct directory and the package is installed:

```bash
# Check if installed
pip list | grep dn-mi

# If not found, install in editable mode
pip install -e .
```

### "ImportError: No module named 'numpy'"

**Solution:** Install dependencies:

```bash
pip install numpy scipy
```

### Matplotlib font warnings

**Solution:** If you see font warnings when running `figures_pub.py`:

```bash
# Clear matplotlib cache
rm -rf ~/.cache/matplotlib

# Or ignore warnings - figures will still generate with fallback fonts
```

### Tests fail with "ModuleNotFoundError"

**Solution:** Install development dependencies:

```bash
pip install -e ".[dev]"
# or
pip install pytest
```

### Virtual environment issues

**Solution:** Make sure you've activated your virtual environment:

```bash
# Check which python is being used
which python3

# Should point to venv/bin/python3 (not /usr/bin/python3)
# If not, reactivate:
source venv/bin/activate
```

## Common Workflows

### As a User (applying DN-MI to your data)

```bash
# One-time setup
pip install git+https://github.com/Arcadia-Science/arcadia-info-standardization.git

# In your Python script or notebook:
# from dn_mi import plugin_mi_3d, mi_to_z_cdf
```

### As a Developer (modifying the package)

```bash
# One-time setup
git clone https://github.com/Arcadia-Science/arcadia-info-standardization.git
cd arcadia-info-standardization
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# After making changes:
pytest tests/  # Run tests
python figures_pub.py --fast --save-dir test/  # Test figures
```

### As a Reviewer (reproducing paper results)

```bash
# One-time setup
git clone https://github.com/Arcadia-Science/arcadia-info-standardization.git
cd arcadia-info-standardization
pip install numpy scipy matplotlib

# Reproduce figures
python figures_pub.py --save-dir paper_figures/

# Figures will be in paper_figures/ as PNG and SVG
```

## System Requirements

### Minimum
- Python 3.8+
- 4 GB RAM
- 1 GB free disk space

### Recommended
- Python 3.10+
- 8 GB RAM (for running all tests and figures)
- 2 GB free disk space

### Operating Systems
- Linux (tested on Ubuntu 20.04+)
- macOS (tested on macOS 12+)
- Windows (should work but less tested)

## Getting Help

If you encounter issues:

1. Check this SETUP.md for troubleshooting
2. Read QUICKSTART.md for usage examples
3. Check README.md for full documentation
4. Open an issue on GitHub: https://github.com/Arcadia-Science/arcadia-info-standardization/issues

## Next Steps

After setup:

1. ✅ Run tests: `pytest tests/ -v`
2. 📊 Try quick example: `python -c "from dn_mi import plugin_mi_3d, mi_to_z_cdf; import numpy as np; print(mi_to_z_cdf(plugin_mi_3d(np.ones((3,2,10))*10), 3, 2, 10, 600))"`
3. 📓 Explore notebook: `jupyter notebook examples/apply_dn_to_your_data.ipynb`
4. 🎨 Generate figures: `python figures_pub.py --fast --save-dir test/`
5. 📖 Read documentation: Open README.md

Happy computing! 🎉
