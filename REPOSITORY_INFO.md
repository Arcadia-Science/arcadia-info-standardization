# Repository Information

## Names and URLs

### GitHub Repository
- **Name**: `arcadia-info-standardization`
- **URL**: `https://github.com/Arcadia-Science/arcadia-info-standardization`
- **Clone**: `git clone https://github.com/Arcadia-Science/arcadia-info-standardization.git`

### Python Package
- **Package name**: `dn-mi` (on PyPI if/when published)
- **Import name**: `dn_mi` (with underscore)
- **Usage**: `from dn_mi import plugin_mi_3d, mi_to_z_cdf`

### Local Directory
- **Development folder**: `arcadia-info-standardization/`
- **After clone**: `cd arcadia-info-standardization`

## Why Different Names?

This is a common pattern:
- **GitHub repo name** (`arcadia-info-standardization`): Descriptive, indicates Arcadia project scope
- **Python package name** (`dn-mi`): Short, focused on method (Dimensionality Normalization for MI)
- **Import name** (`dn_mi`): Python-valid identifier (hyphens not allowed)

## Quick Reference

```bash
# Clone repository
git clone https://github.com/Arcadia-Science/arcadia-info-standardization.git
cd arcadia-info-standardization

# Install package
pip install -e .

# Use in Python
python3 -c "from dn_mi import mi_to_z_cdf; print('Works!')"
```

## Publishing Checklist

When ready to publish this repository:

- [ ] Create GitHub repository: `Arcadia-Science/arcadia-info-standardization`
- [ ] Push this folder to the repository
- [ ] Add topics/tags: `mutual-information`, `statistics`, `bioinformatics`, `genomics`
- [ ] Test installation: `pip install git+https://github.com/Arcadia-Science/arcadia-info-standardization.git`
- [ ] Run tests: `pytest tests/ -v`
- [ ] Generate and check figures: `python figures_pub.py --fast --save-dir test/`
- [ ] Update citation information in README.md when paper is published
- [ ] (Optional) Publish to PyPI: `python -m build && twine upload dist/*`

## File Structure Summary

```
arcadia-info-standardization/          # GitHub repo name
├── dn_mi/                             # Python package (import as dn_mi)
│   ├── __init__.py
│   ├── estimators.py
│   ├── transforms.py
│   └── data.py
├── pyproject.toml                     # Declares package name as "dn-mi"
├── README.md                          # Main documentation
├── SETUP.md                          # Environment setup
├── QUICKSTART.md                     # Quick reference
└── ...
```

## Support

- **Documentation**: See README.md, SETUP.md, QUICKSTART.md
- **Issues**: https://github.com/Arcadia-Science/arcadia-info-standardization/issues
- **Examples**: See `examples/apply_dn_to_your_data.ipynb`
