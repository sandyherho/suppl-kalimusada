# Supplementary Scripts: `kalimusada`: Python-Based Sensitivity Analysis for the Ma-Chen Financial System

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Scripts for generating publication figures from Ma-Chen chaotic system simulations.

## Links

- **Solver**: [github.com/sandyherho/kalimusada](https://github.com/sandyherho/kalimusada)
- **Data & Outputs**: [doi.org/10.17605/OSF.IO/3ZCA7](https://doi.org/10.17605/OSF.IO/3ZCA7)

## Scripts

| Script | Description |
|--------|-------------|
| `divergence_timeseries.py` | Trajectory divergence metrics (3×1 panels) |
| `phase_space_3d.py` | 3D strange attractor visualization (2×2 panels) |
| `state_variables.py` | State variable time series (4×3 panels) |

## Usage

```bash
pip install numpy matplotlib netCDF4
python divergence_timeseries.py
python phase_space_3d.py
python state_variables.py
```

## License

MIT © Sandy H. S. Herho
