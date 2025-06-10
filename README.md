# PyCronos

A comprehensive Python package for data analysis and time series processing.

## Features

- **Data Loading**: Support for multiple file formats (CSV, Excel, JSON, Parquet)
- **Data Preprocessing**: Missing value handling, normalization, standardization
- **Statistical Analysis**: Descriptive statistics, hypothesis testing, correlation analysis
- **Time Series Analysis**: Decomposition, moving averages, outlier detection
- **Visualization**: Built-in plotting capabilities with matplotlib and seaborn
- **Data Validation**: Comprehensive data quality checks

## Installation

```bash
pip install pycronos
```

For development installation:

```bash
git clone https://github.com/yourusername/pycronos.git
cd pycronos
pip install -e ".[dev]"
```

## Quick Start

```python
import pycronos as pc
import pandas as pd

# Load data
loader = pc.DataLoader()
df = loader.auto_load('data.csv')

# Basic preprocessing
preprocessor = pc.Preprocessor()
df_clean = preprocessor.handle_missing_values(df, strategy='fill_mean')

# Descriptive analysis
analysis = pc.DescriptiveAnalysis()
summary = analysis.summary_statistics(df_clean)
correlation = analysis.correlation_matrix(df_clean)

# Visualization
viz = pc.Visualizer()
fig = viz.correlation_heatmap(df_clean)
```

## Package Structure

- `core/`: Data loading, preprocessing, and validation
- `analysis/`: Statistical and time series analysis
- `visualization/`: Plotting and dashboard utilities
- `utils/`: Helper functions and configuration

## Documentation

Full documentation is available at [https://pycronos.readthedocs.io/](https://pycronos.readthedocs.io/)

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Changelog

### v0.1.0
- Initial release
- Basic data loading and preprocessing
- Statistical analysis capabilities
- Visualization tools
