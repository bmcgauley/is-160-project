# IS-160 Project: CNN-Based Employment Trends Analysis

A deep learning project that uses Convolutional Neural Networks (CNNs) to analyze and predict employment trends from California's Quarterly Census of Employment and Wages (QCEW) data.

## Overview

This project builds a spatio-temporal CNN model that processes multi-dimensional employment data including:
- **Temporal patterns**: Quarterly employment changes and seasonal trends
- **Geographic distributions**: County-level employment variations across California
- **Industry classifications**: Sector-specific employment dynamics

The CNN architecture is specifically adapted for tabular time-series data, combining 1D convolutions for temporal processing with 2D convolutions for spatial patterns to predict quarterly employment changes.

## Key Features

- **Automated Data Pipeline**: Downloads and processes QCEW data from California's open data portal
- **Comprehensive Validation**: Statistical quality checks and anomaly detection for employment data
- **Feature Engineering**: Domain-specific features capturing economic relationships and temporal dynamics
- **CNN Architecture**: Custom neural network combining temporal and spatial convolutions
- **Baseline Comparison**: Performance evaluation against traditional forecasting methods (ARIMA, regression, random forest)
- **Reproducible Research**: Fixed seeds, detailed logging, and modular PyTorch implementation

## Quick Start

### Prerequisites
- Python 3.8+
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd is-160-project
   git checkout 001-build-a-convolutional
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   pip install pandas scikit-learn matplotlib seaborn jupyter pytest
   ```

4. **Download data**
   ```bash
   python src/data_download.py
   ```

### Basic Usage

```python
from src.data_acquisition import load_qcew_data
from src.feature_engineering import create_features
from src.cnn_model import EmploymentCNN
from src.training import train_model

# Load and process data
df = load_qcew_data()
features = create_features(df)

# Train model
model = EmploymentCNN()
trained_model = train_model(model, features)
```

## Project Structure

```
├── specs/001-build-a-convolutional/  # Feature specifications and planning
│   ├── spec.md                       # Feature requirements
│   ├── plan.md                       # Implementation plan
│   ├── research.md                   # Technical research
│   ├── data-model.md                 # Data structures
│   ├── quickstart.md                 # Setup guide
│   ├── tasks.md                      # Development tasks
│   └── contracts/                    # API specifications
├── src/                             # Source code
│   ├── data_acquisition.py          # Data loading and preprocessing
│   ├── validation.py                # Quality checks
│   ├── feature_engineering.py       # Feature creation
│   ├── cnn_model.py                 # Neural network architecture
│   ├── training.py                  # Model training pipeline
│   ├── evaluation.py                # Performance assessment
│   └── utils.py                     # Helper functions
├── tests/                           # Unit and integration tests
├── notebooks/                       # Jupyter notebooks for exploration
├── data/                            # Data directories (gitignored)
│   ├── raw/                         # Original QCEW files
│   ├── processed/                   # Cleaned features
│   └── validated/                   # Quality-checked data
├── models/                          # Saved PyTorch models
├── reports/                         # Validation reports
└── figures/                         # Generated plots
```

## Data Source

The project uses California's [Quarterly Census of Employment and Wages (QCEW)](https://edd.ca.gov/) data, which provides comprehensive employment statistics including:
- Employment counts by industry and geography
- Wage information and averages
- Quarterly updates with historical data
- County-level geographic resolution

## Model Architecture

The CNN employs a hybrid approach:
- **Temporal Processing**: 1D convolutions capture quarterly patterns and seasonality
- **Spatial Processing**: 2D convolutions identify geographic employment correlations
- **Fusion Layer**: Combines temporal and spatial features for prediction
- **Output**: Dense layers predict employment changes with uncertainty estimates

## Evaluation Metrics

Model performance is evaluated against:
- **MAPE** (Mean Absolute Percentage Error)
- **Directional Accuracy** (correctly predicting growth/decline)
- **RMSE** (Root Mean Square Error)
- **Baseline Comparison** vs ARIMA, linear regression, and random forest

## Development Workflow

1. **Data Acquisition**: Automated download and initial processing
2. **Exploration**: Statistical analysis and visualization
3. **Validation**: Quality checks and anomaly detection
4. **Feature Engineering**: Domain-specific feature creation
5. **Model Development**: CNN architecture design and training
6. **Evaluation**: Performance assessment and baseline comparison
7. **Documentation**: Results analysis and reporting

## Team Members

- **Project Lead**: [Your Name]
- **Team Member**: Andrew
- **Team Member**: Alejo

See `specs/001-build-a-convolutional/tasks.md` for detailed work assignments and progress tracking.

## Contributing

1. Review the feature specification in `specs/001-build-a-convolutional/spec.md`
2. Check assigned tasks in `tasks.md`
3. Follow the development workflow and coding standards
4. Submit pull requests with clear descriptions
5. Ensure all tests pass before merging

## License

[Specify license if applicable]

## Resources

- [QCEW Documentation](https://www.bls.gov/cew/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [California Employment Development Department](https://edd.ca.gov/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)</content>
<parameter name="filePath">c:\GitHub\is-160-project\README.md