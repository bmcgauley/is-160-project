# Quick Start: Build a Convolutional Neural Network for Employment Trends Analysis

**Date**: 2025-09-24
**Audience**: Data scientists, ML engineers, economists
**Prerequisites**: Python 3.8+, basic ML knowledge

## Overview
This guide gets you started with the CNN-based employment trend analysis project. The system predicts quarterly employment changes from California's QCEW data using spatio-temporal convolutional neural networks.

## Environment Setup

### 1. Clone and Setup
```bash
git clone <repository-url>
cd is-160-project
git checkout 001-build-a-convolutional
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pandas scikit-learn matplotlib seaborn jupyter
pip install pytest black isort  # development tools
```

### 4. Data Acquisition
```bash
mkdir -p data/raw data/processed data/validated
python src/data_download.py  # Downloads all required QCEW CSV files automatically
# Note: Data files are gitignored and will not be committed to version control
```

## Project Structure
```
├── data/
│   ├── raw/           # Original QCEW CSV files
│   ├── processed/     # Cleaned and engineered features
│   └── validated/     # Quality-checked datasets
├── src/
│   ├── data_acquisition.py    # Data loading scripts
│   ├── validation.py          # Quality checks
│   ├── feature_engineering.py # Feature creation
│   ├── cnn_model.py          # Neural network architecture
│   ├── training.py           # Model training pipeline
│   └── evaluation.py         # Performance assessment
├── notebooks/
│   └── exploration.ipynb     # Data exploration
├── models/                   # Saved model artifacts
├── reports/                  # Validation and results
└── tests/                    # Unit and integration tests
```

## Quick Test Run

### 1. Data Exploration
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load sample data
df = pd.read_csv('data/raw/qcew_sample.csv')
print(df.head())
print(df.describe())

# Basic visualization
df.groupby('quarter')['employment'].sum().plot()
plt.title('California Employment Trends')
plt.show()
```

### 2. Run Validation
```python
from src.validation import validate_employment_data

issues = validate_employment_data(df)
print(f"Found {len(issues)} data quality issues")
```

### 3. Feature Engineering
```python
from src.feature_engineering import create_features

features = create_features(df)
print(features.head())
```

### 4. Model Training (Basic)
```python
from src.cnn_model import EmploymentCNN
from src.training import train_model

model = EmploymentCNN()
# Note: Requires processed tensor data
# train_model(model, train_data, val_data)
```

## Development Workflow

### 1. Code Quality
```bash
# Format code
black src/
isort src/

# Run tests
pytest tests/

# Type checking (if configured)
mypy src/
```

### 2. Data Pipeline
```bash
# Full pipeline execution
python -m src.data_acquisition
python -m src.validation
python -m src.feature_engineering
python -m src.training
```

### 3. Model Evaluation
```python
from src.evaluation import evaluate_model

metrics = evaluate_model(model, test_data)
print(f"MAPE: {metrics['mape']:.2%}")
print(f"Directional Accuracy: {metrics['directional_acc']:.2%}")
```

## Common Issues

### Data Loading
- **Issue**: Encoding errors in CSV files
- **Solution**: Specify encoding='latin1' in pd.read_csv()

### Memory Usage
- **Issue**: Large datasets don't fit in memory
- **Solution**: Process data in chunks or use Dask

### Model Convergence
- **Issue**: CNN not learning spatio-temporal patterns
- **Solution**: Check tensor dimensions and normalization

### Geographic Data
- **Issue**: Missing FIPS codes or invalid geographies
- **Solution**: Cross-reference with official census data

## Next Steps

1. **Complete Data Acquisition**: Ensure all QCEW files are downloaded
2. **Run Full Validation**: Address any data quality issues
3. **Experiment with Features**: Try different engineering approaches
4. **Tune Model Architecture**: Adjust CNN hyperparameters
5. **Compare Baselines**: Implement and evaluate ARIMA, regression models
6. **Document Results**: Create comprehensive analysis report

## Resources

- **QCEW Documentation**: https://www.bls.gov/cew/
- **PyTorch Tutorials**: https://pytorch.org/tutorials/
- **Scikit-learn Guide**: https://scikit-learn.org/stable/user_guide.html
- **California Employment Data**: https://edd.ca.gov/