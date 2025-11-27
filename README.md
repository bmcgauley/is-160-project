# IS-160 Project: RNN/LSTM Employment Forecasting with QCEW Data

A deep learning project that uses Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) networks to analyze and forecast employment trends from California's Quarterly Census of Employment and Wages (QCEW) data.

## Overview

This project builds a temporal deep learning model that processes sequential employment data including:
- **Temporal patterns**: Quarterly employment changes and seasonal trends
- **Geographic distributions**: County-level employment variations across California
- **Industry classifications**: Sector-specific employment dynamics
- **Wage trends**: Average weekly wage patterns and growth rates

The LSTM architecture is specifically designed for time-series forecasting, capturing long-term dependencies in employment data to predict future quarterly employment levels and wage trends.

## Key Features

- **Unified Pipeline**: Single-click execution from raw data consolidation to trained model predictions
- **Comprehensive Data Processing**: Automatic consolidation of multiple CSV files into master dataset
- **Data Protection**: Raw data files are never modified (read-only access)
- **Rich Visualizations**: Automatic generation of exploration plots (employment trends, wage patterns, industry analysis)
- **Feature Engineering**: Temporal and geographic features capturing economic relationships
- **LSTM Architecture**: Multi-layer LSTM with batch normalization and dropout for robust forecasting
- **Interactive Prediction**: User-friendly interface for making employment forecasts
- **Modular Design**: Separate modules for exploration, validation, feature engineering, training, and evaluation

## Quick Start

### Prerequisites
- Python 3.8+
- PyTorch
- pandas, numpy, matplotlib, seaborn, scikit-learn

### Installation

```bash
# Clone repository
git clone https://github.com/bmcgauley/is-160-project.git
cd is-160-project

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (CMD):
.venv\Scripts\activate.bat
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Start Commands

**Activate Virtual Environment:**
```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux/Mac
source .venv/bin/activate
```

**Run the Pipeline:**
```bash
# Interactive menu (recommended)
python main.py

# Full pipeline via CLI
python main.py --cli
```

**Monitor Training with TensorBoard:**
```bash
# Start TensorBoard (run in separate terminal while training)
tensorboard --logdir=runs

# Then open browser to: http://localhost:6006
```

### Data Setup

**IMPORTANT**: The raw data files have been manually downloaded and placed in `/data/raw/`. These files are our baseline and should NEVER be modified.

Raw data files:
- `qcew_2004-2007.csv`
- `qcew_2008-2011.csv`
- `qcew_2012-2015.csv`
- `qcew_2016-2019.csv`
- `qcew-2020-2022.csv`
- `qcew-2023-2024.csv`

### Running the Pipeline

The master orchestrator `main.py` coordinates the entire pipeline:

```bash
# Run the complete pipeline (consolidation → exploration → validation → features → training → prediction)
python main.py

# Run specific stages
python main.py --stage explore      # Data exploration and visualization
python main.py --stage validate     # Data quality validation
python main.py --stage train        # Model training
python main.py --stage predict      # Interactive prediction interface

# Options
python main.py --skip-plots          # Skip plot generation during exploration
python main.py --force-rebuild       # Force rebuild of consolidated dataset
python main.py --launch-interface    # Launch prediction interface after completion
```

## Pipeline Stages

The pipeline consists of 8 coordinated stages:

### Stage 1: Data Consolidation
- Combines all raw CSV files into a single master dataset
- **Input**: `/data/raw/*.csv`
- **Output**: `/data/processed/qcew_master_consolidated.csv`
- **Protection**: Raw files are never modified

### Stage 2: Data Exploration
- Exploratory data analysis with comprehensive statistics
- **Output**: Exploration visualizations in `/data/processed/plots/`
  - `employment_trends.png` - Total employment over time
  - `quarterly_distribution.png` - Employment by quarter
  - `wage_trends.png` - Average weekly wage trends
  - `top_industries.png` - Top 15 industries by employment

### Stage 3: Data Validation
- Quality checks: missing values, duplicates, value ranges
- Statistical anomaly detection
- **Output**: `/data/validated/qcew_validated.csv`

### Stage 4: Feature Engineering
- Quarter-over-quarter growth rates
- Seasonal adjustment factors
- Industry concentration metrics
- Geographic clustering features
- **Output**: `/data/processed/qcew_features.csv`

### Stage 5: Preprocessing
- Normalization using robust scaling
- Categorical encoding
- Sequence window creation for LSTM
- Train/validation/test splits
- **Output**: `/data/processed/qcew_preprocessed.csv`

### Stage 6: Model Training
- Multi-layer LSTM architecture
- Batch normalization and dropout
- Early stopping and checkpointing
- **Output**: `/data/processed/lstm_model.pt`

### Stage 7: Model Evaluation
- Accuracy metrics (MSE, MAE, MAPE)
- Prediction visualizations
- Baseline comparisons (ARIMA, exponential smoothing)

### Stage 8: Interactive Prediction
- User-friendly forecasting interface
- Multi-step ahead predictions
- Uncertainty bands and confidence intervals
- Industry risk dashboards

## Project Structure

```
is-160-project/
├── main.py                          # Master pipeline orchestrator
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── data/
│   ├── raw/                        # Raw CSV files (NEVER MODIFIED)
│   ├── processed/                  # Consolidated and processed data
│   │   └── plots/                  # Exploration visualizations
│   └── validated/                  # Quality-checked datasets
├── src/                            # Source modules
│   ├── exploration.py              # Data exploration and EDA
│   ├── validation.py               # Data quality validation
│   ├── feature_engineering.py      # Feature creation
│   ├── temporal_features.py        # Time-based features
│   ├── geographic_features.py      # Spatial features
│   ├── preprocessing.py            # Data normalization and encoding
│   ├── lstm_model.py               # LSTM architecture
│   ├── training.py                 # Model training loops
│   ├── evaluation.py               # Performance evaluation
│   ├── prediction_visuals.py       # Prediction plots
│   ├── baselines.py                # Baseline models
│   ├── forecasting.py              # Multi-step forecasting
│   └── logging_config.py           # Logging configuration
├── docs/                           # Documentation
│   ├── methodology.md              # LSTM methodology
│   ├── results.md                  # Results analysis
│   └── data_dictionary.md          # Data schema
└── specs/001/                      # Project specifications
    └── tasks.md                    # Task tracking
```

## Data Source

The project uses California's [Quarterly Census of Employment and Wages (QCEW)](https://edd.ca.gov/) data (2004-2024), which provides comprehensive employment statistics including:
- Employment counts by industry and geography
- Wage information and averages
- Quarterly updates with 20+ years of historical data
- Statewide California coverage (FIPS 6000)

## Model Architecture

The LSTM employs a time-series forecasting approach:
- **Sequential Processing**: LSTM layers capture temporal dependencies and long-term patterns
- **Batch Normalization**: Stabilizes training and improves convergence
- **Dropout Regularization**: Prevents overfitting on employment sequences
- **Multi-step Prediction**: Forecasts multiple quarters ahead with uncertainty estimates
- **Output**: Dense layers predict employment levels and growth rates

## Evaluation Metrics

Model performance is evaluated using:
- **MSE** (Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **MAPE** (Mean Absolute Percentage Error)
- **Directional Accuracy** (correctly predicting growth/decline)
- **Baseline Comparison** vs ARIMA, exponential smoothing, and traditional regression

## Development Workflow

The entire workflow is automated through `main.py`:

1. **Data Consolidation**: Combine raw CSV files into master dataset
2. **Exploration**: Statistical analysis and visualization
3. **Validation**: Quality checks and anomaly detection
4. **Feature Engineering**: Temporal and geographic feature creation
5. **Preprocessing**: Normalization, encoding, and sequence preparation
6. **Model Training**: LSTM architecture training with early stopping
7. **Evaluation**: Performance assessment and baseline comparison
8. **Prediction**: Interactive forecasting interface

## Team Members

- **Project Lead**: Overall coordination, setup, exploration, preprocessing, visualization
- **Andrew**: Training infrastructure, loss functions, evaluation
- **Alejo**: Feature engineering, documentation

See `specs/001/tasks.md` for detailed work assignments and progress tracking.

## Usage Examples

### Run Full Pipeline
```bash
python main.py
```

### Run Specific Stage
```bash
# Explore data and generate visualizations
python main.py --stage explore

# Train the LSTM model
python main.py --stage train

# Launch interactive prediction interface
python main.py --stage predict
```

### Skip Plot Generation
```bash
python main.py --skip-plots
```

### Force Rebuild
```bash
# Force rebuild of consolidated dataset (if raw data changed)
python main.py --force-rebuild
```

## Output Files

After running the pipeline, the following files are generated:

- `/data/processed/qcew_master_consolidated.csv` - Combined dataset from all raw files
- `/data/validated/qcew_validated.csv` - Quality-checked dataset
- `/data/processed/qcew_features.csv` - Dataset with engineered features
- `/data/processed/qcew_preprocessed.csv` - Normalized and encoded sequences
- `/data/processed/lstm_model.pt` - Trained PyTorch model
- `/data/processed/plots/*.png` - Exploration visualizations

## Contributing

1. Review the feature specification in `specs/001/spec.md`
2. Check assigned tasks in `specs/001/tasks.md`
3. Follow the development workflow and coding standards
4. Submit pull requests with clear descriptions
5. Ensure all tests pass before merging

## Resources

- [QCEW Documentation](https://www.bls.gov/cew/)
- [California Open Data Portal](https://data.ca.gov/)
- [PyTorch LSTM Tutorial](https://pytorch.org/docs/stable/nn.html#lstm)
- [Time Series Forecasting with PyTorch](https://pytorch.org/tutorials/beginner/timeseries_tutorial.html)
- [Scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)

## License

[Specify license if applicable]
