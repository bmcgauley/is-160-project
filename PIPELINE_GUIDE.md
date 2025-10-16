# QCEW Employment Forecasting Pipeline Guide

## Overview

The pipeline has been completely refactored to be more modular, maintainable, and user-friendly.

## Key Improvements

### 1. Interactive Menu Interface ✅
- **Default mode**: Run `python main.py` to launch an easy-to-use menu
- No need to remember command-line arguments
- Visual pipeline status tracking
- Individual stage execution with dependency checking

### 2. Modular Architecture ✅
- **Smaller main.py**: Now only 120 lines (was 650+)
- **Dedicated modules** in `/src/`:
  - `pipeline_orchestrator.py` - Coordinates all stages
  - `consolidation.py` - Data merging logic
  - `exploration.py` - Clean EDA and plotting (legacy code removed)
  - More modules to be added for training, evaluation, etc.

### 3. Fixed Issues ✅
- ✅ Removed Unicode characters causing Windows PowerShell errors
- ✅ Fixed column name handling (Year vs year, Quarter vs qtr)
- ✅ Column normalization (lowercase with underscores)
- ✅ Automatic column mapping for consistency
- ✅ Raw data protection verified after every consolidation

### 4. Both CLI and Interactive Modes ✅
```bash
# Interactive Menu (Default)
python main.py

# Command-Line Mode
python main.py --cli
python main.py --stage consolidate
python main.py --stage explore
python main.py --force-rebuild
```

## Usage

### Interactive Menu (Recommended)

```bash
python main.py
```

Then select from the menu:
```
1. Run Full Pipeline (All Stages)
2. Data Consolidation
3. Data Exploration & Visualization
4. Data Validation
5. Feature Engineering
6. Data Preprocessing
7. Train LSTM Model
8. Evaluate Model
9. Interactive Prediction Interface
10. View Pipeline Status
0. Exit
```

### Command-Line Mode

```bash
# Run full pipeline
python main.py --cli

# Run specific stage
python main.py --stage consolidate
python main.py --stage explore --skip-plots

# Force rebuild
python main.py --stage consolidate --force-rebuild
```

## Pipeline Stages

### Stage 1: Data Consolidation ✅ WORKING
- Combines 6 raw CSV files (2004-2024)
- **5.4 million records**
- Normalizes column names
- Maps columns for consistency
- Verifies raw data integrity
- **Output**: `/data/processed/qcew_master_consolidated.csv`

### Stage 2: Data Exploration ✅ READY
- Exploratory data analysis
- Summary statistics
- Automatic visualization generation
- **Output**: `/data/processed/plots/`
  - `employment_trends.png`
  - `quarterly_distribution.png`
  - `wage_trends.png`
  - `top_industries.png`

### Stage 3: Data Validation ⏳ TO BE IMPLEMENTED
- Quality checks
- Missing value analysis
- Outlier detection
- **Output**: `/data/validated/qcew_validated.csv`

### Stage 4: Feature Engineering ⏳ TO BE IMPLEMENTED
- Quarter-over-quarter growth rates
- Seasonal adjustments
- Industry metrics
- **Output**: `/data/processed/qcew_features.csv`

### Stage 5: Data Preprocessing ⏳ TO BE IMPLEMENTED
- Normalization
- Encoding
- Sequence preparation
- **Output**: `/data/processed/qcew_preprocessed.csv`

### Stage 6: Model Training ⏳ TO BE IMPLEMENTED
- LSTM architecture
- Training loops
- Early stopping
- **Output**: `/data/processed/lstm_model.pt`

### Stage 7: Model Evaluation ⏳ TO BE IMPLEMENTED
- Performance metrics
- Baseline comparisons
- **Output**: Evaluation reports

### Stage 8: Interactive Prediction ⏳ TO BE IMPLEMENTED
- Forecasting interface
- Visualization
- Export results

## File Structure

```
is-160-project/
├── main.py                          # Simple entry point (120 lines)
├── interactive_menu.py              # Menu interface
├── data/
│   ├── raw/                        # 6 CSV files (NEVER MODIFIED)
│   ├── processed/                  # Pipeline outputs
│   │   └── plots/                  # Visualizations
│   └── validated/                  # Quality-checked data
├── src/                            # Modular pipeline components
│   ├── pipeline_orchestrator.py    # Stage coordinator
│   ├── consolidation.py            # Data merging
│   ├── exploration.py              # EDA & plots (clean)
│   ├── validation.py               # Quality checks
│   ├── feature_engineering.py      # Feature creation
│   ├── preprocessing.py            # Data prep
│   ├── lstm_model.py               # Model architecture
│   └── logging_config.py           # Logging setup
├── logs/                           # Pipeline logs
└── docs/                           # Documentation
```

## Test Results

### Consolidation Stage ✅
```
[OK] Successfully consolidated 6 files
  Total records: 5,430,384
  Columns: 16
  Date range: 2004-2024
  Memory usage: 2437.7 MB
[OK] Consolidated dataset saved successfully
[OK] Raw data files unchanged
```

## Next Steps

1. ✅ Complete data consolidation
2. ✅ Create interactive menu
3. ✅ Refactor to modular architecture
4. ⏳ Implement validation module
5. ⏳ Implement feature engineering pipeline
6. ⏳ Implement preprocessing pipeline
7. ⏳ Implement training pipeline
8. ⏳ Implement evaluation pipeline
9. ⏳ Implement prediction interface

## Notes

- **Raw Data Protection**: All raw CSV files in `/data/raw/` are NEVER modified
- **Column Normalization**: All columns converted to lowercase_with_underscores
- **Column Mapping**: Standardized names (e.g., `time_period` → `quarter`)
- **Legacy Code**: Removed from exploration.py - now clean and modular
- **Windows Compatible**: No Unicode characters causing PowerShell errors

## Team Members

- **Project Lead**: Pipeline orchestration, setup, exploration, visualization
- **Andrew**: Training infrastructure, loss functions, evaluation
- **Alejo**: Feature engineering, documentation

## Support

For issues or questions, refer to:
- `specs/001/tasks.md` - Task tracking
- `docs/` - Project documentation
- This guide - Pipeline usage
