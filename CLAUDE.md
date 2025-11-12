# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Employment forecasting system that uses LSTM/RNN models to predict California quarterly employment trends from QCEW (Quarterly Census of Employment and Wages) data. The project processes 5.4M+ employment records through a multi-stage pipeline, performs feature engineering, and trains deep learning models for time-series forecasting.

## Development Setup

### Environment Setup
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux/Mac
python -m venv .venv
source .venv/bin/activate

# Install PyTorch with CUDA support (for GPU training)
# For CUDA 11.8 (compatible with GTX 1070, RTX series)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1 (newer GPUs)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU-only (no GPU)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements.txt

# Verify GPU detection (should show True if GPU is available)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### Running the Pipeline

**Interactive Mode (Default)**:
```bash
python main.py
```
Provides numbered menu to run individual stages or full pipeline.

**CLI Mode**:
```bash
# Run full pipeline
python main.py --cli

# Run specific stage
python main.py --stage consolidate    # Stage 1: Data consolidation
python main.py --stage explore        # Stage 2: Exploration
python main.py --stage validate       # Stage 3: Validation
python main.py --stage features       # Stage 4: Feature engineering
python main.py --stage preprocess     # Stage 5: Preprocessing
python main.py --stage train          # Stage 6: Training
python main.py --stage evaluate       # Stage 7: Evaluation

# Additional options
python main.py --skip-plots          # Skip plot generation
python main.py --force-rebuild       # Force rebuild consolidated data
```

### Testing
```bash
# Test specific feature engineering tasks
python test_T032.py              # Test county-level filtering
python test_quarterly_filter.py  # Test quarterly filtering
python test_improvements.py      # Test recent improvements
```

## Development Workflow

### Git Workflow
Use feature branching with pull requests:
```bash
# Create feature branch
git checkout -b feature/task-description

# Work on feature, commit changes
git add .
git commit -m "feat: implement feature X"

# Push and create PR
git push -u origin feature/task-description
gh pr create --title "Feature: Task Description" --body "Implementation details"

# After code review and /security-audit, merge to master
gh pr merge --squash
```

### Issue Tracking
Use GitHub CLI for task management:
```bash
# Create issue
gh issue create --title "Task description" --body "Details"

# List issues
gh issue list

# Close issue
gh issue close <issue-number>

# Link PR to issue
gh pr create --title "Fix: issue description" --body "Closes #<issue-number>"
```

### Code Quality Standards
- **Be concise**: Refactor files exceeding ~500 lines into logical modules
- **Security**: Run `/security-audit` before merging PRs to master
- **Testing**: Add tests for new feature engineering functions (follow `test_T032.py` pattern)
- **Documentation**: Update relevant status docs when completing tasks

## Architecture

### Three-Layer Pipeline Architecture

**Layer 1: Core Implementation Modules** (`src/`)
- Individual functions that perform specific transformations
- Examples: `filter_to_county_level()`, `calculate_quarterly_growth_rates()`, `normalize_employment_data()`

**Layer 2: Pipeline Orchestrators** (`src/*_pipeline.py`)
- Coordinate multiple related functions into cohesive workflows
- `feature_pipeline.py`: Orchestrates feature engineering (T032-T042)
- `preprocessing_pipeline.py`: Orchestrates preprocessing (T054-T058)

**Layer 3: Master Orchestrator** (`src/pipeline_orchestrator.py`)
- `QCEWPipeline` class coordinates all pipeline stages
- Manages data flow between stages
- Handles configuration and state tracking

### Pipeline Data Flow

```
Raw CSV files (data/raw/*.csv)
    ↓
[Stage 1] consolidation.py
    → data/processed/qcew_master_consolidated.csv (5.4M records)
    ↓
[Stage 2] exploration.py
    → data/processed/plots/*.png (visualizations)
    ↓
[Stage 3] validation.py
    → data/validated/qcew_validated.csv
    → data/validated/validation_report.txt
    ↓
[Stage 4] feature_pipeline.py → feature_engineering.py
    → data/feature_engineering/T032_county_filtered.csv (4.7M records)
    → data/feature_engineering/T033_quarterly_only.csv
    → data/feature_engineering/T034_quality_filtered.csv
    → data/feature_engineering/final_features.csv
    ↓
[Stage 5] preprocessing_pipeline.py → preprocessing.py
    → data/processed/qcew_preprocessed.csv
    → data/processed/qcew_preprocessed_sequences.npz (LSTM sequences)
    ↓
[Stage 6] training.py + lstm_model.py
    → data/processed/lstm_model.pt
    ↓
[Stage 7] evaluation.py
    → data/processed/evaluation_results.json
    ↓
[Stage 8] prediction interface (future)
```

### Key Modules

**Data Pipeline**:
- `consolidation.py`: Merges raw CSV files, handles schema differences (2004-2019 vs 2020+)
- `exploration.py`: Statistical analysis and visualization
- `validation.py`: Quality checks, anomaly detection, employment/wage validation
- `feature_engineering.py`: Core feature functions (growth rates, lags, filtering)
- `feature_pipeline.py`: Feature engineering orchestrator
- `preprocessing.py`: `EmploymentDataPreprocessor` class for normalization, encoding, sequence transformation
- `preprocessing_pipeline.py`: Preprocessing orchestrator

**Model Components**:
- `lstm_model.py`: LSTM and RNN model architectures
  - `EmploymentLSTM`: Main forecasting model (2-layer LSTM with dropout, batch norm)
  - `EmploymentRNN`: Alternative RNN architecture
- `dataset.py`: PyTorch `EmploymentDataset` class, data loaders, augmentation
- `training.py`: `EmploymentTrainer` class with training loop, validation, checkpointing
- `loss_metrics.py`: Custom loss functions and evaluation metrics

**Orchestration**:
- `pipeline_orchestrator.py`: `QCEWPipeline` master coordinator
- `main.py`: Entry point with CLI and interactive menu support
- `interactive_menu.py`: User-friendly menu interface

## Important Implementation Details

### Feature Engineering (Stage 4)

Feature engineering functions in `feature_engineering.py` follow a consistent pattern:
- Accept DataFrame and optional `output_dir` parameter
- Log progress with specific task IDs (T032, T033, etc.)
- Save intermediate outputs to `data/feature_engineering/`
- Return transformed DataFrame for next stage

Example pattern:
```python
def feature_function(df: pd.DataFrame, output_dir: Path = None) -> pd.DataFrame:
    logger.info("T0XX: TASK DESCRIPTION")
    # Transformation logic
    if output_dir:
        output_file = output_dir / 'T0XX_description.csv'
        df.to_csv(output_file, index=False)
        logger.info(f"[OK] Saved to {output_file.name}")
    return df
```

### Preprocessing and Sequences (Stage 5)

The `EmploymentDataPreprocessor` class (`preprocessing.py`) transforms tabular data into LSTM-ready sequences:
- Groups by county + industry for time-series coherence
- Creates sliding windows of length 12 (default)
- Normalizes features using RobustScaler (T054)
- Handles missing values with forward fill + median imputation (T055)
- Encodes categorical features (T056)
- Validates preprocessing results (T058)

Output format (`.npz` file):
```python
{
    'sequences': np.ndarray,  # Shape: (num_samples, seq_len, num_features)
    'targets': np.ndarray,    # Shape: (num_samples,)
    'metadata': dict          # Feature names, scaler params, etc.
}
```

### Model Training (Stage 6)

The training loop (`training.py`) includes:
- Gradient clipping (max_norm=1.0) to prevent exploding gradients
- Early stopping with patience
- Model checkpointing (saves best model based on validation loss)
- Learning rate scheduling
- Comprehensive logging of metrics

Typical training invocation:
```python
from src.training import EmploymentTrainer
from src.lstm_model import EmploymentLSTM
from src.dataset import build_data_loader, EmploymentDataset

# Load sequences
data = np.load('data/processed/qcew_preprocessed_sequences.npz')
train_dataset = EmploymentDataset(data['sequences'], data['targets'])
train_loader = build_data_loader(train_dataset, batch_size=32)

# Initialize model and trainer
model = EmploymentLSTM(input_size=24, hidden_size=64, num_layers=2)
trainer = EmploymentTrainer(model, train_loader, val_loader)
trainer.train(num_epochs=50)
```

### Data Schema Notes

**QCEW Data Schema Changes**:
- 2004-2019: Uses 'AreaName', 'IndustryName'
- 2020+: Uses 'Area Name', 'Industry Name' (with spaces)
- Consolidation module handles both schemas automatically

**Key Columns**:
- `AreaType`: County-level = 'County', Statewide = 'Statewide'
- `Period`: Quarterly format is '1st Qtr', '2nd Qtr', '3rd Qtr', '4th Qtr' (NOT 'Q1', 'Q2', etc.)
- `AvgAnnualEmployment`: Primary target variable
- `AvgWeeklyWages`: Secondary feature
- `IndustryCode`: NAICS industry classification

**Quality Filters**:
- Employment values: 0 to 5,000,000
- Wage values: 0 to 5,000
- Records with negative values are flagged
- Missing critical columns cause validation warnings

## Project Structure

```
├── src/                          # Source code
│   ├── pipeline_orchestrator.py  # Master pipeline coordinator
│   ├── feature_pipeline.py       # Feature engineering orchestrator
│   ├── preprocessing_pipeline.py # Preprocessing orchestrator
│   ├── feature_engineering.py    # Feature engineering functions
│   ├── preprocessing.py          # Preprocessing class
│   ├── consolidation.py          # Data consolidation
│   ├── exploration.py            # EDA and visualization
│   ├── validation.py             # Data quality validation
│   ├── lstm_model.py             # LSTM/RNN architectures
│   ├── dataset.py                # PyTorch Dataset classes
│   ├── training.py               # Training infrastructure
│   ├── loss_metrics.py           # Custom loss functions
│   ├── evaluation.py             # Model evaluation
│   └── logging_config.py         # Logging configuration
│
├── data/                         # Data directories (gitignored except raw)
│   ├── raw/                      # Original QCEW CSV files
│   ├── processed/                # Consolidated and preprocessed data
│   ├── validated/                # Quality-checked data
│   └── feature_engineering/      # Feature engineering outputs
│
├── specs/001/                    # Feature specification documents
│   ├── tasks.md                  # Master task list with assignments
│   ├── spec.md                   # Feature requirements
│   └── plan.md                   # Implementation plan
│
├── logs/                         # Application logs
├── main.py                       # Entry point
├── interactive_menu.py           # Interactive menu interface
├── requirements.txt              # Python dependencies
│
└── Documentation (*.md files):
    ├── PROJECT_STATUS.md          # Current project status
    ├── PIPELINE_ARCHITECTURE.md   # Detailed architecture docs
    ├── QUICK_REFERENCE.md         # Quick reference guide
    └── FEATURE_ENGINEERING_STATUS.md
```

## Common Development Tasks

### Adding a New Feature Engineering Function

1. Implement function in `src/feature_engineering.py`:
```python
def new_feature_function(df: pd.DataFrame, output_dir: Path = None) -> pd.DataFrame:
    """T0XX: Description of transformation"""
    logger.info("T0XX: TRANSFORMATION NAME")
    # Implementation
    if output_dir:
        output_file = output_dir / 'T0XX_description.csv'
        df.to_csv(output_file, index=False)
        logger.info(f"[OK] Saved to {output_file.name}")
    return df
```

2. Update `src/feature_pipeline.py` to call the new function:
```python
def engineer_features(df, output_file, feature_eng_dir):
    # ... existing functions
    df = new_feature_function(df, output_dir=feature_eng_dir)
    # ... rest of pipeline
```

3. Create a test file following the pattern of `test_T032.py`

### Debugging Pipeline Issues

- Check logs in `logs/is160_project.log` and `logs/is160_project_errors.log`
- Examine intermediate outputs in `data/feature_engineering/` to identify which stage failed
- Use `--force-rebuild` flag if consolidation is causing issues
- Verify data schema matches expectations (check for 'Period' format, column names)

### Working with LSTM Models

When modifying the LSTM architecture:
- Input size must match number of features in preprocessed sequences
- Sequence length is typically 12 (3 years of quarterly data)
- Adjust `hidden_size` and `num_layers` in model initialization
- Remember to update model file paths in `pipeline_orchestrator.py`

## Known Patterns and Conventions

### Logging
- All modules use Python logging with consistent format
- Task IDs (T032, T033, etc.) prefixed to log messages for traceability
- Use `logger.info()` for progress, `logger.warning()` for issues, `logger.error()` for failures
- `[OK]` markers indicate successful completion

### Error Handling
- Pipeline stages continue even if non-critical errors occur
- Validation warnings don't stop pipeline execution
- Use try/except blocks to prevent stage failures from crashing entire pipeline

### File Naming
- Intermediate files named with task IDs: `T032_county_filtered.csv`
- Final outputs use descriptive names: `final_features.csv`, `qcew_preprocessed_sequences.npz`
- Models saved as `.pt` files (PyTorch format)

### Path Handling
- Use `pathlib.Path` for all file paths
- Base directory accessed via `Path(__file__).parent.parent` in orchestrators
- All paths constructed relative to base directory for portability

## Troubleshooting

**Issue**: "File not found" errors
- Solution: Run earlier pipeline stages first (consolidation must complete before exploration)
- Check that `.venv` is activated and requirements are installed

**Issue**: "Schema mismatch" or column errors
- Solution: Check if data uses old (2004-2019) or new (2020+) QCEW schema
- Consolidation should handle this automatically; may indicate corrupted CSV files

**Issue**: "Out of memory" during training
- Solution: Reduce `batch_size` in DataLoader creation
- Consider processing smaller subsets (e.g., filter to specific counties or industries)

**Issue**: Model trains but predictions are poor
- Solution: Verify preprocessing completed successfully (check T054-T058 implementations)
- Ensure features are properly normalized and sequences are correctly formed
- Check for data leakage between train/val/test splits
