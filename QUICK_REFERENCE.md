# Quick Reference: Working with the Pipeline

## Pipeline Module Structure (Corrected)

### ✅ What We Have Now

```
src/
├── pipeline_orchestrator.py      # Master coordinator
├── feature_pipeline.py           # NEW: Feature engineering orchestrator
├── preprocessing_pipeline.py     # NEW: Preprocessing orchestrator
├── feature_engineering.py        # Core feature functions (T032-T042)
├── preprocessing.py              # Core preprocessing functions (T054-T058)
├── consolidation.py              # Data consolidation
├── exploration.py                # Data exploration
└── validation.py                 # Data validation
```

### ✅ How They Work Together

```python
# main.py calls:
pipeline_orchestrator.QCEWPipeline()
    ├─> stage_4_feature_engineering()
    │    └─> feature_pipeline.engineer_features()
    │         └─> feature_engineering.filter_to_county_level()  # T032 ✅
    │         └─> feature_engineering.other_functions()         # T033+ ⏳
    │
    └─> stage_5_preprocessing()
         └─> preprocessing_pipeline.preprocess_for_lstm()
              └─> preprocessing.EmploymentDataPreprocessor()
```

## Running the Pipeline

### Option 1: Run specific stage
```bash
python main.py --stage consolidate
python main.py --stage explore
python main.py --stage validate
python main.py --stage feature_engineering  # Runs T032 only (for now)
```

### Option 2: Test individual tasks
```bash
python test_T032.py  # Test T032 specifically
```

### Option 3: Import and use directly
```python
from src.feature_engineering import filter_to_county_level
from pathlib import Path

# Load data
df = pd.read_csv('data/processed/qcew_master_consolidated.csv')

# Run T032
output_dir = Path('data/feature_engineering')
county_df = filter_to_county_level(df, output_dir=output_dir)
```

## Adding New Feature Engineering Tasks

### Step 1: Implement function in feature_engineering.py
```python
def handle_annual_vs_quarterly(df: pd.DataFrame, output_dir: Path = None) -> pd.DataFrame:
    """T033: Handle Annual vs quarterly records"""
    logger.info("T033: HANDLING ANNUAL VS QUARTERLY RECORDS")
    
    # Your implementation here
    
    # Save intermediate output
    if output_dir:
        output_file = output_dir / 'T033_quarterly_filtered.csv'
        df.to_csv(output_file, index=False)
        logger.info(f"[OK] Saved to {output_file.name}")
    
    return df
```

### Step 2: Add to feature_pipeline.py
```python
def engineer_features(df, output_file, feature_eng_dir):
    # T032
    df = filter_to_county_level(df, output_dir=feature_eng_dir)
    
    # T033 - NEW
    from feature_engineering import handle_annual_vs_quarterly
    df = handle_annual_vs_quarterly(df, output_dir=feature_eng_dir)
    
    # ... rest of pipeline
```

### Step 3: Create test file
```python
# test_T033.py
from src.feature_engineering import handle_annual_vs_quarterly

def main():
    # Load previous stage output
    df = pd.read_csv('data/feature_engineering/T032_county_filtered.csv')
    
    # Run T033
    result = handle_annual_vs_quarterly(df, output_dir=feature_eng_dir)
    
    # Validate results
    # ... your checks here
```

## Data Flow Visualization

```
RAW DATA
  ↓
data/raw/*.csv
  ↓
[CONSOLIDATE] consolidation.py
  ↓
data/processed/qcew_master_consolidated.csv (5.4M records)
  ↓
[VALIDATE] validation.py
  ↓
data/validated/qcew_validated.csv
  ↓
[FEATURE ENGINEERING] feature_pipeline.py
  ├─> T032: filter_to_county_level()
  │    → data/feature_engineering/T032_county_filtered.csv (4.7M records) ✅
  │
  ├─> T033: handle_annual_vs_quarterly()
  │    → data/feature_engineering/T033_quarterly_filtered.csv ⏳
  │
  ├─> T034: data_quality_filter()
  │    → data/feature_engineering/T034_quality_filtered.csv ⏳
  │
  ... (T035-T042)
  │
  └─> final_features.csv
       ↓
[PREPROCESSING] preprocessing_pipeline.py
  ├─> T054: normalize_employment_data()
  ├─> T055: handle_missing_values()
  ├─> T056: create_categorical_encodings()
  ├─> T057: transform_to_sequences()
  └─> T058: validate_preprocessing()
       ↓
data/processed/qcew_preprocessed.csv
data/processed/qcew_preprocessed_sequences.npz
  ↓
[TRAINING] (future)
  ↓
[EVALUATION] (future)
  ↓
[PREDICTION] (future)
```

## Important Files

### Configuration
- `main.py` - Entry point
- `src/pipeline_orchestrator.py` - Master coordinator

### Feature Engineering (Current Work)
- `src/feature_engineering.py` - Implementation
- `src/feature_pipeline.py` - Orchestration
- `test_T032.py` - Testing T032
- `data/feature_engineering/` - Outputs

### Documentation
- `T032_IMPLEMENTATION_SUMMARY.md` - What we did
- `PIPELINE_ARCHITECTURE.md` - How it's organized
- `CLEAN_FORMAT_STANDARDS.md` - Unicode fixes
- `data/feature_engineering/README.md` - Directory guide

## Current Status

✅ **Complete**:
- T001-T031: Setup, exploration, validation, consolidation
- T032: County-level filtering

⏳ **Next Up**:
- T033: Handle Annual vs quarterly records
- T034: Data quality filtering
- T035: Central Valley counties subset
- T036: All California counties with features
- T038-T042: Feature calculations

## Getting Help

- Check `PIPELINE_ARCHITECTURE.md` for module structure
- Check `T032_IMPLEMENTATION_SUMMARY.md` for implementation example
- Check `test_T032.py` for testing pattern
- All modules have comprehensive docstrings
