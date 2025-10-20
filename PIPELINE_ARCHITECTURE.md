# Pipeline Module Architecture

**Date**: October 20, 2025  
**Status**: Corrected and properly organized

## Module Structure

The pipeline is organized into layers for modularity and maintainability:

### Layer 1: Core Implementation Modules
These contain the actual implementation of individual functions:

```
src/
├── feature_engineering.py       # Individual feature engineering functions
│   ├── filter_to_county_level()         # T032
│   ├── calculate_quarterly_growth_rates() # T038
│   ├── create_seasonal_adjustments()    # T039
│   ├── calculate_industry_concentration() # T040
│   ├── build_geographic_clustering()    # T041
│   └── generate_lag_features()          # T042
│
├── temporal_features.py         # Time-based features
│   └── (Future: Rolling stats, cyclical features)
│
├── geographic_features.py       # Spatial features
│   └── (Future: Geographic clustering, regional indicators)
│
├── preprocessing.py             # Data preprocessing functions
│   └── EmploymentDataPreprocessor class
│       ├── normalize_employment_data()  # T054
│       ├── handle_missing_values()      # T055
│       ├── create_categorical_encodings() # T056
│       ├── transform_to_sequences()     # T057
│       └── validate_preprocessing()     # T058
│
├── consolidation.py             # Data consolidation
│   └── consolidate_raw_data()
│
├── exploration.py               # Data exploration
│   └── explore_qcew_data()
│
└── validation.py                # Data validation
    └── validate_data_quality()
```

### Layer 2: Pipeline Orchestration Modules
These orchestrate multiple functions into cohesive workflows:

```
src/
├── feature_pipeline.py          # Feature engineering pipeline
│   └── engineer_features()
│       ├── Calls filter_to_county_level()
│       ├── Calls calculate_quarterly_growth_rates()
│       ├── Calls create_seasonal_adjustments()
│       ├── Saves intermediate outputs
│       └── Returns feature-engineered DataFrame
│
├── preprocessing_pipeline.py    # Preprocessing pipeline
│   └── preprocess_for_lstm()
│       ├── Normalizes data (T054)
│       ├── Handles missing values (T055)
│       ├── Encodes categorical data (T056)
│       ├── Transforms to sequences (T057)
│       ├── Validates preprocessing (T058)
│       └── Returns (X_tensor, y_tensor, preprocessor)
│
└── (Future pipeline modules)
    ├── training_pipeline.py     # Model training orchestration
    ├── evaluation_pipeline.py   # Model evaluation orchestration
    └── prediction_interface.py  # Interactive prediction UI
```

### Layer 3: Master Pipeline Orchestrator

```
src/
└── pipeline_orchestrator.py     # Top-level pipeline coordinator
    └── QCEWPipeline class
        ├── stage_1_consolidate()     → consolidation.py
        ├── stage_2_explore()         → exploration.py
        ├── stage_3_validate()        → validation.py
        ├── stage_4_feature_engineering() → feature_pipeline.py
        ├── stage_5_preprocessing()   → preprocessing_pipeline.py
        ├── stage_6_train_model()     → training_pipeline.py (future)
        ├── stage_7_evaluate_model()  → evaluation_pipeline.py (future)
        └── stage_8_prediction_interface() → prediction_interface.py (future)
```

## Import Flow

```
main.py
  └─> pipeline_orchestrator.py (QCEWPipeline)
       ├─> consolidation.py
       ├─> exploration.py
       ├─> validation.py
       ├─> feature_pipeline.py
       │    └─> feature_engineering.py
       │    └─> temporal_features.py
       │    └─> geographic_features.py
       │
       ├─> preprocessing_pipeline.py
       │    └─> preprocessing.py
       │
       └─> (Future pipelines...)
```

## Data Flow Through Pipeline

```
1. Consolidation (consolidation.py)
   data/raw/*.csv → data/processed/qcew_master_consolidated.csv
   
2. Exploration (exploration.py)
   qcew_master_consolidated.csv → data/processed/plots/*.png
   
3. Validation (validation.py)
   qcew_master_consolidated.csv → data/validated/qcew_validated.csv
   
4. Feature Engineering (feature_pipeline.py)
   qcew_validated.csv → data/feature_engineering/
   ├── T032_county_filtered.csv
   ├── T033_quarterly_filtered.csv
   ├── T034_quality_filtered.csv
   ├── ...
   └── final_features.csv
   
5. Preprocessing (preprocessing_pipeline.py)
   final_features.csv → data/processed/
   ├── qcew_preprocessed.csv
   └── qcew_preprocessed_sequences.npz
   
6. Training (training_pipeline.py - future)
   qcew_preprocessed_sequences.npz → data/processed/lstm_model.pt
   
7. Evaluation (evaluation_pipeline.py - future)
   lstm_model.pt + test_data → data/processed/evaluation_results.json
   
8. Prediction (prediction_interface.py - future)
   lstm_model.pt → Interactive UI
```

## Benefits of This Architecture

1. **Modularity**: Each layer has clear responsibilities
2. **Reusability**: Core functions can be called independently
3. **Testability**: Each module can be tested in isolation
4. **Maintainability**: Changes to one module don't cascade
5. **Clarity**: Import structure is easy to understand
6. **Scalability**: New features added without disrupting existing code

## Current Implementation Status

✅ **Implemented**:
- Layer 1: Core modules (consolidation, exploration, validation, feature_engineering, preprocessing)
- Layer 2: feature_pipeline.py, preprocessing_pipeline.py
- Layer 3: pipeline_orchestrator.py

⏳ **Partially Implemented**:
- feature_pipeline.py: Only T032 implemented, T033-T042 pending
- preprocessing.py: Structure exists, implementation incomplete (T054-T058)

❌ **Not Yet Implemented**:
- training_pipeline.py (T065-T074)
- evaluation_pipeline.py (T076-T085)
- prediction_interface.py (T117-T119)

## Testing Each Layer

### Test Layer 1 (Core Functions)
```bash
python test_T032.py  # Tests filter_to_county_level()
```

### Test Layer 2 (Pipeline Modules)
```bash
python -c "from src.feature_pipeline import engineer_features; print('OK')"
python -c "from src.preprocessing_pipeline import preprocess_for_lstm; print('OK')"
```

### Test Layer 3 (Master Orchestrator)
```bash
python main.py --stage feature_engineering
python main.py --stage preprocessing
```

## Next Steps

As we implement T033 and beyond:
1. Add functions to `feature_engineering.py`
2. Update `feature_pipeline.py` to call new functions
3. Test each function individually
4. Test full pipeline integration
5. Update this documentation
