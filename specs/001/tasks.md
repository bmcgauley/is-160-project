# Tas**Team Work **Total Team Members**: 3
- **Project Lead**: Overall coordination, setup, data exploration, validation, preprocessing, model architecture, visualization, a- [ ] T079 **[Alejo]** Document LSTM methodology for employment data analysis and prediction in docs/methodology.md
- [ ] T080 **[Alejo]** Create comprehensive results analysis with employment trend insights in docs/results.md
- [ ] T081 **[Alejo]** Build reproducible experiment scripts for QCEW data processing in scripts/
- [ ] T082 **[Alejo]** Generate academic-style report on LSTM applications to labor economics in docs/report.pdf
- [ ] T083 **[Alejo]** Validate all results are reproducible and methodology is clearly documented in docs/validation.mdnal documentation
- **Andrew**: Training infrastructure, loss functions, and evaluation
- **Alejo**: Feature engineering, and documentationents**

**Total Team Members**: 3
- **Project Lead**: Overall coordination, setup, data exploration, validation, preprocessing, model architecture, visualization, and final documentation
- **Andrew**: Training infrastructure, loss functions, and evaluation
- **Alejo**: Feature engineering, and documentation

### Workload Distribution
- **Project Lead**: 49 tasks (Setup, Data Exploration/Validation, Preprocessing/Model Architecture, Visualization, Unified Pipeline, Prediction Interface)
- **Andrew**: 20 tasks (Training Infrastructure, Loss Functions/Evaluation)
- **Alejo**: 20 tasks (Feature Engineering, Documentation) RNN/LSTM for Employment Forecasting with Interactive Visualizations

**Input**: Design documents from `/specs/001-build-a-convolutional/`
**Prerequisites**: spec.md (available), plan.md (not yet created), research.md, data-model.md, contracts/

## Team Work Segments

**Total Team Members**: 3
- **Project Lead**: [Your Name] - Overall coordination, setup, preprocessing, visualization, and final documentation
- **Andrew**: Data exploration, validation, training infrastructure, and baseline documentation
- **Alejo**: Feature engineering, model architecture, loss functions, and evaluation

### Workload Distribution
- **Project Lead**: 49 tasks (Setup, Data Exploration/Validation, Preprocessing/Model Architecture, Visualization, Unified Pipeline, Prediction Interface)
- **Andrew**: 20 tasks (Training Infrastructure, Loss Functions/Evaluation)
- **Alejo**: 20 tasks (Feature Engineering, Documentation)

### Phase Assignments
- **Phase 3.1 Setup**: Project Lead
- **Phase 3.2 Data Exploration**: Project Lead
- **Phase 3.3 Feature Engineering**: Alejo
- **Phase 3.4 Data Preprocessing/Model Architecture**: Project Lead
- **Phase 3.5 Training Infrastructure**: Andrew
- **Phase 3.6 Loss Functions/Evaluation**: Andrew
- **Phase 3.7 Visualization/Baselines**: Project Lead
- **Phase 3.8 Documentation**: Alejo

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → If not found: ERROR "No implementation plan found"
   → Extract: tech stack, libraries, structure
2. Load optional design documents:
   → data-model.md: Extract entities → model tasks
   → contracts/: Each file → contract test task
   → research.md: Extract decisions → setup tasks
3. Generate tasks by category:
   → Setup: project init, dependencies, linting
   → Tests: contract tests, integration tests
   → Core: models, services, CLI commands
   → Integration: DB, middleware, logging
   → Polish: unit tests, performance, docs
4. Apply task rules:
   → Different files = mark [P] for parallel
   → Same file = sequential (no [P])
   → Tests before implementation (TDD)
5. Number tasks sequentially (T001, T002...)
6. Generate dependency graph
7. Create parallel execution examples
8. Validate task completeness:
   → All contracts have tests?
   → All entities have models?
   → All endpoints implemented?
9. Return: SUCCESS (tasks ready for execution)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions
- **Single project**: `src/`, `tests/` at repository root
- **Web app**: `backend/src/`, `frontend/src/`
- **Mobile**: `api/src/`, `ios/src/` or `android/src/`
- Paths shown below assume single project - adjust based on plan.md structure

## Phase 3.1: Setup **[Project Lead]**
- [x] T001 **[Project Lead]** Implement automated data download script for QCEW CSV files from California Open Data Portal in src/data_download.py
- [x] T002 **[Project Lead]** Set up PyTorch environment with pandas, scikit-learn, matplotlib, seaborn in src/environment.py
- [x] T003 **[Project Lead]** Create data directory structure for raw, processed, and validated datasets in data/ directory
- [x] T004 **[Project Lead]** Implement automated data fetching/aggregation scripts with error handling and retries in src/data_acquisition.py
- [x] T005 **[Project Lead]** Set up logging system for tracking data processing steps and validation results in src/logging_config.py

## Phase 3.2: Data Exploration and Validation **[Project Lead]**
**Note**: This phase serves as the starting point for the entire workflow. The exploration script (notebooks/exploration.ipynb and src/exploration.py) must first trigger the initial data download if data is not present, ensuring the dataset is available before proceeding with analysis. All subsequent phases will be aggregated into a single unified file that runs the complete pipeline from data download to CNN training and evaluation with one click.

- [x] T006 **[Project Lead]** Load QCEW CSV files and examine data structure, columns, and data types in notebooks/exploration.ipynb and src/exploration.py (trigger data download if needed)
- [x] T007 **[Project Lead]** Merge all loaded QCEW CSV files into a single semi-processed dataset for unified analysis, ensuring raw data remains unmodified and creating a consolidated data object for feature engineering
- [x] T008 **[Project Lead]** Perform exploratory data analysis on employment counts, wages, and geographic coverage in notebooks/exploration.ipynb and src/exploration.py
- [x] T009 **[Project Lead]** Identify missing values, outliers, and data quality issues across quarters in notebooks/exploration.ipynb and src/exploration.py
- [x] T010 **[Project Lead]** Create summary statistics and visualizations for employment trends by industry/region in notebooks/exploration.ipynb and src/exploration.py
- [x] T011 **[Project Lead]** Document data schema and create data dictionary for employment variables in docs/data_dictionary.md
- [x] T012 **[Project Lead]** Create automated validation functions for employment count ranges and wage consistency in src/validation.py
- [x] T013 **[Project Lead]** Implement statistical tests for detecting anomalies in quarterly employment changes in src/validation.py
- [x] T014 **[Project Lead]** Build data quality scorecards for each geographic area and industry sector in src/validation.py
- [x] T015 **[Project Lead]** Validate temporal continuity and identify gaps in quarterly reporting in src/validation.py
- [x] T016 **[Project Lead]** Create validation reports with flagged records and quality metrics in src/validation.py

## Phase 3.3: Feature Engineering **[Alejo]**
- [ ] T017 **[Alejo]** Calculate quarter-over-quarter employment growth rates and percentage changes in src/feature_engineering.py
- [ ] T018 **[Alejo]** Create seasonal adjustment factors using historical employment patterns in src/feature_engineering.py
- [ ] T019 **[Alejo]** Engineer industry concentration metrics and economic diversity indices in src/feature_engineering.py
- [ ] T020 **[Alejo]** Build geographic clustering features based on employment similarity in src/feature_engineering.py
- [ ] T021 **[Alejo]** Generate lag features for temporal dependencies in employment trends in src/feature_engineering.py
- [ ] T022 **[Alejo]** Create rolling window statistics (3, 6, 12 quarter averages) for employment stability in src/temporal_features.py
- [ ] T023 **[Alejo]** Engineer cyclical features (quarter, year) and economic cycle indicators in src/temporal_features.py
- [ ] T024 **[Alejo]** Calculate employment volatility measures and trend strength indicators in src/temporal_features.py
- [ ] T025 **[Alejo]** Validate temporal features for consistency and economic reasonableness in src/temporal_features.py
- [ ] T026 **[Alejo]** Create time-based train/validation/test splits preserving temporal order in src/temporal_features.py
- [ ] T027 **[Alejo]** Create geographic feature maps for counties/regions with employment density in src/geographic_features.py
- [ ] T028 **[Alejo]** Engineer industry classification features and sector similarity matrices in src/geographic_features.py
- [ ] T029 **[Alejo]** Build regional economic indicators and metropolitan area classifications in src/geographic_features.py
- [ ] T030 **[Alejo]** Calculate spatial autocorrelation features for neighboring region employment in src/geographic_features.py
- [ ] T031 **[Alejo]** Validate geographic features against known economic geography patterns in src/geographic_features.py

- [ ] T093 **[Project Lead]** Set up feature engineering structure and initial files for team collaboration

## Phase 3.4: Data Preprocessing and Model Architecture **[Project Lead]**
- [ ] T032 **[Project Lead]** Normalize employment counts and wage data using robust scaling techniques in src/preprocessing.py
- [ ] T033 **[Project Lead]** Handle missing values with domain-appropriate imputation strategies in src/preprocessing.py
- [ ] T034 **[Project Lead]** Create categorical encodings for industry codes and geographic identifiers in src/preprocessing.py
- [ ] T035 **[Project Lead]** Transform tabular data into sequence format suitable for RNN/LSTM processing in src/preprocessing.py
- [ ] T036 **[Project Lead]** Validate preprocessing steps maintain data distribution properties in src/preprocessing.py
- [ ] T037 **[Project Lead]** Design LSTM layers for temporal employment sequence processing in src/lstm_model.py
- [ ] T038 **[Project Lead]** Implement RNN architecture for sequential employment pattern recognition in src/lstm_model.py
- [ ] T039 **[Project Lead]** Create custom LSTM architecture combining temporal dependencies and spatial features in src/lstm_model.py
- [ ] T040 **[Project Lead]** Add batch normalization and dropout layers appropriate for employment data in src/lstm_model.py
- [ ] T041 **[Project Lead]** Validate LSTM architecture dimensions match processed employment sequence shapes in src/lstm_model.py

- [ ] T094 **[Project Lead]** Set up data preprocessing and model architecture structure and initial files for team collaboration

## Phase 3.5: Training Infrastructure **[Andrew]**
- [ ] T042 **[Andrew]** Create PyTorch Dataset class for efficient QCEW data loading and batching in src/dataset.py
- [ ] T043 **[Andrew]** Implement data augmentation techniques appropriate for employment time series in src/dataset.py
- [ ] T044 **[Andrew]** Build DataLoader with proper batch sizes for employment tensor processing in src/dataset.py
- [ ] T045 **[Andrew]** Create train/validation data splits preserving temporal and geographic balance in src/dataset.py
- [ ] T046 **[Andrew]** Validate batch processing maintains employment data integrity and relationships in src/dataset.py
- [ ] T047 **[Andrew]** Implement training loop with employment-specific loss functions (MSE, MAE) in src/training.py
- [ ] T048 **[Andrew]** Create validation loop with employment forecasting accuracy metrics in src/training.py
- [ ] T049 **[Andrew]** Add model checkpointing for best employment prediction performance in src/training.py
- [ ] T050 **[Andrew]** Implement early stopping based on employment prediction validation loss in src/training.py
- [ ] T051 **[Andrew]** Build learning rate scheduling appropriate for employment data convergence in src/training.py

- [ ] T095 **[Project Lead]** Set up training infrastructure structure and initial files for team collaboration

## Phase 3.6: Loss Functions and Evaluation **[Andrew]**
- [ ] T052 **[Andrew]** Implement weighted loss functions emphasizing recent employment trends in src/loss_metrics.py
- [ ] T053 **[Andrew]** Create custom metrics for employment forecasting accuracy (MAPE, directional accuracy) in src/loss_metrics.py
- [ ] T054 **[Andrew]** Add employment volatility prediction loss for capturing uncertainty in src/loss_metrics.py
- [ ] T055 **[Andrew]** Build industry-weighted loss functions for sector-specific prediction importance in src/loss_metrics.py
- [ ] T056 **[Andrew]** Validate loss functions align with employment forecasting evaluation standards in src/loss_metrics.py
- [ ] T057 **[Andrew]** Calculate employment prediction accuracy across different time horizons in src/evaluation.py
- [ ] T058 **[Andrew]** Create confusion matrices for employment growth/decline classification in src/evaluation.py
- [ ] T059 **[Andrew]** Plot predicted vs actual employment trends by industry and region in src/evaluation.py
- [ ] T060 **[Andrew]** Generate employment volatility prediction accuracy assessments in src/evaluation.py
- [ ] T061 **[Andrew]** Validate model performance against employment forecasting benchmarks in src/evaluation.py

- [ ] T096 **[Project Lead]** Set up loss functions and evaluation structure and initial files for team collaboration

## Phase 3.7: Visualization and Comparison **[Project Lead]**
- [ ] T062 **[Project Lead]** Implement feature attribution techniques for employment factor importance in src/visualization.py
- [ ] T063 **[Project Lead]** Visualize LSTM learned patterns and their relationship to employment sequences in src/visualization.py
- [ ] T064 **[Project Lead]** Create employment trend visualizations showing model predictions vs reality in src/visualization.py
- [ ] T065 **[Project Lead]** Generate geographic heat maps of employment prediction accuracy in src/visualization.py
- [ ] T066 **[Project Lead]** Validate feature importance aligns with known employment economic factors in src/visualization.py
- [ ] T067 **[Project Lead]** Implement traditional employment forecasting models (ARIMA, exponential smoothing) in src/baselines.py
- [ ] T068 **[Project Lead]** Compare LSTM performance against econometric employment prediction models in src/baselines.py
- [ ] T069 **[Project Lead]** Create ensemble methods combining LSTM with traditional employment forecasting in src/baselines.py
- [ ] T070 **[Project Lead]** Benchmark computational efficiency for large-scale employment data processing in src/baselines.py
- [ ] T071 **[Project Lead]** Validate LSTM provides meaningful improvement over employment forecasting baselines in src/baselines.py
- [ ] T072 **[Project Lead]** Create visual predictions vs actuals plots showing predicted employment alongside actual values in src/prediction_visuals.py
- [ ] T073 **[Project Lead]** Implement multi-step ahead forecasts with 4-quarter predictions and uncertainty bands in src/forecasting.py
- [ ] T074 **[Project Lead]** Build industry risk dashboard displaying growth/decline status for each industry code in src/dashboard.py
- [ ] T075 **[Project Lead]** Develop county-level comparison visualizations for Central Valley counties employment growth vs decline in src/county_comparisons.py
- [ ] T076 **[Project Lead]** Create early warning system flagging industries predicted to lose >5% employment in next 2 quarters in src/early_warning.py
- [ ] T077 **[Project Lead]** Generate wage growth predictions showing industries with highest wage increases in src/wage_predictions.py
- [ ] T078 **[Project Lead]** Produce policy insights with actionable recommendations based on employment predictions in src/policy_insights.py

- [ ] T097 **[Project Lead]** Set up visualization and comparison structure and initial files for team collaboration

## Phase 3.8: Documentation and Reporting **[Alejo]**
- [ ] T079 **[Alejo]** Document LSTM methodology for employment data analysis and prediction in docs/methodology.md
- [ ] T073 **[Alejo]** Create comprehensive results analysis with employment trend insights in docs/results.md
- [ ] T074 **[Alejo]** Build reproducible experiment scripts for QCEW data processing in scripts/
- [ ] T075 **[Alejo]** Generate academic-style report on CNN applications to labor economics in docs/report.pdf
- [ ] T076 **[Alejo]** Validate all results are reproducible and methodology is clearly documented in docs/validation.md

- [ ] T098 **[Project Lead]** Set up documentation and reporting structure and initial files for team collaboration

### Unified Pipeline Development
To achieve the goal of a single-click execution, all components will be developed in separate files initially for modularity, then aggregated into one comprehensive script.

- [ ] T084 **[Project Lead]** Develop modular components in separate files (data download, exploration, feature engineering, preprocessing, model architecture, training, evaluation, visualization)
- [ ] T085 **[Project Lead]** Create integration functions to combine all modules into a single workflow
- [ ] T086 **[Project Lead]** Build a unified script (src/unified_pipeline.py) that executes the entire pipeline from data download to LSTM training and evaluation
- [ ] T087 **[Project Lead]** Add command-line interface and configuration options to the unified script for flexibility
- [ ] T088 **[Project Lead]** Test the unified script end-to-end and ensure it runs with one click
- [ ] T089 **[Project Lead]** Document the unified pipeline usage and deployment instructions
- [ ] T090 **[Project Lead]** Build interactive prediction interface allowing user input of future time and displaying forecasts with visualizations in src/prediction_interface.py
- [ ] T091 **[Project Lead]** Integrate maps, charts, graphs, and confidence bands into the prediction interface output
- [ ] T092 **[Project Lead]** Add uncertainty estimation and error bands to all prediction visualizations

## File Location Reference

### Core Source Files (src/)
**Data Acquisition & Setup:**
- `src/data_download.py` - Automated QCEW CSV download from California Open Data Portal (T001)
- `src/data_acquisition.py` - Data fetching/aggregation with error handling (T004)
- `src/environment.py` - PyTorch environment setup with dependencies (T002)
- `src/logging_config.py` - Logging system for data processing (T005)

**Data Exploration & Validation:**
- `src/exploration.py` - Data structure examination and EDA (T006-T010)
- `src/validation.py` - Data quality validation and statistical tests (T012-T016)
- `notebooks/exploration.ipynb` - Jupyter notebook version of exploration (T006-T010)

**Feature Engineering (Phase 3.3):**
- `src/feature_engineering.py` - Core feature calculations (T017-T021)
- `src/temporal_features.py` - Rolling statistics and cyclical features (T022-T026)
- `src/geographic_features.py` - Spatial features and industry classifications (T027-T031)

**Data Preprocessing & Model Architecture (Phase 3.4):**
- `src/preprocessing.py` - Data normalization, imputation, encoding, sequence transformation (T032-T036)
- `src/lstm_model.py` - LSTM/RNN architectures and validation (T037-T041)

**Training Infrastructure (Phase 3.5):**
- `src/dataset.py` - PyTorch Dataset/DataLoader classes (T042-T046)
- `src/training.py` - Training loops, validation, checkpointing (T047-T051)

**Loss Functions & Evaluation (Phase 3.6):**
- `src/loss_metrics.py` - Custom loss functions and metrics (T052-T056)
- `src/evaluation.py` - Model evaluation and baseline comparisons (T057-T061)

**Visualization & Comparison (Phase 3.7):**
- `src/visualization.py` - Feature attribution and LSTM pattern visualization (T062-T066)
- `src/baselines.py` - ARIMA and exponential smoothing models (T067-T071)
- `src/prediction_visuals.py` - Predictions vs actuals plots (T072)
- `src/forecasting.py` - Multi-step forecasting with uncertainty (T073)
- `src/dashboard.py` - Industry risk dashboards (T074)
- `src/county_comparisons.py` - Central Valley county comparisons (T075)
- `src/early_warning.py` - Early warning systems (T076)
- `src/wage_predictions.py` - Wage growth predictions (T077)
- `src/policy_insights.py` - Policy recommendations (T078)

**Unified Pipeline:**
- `src/unified_pipeline.py` - Single-click execution pipeline (T086)
- `src/prediction_interface.py` - Interactive prediction interface (T090-T092)

### Documentation Files (docs/)
- `docs/data_dictionary.md` - Employment variables documentation (T011)
- `docs/methodology.md` - LSTM methodology documentation (T079)
- `docs/results.md` - Comprehensive results analysis (T080)
- `docs/report.pdf` - Academic-style final report (T082)
- `docs/validation.md` - Reproducibility and validation documentation (T083)

### Scripts Directory (scripts/)
- `scripts/` - Directory for reproducible experiment scripts (T081)

### Data Directories
- `data/raw/` - Raw QCEW CSV files
- `data/processed/` - Consolidated and cleaned datasets
- `data/validated/` - Quality-checked datasets

## Setup Tasks Status

The following setup tasks have been completed to establish project structure:

- [x] T093 **[Alejo]** Set up feature engineering structure and initial files for team collaboration
- [x] T094 **[Project Lead]** Set up data preprocessing and model architecture structure and initial files for team collaboration
- [x] T095 **[Project Lead]** Set up training infrastructure structure and initial files for team collaboration
- [x] T096 **[Project Lead]** Set up loss functions and evaluation structure and initial files for team collaboration
- [x] T097 **[Project Lead]** Set up visualization and comparison structure and initial files for team collaboration
- [x] T098 **[Project Lead]** Set up documentation and reporting structure and initial files for team collaboration

## Issues and Resolutions

### Issue 1: Jupyter Notebook Conversion to Python Scripts
The specs documentation mentions performing tasks in .ipynb files, which is acceptable for interactive development. However, to ensure redundancy and compatibility, each Jupyter notebook should be converted to a corresponding Python script as part of the workflow. This allows for easier integration into automated pipelines and provides alternative execution methods.

- [x] T077 **[Project Lead]** Convert notebooks/exploration.ipynb to src/exploration.py with equivalent functionality
- [ ] T078 **[Project Lead]** Ensure all notebook-based tasks include conversion steps to Python scripts
- [ ] T079 **[Project Lead]** Update task descriptions to reference both notebook and script versions where applicable

### Issue 2: Preventing CSV Data Files from Being Pushed to GitHub
Despite .gitignore configurations, CSV data files in the data/ directory are being pushed to GitHub. Since each team member downloads their own data using the scripts, these large files should not be version-controlled. Need to properly configure .gitignore and potentially remove already committed files.

- [x] T080 **[Andrew]** Update .gitignore to exclude all .csv files in data/ subdirectories (data/raw/*.csv, data/processed/*.csv, data/validated/*.csv)
- [x] T081 **[Andrew]** Check for and remove any already committed CSV files from the repository
- [x] T082 **[Andrew]** Verify that data download scripts still function correctly without committed CSV files

### Issue 3: Limited Data Download Range
The current data download script only retrieves QCEW data from 2020 Q1 to 2024 Q4, despite the California Open Data Portal having data available from 2004-2025. This significantly limits the historical context available for time series modeling and forecasting.

- [x] T099 **[Project Lead]** Investigate why data download is limited to 2020-2024 and identify methods to access full historical dataset (2004-2025) from California Open Data Portal - COMPLETED: Successfully switched data source to California Open Data Portal, implemented format conversion, and expanded dataset from 2020-2024 (64K rows) to 2004-2024 (243K rows)

## Parallel Execution Examples
Tasks that can run in parallel (marked [P]) are limited in this sequential workflow, but some setup tasks can be parallelized. Team members can work on their assigned phases simultaneously where dependencies allow.

**Project Lead** can work on:
- T001, T002, T003, T004, T005 (setup tasks)
- T006-T016 (data exploration/validation) after setup
- T032-T036 (preprocessing) after Phase 3.3 complete
- T037-T041 (model architecture) after preprocessing
- T062-T071 (visualization/baselines) after Phase 3.6 complete
- T072-T076 (documentation) - can start early

**Andrew** can work on:
- T042-T046 (dataset) after Phase 3.4 complete
- T047-T051 (training) after dataset development
- T052-T061 (loss/metrics and evaluation) after training

**Alejo** can work on:
- T017-T031 (feature engineering) after Phase 3.2 complete
- T072-T076 (documentation) after Phase 3.7 complete

## Dependency Graph
All tasks follow a strict sequential dependency:
Setup (T001-T005) → Data Exploration (T006-T016) → Feature Engineering (T017-T031) → Preprocessing (T032-T036) → Architecture (T037-T041) → Dataset (T042-T046) → Training (T047-T051) → Loss/Metrics (T052-T056) → Evaluation (T057-T061) → Visualization (T062-T066) → Baselines (T067-T071) → Documentation (T072-T076)

No tasks can be executed out of this order due to data dependencies and iterative refinement requirements.