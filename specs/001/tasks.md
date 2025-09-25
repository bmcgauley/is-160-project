# Tasks: Build a Convolutional Neural Network for Employment Trends Analysis

**Input**: Design documents from `/specs/001-build-a-convolutional/`
**Prerequisites**: spec.md (available), plan.md (not yet created), research.md, data-model.md, contracts/

## Team Work Segments

**Total Team Members**: 3
- **Project Lead**: [Your Name] - Overall coordination, setup, preprocessing, visualization, and final documentation
- **Andrew**: Data exploration, validation, training infrastructure, and baseline documentation
- **Alejo**: Feature engineering, model architecture, loss functions, and evaluation

### Workload Distribution
- **Project Lead**: 25 tasks (Setup, Preprocessing/Model Architecture, Visualization, Documentation)
- **Andrew**: 25 tasks (Data Exploration, Training Infrastructure, Baseline Comparison)
- **Alejo**: 25 tasks (Feature Engineering, Loss Functions/Evaluation)

### Phase Assignments
- **Phase 3.1 Setup**: Project Lead
- **Phase 3.2 Data Exploration**: Andrew
- **Phase 3.3 Feature Engineering**: Alejo
- **Phase 3.4 Data Preprocessing/Model Architecture**: Project Lead
- **Phase 3.5 Training Infrastructure**: Andrew
- **Phase 3.6 Loss Functions/Evaluation**: Alejo
- **Phase 3.7 Visualization/Baselines**: Project Lead
- **Phase 3.8 Documentation**: Andrew

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

## Phase 3.2: Data Exploration and Validation **[Andrew]**
**Note**: This phase serves as the starting point for the entire workflow. The exploration script (notebooks/exploration.ipynb and src/exploration.py) must first trigger the initial data download if data is not present, ensuring the dataset is available before proceeding with analysis. All subsequent phases will be aggregated into a single unified file that runs the complete pipeline from data download to CNN training and evaluation with one click.

- [x] T006 **[Andrew]** Load QCEW CSV files and examine data structure, columns, and data types in notebooks/exploration.ipynb and src/exploration.py (trigger data download if needed)
- [ ] T007 **[Andrew]** Perform exploratory data analysis on employment counts, wages, and geographic coverage in notebooks/exploration.ipynb and src/exploration.py
- [ ] T008 **[Andrew]** Identify missing values, outliers, and data quality issues across quarters in notebooks/exploration.ipynb and src/exploration.py
- [ ] T009 **[Andrew]** Create summary statistics and visualizations for employment trends by industry/region in notebooks/exploration.ipynb and src/exploration.py
- [ ] T010 **[Andrew]** Document data schema and create data dictionary for employment variables in docs/data_dictionary.md
- [ ] T011 **[Andrew]** Create automated validation functions for employment count ranges and wage consistency in src/validation.py
- [ ] T012 **[Andrew]** Implement statistical tests for detecting anomalies in quarterly employment changes in src/validation.py
- [ ] T013 **[Andrew]** Build data quality scorecards for each geographic area and industry sector in src/validation.py
- [ ] T014 **[Andrew]** Validate temporal continuity and identify gaps in quarterly reporting in src/validation.py
- [ ] T015 **[Andrew]** Create validation reports with flagged records and quality metrics in src/validation.py

## Phase 3.3: Feature Engineering **[Alejo]**
- [ ] T016 **[Alejo]** Calculate quarter-over-quarter employment growth rates and percentage changes in src/feature_engineering.py
- [ ] T017 **[Alejo]** Create seasonal adjustment factors using historical employment patterns in src/feature_engineering.py
- [ ] T018 **[Alejo]** Engineer industry concentration metrics and economic diversity indices in src/feature_engineering.py
- [ ] T019 **[Alejo]** Build geographic clustering features based on employment similarity in src/feature_engineering.py
- [ ] T020 **[Alejo]** Generate lag features for temporal dependencies in employment trends in src/feature_engineering.py
- [ ] T021 **[Alejo]** Create rolling window statistics (3, 6, 12 quarter averages) for employment stability in src/temporal_features.py
- [ ] T022 **[Alejo]** Engineer cyclical features (quarter, year) and economic cycle indicators in src/temporal_features.py
- [ ] T023 **[Alejo]** Calculate employment volatility measures and trend strength indicators in src/temporal_features.py
- [ ] T024 **[Alejo]** Validate temporal features for consistency and economic reasonableness in src/temporal_features.py
- [ ] T025 **[Alejo]** Create time-based train/validation/test splits preserving temporal order in src/temporal_features.py
- [ ] T026 **[Alejo]** Create geographic feature maps for counties/regions with employment density in src/geographic_features.py
- [ ] T027 **[Alejo]** Engineer industry classification features and sector similarity matrices in src/geographic_features.py
- [ ] T028 **[Alejo]** Build regional economic indicators and metropolitan area classifications in src/geographic_features.py
- [ ] T029 **[Alejo]** Calculate spatial autocorrelation features for neighboring region employment in src/geographic_features.py
- [ ] T030 **[Alejo]** Validate geographic features against known economic geography patterns in src/geographic_features.py

## Phase 3.4: Data Preprocessing and Model Architecture **[Project Lead]**
- [ ] T031 **[Project Lead]** Normalize employment counts and wage data using robust scaling techniques in src/preprocessing.py
- [ ] T032 **[Project Lead]** Handle missing values with domain-appropriate imputation strategies in src/preprocessing.py
- [ ] T033 **[Project Lead]** Create categorical encodings for industry codes and geographic identifiers in src/preprocessing.py
- [ ] T034 **[Project Lead]** Transform tabular data into tensor format suitable for CNN processing in src/preprocessing.py
- [ ] T035 **[Project Lead]** Validate preprocessing steps maintain data distribution properties in src/preprocessing.py
- [ ] T036 **[Project Lead]** Design 1D CNN layers for temporal employment sequence processing in src/cnn_model.py
- [ ] T037 **[Project Lead]** Implement 2D CNN layers for geographic-temporal employment pattern recognition in src/cnn_model.py
- [ ] T038 **[Project Lead]** Create custom CNN architecture combining temporal and spatial convolutions in src/cnn_model.py
- [ ] T039 **[Project Lead]** Add batch normalization and dropout layers appropriate for employment data in src/cnn_model.py
- [ ] T040 **[Project Lead]** Validate CNN architecture dimensions match processed employment tensor shapes in src/cnn_model.py

## Phase 3.5: Training Infrastructure **[Andrew]**
- [ ] T041 **[Andrew]** Create PyTorch Dataset class for efficient QCEW data loading and batching in src/dataset.py
- [ ] T042 **[Andrew]** Implement data augmentation techniques appropriate for employment time series in src/dataset.py
- [ ] T043 **[Andrew]** Build DataLoader with proper batch sizes for employment tensor processing in src/dataset.py
- [ ] T044 **[Andrew]** Create train/validation data splits preserving temporal and geographic balance in src/dataset.py
- [ ] T045 **[Andrew]** Validate batch processing maintains employment data integrity and relationships in src/dataset.py
- [ ] T046 **[Andrew]** Implement training loop with employment-specific loss functions (MSE, MAE) in src/training.py
- [ ] T047 **[Andrew]** Create validation loop with employment forecasting accuracy metrics in src/training.py
- [ ] T048 **[Andrew]** Add model checkpointing for best employment prediction performance in src/training.py
- [ ] T049 **[Andrew]** Implement early stopping based on employment prediction validation loss in src/training.py
- [ ] T050 **[Andrew]** Build learning rate scheduling appropriate for employment data convergence in src/training.py

## Phase 3.6: Loss Functions and Evaluation **[Alejo]**
- [ ] T051 **[Alejo]** Implement weighted loss functions emphasizing recent employment trends in src/loss_metrics.py
- [ ] T052 **[Alejo]** Create custom metrics for employment forecasting accuracy (MAPE, directional accuracy) in src/loss_metrics.py
- [ ] T053 **[Alejo]** Add employment volatility prediction loss for capturing uncertainty in src/loss_metrics.py
- [ ] T054 **[Alejo]** Build industry-weighted loss functions for sector-specific prediction importance in src/loss_metrics.py
- [ ] T055 **[Alejo]** Validate loss functions align with employment forecasting evaluation standards in src/loss_metrics.py
- [ ] T056 **[Alejo]** Calculate employment prediction accuracy across different time horizons in src/evaluation.py
- [ ] T057 **[Alejo]** Create confusion matrices for employment growth/decline classification in src/evaluation.py
- [ ] T058 **[Alejo]** Plot predicted vs actual employment trends by industry and region in src/evaluation.py
- [ ] T059 **[Alejo]** Generate employment volatility prediction accuracy assessments in src/evaluation.py
- [ ] T060 **[Alejo]** Validate model performance against employment forecasting benchmarks in src/evaluation.py

## Phase 3.7: Visualization and Comparison **[Project Lead]**
- [ ] T061 **[Project Lead]** Implement feature attribution techniques for employment factor importance in src/visualization.py
- [ ] T062 **[Project Lead]** Visualize CNN learned filters and their relationship to employment patterns in src/visualization.py
- [ ] T063 **[Project Lead]** Create employment trend visualizations showing model predictions vs reality in src/visualization.py
- [ ] T064 **[Project Lead]** Generate geographic heat maps of employment prediction accuracy in src/visualization.py
- [ ] T065 **[Project Lead]** Validate feature importance aligns with known employment economic factors in src/visualization.py
- [ ] T066 **[Project Lead]** Implement traditional employment forecasting models (ARIMA, exponential smoothing) in src/baselines.py
- [ ] T067 **[Project Lead]** Compare CNN performance against econometric employment prediction models in src/baselines.py
- [ ] T068 **[Project Lead]** Create ensemble methods combining CNN with traditional employment forecasting in src/baselines.py
- [ ] T069 **[Project Lead]** Benchmark computational efficiency for large-scale employment data processing in src/baselines.py
- [ ] T070 **[Project Lead]** Validate CNN provides meaningful improvement over employment forecasting baselines in src/baselines.py

## Phase 3.8: Documentation and Reporting **[Andrew]**
- [ ] T071 **[Andrew]** Document CNN methodology for employment data analysis and prediction in docs/methodology.md
- [ ] T072 **[Andrew]** Create comprehensive results analysis with employment trend insights in docs/results.md
- [ ] T073 **[Andrew]** Build reproducible experiment scripts for QCEW data processing in scripts/
- [ ] T074 **[Andrew]** Generate academic-style report on CNN applications to labor economics in docs/report.pdf
- [ ] T075 **[Andrew]** Validate all results are reproducible and methodology is clearly documented in docs/validation.md

### Unified Pipeline Development
To achieve the goal of a single-click execution, all components will be developed in separate files initially for modularity, then aggregated into one comprehensive script.

- [ ] T082 **[Project Lead]** Develop modular components in separate files (data download, exploration, feature engineering, preprocessing, model architecture, training, evaluation, visualization)
- [ ] T083 **[Project Lead]** Create integration functions to combine all modules into a single workflow
- [ ] T084 **[Project Lead]** Build a unified script (src/unified_pipeline.py) that executes the entire pipeline from data download to CNN training and evaluation
- [ ] T085 **[Project Lead]** Add command-line interface and configuration options to the unified script for flexibility
- [ ] T086 **[Project Lead]** Test the unified script end-to-end and ensure it runs with one click
- [ ] T087 **[Project Lead]** Document the unified pipeline usage and deployment instructions

## Issues and Resolutions

### Issue 1: Jupyter Notebook Conversion to Python Scripts
The specs documentation mentions performing tasks in .ipynb files, which is acceptable for interactive development. However, to ensure redundancy and compatibility, each Jupyter notebook should be converted to a corresponding Python script as part of the workflow. This allows for easier integration into automated pipelines and provides alternative execution methods.

- [x] T076 **[Project Lead]** Convert notebooks/exploration.ipynb to src/exploration.py with equivalent functionality
- [ ] T077 **[Project Lead]** Ensure all notebook-based tasks include conversion steps to Python scripts
- [ ] T078 **[Project Lead]** Update task descriptions to reference both notebook and script versions where applicable

### Issue 2: Preventing CSV Data Files from Being Pushed to GitHub
Despite .gitignore configurations, CSV data files in the data/ directory are being pushed to GitHub. Since each team member downloads their own data using the scripts, these large files should not be version-controlled. Need to properly configure .gitignore and potentially remove already committed files.

- [x] T079 **[Andrew]** Update .gitignore to exclude all .csv files in data/ subdirectories (data/raw/*.csv, data/processed/*.csv, data/validated/*.csv)
- [x] T080 **[Andrew]** Check for and remove any already committed CSV files from the repository
- [x] T081 **[Andrew]** Verify that data download scripts still function correctly without committed CSV files

## Parallel Execution Examples
Tasks that can run in parallel (marked [P]) are limited in this sequential workflow, but some setup tasks can be parallelized. Team members can work on their assigned phases simultaneously where dependencies allow.

**Project Lead** can work on:
- T001, T002, T003, T004, T005 (setup tasks)
- T031-T035 (preprocessing) after Phase 3.3 complete
- T036-T040 (model architecture) after preprocessing
- T061-T070 (visualization/baselines) after Phase 3.6 complete
- T071-T075 (documentation) - can start early

**Andrew** can work on:
- T006-T010 (exploration) after setup
- T011-T015 (validation) in parallel with exploration
- T041-T045 (dataset) after Phase 3.4 complete
- T046-T050 (training) after dataset development

**Alejo** can work on:
- T016-T030 (feature engineering) after Phase 3.2 complete
- T051-T055 (loss/metrics) after Phase 3.5 complete
- T056-T060 (evaluation) after loss/metrics development

## Dependency Graph
All tasks follow a strict sequential dependency:
Setup (T001-T005) → Data Exploration (T006-T010) → Validation (T011-T015) → Feature Engineering (T016-T030) → Preprocessing (T031-T035) → Architecture (T036-T040) → Dataset (T041-T045) → Training (T046-T050) → Loss/Metrics (T051-T055) → Evaluation (T056-T060) → Visualization (T061-T065) → Baselines (T066-T070) → Documentation (T071-T075)

No tasks can be executed out of this order due to data dependencies and iterative refinement requirements.