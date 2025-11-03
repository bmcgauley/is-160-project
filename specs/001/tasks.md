# Task Management for IS-160 Project

> **⚠️ IMPORTANT**: As of November 2, 2025, this repository requires pull requests for ALL changes, including this file. When updating tasks.md, you must:
> 1. Create a branch: `git checkout -b docs/update-tasks-md`
> 2. Make your changes to this file
> 3. Commit and push: `git commit -m "docs: update tasks.md" && git push -u origin docs/update-tasks-md`
> 4. Create PR: `gh pr create --title "Update tasks.md" --body "Task status updates"`
> 5. After approval, merge the PR
>
> See [Repository Protection & PR Workflow](#-repository-protection--pr-workflow-effective-november-2-2025) section below for full details.

## Progress Summary (Updated: November 2, 2025)

### Current Status: **Phase 3.5 - Training Infrastructure (Implementation Complete, Debugging Issues)**

**📋 PROJECT STATUS NOTE:**
Work through Phase 3.6 (T001-T080) has been completed by Brian (Project Lead). Due to scheduling challenges and team availability, task responsibilities were adjusted during implementation. Original assignments are preserved below for historical documentation purposes.

**Completed Phases:**
- ✅ Phase 3.1: Setup (T001-T005) - 5/5 tasks complete
- ✅ Phase 3.2: Data Exploration & Validation (T006-T031) - 26/26 tasks complete
- ✅ Phase 3.3: Feature Engineering (T032-T042) - **5/11 core tasks complete** (T039-T041 deferred as advanced features)
- ✅ Phase 3.4: Data Preprocessing & Model Architecture (T054-T063) - **All core tasks complete** (Stages 1-5 working correctly)
- 🔧 Phase 3.5: Training Infrastructure (T065-T074) - **Implementation complete but has runtime issues**
- 🔧 Phase 3.6: Loss Functions (T076-T080) - **Implementation complete but not validated**

**Recent Achievements (Nov 2, 2025):**
- ✅ Completed T054-T058: All preprocessing functions (normalization, imputation, encoding, sequences, validation)
- ✅ Completed T059-T063: LSTM/RNN model architectures (EmploymentLSTM, EmploymentRNN, CustomLSTM)
- ✅ Completed T065-T069: Dataset infrastructure (EmploymentDataset, augmentation, DataLoader, splits)
- ✅ Completed T070-T074: Training loop infrastructure (EmploymentTrainer, validation, checkpointing, early stopping)
- ✅ Completed T076-T080: Custom loss functions (WeightedEmploymentLoss, DirectionalAccuracyLoss, MAPE)
- ✅ Pipeline Stages 1-5 validated as working correctly
- ⚠️ Stage 6 (Training) has runtime issues requiring debugging

**Immediate Priorities:**
1. 🔥 Debug training stage (Stage 6) runtime issues - **BLOCKING ISSUE**
2. Validate loss functions and evaluation metrics work correctly
3. Provide team members opportunity for meaningful contribution (see contribution opportunities below)

---

## 🔒 Repository Protection & PR Workflow (Effective: November 2, 2025)

**IMPORTANT**: The repository now enforces mandatory pull requests and code review for all changes to the `master` branch. **Direct commits to master are blocked.**

### Required Workflow for ALL Changes

**⚠️ This applies to EVERYONE, including project lead. No exceptions.**

1. **Create a feature branch** for any changes (even documentation updates):
   ```bash
   git checkout -b feature/brief-description
   # or for bug fixes:
   git checkout -b fix/brief-description
   # or for documentation:
   git checkout -b docs/brief-description
   ```

2. **Make your changes** and commit to the branch:
   ```bash
   git add .
   git commit -m "descriptive commit message"
   git push -u origin feature/brief-description
   ```

3. **Create a Pull Request**:
   ```bash
   gh pr create --title "Brief PR Title" --body "Description of changes"
   # or use the GitHub web interface
   ```

4. **Code Review**:
   - At least one approval required before merging
   - All CI checks must pass (if configured)
   - No direct push to master allowed

5. **Merge** (after approval):
   ```bash
   gh pr merge --squash
   # or use the GitHub web interface
   ```

### Branch Naming Conventions
- `feature/TXXX-short-description` - New features or enhancements
- `fix/TXXX-bug-description` - Bug fixes
- `docs/TXXX-doc-update` - Documentation updates
- `test/TXXX-test-description` - Test additions or modifications

### Example Workflow
```bash
# Starting work on Task T092 (ARIMA baseline)
git checkout master
git pull origin master
git checkout -b feature/T092-arima-baseline

# Make changes to src/baselines.py
git add src/baselines.py
git commit -m "feat(T092): implement ARIMA forecasting baseline model

- Add ARIMA model using statsmodels
- Create fit() and predict() methods
- Add example usage in docstring"

git push -u origin feature/T092-arima-baseline

# Create PR
gh pr create \
  --title "Feature T092: Implement ARIMA Baseline Model" \
  --body "Implements traditional ARIMA forecasting model for employment data.

## Changes
- New ARIMA model class in src/baselines.py
- Forecast method with configurable parameters
- Documentation and usage examples

## Testing
Tested with sample employment data from Fresno County.

Related to #92"

# After approval and merge, delete branch
git checkout master
git pull origin master
git branch -d feature/T092-arima-baseline
git push origin --delete feature/T092-arima-baseline
```

### GitHub Issues Integration
- All tasks from tasks.md have been created as GitHub issues
- Issues are organized into milestones by phase
- Reference issue numbers in commits: `"feat(T092): ..."` or `"Closes #92"`
- Link PRs to issues using "Closes #92" or "Fixes #92" in PR description

### Why This Matters
- **Quality Control**: All code gets reviewed before merging
- **Accountability**: Clear audit trail of who contributed what
- **Documentation**: PR descriptions provide context for changes
- **Protection**: Prevents accidental overwrites or breaking changes
- **Collaboration**: Team members can review and learn from each other's code

### Emergency Procedures
If you absolutely must bypass (e.g., fixing broken CI), contact project lead Brian. Repository admin access may temporarily adjust rules, but this should be extremely rare.

---

## 🤝 Contribution Opportunities for Team Members

**As of 11/2/2025**, team members have been provided with git workflow training (branches, PRs) to enable safe contribution. The following enhancement tasks are available for team members who wish to contribute to the project.

**Note on Critical Path Items**: To ensure project deliverables are met on schedule, critical path items (training debugging, core evaluation, and final documentation) will be handled by Brian. The tasks below represent meaningful enhancement opportunities that add value to the project without blocking core deliverables.

### 🎨 **ENHANCEMENT OPPORTUNITIES**
These tasks enhance the project's quality, visualization, and analytical depth. While optional, they provide valuable contributions and learning opportunities.

**Category 1: Feature Engineering Visualizations** (Alejo - Good fit for visualization interest)
- [ ] **T042a**: Add distribution plots for growth rate features showing before/after histograms
  - Tools: matplotlib, seaborn
  - Estimated effort: 2-3 hours
  - Value: Helps validate feature engineering transformations

- [ ] **T042b**: Create time series plots showing employment trends with lag features overlaid
  - Tools: matplotlib
  - Estimated effort: 2-3 hours
  - Value: Visual validation of temporal patterns

- [ ] **T042c**: Generate correlation heatmap between original employment and all lag features
  - Tools: seaborn
  - Estimated effort: 1-2 hours
  - Value: Identifies most predictive lag periods

- [ ] **T042d**: Add statistical summary tables after each feature engineering step
  - Tools: pandas
  - Estimated effort: 2-3 hours
  - Value: Provides data quality insights at each transformation

- [ ] **T042e**: Create comprehensive data quality report for feature engineering pipeline
  - Tools: pandas, matplotlib
  - Estimated effort: 3-4 hours
  - Value: Documents records removed, nulls introduced, outliers detected

- [ ] **T042f**: Generate before/after comparison plots for filtering steps
  - Tools: matplotlib, seaborn
  - Estimated effort: 2-3 hours
  - Value: Shows impact of T033 quarterly filtering and T034 quality filtering

**Category 2: Advanced Baseline Comparisons** (Andrew - Good fit for model evaluation interest)
- [ ] **T092**: Implement traditional ARIMA forecasting model for employment data
  - Tools: statsmodels
  - Estimated effort: 4-5 hours
  - Value: Provides econometric baseline for comparison

- [ ] **T093**: Implement exponential smoothing model for employment trends
  - Tools: statsmodels
  - Estimated effort: 3-4 hours
  - Value: Additional baseline for time series forecasting

- [ ] **T094**: Create performance comparison tables: LSTM vs ARIMA vs Exponential Smoothing
  - Tools: pandas, matplotlib
  - Estimated effort: 2-3 hours
  - Value: Demonstrates LSTM advantages over traditional methods

- [ ] **T095**: Benchmark computational efficiency of different forecasting approaches
  - Tools: Python time module
  - Estimated effort: 2-3 hours
  - Value: Documents trade-offs between accuracy and speed

**Category 3: Advanced Temporal Features** (Either member)
- [ ] **T043**: Create rolling window statistics (3, 6, 12 quarter averages) for employment stability metrics
  - Tools: pandas
  - Estimated effort: 3-4 hours
  - Value: Adds smoothed trend features to model

- [ ] **T044**: Engineer cyclical features (quarter encoding, year encoding) and economic cycle indicators
  - Tools: pandas, numpy
  - Estimated effort: 3-4 hours
  - Value: Captures seasonal patterns in employment

- [ ] **T045**: Calculate employment volatility measures and trend strength indicators
  - Tools: pandas, numpy
  - Estimated effort: 3-4 hours
  - Value: Quantifies employment stability for each county/industry

**Category 4: Documentation Enhancements** (Either member)
- [ ] **T107**: Build reproducible experiment scripts with documented parameter settings
  - Tools: Python argparse
  - Estimated effort: 3-4 hours
  - Value: Enables easy replication of results

- [ ] **T120-NEW**: Create video walkthrough of pipeline execution (screen recording)
  - Tools: OBS Studio or similar
  - Estimated effort: 2-3 hours
  - Value: Visual documentation for presentation

- [ ] **T121-NEW**: Build interactive Jupyter notebook demonstrating key pipeline stages
  - Tools: Jupyter
  - Estimated effort: 4-5 hours
  - Value: Educational resource showing data transformations

### ⏱️ Timeline and Checkpoints
**Checkpoint Meeting: November 9, 2025**
- Team members should attend and provide status updates regarding tasks.
- Review progress on any claimed enhancement tasks
- Answer questions about git workflow, implementation patterns, or project structure

**Final Deadline: November 15, 2025**
- Enhancement tasks will be evaluated by this date. Progress will determine which will be included in final deliverables
- Focus should be on quality & demonstratable understanding over quantity

### 📋 How to Contribute (Git Workflow Guide)

**⚠️ MANDATORY PR WORKFLOW**: As of November 2, 2025, all changes require pull requests and code review. Direct commits to `master` are blocked. See [Repository Protection & PR Workflow](#-repository-protection--pr-workflow-effective-november-2-2025) section above for full details.

**Claiming a Task:**
1. Find a task from the enhancement opportunities above
2. GitHub issues have already been created for all tasks (see milestones)
3. Assign yourself to the issue: `gh issue edit <issue-number> --add-assignee @me`
4. Or comment on the issue stating you're working on it

**Development Workflow (MANDATORY STEPS):**
1. **Start from master**:
   ```bash
   git checkout master
   git pull origin master
   ```

2. **Create feature branch** (REQUIRED - no direct commits to master):
   ```bash
   git checkout -b feature/TXXX-brief-description
   ```

3. **Implement task** following existing code patterns in src/ directory

4. **Test implementation** thoroughly (run relevant pipeline stages)

5. **Commit changes** with descriptive message:
   ```bash
   git add .
   git commit -m "feat(TXXX): brief description

   - Detailed change 1
   - Detailed change 2

   Related to #<issue-number>"
   ```

6. **Push branch**:
   ```bash
   git push -u origin feature/TXXX-brief-description
   ```

7. **Create Pull Request** (REQUIRED):
   ```bash
   gh pr create --title "Task TXXX: Description" --body "Implementation details

   ## Changes
   - List key changes

   ## Testing
   - How you tested

   Closes #<issue-number>"
   ```

8. **Wait for code review** - at least one approval required

9. **Address code review feedback** if requested:
   ```bash
   # Make changes based on feedback
   git add .
   git commit -m "fix: address code review feedback"
   git push
   ```

10. **Merge after approval** (automatic or manual):
    ```bash
    gh pr merge --squash
    ```

11. **Clean up** (after merge):
    ```bash
    git checkout master
    git pull origin master
    git branch -d feature/TXXX-brief-description
    ```

**Getting Help:**
- Attend November 9 checkpoint meeting for implementation assistance
- Review existing code patterns in src/ directory (e.g., src/exploration.py for plotting examples)
- Refer to CLAUDE.md for architecture overview and development patterns
- Ask questions in PR comments or meeting discussions

**Important Notes:**
- ✅ **Always create a branch first** - never commit directly to master
- ✅ **All changes require PRs** - even documentation updates
- ✅ **Link PRs to issues** - use "Closes #123" or "Related to #123"
- ✅ **Write good commit messages** - explain what and why
- ✅ **Test before pushing** - ensure code works as expected

**Pipeline Architecture:**
- 3-layer design: Core implementations → Pipeline orchestrators → Master coordinator
- Data flow: Consolidation → Validation → Feature Engineering → Preprocessing → Model Training
- Output files tracked at each stage for debugging and validation

**Total Team Members**: 3
- **Brian (Project Lead)**: Overall coordination, setup, data exploration, validation, preprocessing, model architecture, feature engineering, training infrastructure, loss functions, visualization, and documentation
- **Andrew**: Original assignment - training infrastructure and loss functions/evaluation
- **Alejo**: Original assignment - feature engineering and documentation

### Workload Distribution
- **Brian (Project Lead)**: Implemented phases 3.1-3.6 (T001-T080) plus infrastructure tasks
- **Andrew**: Enhancement opportunities available (see contribution opportunities section)
- **Alejo**: Enhancement opportunities available (see contribution opportunities section)

### 📅 Team Participation Record
**Meeting Attendance (Sep 27 - Nov 1, 2025):**
- Brian: 5/5 meetings (100%)
- Andrew: 2/5 meetings (40%) - Sept 27, Nov 1
- Alejo: 2/5 meetings (40%) - Sept 27, Nov 1

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

### Data Quality Issues Discovered During Exploration
- [x] T017 **[Project Lead]** Investigate why oty_month1_emplvl_pct_chg shows NaN for years 2004-2019 - FIXED: Older data files have different schema (16 vs 42 columns)
- [x] T018 **[Project Lead]** Compare data schema between older CSV files (2004-2019) and newer ones (2020+) to identify missing lq_/oty_ columns
- [x] T019 **[Project Lead]** Fix year-over-year percentage calculations for early data by implementing manual calculations for older files
- [x] T020 **[Project Lead]** Fix duplicate record detection to include agglvl_code and size_code - these are different aggregation levels, not true duplicates
- [x] T021 **[Project Lead]** Investigate 6,270 records with establishments but zero employment - RESOLVED: Legacy data schema differences handled in consolidation
- [x] T022 **[Project Lead]** Investigate 295 records with positive employment but zero average wages - RESOLVED: Legacy data schema differences handled in consolidation
- [x] T023 **[Project Lead]** Review 33,279+ employment outliers outside IQR bounds - RESOLVED: Legacy data schema differences handled in consolidation
- [x] T024 **[Project Lead]** Review 10,448 wage outliers with extreme values - RESOLVED: Legacy data schema differences handled in consolidation
- [x] T025 **[Project Lead]** Improve data quality checks in exploration.py to handle missing data in older files gracefully - COMPLETED: Exploration now handles all schema variations

### Data Consolidation Verification (Added Oct 16, 2025)
The consolidated dataset includes multiple aggregation levels (County: 4.7M records, California Statewide: 299K records, United States: 399K records). Need to verify accuracy of consolidation process.

- [x] T026-NEW **[Project Lead]** Verify row counts match between raw CSV files and consolidated dataset - ensure no records lost or duplicated during consolidation in src/consolidation.py - COMPLETED: 5.4M rows match exactly
- [x] T027-NEW **[Project Lead]** Sample 100 random records from raw CSV files and verify they appear correctly in consolidated dataset with proper column mapping - COMPLETED: Random samples verified
- [x] T028-NEW **[Project Lead]** Validate aggregation level distribution (County/State/National) matches expected proportions from raw data - COMPLETED: 87% county, 7% US, 6% CA
- [x] T029-NEW **[Project Lead]** Verify wage statistics at county level only (excluding state/national aggregates) are reasonable for California employment data - COMPLETED: County-level stats are reasonable
- [x] T030-NEW **[Project Lead]** Check that data consolidation preserves all unique combinations of area_name + year + quarter + industry_code - COMPLETED: All 5.4M records are unique when considering ownership/NAICS levels
- [x] T031-NEW **[Project Lead]** Document aggregation levels and their implications for modeling in docs/data_dictionary.md - COMPLETED: docs/data_aggregation_levels.md created

## Phase 3.3: Feature Engineering **[Alejo]**

### Data Filtering and Preparation (Added Oct 16, 2025)
**Critical preprocessing before feature engineering**: The consolidated data contains "Annual" records mixed with quarterly data, state/national aggregates that must be excluded, and incomplete records (zero employment/wages) that will corrupt features.

**Status Update (Oct 20, 2025)**: Core filtering and feature engineering tasks implemented. T032-T034, T038, T042 are complete and tested. Advanced features (T039-T041) deferred as non-critical.

- [x] T032 **[Alejo]** Filter consolidated data to county-level records only (drop area_type='United States' and 'California - Statewide') in src/feature_engineering.py - COMPLETED: filter_to_county_level() fully implemented and tested (5.4M→4.7M records)
- [x] T033 **[Alejo]** Handle "Annual" vs quarterly records - decided to drop Annual records to keep only quarterly data ('1st Qtr', '2nd Qtr', '3rd Qtr', '4th Qtr') in src/feature_engineering.py - COMPLETED: handle_annual_vs_quarterly() implemented
- [x] T034 **[Alejo]** Create data quality filter to remove incomplete records (negative employment, negative wages, zero employment with establishments) in src/feature_engineering.py - COMPLETED: data_quality_filter() implemented with 3 rules
- [~] T035 **[Alejo]** Create Central Valley counties reference file containing array of county names (Fresno, Kern, Kings, Madera, Merced, San Joaquin, Stanislaus, Tulare) in data/central_valley_counties.json - DEFERRED: File exists, subset creation not critical for initial model
- [~] T036 **[Alejo]** Generate two processed datasets: (1) all California counties with features, (2) Central Valley counties only subset in data/processed/ - DEFERRED: All counties dataset created (final_features.csv), CV subset not needed initially
- [~] T037 **[Alejo]** Validate filtered datasets have consistent temporal coverage and no data gaps in src/feature_engineering.py - PARTIALLY COMPLETE: Basic validation in place, comprehensive checks TODO

### Feature Calculations
- [x] T038 **[Alejo]** Calculate quarter-over-quarter employment growth rates and percentage changes in src/feature_engineering.py - COMPLETED: calculate_quarterly_growth_rates() adds 6 growth columns (QoQ change/%, YoY change/%)
- [~] T039 **[Alejo]** Create seasonal adjustment factors using historical employment patterns in src/feature_engineering.py - DEFERRED: Advanced feature requiring statsmodels, not critical for baseline model
- [~] T040 **[Alejo]** Engineer industry concentration metrics and economic diversity indices in src/feature_engineering.py - DEFERRED: Advanced feature (HHI calculation), low priority for initial model
- [~] T041 **[Alejo]** Build geographic clustering features based on employment similarity in src/feature_engineering.py - DEFERRED: Advanced spatial analysis, low priority for initial model
- [x] T042 **[Alejo]** Generate lag features for temporal dependencies in employment trends in src/feature_engineering.py - COMPLETED: generate_lag_features() adds 4 lag columns (1-4 quarters back)

### Feature Engineering Quality & Visualization (Added Oct 20, 2025)
**Enhancement**: Add comprehensive statistics and visualizations at each feature engineering step to help users understand data transformations and validate feature quality.

- [ ] T042a **[Alejo]** Add distribution plots for growth rate features (QoQ%, YoY%) showing before/after histograms in src/feature_engineering.py with save to data/feature_engineering/plots/
- [ ] T042b **[Alejo]** Create time series plots showing employment trends with lag features overlaid for sample counties/industries in src/feature_engineering.py
- [ ] T042c **[Alejo]** Generate correlation heatmap between original employment and all lag features in src/feature_engineering.py
- [ ] T042d **[Alejo]** Add statistical summary tables (mean, median, std, min, max, null count) printed to console after each feature engineering step in src/feature_engineering.py - PARTIALLY COMPLETE: Added for T038 and T042
- [ ] T042e **[Alejo]** Create data quality report showing: records removed at each filter step, null values introduced by lag/growth features, outlier detection in new features in src/feature_engineering.py
- [ ] T042f **[Alejo]** Generate before/after comparison plots for T033 (quarterly distribution) and T034 (quality filtering impact) in src/feature_engineering.py
- [ ] T042g **[Alejo]** Add feature importance estimation using correlation with target variable (future employment) in src/feature_engineering.py

- [ ] T043 **[Alejo]** Create rolling window statistics (3, 6, 12 quarter averages) for employment stability in src/temporal_features.py
- [ ] T044 **[Alejo]** Engineer cyclical features (quarter, year) and economic cycle indicators in src/temporal_features.py
- [ ] T045 **[Alejo]** Calculate employment volatility measures and trend strength indicators in src/temporal_features.py
- [ ] T046 **[Alejo]** Validate temporal features for consistency and economic reasonableness in src/temporal_features.py
- [ ] T047 **[Alejo]** Create time-based train/validation/test splits preserving temporal order in src/temporal_features.py
- [ ] T048 **[Alejo]** Create geographic feature maps for counties/regions with employment density in src/geographic_features.py
- [ ] T049 **[Alejo]** Engineer industry classification features and sector similarity matrices in src/geographic_features.py
- [ ] T050 **[Alejo]** Build regional economic indicators and metropolitan area classifications in src/geographic_features.py
- [ ] T051 **[Alejo]** Calculate spatial autocorrelation features for neighboring region employment in src/geographic_features.py
- [ ] T052 **[Alejo]** Validate geographic features against known economic geography patterns in src/geographic_features.py

- [x] T053 **[Project Lead]** Set up feature engineering structure and initial files for team collaboration

## Phase 3.4: Data Preprocessing and Model Architecture **[Originally Assigned: Project Lead | Actually Completed By: Brian]**

**Status Update (Nov 2, 2025)**: ALL preprocessing and model architecture tasks complete. Full implementations verified in src/preprocessing.py and src/lstm_model.py. Part of working pipeline (Stages 1-5).

- [x] T054 **[Project Lead → Brian]** Normalize employment counts and wage data using robust scaling techniques in src/preprocessing.py - **COMPLETED**: EmploymentDataPreprocessor.normalize_employment_data() uses RobustScaler, fully functional
- [x] T055 **[Project Lead → Brian]** Handle missing values with domain-appropriate imputation strategies in src/preprocessing.py - **COMPLETED**: EmploymentDataPreprocessor.handle_missing_values() uses SimpleImputer with median strategy
- [x] T056 **[Project Lead → Brian]** Create categorical encodings for industry codes and geographic identifiers in src/preprocessing.py - **COMPLETED**: EmploymentDataPreprocessor.create_categorical_encodings() uses LabelEncoder
- [x] T057 **[Project Lead → Brian]** Transform tabular data into sequence format suitable for RNN/LSTM processing in src/preprocessing.py - **COMPLETED**: transform_to_sequences() fully implemented with sliding window algorithm (groups by county+industry, creates 12-quarter sequences)
- [x] T058 **[Project Lead → Brian]** Validate preprocessing steps maintain data distribution properties in src/preprocessing.py - **COMPLETED**: Validation functions implemented in preprocessing_pipeline.py
- [x] T059 **[Project Lead → Brian]** Design LSTM layers for temporal employment sequence processing in src/lstm_model.py - **COMPLETED**: EmploymentLSTM class with 2-layer LSTM, batch norm, dropout
- [x] T060 **[Project Lead → Brian]** Implement RNN architecture for sequential employment pattern recognition in src/lstm_model.py - **COMPLETED**: EmploymentRNN class fully implemented
- [x] T061 **[Project Lead → Brian]** Create custom LSTM architecture combining temporal dependencies and spatial features in src/lstm_model.py - **COMPLETED**: CustomLSTM class with temporal/spatial fusion
- [x] T062 **[Project Lead → Brian]** Add batch normalization and dropout layers appropriate for employment data in src/lstm_model.py - **COMPLETED**: Integrated into EmploymentLSTM and CustomLSTM architectures
- [x] T063 **[Project Lead → Brian]** Validate LSTM architecture dimensions match processed employment sequence shapes in src/lstm_model.py - **COMPLETED**: validate_lstm_architecture() function with 5 validation checks

- [x] T064 **[Project Lead]** Set up data preprocessing and model architecture structure and initial files for team collaboration

## Phase 3.5: Training Infrastructure **[Original: Andrew → Brian]**

**⚠️ CURRENT BLOCKING ISSUE**: Implementation complete but Stage 6 (Training) has runtime errors preventing execution. Debugging in progress.

- [x] T065 **[Andrew → Brian]** Create PyTorch Dataset class for efficient QCEW data loading and batching in src/dataset.py - **COMPLETED**: EmploymentDataset class fully implemented
- [x] T066 **[Andrew → Brian]** Implement data augmentation techniques appropriate for employment time series in src/dataset.py - **COMPLETED**: create_data_augmentation() with Gaussian noise (1%)
- [x] T067 **[Andrew → Brian]** Build DataLoader with proper batch sizes for employment tensor processing in src/dataset.py - **COMPLETED**: build_data_loader() with configurable batch size, shuffle, workers
- [x] T068 **[Andrew → Brian]** Create train/validation data splits preserving temporal and geographic balance in src/dataset.py - **COMPLETED**: create_train_val_splits() with temporal ordering preservation
- [x] T069 **[Andrew → Brian]** Validate batch processing maintains employment data integrity and relationships in src/dataset.py - **COMPLETED**: validate_batch_processing() with 5 integrity checks
- [x] T070 **[Andrew → Brian]** Implement training loop with employment-specific loss functions (MSE, MAE) in src/training.py - **COMPLETED**: EmploymentTrainer.train_epoch() with gradient clipping
- [x] T071 **[Andrew → Brian]** Create validation loop with employment forecasting accuracy metrics in src/training.py - **COMPLETED**: EmploymentTrainer.validate_epoch()
- [x] T072 **[Andrew → Brian]** Add model checkpointing for best employment prediction performance in src/training.py - **COMPLETED**: Integrated in EmploymentTrainer.train_model()
- [x] T073 **[Andrew → Brian]** Implement early stopping based on employment prediction validation loss in src/training.py - **COMPLETED**: Early stopping with configurable patience in train_model()
- [x] T074 **[Andrew → Brian]** Build learning rate scheduling appropriate for employment data convergence in src/training.py - **COMPLETED**: Scheduler support in EmploymentTrainer class

- [x] T075 **[Project Lead]** Set up training infrastructure structure and initial files for team collaboration

## Phase 3.6: Loss Functions and Evaluation **[Original: Andrew → Brian]**

**Status**: Loss functions implemented but not yet validated. Evaluation tasks pending until training issues resolved.

- [x] T076 **[Andrew → Brian]** Implement weighted loss functions emphasizing recent employment trends in src/loss_metrics.py - **COMPLETED**: WeightedEmploymentLoss class with temporal weighting
- [x] T077 **[Andrew → Brian]** Create custom metrics for employment forecasting accuracy (MAPE, directional accuracy) in src/loss_metrics.py - **COMPLETED**: DirectionalAccuracyLoss and MAPE functions implemented
- [x] T078 **[Andrew → Brian]** Add employment volatility prediction loss for capturing uncertainty in src/loss_metrics.py - **COMPLETED**: Custom volatility loss components in loss_metrics.py
- [x] T079 **[Andrew → Brian]** Build industry-weighted loss functions for sector-specific prediction importance in src/loss_metrics.py - **COMPLETED**: Industry weighting capability in custom loss classes
- [x] T080 **[Andrew → Brian]** Validate loss functions align with employment forecasting evaluation standards in src/loss_metrics.py - **COMPLETED**: Validation logic implemented (not yet tested due to training issues)
- [ ] T081 **[Andrew]** Calculate employment prediction accuracy across different time horizons in src/evaluation.py - **BLOCKED**: Requires working training pipeline
- [ ] T082 **[Andrew]** Create confusion matrices for employment growth/decline classification in src/evaluation.py - **BLOCKED**: Requires working training pipeline
- [ ] T083 **[Andrew]** Plot predicted vs actual employment trends by industry and region in src/evaluation.py - **BLOCKED**: Requires working training pipeline
- [ ] T084 **[Andrew]** Generate employment volatility prediction accuracy assessments in src/evaluation.py - **BLOCKED**: Requires working training pipeline
- [ ] T085 **[Andrew]** Validate model performance against employment forecasting benchmarks in src/evaluation.py - **BLOCKED**: Requires working training pipeline

- [x] T086 **[Project Lead]** Set up loss functions and evaluation structure and initial files for team collaboration

## Phase 3.7: Visualization and Comparison **[Project Lead]**
- [ ] T087 **[Project Lead]** Implement feature attribution techniques for employment factor importance in src/visualization.py
- [ ] T088 **[Project Lead]** Visualize LSTM learned patterns and their relationship to employment sequences in src/visualization.py
- [ ] T089 **[Project Lead]** Create employment trend visualizations showing model predictions vs reality in src/visualization.py
- [ ] T090 **[Project Lead]** Generate geographic heat maps of employment prediction accuracy in src/visualization.py
- [ ] T091 **[Project Lead]** Validate feature importance aligns with known employment economic factors in src/visualization.py
- [ ] T092 **[Project Lead]** Implement traditional employment forecasting models (ARIMA, exponential smoothing) in src/baselines.py
- [ ] T093 **[Project Lead]** Compare LSTM performance against econometric employment prediction models in src/baselines.py
- [ ] T094 **[Project Lead]** Create ensemble methods combining LSTM with traditional employment forecasting in src/baselines.py
- [ ] T095 **[Project Lead]** Benchmark computational efficiency for large-scale employment data processing in src/baselines.py
- [ ] T096 **[Project Lead]** Validate LSTM provides meaningful improvement over employment forecasting baselines in src/baselines.py
- [ ] T097 **[Project Lead]** Create visual predictions vs actuals plots showing predicted employment alongside actual values in src/prediction_visuals.py
- [ ] T098 **[Project Lead]** Implement multi-step ahead forecasts with 4-quarter predictions and uncertainty bands in src/forecasting.py
- [ ] T099 **[Project Lead]** Build industry risk dashboard displaying growth/decline status for each industry code in src/dashboard.py
- [ ] T100 **[Project Lead]** Develop county-level comparison visualizations for Central Valley counties employment growth vs decline in src/county_comparisons.py
- [ ] T101 **[Project Lead]** Create early warning system flagging industries predicted to lose >5% employment in next 2 quarters in src/early_warning.py
- [ ] T102 **[Project Lead]** Generate wage growth predictions showing industries with highest wage increases in src/wage_predictions.py
- [ ] T103 **[Project Lead]** Produce policy insights with actionable recommendations based on employment predictions in src/policy_insights.py

- [x] T104 **[Project Lead]** Set up visualization and comparison structure and initial files for team collaboration

## Phase 3.8: Documentation and Reporting **[Alejo]**
- [ ] T105 **[Alejo]** Document LSTM methodology for employment data analysis and prediction in docs/methodology.md
- [ ] T106 **[Alejo]** Create comprehensive results analysis with employment trend insights in docs/results.md
- [ ] T107 **[Alejo]** Build reproducible experiment scripts for QCEW data processing in scripts/
- [ ] T108 **[Alejo]** Generate academic-style report on LSTM applications to labor economics in docs/report.pdf
- [ ] T109 **[Alejo]** Validate all results are reproducible and methodology is clearly documented in docs/validation.md

- [x] T110 **[Project Lead]** Set up documentation and reporting structure and initial files for team collaboration

### Unified Pipeline Development
To achieve the goal of a single-click execution, all components will be developed in separate files initially for modularity, then aggregated into one comprehensive script.

- [x] T111 **[Project Lead → Brian]** Develop modular components in separate files (data download, exploration, feature engineering, preprocessing, model architecture, training, evaluation, visualization) - **COMPLETED**: All modules implemented in src/
- [x] T112 **[Project Lead → Brian]** Create integration functions to combine all modules into a single workflow - **COMPLETED**: pipeline_orchestrator.py implements QCEWPipeline class
- [x] T113 **[Project Lead → Brian]** Build a unified script (main.py in root directory) that executes the entire pipeline from data consolidation to LSTM training and evaluation - **COMPLETED**: main.py with 8 stages (consolidate, explore, validate, features, preprocess, train, evaluate, predict)
- [x] T114 **[Project Lead → Brian]** Add command-line interface and configuration options to the unified script for flexibility - **COMPLETED**: CLI with --stage, --skip-plots, --force-rebuild, --launch-interface options
- [ ] T115 **[Project Lead → Brian]** Test the unified script end-to-end and ensure it runs with one click - **IN PROGRESS**: Stages 1-5 work, Stage 6 (training) has issues being debugged, Stages 7-8 pending
- [ ] T116 **[Project Lead → Brian]** Document the unified pipeline usage and deployment instructions - **PENDING**: Will complete after end-to-end testing
- [ ] T117 **[Project Lead]** Build interactive prediction interface allowing user input of future time and displaying forecasts with visualizations in src/prediction_interface.py - **BLOCKED**: Requires working training
- [ ] T118 **[Project Lead]** Integrate maps, charts, graphs, and confidence bands into the prediction interface output - **BLOCKED**: Requires working training
- [ ] T119 **[Project Lead]** Add uncertainty estimation and error bands to all prediction visualizations - **BLOCKED**: Requires working training

## Master Pipeline Orchestrator (main.py)

**Location**: `/main.py` (root directory)

**Purpose**: Single entry point for the entire QCEW employment forecasting pipeline that coordinates all stages from data consolidation through model training and prediction.

**Key Features**:
- ✅ Protects raw data files (read-only access, never modified)
- ✅ Consolidates multiple CSV files into master dataset
- ✅ Stages: Consolidate → Explore → Validate → Feature Engineering → Preprocess → Train → Evaluate → Predict
- ✅ Automatic directory creation (processed/, validated/, plots/)
- ✅ Comprehensive logging and progress tracking
- ✅ Stage-specific execution (run individual stages as needed)
- ✅ Command-line interface with options (--stage, --skip-plots, --force-rebuild)
- ✅ Generates exploration visualizations automatically

**Usage**:
```bash
# Run full pipeline
python main.py

# Run specific stage
python main.py --stage explore
python main.py --stage train
python main.py --stage predict

# Options
python main.py --skip-plots          # Skip visualization generation
python main.py --force-rebuild       # Force rebuild of consolidated data
python main.py --launch-interface    # Launch prediction interface after completion
```

**Data Flow**:
1. **Input**: Raw CSV files in `/data/raw/` (manually downloaded, never modified)
2. **Consolidation**: Combines all CSVs → `/data/processed/qcew_master_consolidated.csv`
3. **Exploration**: Analysis + visualizations → `/data/processed/plots/`
4. **Validation**: Quality checks → `/data/validated/qcew_validated.csv`
5. **Features**: Engineered features → `/data/processed/qcew_features.csv`
6. **Preprocessing**: Normalized sequences → `/data/processed/qcew_preprocessed.csv`
7. **Training**: Model training → `/data/processed/lstm_model.pt`
8. **Evaluation**: Performance metrics and comparison plots
9. **Prediction**: Interactive forecasting interface

**Visualization Outputs** (automatically generated in `/data/processed/plots/`):
- `employment_trends.png` - Total employment over time
- `quarterly_distribution.png` - Employment by quarter
- `wage_trends.png` - Average weekly wage trends
- `top_industries.png` - Top 15 industries by employment

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

- [x] T053 **[Project Lead]** Set up feature engineering structure and initial files for team collaboration
- [x] T064 **[Project Lead]** Set up data preprocessing and model architecture structure and initial files for team collaboration
- [x] T075 **[Project Lead]** Set up training infrastructure structure and initial files for team collaboration
- [x] T086 **[Project Lead]** Set up loss functions and evaluation structure and initial files for team collaboration
- [x] T104 **[Project Lead]** Set up visualization and comparison structure and initial files for team collaboration
- [x] T110 **[Project Lead]** Set up documentation and reporting structure and initial files for team collaboration

## 📊 Project Completion Statistics (As of November 2, 2025)

### Overall Progress
- **Total Tasks Defined**: 119 tasks (T001-T119)
- **Tasks Completed**: 80 tasks (67%)
- **Tasks In Progress/Blocked**: 8 tasks (7%)
- **Tasks Not Started**: 31 tasks (26%)

### Completion by Phase
| Phase | Tasks | Completed | In Progress | Not Started | % Complete |
|-------|-------|-----------|-------------|-------------|-----------|
| 3.1 Setup | 5 | 5 | 0 | 0 | 100% |
| 3.2 Exploration & Validation | 26 | 26 | 0 | 0 | 100% |
| 3.3 Feature Engineering | 31 | 5 | 3 | 23 | 16% (core: 100%) |
| 3.4 Preprocessing & Models | 10 | 10 | 0 | 0 | 100% |
| 3.5 Training Infrastructure | 10 | 10 | 0 | 0 | 100% (has runtime issues) |
| 3.6 Loss & Evaluation | 10 | 5 | 5 | 0 | 50% |
| 3.7 Visualization | 17 | 0 | 0 | 17 | 0% |
| 3.8 Documentation | 5 | 0 | 0 | 5 | 0% |
| Unified Pipeline | 5 | 4 | 0 | 1 | 80% |

### Critical Path Status
✅ **Working (Stages 1-5)**: Data consolidation, exploration, validation, feature engineering, preprocessing
⚠️ **Blocked (Stage 6)**: Model training (implementation complete but runtime errors)
❌ **Not Started (Stages 7-8)**: Evaluation, prediction interface

### Completion by Team Member

| Member | Planned Tasks | Completed Tasks | Completion Rate | Notes |
|--------|---------------|-----------------|-----------------|-------|
| **Brian (Project Lead)** | 49 tasks | **80 tasks** | **163%** | Implemented phases 3.1-3.6 plus infrastructure |
| **Andrew** | 20 tasks | **0 tasks** | **0%** | Enhancement opportunities available |
| **Alejo** | 20 tasks | **0 tasks** | **0%** | Enhancement opportunities available |

### Historical Assignment Record
This section preserves the original task assignments for accountability and contribution tracking:

**Phase 3.1 (Setup)**: Original: Brian → Implemented: Brian ✅
**Phase 3.2 (Exploration/Validation)**: Original: Brian → Implemented: Brian ✅
**Phase 3.3 (Feature Engineering)**: Original: Alejo → Implemented: Brian ✅
**Phase 3.4 (Preprocessing/Models)**: Original: Brian → Implemented: Brian ✅
**Phase 3.5 (Training Infrastructure)**: Original: Andrew → Implemented: Brian ✅
**Phase 3.6 (Loss/Evaluation)**: Original: Andrew → Implemented: Brian (partial) 🔄
**Phase 3.7 (Visualization)**: Original: Brian → Status: In progress 🔄
**Phase 3.8 (Documentation)**: Original: Alejo → Status: Pending ⏳

### Forensic Work Log Summary

**Week of Sep 27, 2025**:
- Brian: Project setup, environment configuration, work breakdown structure, task assignments
- Andrew: Attended for project setup & introductory training using github
- Alejo: Attended for project setup & introductory training using github

**Week of Oct 11, 2025**:
- Brian: Data sourcing attempts (API-based approach)
- Andrew: Absent
- Alejo: Absent

**Week of Oct 18, 2025**:
- Brian: Completed data sourcing, began data consolidation implementation
- Andrew: Absent
- Alejo: Absent

**Week of Oct 25, 2025**:
- Brian: Completed Phases 3.1-3.2 (Setup, Exploration, Validation), began Phase 3.3 (Feature Engineering)
- Andrew: Absent
- Alejo: Absent

**Week of Nov 1, 2025**:
- Brian: Completed Phases 3.3-3.6 (Feature Engineering, Preprocessing, Models, Training Infrastructure, Loss Functions); attended meeting at 10am
- Andrew: Attended rescheduled meeting at 10:30am
- Alejo: Attended rescheduled meeting at 10:20am

**Week of Nov 2, 2025** (current):
- Brian: Documentation updates, task validation, debugging training stage
- Andrew: Enhancement opportunities available (see contribution section)
- Alejo: Enhancement opportunities available (see contribution section)

### Next Steps for Project Success
1. **Immediate (This Week)**: Debug Stage 6 (Training) runtime issues - Brian
2. **Checkpoint Meeting (Nov 9)**: Team meeting to assist with enhancement tasks if claimed
3. **Short-term (By Nov 15)**: Complete evaluation tasks (T081-T085) - Brian
4. **Documentation (By Nov 15)**: Complete methodology and results documentation (T105-T109) - Brian
5. **Final Integration (By Nov 15)**: End-to-end pipeline testing (T115-T116) - Brian
6. **Optional Enhancements (By Nov 15)**: Visualization and advanced features available for team contribution

### Risk Assessment
🔴 **High Risk**: Training stage blocking progress on evaluation and prediction interface
🟡 **Medium Risk**: Team member non-contribution may delay documentation deliverables
🟢 **Low Risk**: Core pipeline (Stages 1-5) is stable and working correctly

## Issues and Resolutions

### Issue 1: Jupyter Notebook Conversion to Python Scripts
The specs documentation mentions performing tasks in .ipynb files, which is acceptable for interactive development. However, to ensure redundancy and compatibility, each Jupyter notebook should be converted to a corresponding Python script as part of the workflow. This allows for easier integration into automated pipelines and provides alternative execution methods.

- [x] **[Project Lead]** Convert notebooks/exploration.ipynb to src/exploration.py with equivalent functionality
- [ ] **[Project Lead]** Ensure all notebook-based tasks include conversion steps to Python scripts
- [ ] **[Project Lead]** Update task descriptions to reference both notebook and script versions where applicable

### Issue 2: Preventing CSV Data Files from Being Pushed to GitHub
Despite .gitignore configurations, CSV data files in the data/ directory are being pushed to GitHub. Since each team member downloads their own data using the scripts, these large files should not be version-controlled. Need to properly configure .gitignore and potentially remove already committed files.

- [x] **[Andrew]** Update .gitignore to exclude all .csv files in data/ subdirectories (data/raw/*.csv, data/processed/*.csv, data/validated/*.csv)
- [x] **[Andrew]** Check for and remove any already committed CSV files from the repository
- [x] **[Andrew]** Verify that data download scripts still function correctly without committed CSV files

### Issue 3: Limited Data Download Range
The current data download script only retrieves QCEW data from 2020 Q1 to 2024 Q4, despite the California Open Data Portal having data available from 2004-2025. This significantly limits the historical context available for time series modeling and forecasting.

- [x] **[Project Lead]** Investigate why data download is limited to 2020-2024 and identify methods to access full historical dataset (2004-2025) from California Open Data Portal - COMPLETED: Successfully switched data source to California Open Data Portal, implemented format conversion, and expanded dataset from 2020-2024 (64K rows) to 2004-2024 (243K rows)

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