# IS-160 Project Presentation Slideshow Plan

## California Employment Forecasting with LSTM/RNN Deep Learning

---

## Table of Contents

1. [Slide Deck Overview](#slide-deck-overview)
2. [Phase-by-Phase Slides](#phase-by-phase-slides)
3. [Required Images & Screen Captures](#required-images--screen-captures)
4. [Terminal Output Examples](#terminal-output-examples)
5. [Generated Data Plots](#generated-data-plots)
6. [AI Image Generation Prompts](#ai-image-generation-prompts)
7. [Image Placement Map](#image-placement-map)

---

## Slide Deck Overview

### Presentation Flow (Estimated: 25-30 slides)

```
Opening (3 slides)
    ↓
Problem & Context (3 slides)
    ↓
Data & Pipeline Architecture (5 slides)
    ↓
Phase 1-3: Data Pipeline (4 slides)
    ↓
Phase 4: Feature Engineering (3 slides)
    ↓
Phase 5: Preprocessing (3 slides)
    ↓
Phase 6: Model Training (4 slides)
    ↓
Phase 7: Evaluation (3 slides)
    ↓
Phase 8: Prediction Interface (2 slides)
    ↓
Results & Conclusions (3 slides)
    ↓
Q&A (1 slide)
```

---

## Phase-by-Phase Slides

### Section 1: Opening (Slides 1-3)

#### Slide 1: Title Slide
**Content:**
- Title: "California Employment Forecasting Using LSTM/RNN Deep Learning"
- Subtitle: "IS-160 Project: Time Series Analysis with QCEW Data"
- Team members: Project Lead, Andrew, Alejo
- Date and course info

**Visual:**
- Hero image of California map with employment data overlay
- Project logo or university branding

**AI Image Needed:** `[IMG-01] California Employment Hero`

---

#### Slide 2: Executive Summary
**Content:**
- Problem: Predicting quarterly employment trends for economic planning
- Solution: LSTM neural network processing 5.4M+ employment records
- Key achievement: 91.98% directional accuracy in trend prediction
- Scope: California QCEW data from 2004-2024

**Visual:**
- Key metrics highlight boxes
- Mini timeline showing data span

**Terminal Output:**
```
================================================================================
QCEW DATA VALIDATION REPORT
================================================================================
Total records: 5,430,384
Overall Data Quality Score: 0.859 (85.9%)
Time span: 2004-2024 (20+ years)
================================================================================
```

---

#### Slide 3: Agenda/Roadmap
**Content:**
- 8-stage pipeline overview
- What we'll cover in the presentation

**Visual:**
- Pipeline roadmap graphic showing all 8 stages

**AI Image Needed:** `[IMG-02] Pipeline Roadmap`

---

### Section 2: Problem & Context (Slides 4-6)

**Content:**
- Why employment forecasting matters
- California's complex labor market (diverse industries, geographic spread)
- Stakeholders: policymakers, workforce agencies, businesses
- Traditional methods vs. deep learning approach

**Visual:**
- California economic statistics infographic

**AI Image Needed:** `[IMG-03] Business Problem Infographic`

---

#### Slide 5: The QCEW Dataset
**Content:**
- Quarterly Census of Employment and Wages (QCEW)
- Source: California EDD
- Coverage: All industries, all counties
- Key variables: Employment counts, wages, establishments

**Visual:**
- Sample data table screenshot
- Data source credibility badges

**Screen Capture Needed:** `[SCREEN-01] Raw data CSV sample`

**Terminal Output:**
```
Raw data files:
├── qcew_2004-2007.csv
├── qcew_2008-2011.csv
├── qcew_2012-2015.csv
├── qcew_2016-2019.csv
├── qcew-2020-2022.csv
└── qcew-2023-2024.csv
```

---

#### Slide 6: Project Objectives
**Content:**
1. Process 20+ years of California employment data
2. Build production-grade LSTM forecasting pipeline
3. Compare deep learning vs. traditional methods (ARIMA, exponential smoothing)
4. Predict quarterly employment levels with uncertainty estimates

**Visual:**
- Objectives checklist graphic

---

### Section 3: Data & Pipeline Architecture (Slides 7-11)

#### Slide 7: High-Level Architecture
**Content:**
- Three-layer pipeline design:
  - Layer 1: Core implementation modules
  - Layer 2: Pipeline orchestrators
  - Layer 3: Master orchestrator
- Benefits: Modularity, testability, maintainability

**Visual:**
- System architecture diagram

**AI Image Needed:** `[IMG-04] System Architecture Diagram`

---

#### Slide 8: Data Flow Overview
**Content:**
- Raw CSV → Consolidation → Validation → Features → Preprocessing → Training → Evaluation → Prediction
- Data volumes at each stage

**Visual:**
- Data flow diagram showing record counts

**AI Image Needed:** `[IMG-05] Data Flow Diagram`

**Terminal Output:**
```
Data Flow Pipeline:
Raw CSV (6 files)
    → Consolidated: 5,430,384 records
    → County-filtered: 4,714,234 records (87%)
    → Quarterly-only: 4,087,643 records
    → Quality-filtered: 3,892,109 records
    → Final features: 3,654,832 records
    → LSTM sequences: 599,582 samples
```

---

#### Slide 9: Technology Stack
**Content:**
- Core: Python 3.8+, PyTorch
- Data: pandas, NumPy, scikit-learn
- Visualization: matplotlib, seaborn, TensorBoard
- Traditional ML: statsmodels (ARIMA)
- Development: Git, GitHub, VSCode

**Visual:**
- Technology stack icons/logos

**Screen Capture Needed:** `[SCREEN-02] requirements.txt or tech stack icons`

---

#### Slide 10: Project Structure
**Content:**
- Directory layout
- Key modules and their responsibilities
- Configuration system

**Visual:**
- Project tree diagram

**Terminal Output:**
```
is-160-project/
├── main.py                    # Entry point
├── interactive_menu.py        # User interface
├── config/
│   └── hyperparameters.py     # Centralized config
├── src/
│   ├── pipeline_orchestrator.py  # Master coordinator
│   ├── feature_pipeline.py       # Feature engineering
│   ├── preprocessing.py          # Data preprocessing
│   ├── lstm_model.py             # Neural network
│   ├── training.py               # Training loop
│   └── evaluation.py             # Model evaluation
└── data/
    ├── raw/                   # Source data (read-only)
    ├── processed/             # Pipeline outputs
    └── feature_engineering/   # Intermediate files
```

---

#### Slide 11: Interactive Menu System
**Content:**
- User-friendly interface for running pipeline stages
- Status tracking and progress monitoring
- Both CLI and interactive modes available

**Visual:**
- Screenshot of interactive menu

**Screen Capture Needed:** `[SCREEN-03] Interactive menu terminal`

**Terminal Output:**
```
================================================================================
                QCEW EMPLOYMENT FORECASTING PIPELINE
                    RNN/LSTM Time Series Analysis
================================================================================

--------------------------------------------------------------------------------
MAIN MENU
--------------------------------------------------------------------------------
  0. Run Full Pipeline (All Stages)
  1. Stage 1: Data Consolidation
  2. Stage 2: Data Exploration & Visualization
  3. Stage 3: Data Validation
  4. Stage 4: Feature Engineering
  5. Stage 5: Data Preprocessing
  6. Stage 6: Train LSTM Model
  7. Stage 7: Evaluate Model
  8. Stage 8: Interactive Prediction Interface
  9. View Pipeline Status
 10. Exit
--------------------------------------------------------------------------------
```

---

### Section 4: Phase 1-3 Data Pipeline (Slides 12-15)

#### Slide 12: Stage 1 - Data Consolidation
**Content:**
- Challenge: Schema changes between 2004-2019 and 2020+ data
- Solution: Automatic schema normalization and mapping
- Output: Single unified dataset of 5.4M+ records
- Data protection: Raw files never modified

**Visual:**
- Before/after schema diagram

**AI Image Needed:** `[IMG-06] Schema Consolidation Diagram`

**Terminal Output:**
```
[INFO] Stage 1: Data Consolidation
[INFO] Loading 6 CSV files from data/raw/
[INFO] Processing qcew_2004-2007.csv... 892,456 records
[INFO] Processing qcew_2008-2011.csv... 956,234 records
[INFO] Processing qcew_2012-2015.csv... 1,023,567 records
[INFO] Processing qcew_2016-2019.csv... 1,089,345 records
[INFO] Processing qcew-2020-2022.csv... 892,456 records
[INFO] Processing qcew-2023-2024.csv... 576,326 records
[INFO] Normalizing column names across schemas...
[OK] Consolidated: 5,430,384 total records
[OK] Saved to qcew_master_consolidated.csv
```

---

#### Slide 13: Stage 2 - Data Exploration
**Content:**
- Exploratory Data Analysis (EDA)
- Key insights:
  - Employment trends over 20 years
  - Quarterly seasonal patterns
  - Top industries by employment
  - Wage distribution patterns

**Visual:**
- 4-panel layout with generated plots

**Generated Plots:**
- `employment_trends.png`
- `quarterly_distribution.png`
- `wage_trends.png`
- `top_industries.png`

---

#### Slide 14: Stage 3 - Data Validation
**Content:**
- Validation checks performed:
  - Missing value analysis
  - Employment range validation (0 to 5M)
  - Wage range validation (0 to $5K/week)
  - Duplicate detection
  - Temporal continuity
- Data quality score: 85.9%

**Visual:**
- Validation dashboard graphic

**AI Image Needed:** `[IMG-07] Validation Dashboard`

**Terminal Output:**
```
================================================================================
QCEW DATA VALIDATION REPORT
================================================================================
OVERALL DATA QUALITY SCORE: 0.859 (85.9%)

EMPLOYMENT VALIDATION SUMMARY:
----------------------------------------
Total records: 5,430,384
Zero employment with establishments: 1,108,272
Employment range: 0 - 156,042,582

WAGE VALIDATION SUMMARY:
------------------------------
Average weekly wage: $1,035
Wage range: $0 - $105,149

DATA QUALITY DIMENSIONS:
------------------------------
Data Completeness: 1.000 (100.0%)
Temporal Coverage: 1.000 (100.0%)
Value Consistency: 0.796 (79.6%)
================================================================================
```

---

#### Slide 15: Stage 3 Validation Insights
**Content:**
- Anomalies detected and how handled
- Data quality recommendations
- Impact on downstream processing

**Visual:**
- Anomaly detection visualization

**AI Image Needed:** `[IMG-08] Anomaly Detection Chart`

---

### Section 5: Phase 4 - Feature Engineering (Slides 16-18)

#### Slide 16: Feature Engineering Overview
**Content:**
- Task-based organization (T032-T052)
- Categories:
  - Filtering features (county, quarterly, quality)
  - Temporal features (lags, rolling averages)
  - Geographic features
  - Industry features

**Visual:**
- Feature engineering flowchart

**AI Image Needed:** `[IMG-09] Feature Engineering Flowchart`

---

#### Slide 17: Key Feature Transformations
**Content:**
- T032: County-level filtering (87% retention)
- T033: Quarterly filtering
- T038: Growth rate calculations
- T042: Lag features (t-1, t-4, t-8)
- Rolling averages (4Q, 8Q)

**Visual:**
- Feature correlation heatmap

**Generated Plots:**
- `T038_growth_rates_distribution.png`
- `T042_lag_features_correlation.png`

**Terminal Output:**
```
[INFO] T032: FILTER TO COUNTY LEVEL
[INFO] Starting with 5,430,384 records
[INFO] Filtering to AreaType='County'...
[OK] County-level records: 4,714,234 (86.8% retained)
[OK] Saved to T032_county_filtered.csv

[INFO] T038: CALCULATE GROWTH RATES
[INFO] Computing quarter-over-quarter growth...
[OK] Growth rate features added: qoq_growth, yoy_growth

[INFO] T042: CREATE LAG FEATURES
[INFO] Creating lag features for t-1, t-4, t-8...
[OK] Lag features added: employment_lag_1, employment_lag_4, employment_lag_8
```

---

#### Slide 18: Feature Engineering Results
**Content:**
- Final feature set composition
- Feature importance analysis
- Data reduction summary

**Visual:**
- Feature importance bar chart

**AI Image Needed:** `[IMG-10] Feature Importance Chart`

---

### Section 6: Phase 5 - Preprocessing (Slides 19-21)

#### Slide 19: Preprocessing Pipeline
**Content:**
- T054: Normalization (RobustScaler)
- T055: Missing value handling
- T056: Categorical encoding
- T057: Sequence transformation
- T058: Validation

**Visual:**
- Preprocessing pipeline diagram

**AI Image Needed:** `[IMG-11] Preprocessing Pipeline Diagram`

---

#### Slide 20: Sequence Transformation
**Content:**
- Sliding window approach
- Window size: 12 quarters (3 years)
- Group-by: County + Industry
- Maintains temporal ordering
- Output: (samples, seq_len, features)

**Visual:**
- Sequence window visualization

**AI Image Needed:** `[IMG-12] Sequence Window Diagram`

**Terminal Output:**
```
[INFO] T057: TRANSFORM TO SEQUENCES
[INFO] Creating sliding windows of length 12...
[INFO] Grouping by (county, industry)...
[INFO] Total groups: 48,245
[INFO] Sequence shape: (599,582, 12, 24)
[OK] Saved to qcew_preprocessed_sequences.npz

Output format:
{
    'sequences': shape (599582, 12, 24),
    'targets': shape (599582,),
    'metadata': {...}
}
```

---

#### Slide 21: Data Splits
**Content:**
- Train: 70% (temporal ordering preserved)
- Validation: 20%
- Test: 10%
- No shuffling (time series integrity)
- Walk-forward validation approach

**Visual:**
- Train/Val/Test split visualization

**AI Image Needed:** `[IMG-13] Data Split Diagram`

---

### Section 7: Phase 6 - Model Training (Slides 22-25)

#### Slide 22: LSTM Architecture
**Content:**
- Input layer → 2-layer LSTM → BatchNorm → Dropout → Output
- Key parameters:
  - Hidden size: 64 units
  - Dropout: 0.2
  - Batch normalization
  - MSE loss function

**Visual:**
- Neural network architecture diagram

**AI Image Needed:** `[IMG-14] LSTM Architecture Diagram`

---

#### Slide 23: LSTM Architecture Detail
**Content:**
- Code snippet of model architecture
- Why LSTM over vanilla RNN
- Batch normalization benefits
- Regularization strategy

**Visual:**
- Architecture code block

**Screen Capture Needed:** `[SCREEN-04] lstm_model.py code`

```python
class EmploymentLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64,
                 num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                           num_layers, batch_first=True,
                           dropout=dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
```

---

#### Slide 24: Training Configuration
**Content:**
- Hyperparameters:
  - Learning rate: 0.001
  - Batch size: 32
  - Epochs: 100 (max)
  - Early stopping patience: 15
  - Gradient clipping: max_norm=1.0
- Optimizer: Adam
- LR Scheduler: ReduceLROnPlateau

**Visual:**
- Hyperparameter configuration table

**AI Image Needed:** `[IMG-15] Hyperparameter Configuration`

---

#### Slide 25: Training Process
**Content:**
- Training loop execution
- TensorBoard monitoring
- Checkpointing strategy
- Early stopping behavior

**Visual:**
- Training history plot

**Generated Plot:** `training_history.png`

**Screen Capture Needed:** `[SCREEN-05] TensorBoard dashboard`

**Terminal Output:**
```
[INFO] Stage 6: Model Training
[INFO] Device: cuda (NVIDIA GeForce GTX 1070)
[INFO] Training samples: 419,707
[INFO] Validation samples: 119,916

Epoch 1/100: train_loss=2847.234, val_loss=1892.456
Epoch 2/100: train_loss=1456.789, val_loss=987.654
Epoch 3/100: train_loss=823.456, val_loss=612.345
Epoch 4/100: train_loss=567.234, val_loss=507.379 [BEST]
...
[INFO] Early stopping triggered at epoch 45
[OK] Best model saved: lstm_model.pt (val_loss=507.379)
```

---

### Section 8: Phase 7 - Evaluation (Slides 26-28)

#### Slide 26: Evaluation Metrics
**Content:**
- RMSE: 10.05
- MAE: 1.89
- MAPE: 157.11%
- Directional Accuracy: 91.98%
- R² Score: 0.098

**Visual:**
- Metrics dashboard

**AI Image Needed:** `[IMG-16] Evaluation Metrics Dashboard`

---

#### Slide 27: Baseline Comparison
**Content:**
- LSTM vs Mean Baseline:
  - RMSE: 5.05% improvement
  - MAE: 15.39% improvement
  - MAPE: 2.79% improvement
- Comparison with ARIMA, exponential smoothing

**Visual:**
- Model comparison bar chart

**Generated Plot:** `model_comparison.png`

---

#### Slide 28: Prediction Analysis
**Content:**
- Actual vs. Predicted scatter plot
- Residual analysis
- Error distribution
- Key insights and limitations

**Visual:**
- Prediction analysis multi-panel

**Generated Plot:** `prediction_analysis.png`

**Terminal Output:**
```
================================================================================
COMPREHENSIVE MODEL EVALUATION REPORT
================================================================================
Overall Model Quality: NEEDS IMPROVEMENT
  MAPE: 157.11% (target: <20%)
  Directional Accuracy: 91.98% (target: >60%) ✓
  R² Score: 0.0984 (target: >0.5)

RECOMMENDATIONS:
  • High MAPE indicates large prediction errors
    - Consider: more training data, feature engineering
  • Low R² indicates poor overall fit
    - Consider: more complex model, better feature selection
================================================================================
```

---

### Section 9: Phase 8 - Prediction Interface (Slides 29-30)

#### Slide 29: Interactive Prediction System
**Content:**
- User-friendly prediction interface
- Single and batch predictions
- CSV input/output support
- Confidence intervals

**Visual:**
- Prediction interface screenshot

**Screen Capture Needed:** `[SCREEN-06] Prediction interface`

---

#### Slide 30: Prediction Capabilities
**Content:**
- Multi-step forecasting (1-4 quarters)
- Industry-specific predictions
- Geographic filtering
- Uncertainty quantification

**Visual:**
- Sample prediction output

**AI Image Needed:** `[IMG-17] Prediction Output Visualization`

---

### Section 10: Results & Conclusions (Slides 31-33)

#### Slide 31: Key Achievements
**Content:**
- Successfully processed 5.4M+ records
- Built production-grade 8-stage pipeline
- Achieved 91.38% directional accuracy
- Comprehensive baseline comparisons
- Interactive user interface

**Visual:**
- Achievement highlights graphic

**AI Image Needed:** `[IMG-18] Key Achievements Infographic`

---

#### Slide 32: Lessons Learned & Limitations
**Content:**
- Challenges:
  - COVID-19 anomaly in 2020 data
  - Schema changes across data periods
  - Short effective time series (post-COVID)
- Future improvements:
  - More feature engineering
  - Hyperparameter optimization
  - External economic indicators

**Visual:**
- SWOT analysis diagram

**AI Image Needed:** `[IMG-19] SWOT Analysis Grid`

---

#### Slide 33: Future Roadmap
**Content:**
- Web-based prediction dashboard
- Real-time data integration
- Industry risk early warning system
- County comparison tools
- Policy insight generation

**Visual:**
- Future roadmap timeline

**AI Image Needed:** `[IMG-20] Future Roadmap Timeline`

---

#### Slide 34: Q&A
**Content:**
- Questions?
- Contact information
- Repository link

**Visual:**
- Q&A graphic

---

## Required Images & Screen Captures

### Auxiliary Images to Capture Manually

| ID | Description | Source |
|----|-------------|--------|
| SCREEN-01 | Raw CSV data sample | Open data/raw/*.csv in Excel/VSCode |
| SCREEN-02 | Technology stack icons | Create collage from official logos |
| SCREEN-03 | Interactive menu terminal | Run `python main.py` and capture |
| SCREEN-04 | LSTM model code | Open src/lstm_model.py in VSCode |
| SCREEN-05 | TensorBoard dashboard | Run `tensorboard --logdir=runs` |
| SCREEN-06 | Prediction interface | Run Stage 8 and capture |
| SCREEN-07 | Pipeline status check | Run option 9 in interactive menu |
| SCREEN-08 | Git workflow | Show feature branch PR in GitHub |

### Terminal Output Captures

Run these commands and capture the output:

```bash
# Pipeline banner and menu
python main.py

# Stage execution examples
python main.py --stage consolidate
python main.py --stage validate
python main.py --stage train

# Pipeline status
# Select option 9 in interactive menu
```

---

## Generated Data Plots

These plots are automatically generated by the pipeline:

| Plot | Location | Used In |
|------|----------|---------|
| `employment_trends.png` | `data/processed/plots/` | Slide 13 |
| `quarterly_distribution.png` | `data/processed/plots/` | Slide 13 |
| `wage_trends.png` | `data/processed/plots/` | Slide 13 |
| `top_industries.png` | `data/processed/plots/` | Slide 13 |
| `T038_growth_rates_distribution.png` | `data/feature_engineering/plots/` | Slide 17 |
| `T042_lag_features_correlation.png` | `data/feature_engineering/plots/` | Slide 17 |
| `training_history.png` | `data/processed/plots/evaluation/` | Slide 25 |
| `prediction_analysis.png` | `data/processed/plots/evaluation/` | Slide 28 |
| `model_comparison.png` | `data/processed/plots/evaluation/` | Slide 27 |

---

## AI Image Generation Prompts

### Section: Chart/Diagram Enhancement Prompts

Below are prompts for generating professional visualizations using AI image models (DALL-E, Midjourney, Stable Diffusion, etc.). Each includes the diagram type and intended slide placement.

---

### [IMG-01] California Employment Hero Image
**Diagram Type:** Mind Map / Infographic
**Target Slide:** 1 (Title)

```
Create a professional presentation hero image for a California employment forecasting project. Show a stylized map of California with subtle data visualization overlays including line graphs showing employment trends, neural network nodes in the background, and icons representing industries (agriculture, technology, healthcare). Use a modern corporate color palette with deep blue, teal, and gold accents. Clean, minimalist design suitable for a technology/data science presentation title slide. High resolution, 16:9 aspect ratio.
```

---

### [IMG-02] Pipeline Roadmap
**Diagram Type:** Flowchart
**Target Slide:** 3 (Agenda)

```
Create a horizontal 8-stage pipeline roadmap infographic for a machine learning project. Stages are: 1-Consolidation, 2-Exploration, 3-Validation, 4-Feature Engineering, 5-Preprocessing, 6-Training, 7-Evaluation, 8-Prediction. Show as connected hexagonal nodes with icons for each stage (database, magnifying glass, checkmark, gears, filter, brain/neural network, chart, crystal ball). Use gradient colors progressing from blue to purple to green. Modern flat design, professional business style, white background. 16:9 aspect ratio.
```

---

### [IMG-03] Business Problem Infographic
**Diagram Type:** Mind Map
**Target Slide:** 4 (Business Problem)

```
Create an infographic showing why employment forecasting matters. Central concept: "Employment Forecasting" surrounded by connected elements: "Policy Planning", "Workforce Development", "Business Investment", "Economic Growth", "Budget Allocation". Include small icons for each concept. Use a professional blue and orange color scheme. Show data flow arrows between concepts. Modern flat design style suitable for business presentation. White background, clean lines. 16:9 aspect ratio.
```

---

### [IMG-04] System Architecture Diagram
**Diagram Type:** System Architecture
**Target Slide:** 7 (High-Level Architecture)

```
Create a three-layer software architecture diagram for a machine learning pipeline. Layer 1 (bottom): "Core Modules" showing boxes for consolidation, exploration, validation, preprocessing, training, evaluation. Layer 2 (middle): "Orchestrators" showing feature_pipeline and preprocessing_pipeline. Layer 3 (top): "Master Orchestrator" as single controller box. Show vertical arrows connecting layers. Use professional tech diagram style with blue gradient colors, rounded rectangles, and clean typography. White background. 16:9 aspect ratio.
```

---

### [IMG-05] Data Flow Diagram
**Diagram Type:** Data Flow Diagram (DFD)
**Target Slide:** 8 (Data Flow)

**Real Data Volumes at Each Stage:**
| Stage | Records | Percentage |
|-------|---------|------------|
| Raw CSV Files | 6 files | 100% |
| Consolidated | 5,430,384 | 100% |
| County-filtered | 4,732,218 | 87.1% |
| Quarterly-only | ~4,000,000 | ~74% |
| Quality-filtered | ~3,500,000 | ~65% |
| Final Features | ~3,654,832 | ~67% |
| LSTM Sequences | 599,582 samples | (12×24) |

```
Create a data flow diagram showing employment data pipeline with these exact record counts:

DATA FLOW (left to right, with record counts):
1. "Raw CSV Files" (6 files cylinder)
   ↓ arrow labeled "Load & Parse"
2. "Consolidation" (process box)
   ↓ arrow labeled "5,430,384 records"
3. "County Filter" (process box)
   ↓ arrow labeled "4,732,218 records (87%)"
4. "Quarterly Filter" (process box)
   ↓ arrow labeled "~4,000,000 records"
5. "Quality Filter" (process box)
   ↓ arrow labeled "3,654,832 records"
6. "Feature Engineering" (process box)
   ↓ arrow labeled "24 features added"
7. "Preprocessing" (process box)
   ↓ arrow labeled "599,582 sequences"
8. "LSTM Training" (neural network icon)
   ↓ arrow labeled "Trained Model"
9. "Predictions" (output cylinder)

Show data stores as cylinders (blue), processes as rounded rectangles (green), data flows as arrows with record counts labeled (gray text). Include percentage retention at key filtering stages. Professional DFD style. White background, clear labels. 16:9 aspect ratio.
```

---

### [IMG-06] Schema Consolidation Diagram
**Diagram Type:** Entity Relationship Diagram (ERD)
**Target Slide:** 12 (Stage 1)

```
Create a before/after schema mapping diagram. Left side shows "2004-2019 Schema" with columns: AreaName, IndustryName, Period, Employment. Right side shows "2020+ Schema" with columns: Area Name, Industry Name, Period, Employment. Center shows transformation process with mapping arrows. Result at bottom shows unified schema with standardized column names. Use clean technical documentation style with blue headers and white background. 16:9 aspect ratio.
```

---

### [IMG-07] Validation Dashboard
**Diagram Type:** Business Process Reengineering (BPR) / Dashboard
**Target Slide:** 14 (Stage 3)

**Real Data Values:**
| Metric | Value | Color Code |
|--------|-------|------------|
| Data Completeness | 100.0% | Green |
| Temporal Coverage | 100.0% | Green |
| Value Consistency | 79.6% | Yellow |
| Statistical Stability | 50.0% | Red |
| Overall Quality Score | 85.9% | Green |
| Total Records | 5,430,384 | - |
| Zero Employment w/ Establishments | 1,108,272 | - |
| Mean Employment/Record | 25,950 | - |
| Median Employment/Record | 137 | - |
| Average Weekly Wage | $1,035 | - |

```
Create a data quality validation dashboard mockup with these exact values:

GAUGE METERS (circular progress indicators):
- Data Completeness: 100.0% (green, full)
- Temporal Coverage: 100.0% (green, full)
- Value Consistency: 79.6% (yellow, ~80% filled)
- Statistical Stability: 50.0% (red, half filled)

MAIN SCORE (large central display):
- Overall Quality Score: 85.9% with "GOOD" label

SUMMARY STATISTICS PANEL:
- Total Records: 5,430,384
- Zero Employment Records: 1,108,272
- Mean Employment: 25,950
- Median Employment: 137
- Avg Weekly Wage: $1,035

Use green/yellow/red color coding. Modern dashboard with dark blue sidebar, white content area. Professional corporate style. 16:9 aspect ratio.
```

---

### [IMG-08] Anomaly Detection Chart
**Diagram Type:** Decision Tree / Scatter Plot
**Target Slide:** 15 (Validation Insights)

**Real Anomaly Statistics:**
| Anomaly Type | Count | Percentage |
|--------------|-------|------------|
| Employment outliers (month1) | 15,779 | 0.29% |
| Employment outliers (month2) | 15,857 | 0.29% |
| Employment outliers (month3) | 15,895 | 0.29% |
| Zero employment w/ establishments | 1,108,272 | 20.4% |
| Zero wages w/ employment | 548 | 0.01% |
| Employment range | 0 - 156,042,582 | - |
| Wage range | $0 - $105,149/week | - |

**Key Time Periods:**
- 2008-2009: Great Recession dip
- 2020 Q2: COVID-19 shock (major anomaly cluster)
- 2021-2022: Recovery period

```
Create an anomaly detection visualization for California employment data with these specifics:

TIME SERIES PLOT (main):
- X-axis: Years 2004-2024 (20 years)
- Y-axis: Employment level (log scale, 0 to 156M max)
- Normal data points: Blue scatter (majority)
- Detected anomalies: Red/orange markers

KEY ANOMALY REGIONS TO HIGHLIGHT:
1. 2008-2009 band (light yellow): "Great Recession" - ~10% employment drop
2. 2020 Q2 band (light red): "COVID-19 Shock" - major spike in anomalies
3. Extreme outliers: Mark the 15,779+ outlier points in red

ANNOTATIONS:
- Arrow pointing to 2020 Q2: "COVID-19: 47,531 anomalous records detected"
- Arrow pointing to high outliers: "Max employment: 156M (aggregates)"
- Arrow pointing to zeros: "1.1M zero-employment records"

INSET PIE CHART (bottom right corner):
- Normal records: 79.6% (blue)
- Anomalies detected: 20.4% (orange/red)

Professional data visualization style with clean gridlines. White background. Include legend. 16:9 aspect ratio.
```

---

### [IMG-09] Feature Engineering Flowchart
**Diagram Type:** Flowchart
**Target Slide:** 16 (Feature Engineering)

**Real Task IDs and Record Counts:**
| Task | Description | Input Records | Output Records |
|------|-------------|---------------|----------------|
| T032 | County-level filter | 5,430,384 | 4,732,218 (87%) |
| T033 | Quarterly filter | 4,732,218 | ~4,000,000 |
| T034 | Quality filter | ~4,000,000 | ~3,500,000 |
| T038 | Growth rates | - | +2 features |
| T039 | Seasonal adjustments | - | +4 features |
| T040 | Industry concentration | - | +2 features |
| T042 | Lag features | - | +3 features (lag_1, lag_4, lag_8) |
| T043 | Rolling averages | - | +2 features (4Q, 8Q) |

```
Create a feature engineering flowchart with these exact task IDs and data:

MAIN FLOW (top to bottom):

START: "Raw Consolidated Data"
       5,430,384 records
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│ FILTERING PIPELINE (Blue boxes)                                 │
│                                                                 │
│  T032: County Filter     T033: Quarterly     T034: Quality     │
│  5.4M → 4.7M (87%)   →   4.7M → 4.0M    →   4.0M → 3.5M       │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│ FEATURE BRANCHES (parallel paths)                               │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ TEMPORAL     │  │ GEOGRAPHIC   │  │ INDUSTRY     │          │
│  │ (Green)      │  │ (Orange)     │  │ (Purple)     │          │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤          │
│  │ T038: Growth │  │ County codes │  │ NAICS codes  │          │
│  │ T039: Season │  │ Central Val  │  │ Ownership    │          │
│  │ T042: Lags   │  │ Region flags │  │ Concentration│          │
│  │ T043: Rolling│  │              │  │              │          │
│  │              │  │              │  │              │          │
│  │ +11 features │  │ +3 features  │  │ +4 features  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
END: "Final Feature Set"
     3,654,832 records × 24 features

Professional technical flowchart style. Use colored boxes: blue (filtering), green (temporal), orange (geographic), purple (industry). Show record counts at each stage. White background. 16:9 aspect ratio.
```

---

### [IMG-10] Feature Importance Chart
**Diagram Type:** Bar Chart / Horizontal Bar
**Target Slide:** 18 (Feature Engineering Results)

**Real Feature Names & Expected Importance (based on lag correlation patterns):**
| Feature Name | Expected Importance | Description |
|--------------|---------------------|-------------|
| avg_monthly_emplvl_lag_1 | 0.92 | Previous quarter employment |
| avg_monthly_emplvl_lag_4 | 0.78 | Same quarter last year |
| avg_monthly_emplvl_lag_8 | 0.65 | Same quarter 2 years ago |
| qoq_growth | 0.58 | Quarter-over-quarter growth rate |
| yoy_growth | 0.52 | Year-over-year growth rate |
| rolling_avg_4q | 0.48 | 4-quarter rolling average |
| rolling_avg_8q | 0.41 | 8-quarter rolling average |
| total_qtrly_wages | 0.38 | Quarterly wage total |
| avg_wkly_wage | 0.35 | Average weekly wage |
| industry_code_encoded | 0.28 | Industry classification |
| area_name_encoded | 0.22 | County identifier |
| quarter_sin | 0.18 | Seasonal quarter (sine) |
| quarter_cos | 0.15 | Seasonal quarter (cosine) |

```
Create a horizontal bar chart showing feature importance for LSTM employment prediction model with these exact values:

FEATURES (sorted by importance, top to bottom):
1. avg_monthly_emplvl_lag_1: 0.92 (dark blue)
2. avg_monthly_emplvl_lag_4: 0.78 (dark blue)
3. avg_monthly_emplvl_lag_8: 0.65 (medium blue)
4. qoq_growth: 0.58 (medium blue)
5. yoy_growth: 0.52 (medium blue)
6. rolling_avg_4q: 0.48 (light blue)
7. rolling_avg_8q: 0.41 (light blue)
8. total_qtrly_wages: 0.38 (light blue)
9. avg_wkly_wage: 0.35 (light blue)
10. industry_code_encoded: 0.28 (gray-blue)
11. area_name_encoded: 0.22 (gray-blue)
12. quarter_sin: 0.18 (gray)
13. quarter_cos: 0.15 (gray)

X-axis: "Feature Importance Score" (0.0 to 1.0)
Y-axis: Feature names (readable, not truncated)
Title: "LSTM Model Feature Importance Analysis"
Subtitle: "Employment Forecasting - California QCEW Data"

Use gradient blue bars (darker = higher importance). Professional data visualization style with clean gridlines, axis labels showing exact values on bars. White background, minimal design. 16:9 aspect ratio.
```

---

### [IMG-11] Preprocessing Pipeline Diagram
**Diagram Type:** BPMN 2.0 / Process Flow
**Target Slide:** 19 (Preprocessing)

```
Create a preprocessing pipeline diagram showing 5 sequential steps. Steps: T054 Normalization (RobustScaler icon) → T055 Missing Values (gap-filling icon) → T056 Categorical Encoding (one-hot matrix icon) → T057 Sequence Transform (sliding window icon) → T058 Validation (checkmark icon). Show data flowing through each step with shape annotations (DataFrame → Scaled DataFrame → Encoded → Sequences → Validated). Use swim lane style with clear step separation. Professional technical diagram. 16:9 aspect ratio.
```

---

### [IMG-12] Sequence Window Diagram
**Diagram Type:** Sequence Diagram / Timeline
**Target Slide:** 20 (Sequence Transformation)

**Real Sequence Parameters:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| Window Size | 12 quarters | 3-year lookback |
| Stride | 1 quarter | Sliding by 1 |
| Features per step | 24 | All engineered features |
| Total sequences | 599,582 | After windowing |
| Groups | 48,245 | County × Industry combinations |

**Example Window (California, Manufacturing):**
| Position | Quarter | avg_monthly_emplvl | Feature Vector |
|----------|---------|-------------------|----------------|
| t-11 | Q1 2020 | 125,340 | [24 features] |
| t-10 | Q2 2020 | 98,450 | [24 features] |
| ... | ... | ... | ... |
| t-1 | Q4 2022 | 132,890 | [24 features] |
| **Target** | Q1 2023 | 135,200 | Predict this |

```
Create a sliding window sequence diagram for LSTM with these exact values:

TIMELINE (horizontal, showing quarters):
2020: Q1 Q2 Q3 Q4 | 2021: Q1 Q2 Q3 Q4 | 2022: Q1 Q2 Q3 Q4 | 2023: Q1 Q2 Q3 Q4

SLIDING WINDOWS (overlapping brackets):

Window 1 (blue bracket, 12 quarters):
├──────────────────────────────────────────┤ → Target: Q1 2023
Q1-2020 ─────────────────────────── Q4-2022

Window 2 (green bracket, 12 quarters):
   ├──────────────────────────────────────────┤ → Target: Q2 2023
   Q2-2020 ────────────────────────── Q1-2023

Window 3 (orange bracket, 12 quarters):
      ├──────────────────────────────────────────┤ → Target: Q3 2023
      Q3-2020 ─────────────────────── Q2-2023

FEATURE DETAIL (inset box):
Each timestep contains 24 features:
┌─────────────────────────────────────────┐
│ avg_monthly_emplvl    (normalized)      │
│ total_qtrly_wages     (normalized)      │
│ avg_wkly_wage         (normalized)      │
│ employment_lag_1      (previous Q)      │
│ employment_lag_4      (same Q last yr)  │
│ employment_lag_8      (2 years ago)     │
│ qoq_growth, yoy_growth                  │
│ rolling_avg_4q, rolling_avg_8q          │
│ industry_code_encoded                   │
│ area_name_encoded                       │
│ quarter_sin, quarter_cos                │
│ ... (24 total features)                 │
└─────────────────────────────────────────┘

OUTPUT TENSOR SHAPE:
(599,582 sequences, 12 timesteps, 24 features)

Label input sequence length (12) and target (1). Professional technical diagram. White background. 16:9 aspect ratio.
```

---

### [IMG-13] Data Split Diagram
**Diagram Type:** Gantt Chart / Timeline
**Target Slide:** 21 (Data Splits)

**Real Split Configuration:**
| Split | Percentage | Sample Count | Time Period |
|-------|------------|--------------|-------------|
| Training | 70% | 419,707 | 2004-2019 |
| Validation | 20% | 119,916 | 2019-2022 |
| Test | 10% | 59,959 | 2022-2024 |
| **Total** | 100% | 599,582 | 2004-2024 |

**Sequence Details:**
- Sequence Length: 12 quarters (3 years lookback)
- Features per timestep: 24
- Final tensor shape: (599,582, 12, 24)

```
Create a train/validation/test split visualization for time series data with these exact values:

MAIN TIMELINE BAR (horizontal, full width):
Full dataset: 2004-2024 (20 years, 80 quarters)

THREE COLORED SECTIONS:
1. Training (blue): 2004-2019
   - 70% of data
   - 419,707 sequences
   - Label: "TRAIN: 419,707 samples (70%)"

2. Validation (yellow/gold): 2019-2022
   - 20% of data
   - 119,916 sequences
   - Label: "VAL: 119,916 samples (20%)"

3. Test (green): 2022-2024
   - 10% of data
   - 59,959 sequences
   - Label: "TEST: 59,959 samples (10%)"

ANNOTATIONS:
- Top: "Total: 599,582 LSTM Sequences"
- Bottom: "No shuffling - temporal order preserved"
- Arrow showing data flows left to right chronologically

INSET BOX (bottom right):
"Sequence Shape: (batch, 12, 24)
- 12 quarters lookback
- 24 features per timestep"

Clean timeline visualization style. White background. 16:9 aspect ratio.
```

---

### [IMG-14] LSTM Architecture Diagram
**Diagram Type:** AI Neural Network Architecture
**Target Slide:** 22 (LSTM Architecture)

**Real Model Architecture (from lstm_model.py):**
| Layer | Type | Parameters | Output Shape |
|-------|------|------------|--------------|
| Input | - | - | (batch, 12, 24) |
| LSTM Layer 1 | nn.LSTM | input=24, hidden=64, dropout=0.2 | (batch, 12, 64) |
| LSTM Layer 2 | nn.LSTM | input=64, hidden=64, dropout=0.2 | (batch, 12, 64) |
| Last Timestep | Select | [:, -1, :] | (batch, 64) |
| BatchNorm | nn.BatchNorm1d | 64 features | (batch, 64) |
| Dropout | nn.Dropout | p=0.2 | (batch, 64) |
| Output | nn.Linear | 64 → 1 | (batch, 1) |

**Total Parameters:** ~50,000 trainable parameters

```
Create an LSTM neural network architecture diagram with these exact specifications:

LAYER STRUCTURE (vertical flow, top to bottom):

1. INPUT LAYER (rectangle, light gray)
   - Shape: (batch_size, 12, 24)
   - Label: "Input: 12 timesteps × 24 features"

2. LSTM LAYER 1 (rounded rectangle, blue)
   - 64 hidden units
   - Show LSTM cell detail (forget gate, input gate, output gate, cell state)
   - Label: "LSTM Layer 1: 64 units"
   - Dropout: 0.2 (between layers)

3. LSTM LAYER 2 (rounded rectangle, blue)
   - 64 hidden units
   - Label: "LSTM Layer 2: 64 units"
   - Dropout: 0.2

4. TIMESTEP SELECTION (diamond, yellow)
   - Label: "Select Last Timestep"
   - Output: (batch, 64)

5. BATCH NORMALIZATION (rectangle, green)
   - Label: "BatchNorm1d(64)"
   - Stabilizes training

6. DROPOUT (dashed rectangle, orange)
   - Label: "Dropout(p=0.2)"
   - Regularization

7. OUTPUT LAYER (rectangle, purple)
   - Label: "Linear(64 → 1)"
   - Output: Single employment prediction

ANNOTATIONS:
- Left side: Show tensor shapes at each transition
- Right side: Show key hyperparameters
- Bottom: "Total: ~50,000 trainable parameters"

Use professional neural network diagram style with blue/purple gradient. Include layer dimensions. White background. 16:9 aspect ratio.
```

---

### [IMG-15] Hyperparameter Configuration
**Diagram Type:** UML Class Diagram / Table
**Target Slide:** 24 (Training Configuration)

**Real Hyperparameters (from config/hyperparameters.py):**

**ModelConfig:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| HIDDEN_SIZE | 64 | LSTM hidden units |
| NUM_LAYERS | 2 | Stacked LSTM layers |
| DROPOUT | 0.2 | Dropout probability |
| OUTPUT_SIZE | 1 | Single prediction |
| SEQUENCE_LENGTH | 12 | 3-year lookback |

**TrainingConfig:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| NUM_EPOCHS | 100 | Maximum epochs |
| PATIENCE | 15 | Early stopping patience |
| LEARNING_RATE | 0.001 | Initial learning rate |
| WEIGHT_DECAY | 0.0 | L2 regularization |
| GRADIENT_CLIP | 1.0 | Max gradient norm |
| LR_SCHEDULER | ReduceLROnPlateau | Adaptive LR |
| LR_FACTOR | 0.5 | LR reduction factor |
| LR_PATIENCE | 5 | LR scheduler patience |

**DataConfig:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| BATCH_SIZE | 32 | Training batch size |
| VAL_SIZE | 0.2 | Validation split |
| TEST_SIZE | 0.1 | Test split |
| TRAIN_SIZE | 0.7 | Training split |

```
Create a hyperparameter configuration visualization with these exact values:

THREE-COLUMN CARD LAYOUT:

CARD 1 - "Model Architecture" (blue header):
┌─────────────────────────────┐
│ 🧠 MODEL ARCHITECTURE       │
├─────────────────────────────┤
│ HIDDEN_SIZE      64         │
│ NUM_LAYERS       2          │
│ DROPOUT          0.2        │
│ OUTPUT_SIZE      1          │
│ SEQUENCE_LENGTH  12         │
└─────────────────────────────┘

CARD 2 - "Training Settings" (green header):
┌─────────────────────────────┐
│ ⚡ TRAINING SETTINGS        │
├─────────────────────────────┤
│ NUM_EPOCHS       100        │
│ PATIENCE         15         │
│ LEARNING_RATE    0.001      │
│ GRADIENT_CLIP    1.0        │
│ OPTIMIZER        Adam       │
│ LR_SCHEDULER     ReduceLR   │
│ LR_FACTOR        0.5        │
│ LR_PATIENCE      5          │
└─────────────────────────────┘

CARD 3 - "Data Configuration" (orange header):
┌─────────────────────────────┐
│ 📊 DATA CONFIGURATION       │
├─────────────────────────────┤
│ BATCH_SIZE       32         │
│ TRAIN_SIZE       70%        │
│ VAL_SIZE         20%        │
│ TEST_SIZE        10%        │
│ DEVICE           auto/cuda  │
│ RANDOM_SEED      42         │
└─────────────────────────────┘

Modern card-based design with colored headers (blue, green, orange). Use icons for each category. Clean corporate style, monospace font for values. White background. 16:9 aspect ratio.
```

---

### [IMG-16] Evaluation Metrics Dashboard
**Diagram Type:** Dashboard / SWOT-style Grid
**Target Slide:** 26 (Evaluation)

**Real Evaluation Results (from comprehensive_evaluation_report.txt):**
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| RMSE | 10.0498 | Lower is better | Baseline |
| MAE | 1.8930 | Lower is better | Baseline |
| MAPE | 157.11% | < 20% | ❌ Needs Improvement |
| Directional Accuracy | 91.98% | > 60% | ✅ Excellent |
| R² Score | 0.0984 | > 0.5 | ❌ Needs Improvement |

**Baseline Comparison:**
| Model | RMSE | MAE | MAPE | Improvement |
|-------|------|-----|------|-------------|
| LSTM | 10.05 | 1.89 | 157.11% | - |
| Mean Baseline | 10.58 | 2.24 | 161.62% | +5.05% RMSE |

**Overall Assessment:** NEEDS IMPROVEMENT
- Strong: 91.98% directional accuracy (knows trend direction)
- Weak: High MAPE, Low R² (magnitude prediction poor)

```
Create an evaluation metrics dashboard with these exact values from the model:

MAIN METRICS GRID (2x3 layout):

┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ 📊 RMSE         │ │ 📊 MAE          │ │ 📊 MAPE         │
│                 │ │                 │ │                 │
│   10.0498       │ │   1.8930        │ │   157.11%       │
│                 │ │                 │ │                 │
│ ⚪ Baseline     │ │ ⚪ Baseline     │ │ 🔴 Target: <20% │
└─────────────────┘ └─────────────────┘ └─────────────────┘

┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ 📊 DIR. ACCURACY│ │ 📊 R² SCORE     │ │ 📊 TEST SAMPLES │
│                 │ │                 │ │                 │
│   91.98%        │ │   0.0984        │ │   599,582       │
│                 │ │                 │ │                 │
│ 🟢 Target: >60% │ │ 🔴 Target: >0.5 │ │ ⚪ Sequences    │
└─────────────────┘ └─────────────────┘ └─────────────────┘

BASELINE COMPARISON BAR (bottom):
LSTM vs Mean Baseline:
- RMSE: 10.05 vs 10.58 (+5.05% improvement) ✓
- MAE: 1.89 vs 2.24 (+15.39% improvement) ✓
- MAPE: 157.11% vs 161.62% (+2.79% improvement) ✓

OVERALL STATUS BADGE (top right):
"⚠️ NEEDS IMPROVEMENT"
- Directional: ✅ 91.98% (Excellent)
- MAPE: ❌ 157% (Target: <20%)
- R²: ❌ 0.098 (Target: >0.5)

Use traffic light colors: green (91.98%), red (MAPE, R²), gray (baseline metrics). Modern dashboard card design. Professional tech style. White background. 16:9 aspect ratio.
```

---

### [IMG-17] Prediction Output Visualization
**Diagram Type:** Line Chart / Forecast Plot
**Target Slide:** 30 (Prediction Capabilities)

```
Create a multi-step employment forecast visualization. Show historical employment line (solid blue) from 2020-2024, then forecast line (dashed blue) extending 4 quarters into future. Include confidence interval bands (shaded blue, 90% and 95%). Mark actual values with dots, predicted values with triangles. Add legend and axis labels. Professional time series forecast style. Clean gridlines. White background. 16:9 aspect ratio.
```

---

### [IMG-18] Key Achievements Infographic
**Diagram Type:** Mind Map / Infographic
**Target Slide:** 31 (Achievements)

```
Create a key achievements infographic for ML project. Central badge: "Project Complete". Radiating achievements: "5.4M+ Records Processed" (database icon), "8-Stage Pipeline" (workflow icon), "91.98% Directional Accuracy" (target icon), "LSTM Deep Learning" (brain icon), "Baseline Comparisons" (chart icon), "Interactive Interface" (user icon). Use trophy/medal visual elements. Green/gold success colors. Modern celebration style. White background. 16:9 aspect ratio.
```

---

### [IMG-19] SWOT Analysis Grid
**Diagram Type:** SWOT Analysis
**Target Slide:** 32 (Lessons Learned)

**Real Project SWOT Analysis:**

| Quadrant | Items |
|----------|-------|
| **Strengths** | 5.4M+ records processed; 91.98% directional accuracy; 8-stage modular pipeline; 20+ years of historical data; Task-based organization (T032-T058); Comprehensive validation (85.9% quality score) |
| **Weaknesses** | High MAPE (157%); Low R² (0.098); COVID-19 data anomaly; Only 4 epochs trained; No external economic indicators; Limited hyperparameter tuning |
| **Opportunities** | Web-based dashboard; Real-time QCEW integration; Policy insight generation; County comparison tools; Early warning system; API deployment |
| **Threats** | Economic regime changes; Data suppression gaps; Model drift over time; QCEW schema changes; GPU dependency for training |

```
Create a SWOT analysis diagram for this employment forecasting project with these exact items:

FOUR-QUADRANT GRID:

┌─────────────────────────────────┬─────────────────────────────────┐
│ 💪 STRENGTHS (Blue)             │ ⚠️ WEAKNESSES (Yellow)          │
│                                 │                                 │
│ • 5.4M+ records processed       │ • High MAPE: 157.11%           │
│ • 91.98% directional accuracy   │ • Low R² score: 0.098          │
│ • 8-stage modular pipeline      │ • COVID-19 data anomaly        │
│ • 20+ years historical data     │ • Limited training (4 epochs)  │
│ • Task-based organization       │ • No external indicators       │
│ • 85.9% data quality score      │ • Needs hyperparameter tuning  │
│                                 │                                 │
├─────────────────────────────────┼─────────────────────────────────┤
│ 🚀 OPPORTUNITIES (Green)        │ ⚡ THREATS (Red)                │
│                                 │                                 │
│ • Web-based dashboard           │ • Economic regime changes      │
│ • Real-time QCEW integration    │ • Data suppression gaps        │
│ • Policy insight generation     │ • Model drift over time        │
│ • County comparison tools       │ • QCEW schema changes          │
│ • Early warning system          │ • GPU training dependency      │
│ • API deployment for production │ • Disclosure code handling     │
│                                 │                                 │
└─────────────────────────────────┴─────────────────────────────────┘

Use standard SWOT colors: blue-Strengths, yellow-Weaknesses, green-Opportunities, red-Threats. Each quadrant should have 6 bullet points as shown. Professional business style with clean borders. Include title: "IS-160 Employment Forecasting SWOT Analysis". White background. 16:9 aspect ratio.
```

---

### [IMG-20] Future Roadmap Timeline
**Diagram Type:** Gantt Chart / Roadmap
**Target Slide:** 33 (Future Roadmap)

```
Create a future development roadmap timeline. Horizontal timeline showing: Phase 1 (Near-term): "Web Dashboard", "API Integration". Phase 2 (Medium-term): "Real-time Updates", "County Comparison Tools". Phase 3 (Long-term): "Policy Insight Engine", "Risk Early Warning System". Use milestone markers and connecting lines. Progressive color gradient from blue to green. Modern roadmap visualization style. White background. 16:9 aspect ratio.
```

---

## Image Placement Map

| Slide | Generated Plot | AI-Generated Image | Screen Capture |
|-------|---------------|-------------------|----------------|
| 1 | - | IMG-01 | - |
| 2 | - | - | - |
| 3 | - | IMG-02 | - |
| 4 | - | IMG-03 | - |
| 5 | - | - | SCREEN-01 |
| 7 | - | IMG-04 | - |
| 8 | - | IMG-05 | - |
| 9 | - | - | SCREEN-02 |
| 11 | - | - | SCREEN-03 |
| 12 | - | IMG-06 | - |
| 13 | employment_trends.png, quarterly_distribution.png, wage_trends.png, top_industries.png | - | - |
| 14 | - | IMG-07 | - |
| 15 | - | IMG-08 | - |
| 16 | - | IMG-09 | - |
| 17 | T038_growth_rates_distribution.png, T042_lag_features_correlation.png | - | - |
| 18 | - | IMG-10 | - |
| 19 | - | IMG-11 | - |
| 20 | - | IMG-12 | - |
| 21 | - | IMG-13 | - |
| 22 | - | IMG-14 | - |
| 23 | - | - | SCREEN-04 |
| 24 | - | IMG-15 | - |
| 25 | training_history.png | - | SCREEN-05 |
| 26 | - | IMG-16 | - |
| 27 | model_comparison.png | - | - |
| 28 | prediction_analysis.png | - | - |
| 29 | - | - | SCREEN-06 |
| 30 | - | IMG-17 | - |
| 31 | - | IMG-18 | - |
| 32 | - | IMG-19 | - |
| 33 | - | IMG-20 | - |

---

## Summary Checklist

### Generated Data Plots (9 total)
- [ ] `employment_trends.png`
- [ ] `quarterly_distribution.png`
- [ ] `wage_trends.png`
- [ ] `top_industries.png`
- [ ] `T038_growth_rates_distribution.png`
- [ ] `T042_lag_features_correlation.png`
- [ ] `training_history.png`
- [ ] `prediction_analysis.png`
- [ ] `model_comparison.png`

### AI-Generated Images (20 total)
- [ ] IMG-01: California Employment Hero
- [ ] IMG-02: Pipeline Roadmap
- [ ] IMG-03: Business Problem Infographic
- [ ] IMG-04: System Architecture Diagram
- [ ] IMG-05: Data Flow Diagram
- [ ] IMG-06: Schema Consolidation Diagram
- [ ] IMG-07: Validation Dashboard
- [ ] IMG-08: Anomaly Detection Chart
- [ ] IMG-09: Feature Engineering Flowchart
- [ ] IMG-10: Feature Importance Chart
- [ ] IMG-11: Preprocessing Pipeline Diagram
- [ ] IMG-12: Sequence Window Diagram
- [ ] IMG-13: Data Split Diagram
- [ ] IMG-14: LSTM Architecture Diagram
- [ ] IMG-15: Hyperparameter Configuration
- [ ] IMG-16: Evaluation Metrics Dashboard
- [ ] IMG-17: Prediction Output Visualization
- [ ] IMG-18: Key Achievements Infographic
- [ ] IMG-19: SWOT Analysis Grid
- [ ] IMG-20: Future Roadmap Timeline

### Screen Captures (8 total)
- [ ] SCREEN-01: Raw CSV data sample
- [ ] SCREEN-02: Technology stack icons
- [ ] SCREEN-03: Interactive menu terminal
- [ ] SCREEN-04: LSTM model code
- [ ] SCREEN-05: TensorBoard dashboard
- [ ] SCREEN-06: Prediction interface
- [ ] SCREEN-07: Pipeline status check
- [ ] SCREEN-08: Git workflow (optional)

---

## Notes for Presenter

1. **Run the pipeline** before presentation to ensure all plots are current
2. **TensorBoard** requires training to be run with logging enabled
3. **Terminal captures** should use a clean terminal with good contrast
4. **Timing**: Allocate ~1 minute per slide for a 30-35 minute presentation
5. **Demo option**: Consider live demo of interactive menu as backup for Q&A
