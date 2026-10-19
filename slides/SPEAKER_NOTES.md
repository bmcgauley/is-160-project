# Speaker Talking Points
## California Employment Forecasting with LSTM/RNN Deep Learning
### IS-160 Project Presentation

---

## Slide 1: Title Slide
**Duration:** ~30 seconds

### Talking Points:
- Welcome everyone to our IS-160 project presentation
- Today we're presenting our California Employment Forecasting system using LSTM deep learning
- This project processes over 5.4 million employment records from the QCEW dataset
- Our goal: predict quarterly employment trends to support economic planning and policy decisions

---

## Slide 2: Executive Summary
**Duration:** ~1 minute

### Talking Points:
- Quick overview of what we accomplished:
  - Processed 20+ years of California employment data (2004-2024)
  - Built a production-grade 8-stage data pipeline
  - Achieved **91.98% directional accuracy** in predicting employment trends
  - Compared our LSTM model against traditional baseline methods
- Key technical achievement: transformed raw government data into actionable forecasts
- Data quality score of 85.9% demonstrates robust data validation

---

## Slide 3: Agenda/Roadmap
**Duration:** ~30 seconds

### Talking Points:
- Here's our presentation roadmap - we'll walk through all 8 stages of our pipeline
- Start with the business problem and data context
- Move through data processing, feature engineering, and preprocessing
- Cover our LSTM model architecture and training process
- End with evaluation results and future directions
- Feel free to ask questions as we go

---

## Slide 4: The Business Problem
**Duration:** ~1.5 minutes

### Talking Points:
- **Why does employment forecasting matter?**
  - Policymakers need accurate forecasts for budget planning
  - Workforce agencies rely on predictions for training program allocation
  - Businesses use employment trends for expansion decisions
- **California's challenge:**
  - Largest state economy in the US (~$3.6 trillion GDP)
  - Diverse industries: tech, agriculture, healthcare, entertainment
  - 58 counties with vastly different economic profiles
- **Traditional methods vs. our approach:**
  - Traditional: ARIMA, exponential smoothing - linear assumptions
  - Our approach: LSTM neural networks - captures complex non-linear patterns

---

## Slide 5: Problem Context
**Duration:** ~1 minute

### Talking Points:
- California has unique challenges for employment forecasting:
  - Geographic spread from Silicon Valley to Central Valley agriculture
  - Seasonal patterns vary dramatically by industry
  - Economic shocks (like COVID-19) affect regions differently
- Stakeholders who benefit from better forecasts:
  - California EDD (Employment Development Department)
  - County workforce boards
  - Private sector employers for hiring decisions
  - Educational institutions for program planning

---

## Slide 6: The QCEW Dataset
**Duration:** ~1.5 minutes

### Talking Points:
- **QCEW = Quarterly Census of Employment and Wages**
  - Authoritative government data source from California EDD
  - Covers ALL employers subject to unemployment insurance
  - Approximately 99% of all wage and salary employment
- **Our data spans:**
  - 6 CSV files covering 2004-2024
  - Over 5.4 million individual records
  - All 58 California counties
  - All industry classifications (NAICS codes)
- **Key variables we use:**
  - Employment counts (our prediction target)
  - Average weekly wages
  - Establishment counts
  - Industry and geographic identifiers

---

## Slide 7: Project Objectives
**Duration:** ~1 minute

### Talking Points:
- We set four clear objectives at the start:
  1. **Process 20+ years of data** - handle schema changes, clean data, validate quality
  2. **Build production-grade pipeline** - not just a research prototype, but maintainable code
  3. **Compare methods** - LSTM vs. ARIMA vs. simple baselines
  4. **Predict with uncertainty** - not just point estimates, but confidence intervals
- Success criteria: Beat baseline models, achieve reasonable directional accuracy

---

## Slide 8: Data & Pipeline Architecture Section Header
**Duration:** ~15 seconds

### Talking Points:
- Now let's dive into how we built this system
- We'll cover our three-layer architecture and data flow

---

## Slide 9: High-Level Architecture
**Duration:** ~1.5 minutes

### Talking Points:
- **Three-layer pipeline design:**
  - **Layer 1 - Core Modules:** Individual functions that do specific tasks (filtering, calculating growth rates, etc.)
  - **Layer 2 - Orchestrators:** Coordinate related functions (feature_pipeline.py, preprocessing_pipeline.py)
  - **Layer 3 - Master Orchestrator:** QCEWPipeline class that runs the entire show
- **Why this architecture?**
  - **Modularity:** Each piece can be tested independently
  - **Testability:** We can verify each transformation
  - **Maintainability:** Easy to add new features or fix bugs
- Example: If we need to add a new feature, we add a function to Layer 1, call it from Layer 2

---

## Slide 10: Data Flow Overview
**Duration:** ~1.5 minutes

### Talking Points:
- Let me walk you through how data flows through our pipeline:
  - Start with **6 raw CSV files** from California EDD
  - **Stage 1 - Consolidation:** Merge into 5.4M unified records
  - **Stage 4 - County Filter:** Down to 4.7M (87% retained)
  - **Quarterly filtering:** Focus on quarterly aggregates
  - **Quality filtering:** Remove anomalies and incomplete records
  - **Final features:** 3.6M records with 24 engineered features
  - **LSTM sequences:** 599,582 training samples
- Key insight: We retain most data while ensuring quality - 87% retention at county filter is excellent

---

## Slide 11: Technology Stack
**Duration:** ~1 minute

### Talking Points:
- **Core framework:** Python 3.8+ with PyTorch for deep learning
- **Data processing:** pandas for data manipulation, NumPy for numerical operations
- **Machine learning:** scikit-learn for preprocessing, statsmodels for baseline ARIMA
- **Visualization:** matplotlib, seaborn for plots, TensorBoard for training monitoring
- **Development:** Git/GitHub for version control, VSCode as our IDE
- All dependencies managed via requirements.txt for reproducibility

---

## Slide 12: Project Structure
**Duration:** ~1 minute

### Talking Points:
- Quick tour of our codebase:
  - **main.py** - Entry point, supports both CLI and interactive modes
  - **src/** folder contains all pipeline modules
  - **config/** holds hyperparameters in a centralized location
  - **data/** organized into raw, processed, and feature_engineering subdirectories
- Key design principle: **Raw data is read-only** - we never modify source files
- All outputs are reproducible from the pipeline

---

## Slide 13: Interactive Menu System
**Duration:** ~1 minute

### Talking Points:
- We built a user-friendly interface for running the pipeline
- Users can:
  - Run individual stages (useful for debugging)
  - Run the full pipeline end-to-end
  - Check pipeline status
- Status tracking shows which stages have completed
- This makes it easy for anyone to reproduce our results

---

## Slide 14: Phase 1-3 Data Pipeline Section Header
**Duration:** ~15 seconds

### Talking Points:
- Now let's look at the first three stages: Consolidation, Exploration, and Validation

---

## Slide 15: Stage 1 - Data Consolidation
**Duration:** ~1.5 minutes

### Talking Points:
- **The challenge:** Schema changed between 2004-2019 and 2020+ data
  - Old format: 'AreaName', 'IndustryName' (no spaces)
  - New format: 'Area Name', 'Industry Name' (with spaces)
- **Our solution:** Automatic schema detection and normalization
  - Detects which format each file uses
  - Maps columns to a unified schema
  - Preserves all original data
- **Output:** Single consolidated CSV with 5.4M+ records
- **Data protection:** Raw files are NEVER modified - always work with copies

---

## Slide 16: Stage 1 - Terminal Output
**Duration:** ~30 seconds

### Talking Points:
- This is actual terminal output from running consolidation
- Each file processed with record count logged
- Final consolidation confirmed: 5.4 million records
- Output saved to qcew_master_consolidated.csv

---

## Slides 17-18: Stage 2 - Data Exploration
**Duration:** ~1.5 minutes

### Talking Points:
- **Exploratory Data Analysis (EDA)** - understanding our data before modeling
- **Key insights discovered:**
  - **Employment trends:** Clear growth pattern from 2004-2019, then COVID shock
  - **Quarterly patterns:** Strong seasonality (Q4 typically highest for retail)
  - **Top industries:** Healthcare, retail, and professional services dominate
  - **Wage distribution:** Highly right-skewed (most wages modest, some very high)
- Generated plots saved to data/processed/plots/ for reference
- This exploration informed our feature engineering decisions

---

## Slides 19-20: Stage 3 - Data Validation
**Duration:** ~1.5 minutes

### Talking Points:
- **Validation checks performed:**
  - Missing value analysis - 100% completeness achieved
  - Employment range validation (0 to 5M reasonable, flagged outliers)
  - Wage range validation ($0 to $5K/week reasonable)
  - Duplicate detection
  - Temporal continuity - are all quarters present?
- **Overall quality score: 85.9%**
  - Data Completeness: 100%
  - Temporal Coverage: 100%
  - Value Consistency: 79.6% (some outliers present)
- This gave us confidence to proceed with modeling

---

## Slide 21: Stage 3 - Validation Insights
**Duration:** ~1 minute

### Talking Points:
- **Anomalies detected:**
  - 1.1 million records with zero employment but existing establishments
  - COVID-19 Q2 2020 created a cluster of anomalies
  - Extreme outliers in both high and low directions
- **How we handled them:**
  - Quality filters remove records outside reasonable bounds
  - COVID period flagged but retained (important signal)
  - Recommendations documented for future data collection

---

## Slide 22: Feature Engineering Section Header
**Duration:** ~15 seconds

### Talking Points:
- Phase 4 is where we transform raw data into predictive features

---

## Slide 23: Feature Engineering Overview
**Duration:** ~1 minute

### Talking Points:
- **Task-based organization (T032-T052):**
  - Each transformation has a unique task ID for traceability
  - Outputs saved at each step for debugging
- **Four categories of features:**
  1. **Filtering features:** County, quarterly, quality filters
  2. **Temporal features:** Lags, rolling averages, growth rates
  3. **Geographic features:** County encodings, regional groupings
  4. **Industry features:** NAICS codes, industry concentration

---

## Slides 24-29: Key Feature Transformations
**Duration:** ~2 minutes

### Talking Points:
- **T032 - County-level filtering:**
  - Removed statewide aggregates to focus on county granularity
  - Retained 87% of data (4.7M records)
- **T033 - Quarterly filtering:**
  - Focused on quarterly periods (not monthly)
  - Aligns with QCEW reporting structure
- **T038 - Growth rate calculations:**
  - Quarter-over-quarter (QoQ) growth
  - Year-over-year (YoY) growth
  - These capture momentum in employment changes
- **T042 - Lag features:**
  - Employment at t-1 (previous quarter)
  - Employment at t-4 (same quarter last year)
  - Employment at t-8 (same quarter two years ago)
  - Most predictive features in our model
- **Rolling averages (4Q and 8Q):**
  - Smooth out short-term noise
  - Capture underlying trends

---

## Slide 30: Feature Engineering Results
**Duration:** ~1 minute

### Talking Points:
- **Final feature set: 24 features**
  - Lag features most important (0.92, 0.78, 0.65 importance)
  - Growth rates next (0.58, 0.52)
  - Rolling averages provide trend context
  - Categorical encodings for industry and geography
- **Data reduction summary:**
  - Started: 5.4M records
  - After filtering: 3.6M records
  - Ready for LSTM: 599,582 sequences

---

## Slide 31: Preprocessing Section Header
**Duration:** ~15 seconds

### Talking Points:
- Now we transform tabular data into LSTM-ready sequences

---

## Slides 32-34: Preprocessing Pipeline
**Duration:** ~1.5 minutes

### Talking Points:
- **T054 - Normalization:**
  - RobustScaler to handle outliers
  - More resistant to extreme values than StandardScaler
- **T055 - Missing value handling:**
  - Forward fill for temporal continuity
  - Median imputation for remaining gaps
- **T056 - Categorical encoding:**
  - Label encoding for county and industry
  - Preserves ordinal relationships where appropriate
- **T057 - Sequence transformation:**
  - Creates sliding windows for LSTM input
- **T058 - Validation:**
  - Confirms output shape and value ranges

---

## Slides 35-38: Sequence Transformation
**Duration:** ~1.5 minutes

### Talking Points:
- **Sliding window approach:**
  - Window size: 12 quarters (3 years of lookback)
  - The model sees 3 years of history to predict next quarter
- **Grouping strategy:**
  - Group by County + Industry combinations
  - 48,245 unique groups
  - Maintains temporal ordering within each group
- **Output shape:**
  - 599,582 sequences total
  - Each sequence: 12 timesteps x 24 features
  - Target: single employment value (next quarter)

---

## Slide 39: Data Splits
**Duration:** ~1 minute

### Talking Points:
- **Train/Val/Test split:**
  - Training: 70% (419,707 sequences) - 2004-2019
  - Validation: 20% (119,916 sequences) - 2019-2022
  - Test: 10% (59,959 sequences) - 2022-2024
- **Critical: No shuffling!**
  - Time series requires temporal ordering
  - Can't let future data leak into training
  - Walk-forward validation approach

---

## Slide 40: Model Training Section Header
**Duration:** ~15 seconds

### Talking Points:
- Now the exciting part - our LSTM neural network

---

## Slides 41-42: LSTM Architecture
**Duration:** ~2 minutes

### Talking Points:
- **Why LSTM over vanilla RNN?**
  - Long Short-Term Memory handles long-range dependencies
  - Prevents vanishing gradient problem
  - Perfect for quarterly data with annual patterns
- **Our architecture:**
  - Input: (batch, 12 timesteps, 24 features)
  - Two stacked LSTM layers with 64 hidden units each
  - Dropout (0.2) between layers for regularization
  - BatchNorm1d for training stability
  - Final linear layer outputs single prediction
- **Code snippet shows the clean PyTorch implementation**
- Total: ~50,000 trainable parameters

---

## Slides 43: Training Configuration
**Duration:** ~1 minute

### Talking Points:
- **Hyperparameters:**
  - Learning rate: 0.001 (standard for Adam)
  - Batch size: 32 (balances speed and gradient stability)
  - Max epochs: 100 (with early stopping)
  - Patience: 15 epochs without improvement triggers stop
  - Gradient clipping: 1.0 (prevents exploding gradients)
- **Optimizer:** Adam with ReduceLROnPlateau scheduler
  - Reduces learning rate when validation loss plateaus
  - Factor: 0.5 reduction, Patience: 5 epochs

---

## Slides 44-67: TensorBoard Training Metrics
**Duration:** ~2 minutes (summarize these together)

### Talking Points:
- These slides show our TensorBoard monitoring during training
- **Key metrics tracked:**
  - Training and validation loss per batch
  - Loss curves over epochs
  - Learning rate adjustments
  - Epochs without improvement
- **What we observed:**
  - Loss decreased steadily in early epochs
  - Learning rate reduced automatically when progress slowed
  - Early stopping triggered to prevent overfitting
  - Final model saved at best validation loss

---

## Slide 68: Lessons Learned & Limitations
**Duration:** ~1.5 minutes

### Talking Points:
- **Challenges we faced:**
  - **COVID-19 anomaly:** 2020 Q2 was a massive outlier, disrupted patterns
  - **Schema changes:** Required robust data handling across 20 years
  - **Short effective time series:** Post-COVID patterns still emerging
- **Current limitations:**
  - High MAPE indicates magnitude predictions need improvement
  - No external economic indicators (unemployment rates, GDP, etc.)
  - Model trained for only a few epochs due to time constraints
- **Future improvements:**
  - More aggressive hyperparameter tuning
  - Incorporate external economic data
  - Ensemble methods combining LSTM with traditional models

---

## Slide 69: Future Roadmap
**Duration:** ~1 minute

### Talking Points:
- **Near-term:**
  - Web-based prediction dashboard
  - API for programmatic access
- **Medium-term:**
  - Real-time data integration with EDD feeds
  - County comparison tools
- **Long-term:**
  - Industry risk early warning system
  - Policy insight generation
  - Automated reporting for stakeholders

---

## Slide 70: Q&A / Closing
**Duration:** Variable

### Talking Points:
- Thank you for your attention
- Happy to take questions about:
  - Technical implementation details
  - Data processing decisions
  - Model architecture choices
  - Future development plans
- Our code is available in the GitHub repository
- Contact information for follow-up questions

---

## General Presentation Tips

### Timing Guide
- Target: 25-30 minutes for content
- Leave 5-10 minutes for Q&A
- Pace: approximately 1 minute per slide on average

### Key Messages to Emphasize
1. **Scale:** 5.4 million records processed successfully
2. **Quality:** 85.9% data quality score, 91.98% directional accuracy
3. **Architecture:** Production-grade 8-stage pipeline
4. **Practical value:** Forecasts support real policy decisions

### Potential Questions to Prepare For
1. "Why LSTM instead of Transformer models?"
   - LSTMs work well for moderate-length sequences (12 quarters)
   - Transformers better for very long sequences, more compute intensive
   - LSTM proven effective for economic time series

2. "How did you handle the COVID-19 disruption?"
   - Retained COVID data as important signal
   - Quality filters handled extreme outliers
   - Model learns from disruption for future resilience

3. "What's the computational requirement?"
   - Training: GPU recommended (GTX 1070 or better)
   - Inference: Can run on CPU
   - Full pipeline: ~30-60 minutes on modern hardware

4. "How do you validate the model works?"
   - Baseline comparison (beats mean baseline)
   - Directional accuracy (91.98% - knows trend direction)
   - Walk-forward validation preserves temporal integrity

5. "Why is MAPE so high at 157%?"
   - Small employment values create large percentage errors
   - Directional accuracy is more meaningful for policy
   - Future work will focus on improving magnitude predictions
