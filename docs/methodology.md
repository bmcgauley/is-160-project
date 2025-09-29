# LSTM Methodology for Employment Data Analysis and Prediction

## Overview
This document describes the methodology used for analyzing California employment data 
using Long Short-Term Memory (LSTM) neural networks for time series forecasting.

## Data Sources
- Quarterly Census of Employment and Wages (QCEW) data from California
- Time period: 2020 Q1 - 2024 Q3 (19 quarters)
- Geographic coverage: Central Valley counties (Fresno, Kern, Tulare, Stanislaus, 
  San Joaquin, Merced, Kings, Madera) and industry level (NAICS codes)
- Industries: Focus on top 10-15 industries by employment share

## Data Preprocessing
### Steps:
1. Data acquisition and validation
2. Handling suppressed/missing data:
   - Identify suppressed cells (disclosure restrictions)
   - Forward-fill for short gaps (1-2 quarters)
   - Drop series with excessive missing data
3. Feature engineering:
   - Temporal features: lagged values (t-1 to t-4), rolling averages (4Q, 8Q), 
     YoY changes, quarter indicators (Q1-Q4)
   - Geographic features: county indicators, Central Valley aggregations
   - Industry features: NAICS codes, ownership type, industry concentration
   - Interaction features: county-industry combinations, seasonal-industry patterns
4. Normalization: MinMaxScaler or StandardScaler for numeric features
5. Sequence preparation: Create sliding windows of 8-12 quarters for LSTM input
6. Train/validation/test splits: 70%/15%/15% with temporal ordering preserved

## Model Architecture
### LSTM Network Design:
- Input: Multi-feature time series sequences (8-12 quarter lookback)
- Hidden layers: 2-3 LSTM layers (128, 64, 32 units) with dropout (0.2-0.3)
- Output: Multi-step employment forecasts (1-4 quarters ahead)
- Loss function: Huber Loss (SmoothL1Loss) for robustness to COVID outlier
- Activation: Tanh (LSTM default), ReLU (optional for dense layers)

## Training Procedure
### Hyperparameters:
- Learning rate: 0.001 (initial) with ReduceLROnPlateau scheduler
  - Factor: 0.5, Patience: 10 epochs, Min LR: 1e-6
- Batch size: 32-64 (balanced for memory and gradient stability)
- Sequence length: 8-12 quarters (lookback window)
- Epochs: 100-200 with early stopping (patience: 20)
- Optimizer: Adam (β1=0.9, β2=0.999)
- Regularization: Dropout (0.2-0.3) and gradient clipping (max_norm=1.0)

## Evaluation Metrics
- Mean Absolute Percentage Error (MAPE): Percentage-based accuracy
- Root Mean Squared Error (RMSE): Absolute error in original units
- Mean Absolute Error (MAE): Average absolute deviation
- Directional Accuracy: Percentage of correct trend predictions (up/down/stable)
- Industry-specific performance metrics: Per-industry MAPE and RMSE

## Baseline Comparisons
- ARIMA models
- Exponential smoothing (Holt-Winters)
- Naive seasonal baseline (previous year same quarter)
- Simple persistence model (last observed value)

## Validation and Robustness
- Time series cross-validation (walk-forward validation):
  - Expanding window approach preserves temporal ordering
  - No data leakage: Test set always chronologically after training set
- Train/Validation/Test split: 70%/15%/15% with temporal ordering preserved
- Sensitivity analysis: Test model on different counties and industries
- Out-of-sample testing: Hold out 2024 Q3-Q4 for final evaluation
- Economic plausibility checks: Verify predictions align with domain knowledge

## Computational Requirements
- PyTorch framework (version ≥ 1.12)
- GPU acceleration recommended (CUDA-compatible)
- Memory requirements: ~4-8GB RAM (dependent on sequence length and batch size)
- Training time: ~10-30 minutes per model on GPU

## Limitations and Assumptions
- Data quality: Assumes QCEW data accurately reflects employment levels
- Short time horizon: Only 4.75 years of data (limited training examples)
- COVID anomaly: 2020 shock may not represent typical economic dynamics
- Exogenous factors: Model doesn't include external economic indicators 
  (GDP, interest rates, policy changes, weather, etc.)
- Generalization: Trained on 2020-2024; may not generalize to different economic regimes
- Aggregation: County-level data may hide sub-county spatial variations
- Disclosure suppression: Some industry-county combinations unavailable due to privacy