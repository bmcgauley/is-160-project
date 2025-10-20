# Feature Engineering Data Directory

This directory contains intermediate datasets from each stage of the feature engineering pipeline.

## Directory Structure

```
feature_engineering/
├── T032_county_filtered.csv          # County-level records only (state/national removed)
├── T033_quarterly_filtered.csv       # Annual records handled/removed
├── T034_quality_filtered.csv         # Incomplete records removed
├── T035_central_valley.csv           # Central Valley counties subset
├── T036_all_california.csv           # All California counties with features
├── T038_growth_rates.csv             # Quarter-over-quarter growth rates
├── T039_seasonal_adjusted.csv        # Seasonal adjustment factors
├── T040_industry_concentration.csv   # Industry concentration metrics
├── T041_geographic_clustering.csv    # Geographic clustering features
├── T042_lag_features.csv             # Lag features for temporal dependencies
└── final_features.csv                # Complete feature-engineered dataset
```

## Data Continuity

Each file represents the data state after completing its corresponding task:
- **T032**: Filtered to county-level records only
- **T033**: Annual vs quarterly records handled
- **T034**: Data quality filtering applied
- **T035**: Central Valley counties subset created
- **T036**: All California counties with features
- **T038-T042**: Progressive feature engineering steps
- **final_features.csv**: Ready for preprocessing and model training

## File Naming Convention

`T{task_number}_{descriptive_name}.csv`

Example: `T032_county_filtered.csv` = Output from Task 032

## Usage

The feature engineering pipeline automatically saves intermediate outputs to this directory.
Each stage can be inspected independently for validation and debugging purposes.
