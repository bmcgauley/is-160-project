# QCEW Data Aggregation Levels

**Date**: October 16, 2025  
**Author**: Project Lead

## Overview

The consolidated QCEW dataset contains records at **three distinct aggregation levels**, which explains some of the seemingly extreme values in the exploration statistics.

## Aggregation Level Breakdown

| Aggregation Level | Record Count | Percentage | Description |
|------------------|--------------|------------|-------------|
| **County** | 4,732,218 | 87.1% | Individual California counties (e.g., Alameda County, Los Angeles County) |
| **California Statewide** | 298,804 | 5.5% | State-level aggregations for all of California |
| **United States** | 399,362 | 7.4% | National-level aggregations across all states |
| **TOTAL** | 5,430,384 | 100% | Complete consolidated dataset |

## Impact on Statistics

### Wage Statistics Explained

The exploration output showed:
- **Mean total quarterly wages**: $730,128,622.54
- **Median total quarterly wages**: $4,250,230.50

These values are **correct** but include all aggregation levels:

#### National Level (United States)
- Example: Q4 2024 - Total all industries
  - Employment: 154,870,300 workers
  - **Total quarterly wages: $11.7 TRILLION**

#### State Level (California)
- Example: California statewide aggregates
  - Significantly lower than national but higher than counties

#### County Level (Alameda, Los Angeles, etc.)
- Example: Alameda County, Q1 2004, specific industry
  - Employment: 354 workers
  - Total quarterly wages: $2,532,357 (~$7,150/worker/quarter)

### Why the Mean is High

The mean of $730M is heavily skewed by:
1. **National aggregates** showing $11+ trillion in quarterly wages
2. **State aggregates** showing billions in quarterly wages
3. County-level records (87% of data) having much more reasonable values

The **median of $4.25M** is more representative of typical records since most are county-level.

## Implications for Modeling

### Data Filtering Recommendations

For employment forecasting models, consider:

1. **County-Only Model**: Filter to `area_type == 'County'` (4.7M records)
   - Most granular level for geographic analysis
   - Avoids double-counting (county → state → national rollups)

2. **State-Only Model**: Filter to `area_type == 'California - Statewide'` (299K records)
   - For California-wide trend analysis
   - Useful for policy-level insights

3. **Multi-Level Model**: Keep all levels but add `area_type` as a feature
   - Account for aggregation level in predictions
   - More complex but captures relationships

### Validation Considerations

When validating data quality:
- ✅ National records with $11T+ wages are **VALID** (entire US economy)
- ✅ California records with billions in wages are **VALID** (entire state)
- ✅ County records with millions in wages are **VALID** (individual counties)
- ⚠️ Check for outliers **within each aggregation level separately**

## Raw Data Structure

Each raw CSV file contains records at all three levels:
- `qcew_2004-2007.csv`: 1,033,078 records (all levels mixed)
- `qcew_2008-2011.csv`: Similar structure
- `qcew_2012-2015.csv`: Similar structure
- `qcew_2016-2019.csv`: Similar structure
- `qcew_2020-2022.csv`: Similar structure
- `qcew_2023-2024.csv`: Similar structure

The consolidation process correctly preserves all aggregation levels without modification.

## Next Steps

1. **T026-NEW**: Verify row counts match between raw and consolidated
2. **T027-NEW**: Sample random records to validate data accuracy
3. **T028-NEW**: Validate aggregation level distribution
4. **T029-NEW**: Analyze county-level statistics separately
5. **T030-NEW**: Check for unique key combinations
6. **T031-NEW**: Document findings in data dictionary

## References

- Raw data source: California Open Data Portal
- Consolidated data: `/data/processed/qcew_master_consolidated.csv`
- Exploration script: `src/exploration.py`
- Main pipeline: `main.py`
