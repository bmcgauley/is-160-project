# Feature Engineering Data Filtering Requirements

**Date**: October 16, 2025  
**Status**: Planning Phase - Tasks Created (T032-T037)

## Critical Issues Discovered

### 1. Annual vs Quarterly Records
**Problem**: The `quarter` column contains both quarterly values ("1st Qtr", "2nd Qtr", etc.) AND "Annual" values.
- Annual records aggregate entire year's data
- Mixing annual and quarterly data will corrupt time-series features
- Need to decide: drop Annual records OR process separately

**Impact**: Feature engineering calculations (quarter-over-quarter growth, rolling averages) will be invalid if Annual records are included.

### 2. Multiple Aggregation Levels
**Problem**: Data includes County (87%), State (6%), and National (7%) aggregation levels.
- State and national data are nominal percentage of total records
- Including them creates data leakage (county data rolls up to state/national)
- Model would see same employment counted multiple times

**Solution**: **Filter to county-level only** before feature engineering (T032).

### 3. Incomplete Records (Zero Values)
**Problem**: Many records have:
- Zero employment but non-zero establishments
- Zero wages but non-zero employment
- Both employment and wages at zero

**Examples from exploration**:
- Employment Min: 0
- Wages Min: $0.00

**Impact**: These records represent:
- Data suppression (privacy protection)
- Incomplete reporting
- Seasonal closures
- Data quality issues

**Solution**: Filter out incomplete records (T034) before feature engineering to prevent:
- Division by zero errors
- Invalid percentage calculations
- Corrupted rolling averages
- Misleading growth rates

### 4. Central Valley Focus Required
**Problem**: Project focuses on Central Valley employment forecasting, but data includes all California counties.

**Solution**: Create two processed datasets (T036):
1. **Full California dataset** (all counties, filtered/cleaned)
2. **Central Valley subset** (8 core counties only)

## Task Overview

### Data Filtering Tasks (T032-T037)
Created new section in Phase 3.3 to handle data quality issues BEFORE feature calculations:

- **T032**: Filter to county-level only (drop state/national aggregates)
- **T033**: Handle Annual vs quarterly records
- **T034**: Remove incomplete records (zero employment/wages)
- **T035**: Create Central Valley counties reference file ✅ COMPLETED
- **T036**: Generate two processed datasets (full CA + Central Valley)
- **T037**: Validate filtered datasets for temporal consistency

### Central Valley Counties

**Core 8 Counties** (primary focus):
1. Fresno County
2. Kern County
3. Kings County
4. Madera County
5. Merced County
6. San Joaquin County
7. Stanislaus County
8. Tulare County

**Reference file created**: `/data/central_valley_counties.json`

## Data Filtering Pipeline

```
Raw Consolidated Data (5.4M records)
    ↓
Filter: area_type == 'County'
    ↓ (4.7M county records, 87%)
    ↓
Filter: quarter != 'Annual'
    ↓ (estimated ~4.0M quarterly records)
    ↓
Filter: avg_monthly_emplvl > 0 AND total_qtrly_wages > 0
    ↓ (estimated ~3.5M complete records)
    ↓
Split into two datasets:
    ↓
    ├─→ All California Counties Dataset (~3.5M records)
    │   → Feature engineering
    │   → Preprocessing
    │   → Model training
    │
    └─→ Central Valley Counties Only (~200-300K records)
        → Feature engineering
        → Preprocessing
        → Model training (focus model)
```

## Expected Data Reduction

| Stage | Records | Notes |
|-------|---------|-------|
| Raw consolidated | 5,430,384 | All aggregation levels |
| County-level only | 4,732,218 | 87% of data |
| Quarterly only | ~4,000,000 | Remove Annual records |
| Complete records | ~3,500,000 | Remove zeros |
| Central Valley | ~200,000-300,000 | 8 counties only |

## Impact on Subsequent Phases

### Phase 3.3: Feature Engineering
- **MUST complete T032-T037 FIRST** before any feature calculations
- All subsequent feature tasks (T038+) will operate on filtered data only
- Two parallel feature engineering pipelines (full CA + Central Valley)

### Phase 3.4: Preprocessing
- Preprocessing will use filtered datasets as input
- No need to handle aggregation level issues (already filtered)
- No need to handle Annual records (already removed)
- No need to handle zero values (already filtered)

### Phase 3.5+: Training
- Two separate models can be trained:
  1. All-California model (broader patterns)
  2. Central Valley model (focused predictions)

## Files Created

1. **`/data/central_valley_counties.json`** ✅
   - Reference file with county names
   - Filtering instructions
   - Optional extended county list

2. **Tasks updated in `/specs/001/tasks.md`** ✅
   - T032-T037: Data filtering tasks
   - All subsequent task numbers renumbered
   - Documentation sections updated

## Next Steps for Alejo (Phase 3.3 Owner)

When starting feature engineering:

1. **Start with T032**: Filter to county-level data only
2. **T033**: Decide on Annual record handling (recommend: drop them)
3. **T034**: Implement quality filters (remove zeros)
4. **T035**: Load Central Valley counties from JSON ✅
5. **T036**: Create two output datasets
6. **T037**: Validate both datasets
7. **T038+**: Proceed with feature calculations on filtered data

## Questions to Resolve

1. **Annual records**: Drop completely or process separately?
   - Recommendation: **Drop** - they aggregate quarterly data we already have
   
2. **Zero value threshold**: How aggressive to filter?
   - Recommendation: Remove if **either** employment=0 **or** wages=0
   
3. **Central Valley definition**: Use 8 core counties or expand to 19?
   - Recommendation: Start with **8 core**, expand later if needed

4. **Ownership types**: Keep all (Federal/State/Local/Private) or filter?
   - Recommendation: **Keep all** - provides richer patterns

## References

- Data aggregation analysis: `/docs/data_aggregation_levels.md`
- Verification script: `/scripts/verify_consolidation.py`
- Central Valley counties: `/data/central_valley_counties.json`
- Updated tasks: `/specs/001/tasks.md`
