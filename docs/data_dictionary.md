# QCEW Data Dictionary

## Overview
This document provides a comprehensive data dictionary for the Quarterly Census of Employment and Wages (QCEW) data used in the IS-160 CNN Employment Trends Analysis project. The data covers California state-level employment statistics from Q1 2020 through Q4 2024.

## Data Source
- **Source**: California Open Data Portal - QCEW datasets
- **Geographic Coverage**: California State (FIPS: 06000)
- **Time Period**: 2020 Q1 - 2024 Q4
- **Update Frequency**: Quarterly
- **Data Structure**: 64,036 records across 2,405 unique industry codes

## Dataset Structure

### Core Identifiers
| Field | Type | Description | Example | Notes |
|-------|------|-------------|---------|-------|
| `area_fips` | int64 | Federal Information Processing Standard code for geographic area | 6000 | Always 6000 (California state) |
| `own_code` | int64 | Ownership code indicating type of establishment ownership | 0, 1, 2, 3, 4, 5 | See ownership codes table below |
| `industry_code` | object | North American Industry Classification System (NAICS) code | "10", "1021", "44-45" | Up to 6-digit codes |
| `agglvl_code` | int64 | Aggregation level code | 10, 12, 14, etc. | Indicates level of industry aggregation |
| `size_code` | int64 | Employment size class code | 0, 1, 2, 3, etc. | Based on establishment size ranges |

### Time Variables
| Field | Type | Description | Range | Notes |
|-------|------|-------------|-------|-------|
| `year` | int64 | Calendar year | 2020-2024 | |
| `qtr` | int64 | Quarter of the year | 1-4 | 1=Q1, 2=Q2, 3=Q3, 4=Q4 |

### Employment Variables
| Field | Type | Description | Units | Range | Notes |
|-------|------|-------------|-------|-------|-------|
| `qtrly_estabs` | int64 | Number of establishments in the quarter | Count | 0-18,423,998 | Total establishments reporting |
| `month1_emplvl` | int64 | Employment level in first month of quarter | Count | 0-18,423,998 | March, June, September, or December |
| `month2_emplvl` | int64 | Employment level in second month of quarter | Count | 0-18,428,658 | April, July, October, or January |
| `month3_emplvl` | int64 | Employment level in third month of quarter | Count | 0-18,246,332 | May, August, November, or February |

### Wage Variables
| Field | Type | Description | Units | Range | Notes |
|-------|------|-------------|-------|-------|-------|
| `total_qtrly_wages` | int64 | Total wages paid in the quarter | Dollars | 0-9,269,000,000 | Gross wages before deductions |
| `taxable_qtrly_wages` | int64 | Taxable wages subject to UI tax | Dollars | 0-9,269,000,000 | Wages subject to unemployment insurance |
| `qtrly_contributions` | int64 | UI tax contributions collected | Dollars | 0-184,000,000 | Employer contributions |
| `avg_wkly_wage` | int64 | Average weekly wage per employee | Dollars | 0-96,972 | Calculated from total wages and employment |

### Last Quarter Comparison Variables
| Field | Type | Description | Units | Notes |
|-------|------|-------------|-------|-------|
| `lq_disclosure_code` | object | Disclosure code for last quarter data | "N", "-", null | Data suppression indicator |
| `lq_qtrly_estabs` | float64 | Last quarter establishments | Count | Null if suppressed |
| `lq_month1_emplvl` | float64 | Last quarter month 1 employment | Count | Null if suppressed |
| `lq_month2_emplvl` | float64 | Last quarter month 2 employment | Count | Null if suppressed |
| `lq_month3_emplvl` | float64 | Last quarter month 3 employment | Count | Null if suppressed |
| `lq_total_qtrly_wages` | float64 | Last quarter total wages | Dollars | Null if suppressed |
| `lq_taxable_qtrly_wages` | float64 | Last quarter taxable wages | Dollars | Null if suppressed |
| `lq_qtrly_contributions` | float64 | Last quarter contributions | Dollars | Null if suppressed |
| `lq_avg_wkly_wage` | float64 | Last quarter average wage | Dollars | Null if suppressed |

### Over-the-Year Comparison Variables
| Field | Type | Description | Units | Notes |
|-------|------|-------------|-------|-------|
| `oty_disclosure_code` | object | Disclosure code for over-the-year data | "N", "-", null | Data suppression indicator |
| `oty_qtrly_estabs_chg` | int64 | Year-over-year establishments change | Count | Absolute change |
| `oty_qtrly_estabs_pct_chg` | float64 | Year-over-year establishments percent change | Percent | Percentage change |
| `oty_month1_emplvl_chg` | int64 | Year-over-year month 1 employment change | Count | Absolute change |
| `oty_month1_emplvl_pct_chg` | float64 | Year-over-year month 1 employment percent change | Percent | Percentage change |
| `oty_month2_emplvl_chg` | int64 | Year-over-year month 2 employment change | Count | Absolute change |
| `oty_month2_emplvl_pct_chg` | float64 | Year-over-year month 2 employment percent change | Percent | Percentage change |
| `oty_month3_emplvl_chg` | int64 | Year-over-year month 3 employment change | Count | Absolute change |
| `oty_month3_emplvl_pct_chg` | float64 | Year-over-year month 3 employment percent change | Percent | Percentage change |
| `oty_total_qtrly_wages_chg` | int64 | Year-over-year total wages change | Dollars | Absolute change |
| `oty_total_qtrly_wages_pct_chg` | float64 | Year-over-year total wages percent change | Percent | Percentage change |
| `oty_taxable_qtrly_wages_chg` | int64 | Year-over-year taxable wages change | Dollars | Absolute change |
| `oty_taxable_qtrly_wages_pct_chg` | float64 | Year-over-year taxable wages percent change | Percent | Percentage change |
| `oty_qtrly_contributions_chg` | int64 | Year-over-year contributions change | Dollars | Absolute change |
| `oty_qtrly_contributions_pct_chg` | float64 | Year-over-year contributions percent change | Percent | Percentage change |
| `oty_avg_wkly_wage_chg` | int64 | Year-over-year average wage change | Dollars | Absolute change |
| `oty_avg_wkly_wage_pct_chg` | float64 | Year-over-year average wage percent change | Percent | Percentage change |

### Disclosure Codes
| Field | Type | Description | Values | Notes |
|-------|------|-------------|--------|-------|
| `disclosure_code` | object | Current quarter data suppression code | "N", "-", null | 93.53% missing (expected) |
| `lq_disclosure_code` | object | Last quarter data suppression code | "N", "-", null | 91.00% missing (expected) |
| `oty_disclosure_code` | object | Over-the-year data suppression code | "N", "-", null | 89.12% missing (expected) |

## Code Reference Tables

### Ownership Codes
| Code | Description | Employment Share |
|------|-------------|------------------|
| 0 | Total Covered | 5.8% |
| 1 | Federal Government | 0.7% |
| 2 | State Government | 1.3% |
| 3 | Local Government | 4.5% |
| 4 | Private | N/A |
| 5 | Private Households | 39.6% |
| 6 | Quasi-Public | N/A |
| 7 | Private Education | N/A |
| 8 | Private Health Care | 0.8% |

### Aggregation Level Codes
| Code | Description |
|------|-------------|
| 10 | 1-digit NAICS |
| 12 | 2-digit NAICS |
| 14 | 3-digit NAICS |
| 16 | 4-digit NAICS |
| 18 | 5-digit NAICS |
| 20 | 6-digit NAICS |

### Size Class Codes
| Code | Employment Size Range |
|------|----------------------|
| 0 | All sizes |
| 1 | 1-4 employees |
| 2 | 5-9 employees |
| 3 | 10-19 employees |
| 4 | 20-49 employees |
| 5 | 50-99 employees |
| 6 | 100-249 employees |
| 7 | 250-499 employees |
| 8 | 500-999 employees |
| 9 | 1,000+ employees |

## Data Quality Notes

### Missing Values
- Disclosure codes are intentionally missing (89-94%) to protect sensitive employment data
- No missing values in core employment and wage variables
- All temporal and geographic identifiers are complete

### Outliers
- Employment variables show 13.5% outliers (expected due to industry size variation)
- Wage variables show 4.5% outliers (some very high wages in specialized industries)
- Outliers are generally legitimate and should be retained for analysis

### Data Consistency
- No negative employment values
- 4494 records (7%) have establishments but zero employment (seasonal businesses)
- 26 records have employment but zero wages (possible data entry issues)

## Key Statistics Summary

| Metric | Value |
|--------|-------|
| Total Records | 64,036 |
| Date Range | 2020 Q1 - 2024 Q4 |
| Unique Industries | 2,405 |
| Total Employment (2024 Q4) | 168,278,513 |
| Average Employment per Record | 49,462 |
| Average Weekly Wage | $1,522 |
| Employment Growth (2020-2024) | +12.6% |
| Data Completeness Score | 95.2% |

## Usage Notes for CNN Model

### Feature Engineering Considerations
- **Temporal Features**: Use quarter, year, and month variables for cyclical patterns
- **Industry Features**: NAICS codes can be encoded hierarchically (1-6 digits)
- **Size Features**: Employment size classes provide categorical business scale information
- **Change Variables**: Year-over-year changes are pre-calculated and reliable

### Data Preprocessing Requirements
- **Normalization**: Employment and wage variables need scaling due to wide ranges
- **Encoding**: Industry codes (strings) and ownership codes (integers) need encoding
- **Missing Data**: Disclosure code nulls should be treated as "not suppressed"
- **Outliers**: Retain outliers as they represent legitimate industry variation

### Model Input Preparation
- **Sequence Length**: 20 quarters (2020 Q1 - 2024 Q4) for temporal modeling
- **Feature Dimensions**: 42 raw features, expandable to 100+ engineered features
- **Target Variables**: Employment levels, wage changes, or growth predictions
- **Train/Validation Split**: Use temporal split (earlier quarters for training)

## Data Version History
- **v1.0**: Initial consolidated dataset (2020-2024 Q4)
- **Last Updated**: September 2025
- **Source Files**: 20 quarterly CSV files from California Open Data Portal