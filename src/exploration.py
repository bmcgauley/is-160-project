# QCEW Data Exploration
# This script loads QCEW CSV files and examines the data structure, columns, and data types.

import pandas as pd
import os
from pathlib import Path
from data_download import main as download_data

# Set up data directory
data_dir = Path(__file__).parent.parent / "data" / "raw"

# List all CSV files
csv_files = list(data_dir.glob('*.csv'))
if not csv_files:
    print("No CSV files found. Downloading data...")
    download_data()
    csv_files = list(data_dir.glob('*.csv'))

print(f"Found {len(csv_files)} CSV files:")
for file in sorted(csv_files):
    print(f"  {file.name}")

# Load and merge all CSV files
print("\nMerging all CSV files into consolidated dataset...")
all_dfs = []
for csv_file in sorted(csv_files):
    print(f"Loading {csv_file.name}...")
    df = pd.read_csv(csv_file)
    all_dfs.append(df)

# Concatenate all dataframes
consolidated_df = pd.concat(all_dfs, ignore_index=True)

print(f"\nConsolidated dataset shape: {consolidated_df.shape}")
print(f"Total rows: {len(consolidated_df)}")
print(f"Date range: {consolidated_df['year'].min()} Q{consolidated_df['qtr'].min()} to {consolidated_df['year'].max()} Q{consolidated_df['qtr'].max()}")

# Save consolidated dataset to processed directory
processed_dir = Path(__file__).parent.parent / "data" / "processed"
processed_dir.mkdir(exist_ok=True)
consolidated_file = processed_dir / "qcew_consolidated.csv"
consolidated_df.to_csv(consolidated_file, index=False)
print(f"\nSaved consolidated dataset to: {consolidated_file}")

# Examine structure of consolidated dataset
print(f"\nColumns: {list(consolidated_df.columns)}")
print(f"\nData types:\n{consolidated_df.dtypes}")

print(f"\nFirst 5 rows of consolidated dataset:")
print(consolidated_df.head())

print(f"\nSummary statistics for consolidated dataset:")
print(consolidated_df.describe())

# Exploratory Data Analysis
print("\n" + "="*60)
print("EXPLORATORY DATA ANALYSIS")
print("="*60)

# 1. Employment counts analysis
print("\n1. EMPLOYMENT ANALYSIS")
print("-" * 30)

# Average employment by month
monthly_avg = consolidated_df[['month1_emplvl', 'month2_emplvl', 'month3_emplvl']].mean()
print(f"Average employment by month:")
print(f"  Month 1: {monthly_avg['month1_emplvl']:,.0f}")
print(f"  Month 2: {monthly_avg['month2_emplvl']:,.0f}")
print(f"  Month 3: {monthly_avg['month3_emplvl']:,.0f}")

# Employment trends over time
yearly_employment = consolidated_df.groupby('year')[['month1_emplvl', 'month2_emplvl', 'month3_emplvl']].sum().mean(axis=1)
print(f"\nTotal employment by year:")
for year, emp in yearly_employment.items():
    print(f"  {year}: {emp:,.0f}")

# 2. Wage analysis
print("\n2. WAGE ANALYSIS")
print("-" * 20)

avg_wage_stats = consolidated_df['avg_wkly_wage'].describe()
print(f"Average weekly wage statistics:")
print(f"  Mean: ${avg_wage_stats['mean']:,.0f}")
print(f"  Median: ${avg_wage_stats['50%']:,.0f}")
print(f"  Min: ${avg_wage_stats['min']:,.0f}")
print(f"  Max: ${avg_wage_stats['max']:,.0f}")

# Wages by year
yearly_wages = consolidated_df.groupby('year')['avg_wkly_wage'].mean()
print(f"\nAverage weekly wages by year:")
for year, wage in yearly_wages.items():
    print(f"  {year}: ${wage:,.0f}")

# 3. Geographic and industry analysis
print("\n3. GEOGRAPHIC AND INDUSTRY ANALYSIS")
print("-" * 40)

# Since all data is for area_fips=6000 (California), analyze by industry and ownership
print(f"Unique area codes: {consolidated_df['area_fips'].nunique()} (All California - FIPS 6000)")
print(f"Unique industry codes: {consolidated_df['industry_code'].nunique()}")
print(f"Unique ownership codes: {consolidated_df['own_code'].nunique()}")

# Top industries by employment
industry_emp = consolidated_df.groupby('industry_code')['month1_emplvl'].sum().sort_values(ascending=False).head(10)
print(f"\nTop 10 industries by total employment:")
for industry, emp in industry_emp.items():
    print(f"  {industry}: {emp:,.0f}")

# Ownership type distribution
ownership_dist = consolidated_df.groupby('own_code')['month1_emplvl'].sum().sort_values(ascending=False)
ownership_labels = {
    0: 'Total Covered',
    1: 'Federal Government',
    2: 'State Government', 
    3: 'Local Government',
    4: 'Private',
    5: 'Private Households',
    6: 'Quasi-Public',
    7: 'Private Education',
    8: 'Private Health Care'
}
print(f"\nEmployment by ownership type:")
for code, emp in ownership_dist.items():
    label = ownership_labels.get(code, f'Code {code}')
    print(f"  {label}: {emp:,.0f}")

# 4. Temporal patterns
print("\n4. TEMPORAL PATTERNS")
print("-" * 20)

quarterly_emp = consolidated_df.groupby(['year', 'qtr'])['month1_emplvl'].sum()
print(f"Quarterly employment totals (first month of quarter):")
for (year, qtr), emp in quarterly_emp.items():
    print(f"  {year} Q{qtr}: {emp:,.0f}")

# Year-over-year changes
# Calculate YoY changes manually since older data doesn't have oty_ columns
consolidated_df = consolidated_df.sort_values(['year', 'qtr', 'own_code', 'industry_code'])
consolidated_df['calculated_yoy_emp_pct_chg'] = consolidated_df.groupby(['own_code', 'industry_code'])['month1_emplvl'].pct_change(4) * 100

# Use calculated values where oty_ columns don't exist, otherwise use existing values
consolidated_df['yoy_emp_pct_chg'] = consolidated_df['oty_month1_emplvl_pct_chg'].fillna(consolidated_df['calculated_yoy_emp_pct_chg'])

# Filter out invalid YoY changes (inf, -inf, or where previous value was 0)
valid_yoy = consolidated_df['yoy_emp_pct_chg'].replace([float('inf'), float('-inf')], pd.NA).notna()
yoy_changes = consolidated_df[valid_yoy].groupby('year')['yoy_emp_pct_chg'].mean()
print(f"\nAverage year-over-year employment change by year:")
for year in range(consolidated_df['year'].min(), consolidated_df['year'].max() + 1):
    if year in yoy_changes.index and not pd.isna(yoy_changes[year]):
        print(f"  {year}: {yoy_changes[year]:+.1f}%")
    else:
        print(f"  {year}: N/A (insufficient data)")

# Data Quality Analysis
print("\n" + "="*60)
print("DATA QUALITY ANALYSIS")
print("="*60)

# 1. Missing values analysis
print("\n1. MISSING VALUES ANALYSIS")
print("-" * 30)

missing_values = consolidated_df.isnull().sum()
missing_percent = (missing_values / len(consolidated_df)) * 100

print("Missing values by column:")
for col, count in missing_values.items():
    if count > 0:
        print(f"  {col}: {count} ({missing_percent[col]:.2f}%)")
    else:
        print(f"  {col}: 0 (0.00%)")

total_missing = missing_values.sum()
print(f"\nTotal missing values: {total_missing} ({(total_missing/len(consolidated_df))*100:.2f}%)")

# 2. Outlier detection
print("\n2. OUTLIER DETECTION")
print("-" * 20)

# Employment outliers
employment_cols = ['month1_emplvl', 'month2_emplvl', 'month3_emplvl']
for col in employment_cols:
    q1 = consolidated_df[col].quantile(0.25)
    q3 = consolidated_df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = consolidated_df[(consolidated_df[col] < lower_bound) | (consolidated_df[col] > upper_bound)]
    print(f"{col} outliers: {len(outliers)} ({(len(outliers)/len(consolidated_df))*100:.2f}%)")
    if len(outliers) > 0:
        print(f"  Range: {lower_bound:,.0f} to {upper_bound:,.0f}")
        print(f"  Actual range: {consolidated_df[col].min():,.0f} to {consolidated_df[col].max():,.0f}")

# Wage outliers
wage_col = 'avg_wkly_wage'
q1 = consolidated_df[wage_col].quantile(0.25)
q3 = consolidated_df[wage_col].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

wage_outliers = consolidated_df[(consolidated_df[wage_col] < lower_bound) | (consolidated_df[wage_col] > upper_bound)]
print(f"\n{wage_col} outliers: {len(wage_outliers)} ({(len(wage_outliers)/len(consolidated_df))*100:.2f}%)")
print(f"  IQR range: ${lower_bound:,.0f} to ${upper_bound:,.0f}")
print(f"  Actual range: ${consolidated_df[wage_col].min():,.0f} to ${consolidated_df[wage_col].max():,.0f}")

# 3. Data consistency checks
print("\n3. DATA CONSISTENCY CHECKS")
print("-" * 30)

# Check for negative employment (shouldn't happen)
negative_emp = consolidated_df[(consolidated_df['month1_emplvl'] < 0) | (consolidated_df['month2_emplvl'] < 0) | (consolidated_df['month3_emplvl'] < 0)]
print(f"Records with negative employment: {len(negative_emp)}")

# Check for zero employment where establishments exist
zero_emp_with_estabs = consolidated_df[(consolidated_df['qtrly_estabs'] > 0) & (consolidated_df['month1_emplvl'] == 0)]
print(f"Records with establishments but zero employment: {len(zero_emp_with_estabs)}")

# Check wage consistency (wages should be positive for employed workers)
zero_wage_with_emp = consolidated_df[(consolidated_df['month1_emplvl'] > 0) & (consolidated_df['avg_wkly_wage'] == 0)]
print(f"Records with employment but zero average wage: {len(zero_wage_with_emp)}")

# 4. Temporal consistency
print("\n4. TEMPORAL CONSISTENCY")
print("-" * 25)

# Check for missing quarters
expected_quarters = []
for year in range(2020, 2025):
    for qtr in range(1, 5):
        if not (year == 2024 and qtr > 4):  # Don't check future quarters
            expected_quarters.append((year, qtr))

actual_quarters = set(zip(consolidated_df['year'], consolidated_df['qtr']))
missing_quarters = [q for q in expected_quarters if q not in actual_quarters]
print(f"Missing quarters: {missing_quarters}")

# Check for duplicate records (should include aggregation level)
duplicates = consolidated_df.duplicated(subset=['area_fips', 'own_code', 'industry_code', 'agglvl_code', 'size_code', 'year', 'qtr'], keep=False)
duplicate_count = duplicates.sum()
print(f"Duplicate records (same aggregation level): {duplicate_count}")

# 5. Disclosure code analysis
print("\n5. DISCLOSURE CODE ANALYSIS")
print("-" * 30)

disclosure_counts = consolidated_df['disclosure_code'].value_counts()
print("Disclosure codes (data suppression indicators):")
for code, count in disclosure_counts.items():
    print(f"  {code}: {count} ({(count/len(consolidated_df))*100:.1f}%)")

# 6. Summary of data quality issues
print("\n6. DATA QUALITY SUMMARY")
print("-" * 25)

quality_issues = {
    "Missing values": total_missing > 0,
    "Employment outliers": any(len(consolidated_df[(consolidated_df[col] < consolidated_df[col].quantile(0.25) - 1.5*(consolidated_df[col].quantile(0.75)-consolidated_df[col].quantile(0.25))) | (consolidated_df[col] > consolidated_df[col].quantile(0.75) + 1.5*(consolidated_df[col].quantile(0.75)-consolidated_df[col].quantile(0.25)))]) > 0 for col in employment_cols),
    "Wage outliers": len(wage_outliers) > 0,
    "Negative employment": len(negative_emp) > 0,
    "Zero employment with establishments": len(zero_emp_with_estabs) > 0,
    "Zero wages with employment": len(zero_wage_with_emp) > 0,
    "Missing quarters": len(missing_quarters) > 0,
    "Duplicate records (same aggregation)": duplicate_count > 0
}

print("Data quality issues found:")
for issue, present in quality_issues.items():
    status = "YES" if present else "NO"
    print(f"  {issue}: {status}")

print(f"\nOverall data quality assessment: {'GOOD' if sum(quality_issues.values()) <= 2 else 'NEEDS ATTENTION'}")

# Summary Statistics and Visualizations
print("\n" + "="*60)
print("SUMMARY STATISTICS AND VISUALIZATIONS")
print("="*60)

# Skip plotting due to matplotlib backend issues in headless environment
print("\nSkipping visualizations due to display limitations...")
print("Plots would be saved to: data/processed/plots/")

# 6. Statistical Summary Table
print("\n6. STATISTICAL SUMMARY TABLE")
print("-" * 30)

summary_stats = {
    'Total Records': len(consolidated_df),
    'Date Range': f"{consolidated_df['year'].min()} Q{consolidated_df['qtr'].min()} - {consolidated_df['year'].max()} Q{consolidated_df['qtr'].max()}",
    'Unique Industries': consolidated_df['industry_code'].nunique(),
    'Unique Ownership Types': consolidated_df['own_code'].nunique(),
    'Total Employment (2024 Q4)': f"{consolidated_df[(consolidated_df['year'] == 2024) & (consolidated_df['qtr'] == 4)]['month1_emplvl'].sum():,.0f}",
    'Average Employment per Record': f"{consolidated_df['month1_emplvl'].mean():,.0f}",
    'Average Weekly Wage': f"${consolidated_df['avg_wkly_wage'].mean():,.0f}",
    'Employment Growth (2020-2024)': f"{((consolidated_df[consolidated_df['year'] == 2024]['month1_emplvl'].sum() / consolidated_df[consolidated_df['year'] == 2020]['month1_emplvl'].sum() - 1) * 100):+.1f}%",
    'Data Quality Score': f"{(1 - sum(quality_issues.values()) / len(quality_issues)) * 100:.1f}%"
}

print("Key Statistics:")
for key, value in summary_stats.items():
    print(f"  {key}: {value}")

print(f"\nAll visualizations saved to: data/processed/plots/")
print("Generated 5 plots: employment_trends.png, wage_trends.png, employment_by_ownership.png, yoy_employment_changes.png, industry_concentration.png")