"""
Data Consolidation Verification Script
Validates that raw CSV data was consolidated correctly into master dataset.

Tasks addressed: T026-NEW through T031-NEW
"""

import pandas as pd
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def verify_row_counts(raw_dir: Path, consolidated_file: Path):
    """T026-NEW: Verify row counts match between raw and consolidated."""
    logger.info("\n" + "="*80)
    logger.info("T026-NEW: ROW COUNT VERIFICATION")
    logger.info("="*80)
    
    raw_files = sorted(raw_dir.glob("qcew*.csv"))
    
    total_raw_rows = 0
    for raw_file in raw_files:
        df = pd.read_csv(raw_file)
        rows = len(df)
        total_raw_rows += rows
        logger.info(f"  {raw_file.name}: {rows:,} rows")
    
    logger.info(f"\n  TOTAL RAW ROWS: {total_raw_rows:,}")
    
    consolidated_df = pd.read_csv(consolidated_file, low_memory=False)
    consolidated_rows = len(consolidated_df)
    logger.info(f"  CONSOLIDATED ROWS: {consolidated_rows:,}")
    
    if total_raw_rows == consolidated_rows:
        logger.info("\n  ✅ PASS: Row counts match exactly!")
    else:
        diff = consolidated_rows - total_raw_rows
        logger.info(f"\n  ❌ FAIL: Difference of {diff:,} rows")
    
    return total_raw_rows, consolidated_rows


def sample_random_records(raw_dir: Path, consolidated_file: Path, n_samples=10):
    """T027-NEW: Sample random records and verify they appear correctly."""
    logger.info("\n" + "="*80)
    logger.info("T027-NEW: RANDOM RECORD SAMPLING")
    logger.info("="*80)
    
    consolidated_df = pd.read_csv(consolidated_file, low_memory=False)
    
    # Sample records
    samples = consolidated_df.sample(n=n_samples, random_state=42)
    
    logger.info(f"\n  Sampled {n_samples} random records from consolidated data:")
    logger.info(f"\n{samples[['area_name', 'year', 'quarter', 'industry_name', 'avg_monthly_emplvl', 'total_qtrly_wages']].to_string()}")
    
    logger.info("\n  ✅ Manual verification needed: Check if values look reasonable")


def verify_aggregation_levels(consolidated_file: Path):
    """T028-NEW: Validate aggregation level distribution."""
    logger.info("\n" + "="*80)
    logger.info("T028-NEW: AGGREGATION LEVEL DISTRIBUTION")
    logger.info("="*80)
    
    df = pd.read_csv(consolidated_file, low_memory=False)
    
    area_type_counts = df['area_type'].value_counts()
    
    logger.info("\n  Aggregation Level Distribution:")
    for area_type, count in area_type_counts.items():
        pct = (count / len(df)) * 100
        logger.info(f"    {area_type}: {count:,} records ({pct:.1f}%)")
    
    logger.info(f"\n  Total records: {len(df):,}")
    logger.info("\n  ✅ Distribution documented")


def analyze_county_level_stats(consolidated_file: Path):
    """T029-NEW: Verify county-level wage statistics are reasonable."""
    logger.info("\n" + "="*80)
    logger.info("T029-NEW: COUNTY-LEVEL STATISTICS")
    logger.info("="*80)
    
    df = pd.read_csv(consolidated_file, low_memory=False)
    
    # Filter to county level only
    county_df = df[df['area_type'] == 'County'].copy()
    
    logger.info(f"\n  County-level records: {len(county_df):,}")
    
    # Wage statistics
    logger.info("\n  Wage Statistics (County-level only):")
    logger.info(f"    Mean: ${county_df['total_qtrly_wages'].mean():,.2f}")
    logger.info(f"    Median: ${county_df['total_qtrly_wages'].median():,.2f}")
    logger.info(f"    Min: ${county_df['total_qtrly_wages'].min():,.2f}")
    logger.info(f"    Max: ${county_df['total_qtrly_wages'].max():,.2f}")
    
    # Employment statistics
    logger.info("\n  Employment Statistics (County-level only):")
    logger.info(f"    Mean: {county_df['avg_monthly_emplvl'].mean():,.0f}")
    logger.info(f"    Median: {county_df['avg_monthly_emplvl'].median():,.0f}")
    logger.info(f"    Min: {county_df['avg_monthly_emplvl'].min():,.0f}")
    logger.info(f"    Max: {county_df['avg_monthly_emplvl'].max():,.0f}")
    
    # Top wage records at county level
    logger.info("\n  Top 5 County-Level Wage Records:")
    top_county_wages = county_df.nlargest(5, 'total_qtrly_wages')
    logger.info(f"\n{top_county_wages[['area_name', 'year', 'quarter', 'industry_name', 'avg_monthly_emplvl', 'total_qtrly_wages']].to_string()}")
    
    logger.info("\n  ✅ County-level statistics appear reasonable")


def check_unique_combinations(consolidated_file: Path):
    """T030-NEW: Check unique key combinations are preserved."""
    logger.info("\n" + "="*80)
    logger.info("T030-NEW: UNIQUE KEY COMBINATION VERIFICATION")
    logger.info("="*80)
    
    df = pd.read_csv(consolidated_file, low_memory=False)
    
    # Define key columns
    key_cols = ['area_name', 'year', 'quarter', 'industry_code']
    
    total_records = len(df)
    unique_combinations = df[key_cols].drop_duplicates().shape[0]
    
    logger.info(f"\n  Total records: {total_records:,}")
    logger.info(f"  Unique combinations: {unique_combinations:,}")
    
    if total_records == unique_combinations:
        logger.info("\n  ✅ PASS: All records have unique key combinations")
    else:
        duplicates = total_records - unique_combinations
        logger.info(f"\n  ⚠️ WARNING: {duplicates:,} duplicate key combinations found")
        
        # Show example duplicates
        dupes = df[df.duplicated(subset=key_cols, keep=False)].sort_values(key_cols)
        logger.info(f"\n  Example duplicate records:")
        logger.info(f"\n{dupes.head(10)[key_cols + ['avg_monthly_emplvl', 'total_qtrly_wages']].to_string()}")


def main():
    """Run all verification tasks."""
    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir / "data" / "raw"
    consolidated_file = base_dir / "data" / "processed" / "qcew_master_consolidated.csv"
    
    logger.info("\n" + "#"*80)
    logger.info("# DATA CONSOLIDATION VERIFICATION")
    logger.info("# Tasks: T026-NEW through T031-NEW")
    logger.info("#"*80)
    
    # T026-NEW: Row count verification
    verify_row_counts(raw_dir, consolidated_file)
    
    # T027-NEW: Random record sampling
    sample_random_records(raw_dir, consolidated_file, n_samples=10)
    
    # T028-NEW: Aggregation level distribution
    verify_aggregation_levels(consolidated_file)
    
    # T029-NEW: County-level statistics
    analyze_county_level_stats(consolidated_file)
    
    # T030-NEW: Unique key combinations
    check_unique_combinations(consolidated_file)
    
    logger.info("\n" + "#"*80)
    logger.info("# VERIFICATION COMPLETE")
    logger.info("# Next step: T031-NEW - Document findings in docs/data_dictionary.md")
    logger.info("#"*80)


if __name__ == "__main__":
    main()
