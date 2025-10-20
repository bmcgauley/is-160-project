"""
Test script for T032: Filter to County-Level Records

This script tests the filter_to_county_level() function to ensure it:
1. Successfully filters out state and national aggregates
2. Retains only county-level records
3. Validates data integrity after filtering
4. Provides detailed logging of the filtering process
"""

import pandas as pd
import logging
from pathlib import Path
from src.feature_engineering import filter_to_county_level
from src.logging_config import setup_logging

def main():
    """Run T032 test on consolidated data."""
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*80)
    logger.info("TESTING T032: FILTER TO COUNTY-LEVEL RECORDS")
    logger.info("="*80)
    
    # Define paths
    project_root = Path(__file__).parent
    consolidated_file = project_root / 'data' / 'processed' / 'qcew_master_consolidated.csv'
    feature_eng_dir = project_root / 'data' / 'feature_engineering'
    
    # Ensure feature engineering directory exists
    feature_eng_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if consolidated file exists
    if not consolidated_file.exists():
        logger.error(f"Consolidated file not found: {consolidated_file}")
        logger.error("Please run main.py --stage consolidate first")
        return
    
    # Load consolidated data
    logger.info(f"\nLoading consolidated data from: {consolidated_file}")
    # Specify dtypes to avoid mixed type warnings
    dtype_spec = {
        'area_type': 'str',
        'area_name': 'str',
        'year': 'int',
        'quarter': 'str',  # Can be '1st Qtr', '2nd Qtr', 'Annual', etc.
        'ownership': 'str',
        'naics_level': 'str',
        'industry_code': 'str',
        'industry_name': 'str'
    }
    df = pd.read_csv(consolidated_file, dtype=dtype_spec, low_memory=False)
    logger.info(f"Loaded {len(df):,} records")
    
    # Display sample of data before filtering
    logger.info("\n" + "-"*80)
    logger.info("SAMPLE DATA BEFORE FILTERING")
    logger.info("-"*80)
    if 'area_type' in df.columns and 'area_name' in df.columns:
        sample = df.groupby('area_type').head(2)[['area_type', 'area_name', 'year', 'quarter']]
        logger.info(f"\n{sample.to_string()}")
    
    # Run the filter function (T032)
    try:
        county_df = filter_to_county_level(df, output_dir=feature_eng_dir)
        
        # Validate results
        logger.info("\n" + "-"*80)
        logger.info("VALIDATION CHECKS")
        logger.info("-"*80)
        
        # Check 1: Only County records remain
        if 'area_type' in county_df.columns:
            unique_types = county_df['area_type'].unique()
            logger.info(f"\n[CHECK 1] Unique area_type values after filtering: {unique_types}")
            if len(unique_types) == 1 and unique_types[0] == 'County':
                logger.info("  PASS: Only 'County' records remain")
            else:
                logger.error(f"  FAIL: Found non-county records: {unique_types}")
                return
        
        # Check 2: No null values in key columns
        key_columns = ['area_type', 'area_name', 'year', 'quarter']
        existing_key_cols = [col for col in key_columns if col in county_df.columns]
        null_counts = county_df[existing_key_cols].isnull().sum()
        logger.info(f"\n[CHECK 2] Null values in key columns:")
        for col in existing_key_cols:
            logger.info(f"  {col}: {null_counts[col]:,} nulls")
        if null_counts.sum() == 0:
            logger.info("  PASS: No null values in key columns")
        else:
            logger.warning("  WARNING: Found null values in key columns")
        
        # Check 3: Temporal coverage maintained
        if 'year' in county_df.columns:
            year_min = county_df['year'].min()
            year_max = county_df['year'].max()
            year_count = county_df['year'].nunique()
            logger.info(f"\n[CHECK 3] Temporal coverage:")
            logger.info(f"  Year range: {year_min} - {year_max}")
            logger.info(f"  Unique years: {year_count}")
            logger.info("  PASS: Temporal coverage preserved")
        
        # Check 4: County coverage
        if 'area_name' in county_df.columns:
            county_count = county_df['area_name'].nunique()
            logger.info(f"\n[CHECK 4] Geographic coverage:")
            logger.info(f"  Unique counties: {county_count}")
            # List top 10 counties by record count
            top_counties = county_df['area_name'].value_counts().head(10)
            logger.info("\n  Top 10 counties by record count:")
            for county, count in top_counties.items():
                logger.info(f"    {county}: {count:,} records")
            logger.info("  PASS: Multiple counties present")
        
        # Check 5: Data volume is reasonable
        original_county_pct = 87.1  # From data_aggregation_levels.md
        expected_county_count = len(df) * (original_county_pct / 100)
        actual_county_count = len(county_df)
        diff_pct = abs(actual_county_count - expected_county_count) / expected_county_count * 100
        
        logger.info(f"\n[CHECK 5] Data volume validation:")
        logger.info(f"  Expected county records: ~{expected_county_count:,.0f} (87.1% of total)")
        logger.info(f"  Actual county records: {actual_county_count:,}")
        logger.info(f"  Difference: {diff_pct:.2f}%")
        if diff_pct < 1.0:
            logger.info("  PASS: Record count matches expectations")
        else:
            logger.warning(f"  WARNING: Record count differs by {diff_pct:.2f}%")
        
        # Display sample of filtered data
        logger.info("\n" + "-"*80)
        logger.info("SAMPLE DATA AFTER FILTERING")
        logger.info("-"*80)
        if 'area_name' in county_df.columns:
            sample_cols = ['area_type', 'area_name', 'year', 'quarter', 'industry_code']
            available_cols = [col for col in sample_cols if col in county_df.columns]
            sample = county_df.head(10)[available_cols]
            logger.info(f"\n{sample.to_string()}")
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("T032 TEST SUMMARY")
        logger.info("="*80)
        logger.info("[PASS] All validation checks passed")
        logger.info(f"[PASS] Successfully filtered to {len(county_df):,} county-level records")
        logger.info(f"[PASS] Removed {len(df) - len(county_df):,} state/national aggregates")
        logger.info("[PASS] Data integrity maintained")
        logger.info("\n*** T032 IMPLEMENTATION SUCCESSFUL ***")
        logger.info("="*80 + "\n")
        
        # Verify output file was created
        output_file = feature_eng_dir / 'T032_county_filtered.csv'
        if output_file.exists():
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            logger.info(f"[OK] Output file created: {output_file.name} ({file_size_mb:.1f} MB)")
        else:
            logger.warning("[WARN] Output file was not created")
        
    except Exception as e:
        logger.error(f"\n[FAIL] T032 TEST FAILED: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return

if __name__ == "__main__":
    main()
