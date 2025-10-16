"""
Data Consolidation Module

Handles combining multiple raw CSV files into a single consolidated dataset.
"""

import pandas as pd
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


def consolidate_raw_data(raw_dir: Path, output_file: Path, force_rebuild: bool = False) -> pd.DataFrame:
    """
    Consolidate all raw CSV files into a single master dataset.
    
    Args:
        raw_dir: Directory containing raw CSV files
        output_file: Path to save consolidated file
        force_rebuild: Force rebuild even if file exists
        
    Returns:
        Consolidated DataFrame
    """
    logger.info("\n" + "="*80)
    logger.info("STAGE 1: DATA CONSOLIDATION")
    logger.info("="*80)

    # Check if consolidated file already exists
    if output_file.exists() and not force_rebuild:
        logger.info(f"Loading existing consolidated file: {output_file}")
        df = pd.read_csv(output_file)
        logger.info(f"Loaded {len(df):,} records from consolidated file")
        return df

    # Find all CSV files in raw directory (excluding metadata)
    csv_files = sorted([
        f for f in raw_dir.glob('*.csv')
        if 'metadata' not in f.name.lower()
    ])

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}. "
            "Please ensure raw data files are present."
        )

    logger.info(f"Found {len(csv_files)} raw CSV files:")
    for file in csv_files:
        logger.info(f"  - {file.name} ({file.stat().st_size / (1024*1024):.1f} MB)")

    # Load and concatenate all files
    logger.info("\nLoading and combining all CSV files...")
    all_dfs = []
    total_rows = 0

    for csv_file in csv_files:
        logger.info(f"Reading {csv_file.name}...")
        df_chunk = pd.read_csv(csv_file, low_memory=False)
        rows = len(df_chunk)
        total_rows += rows
        all_dfs.append(df_chunk)
        logger.info(f"  [OK] Loaded {rows:,} rows")

    # Concatenate all dataframes
    logger.info("\nCombining all datasets...")
    consolidated_df = pd.concat(all_dfs, ignore_index=True)

    # Normalize column names to lowercase with underscores
    consolidated_df.columns = consolidated_df.columns.str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
    
    # Map column names for consistency
    column_mapping = {
        'time_period': 'quarter',
        'naics_level': 'naics_level',
        'naics_code': 'industry_code',
        'establishments': 'qtrly_estabs',
        'average_monthly_employment': 'avg_monthly_emplvl',
        '1st_month_emp': 'month1_emplvl',
        '2nd_month_emp': 'month2_emplvl',
        '3rd_month_emp': 'month3_emplvl',
        'total_wages_all_workers': 'total_qtrly_wages',
        'average_weekly_wages': 'avg_wkly_wage'
    }
    consolidated_df.rename(columns=column_mapping, inplace=True)

    # Basic info
    logger.info(f"\n[OK] Successfully consolidated {len(csv_files)} files")
    logger.info(f"  Total records: {len(consolidated_df):,}")
    logger.info(f"  Columns: {len(consolidated_df.columns)}")
    
    # Check if year column exists
    if 'year' in consolidated_df.columns:
        logger.info(f"  Date range: {consolidated_df['year'].min()}-{consolidated_df['year'].max()}")
    logger.info(f"  Memory usage: {consolidated_df.memory_usage(deep=True).sum() / (1024*1024):.1f} MB")

    # Save consolidated dataset
    logger.info(f"\nSaving consolidated dataset to: {output_file}")
    consolidated_df.to_csv(output_file, index=False)
    logger.info("[OK] Consolidated dataset saved successfully")

    # VERIFY RAW DATA REMAINS UNCHANGED
    logger.info("\n[WARNING] VERIFICATION: Ensuring raw data files remain unmodified...")
    for csv_file in csv_files:
        if csv_file.stat().st_mtime > datetime.now().timestamp() - 60:
            logger.warning(f"  WARNING: {csv_file.name} was recently modified!")
        else:
            logger.info(f"  [OK] {csv_file.name} - unchanged")

    return consolidated_df
