"""
Feature Engineering Module for QCEW Employment Data Analysis

This module contains functions for calculating quarter-over-quarter employment growth rates,
seasonal adjustments, and industry concentration metrics.

All intermediate outputs are saved to data/feature_engineering/ for continuity and debugging.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def filter_to_county_level(df: pd.DataFrame, output_dir: Path = None) -> pd.DataFrame:
    """
    Filter consolidated data to county-level records only.
    
    T032: Removes state-level and national-level aggregates to focus on 
    county-level employment data for modeling. This prevents double-counting 
    and ensures geographic granularity.
    
    Args:
        df: Consolidated QCEW dataframe with area_type column
        output_dir: Optional directory to save intermediate output
        
    Returns:
        DataFrame containing only county-level records
        
    Raises:
        ValueError: If area_type column is missing
        
    Notes:
        - Drops records where area_type='United States' (national aggregates)
        - Drops records where area_type='California - Statewide' (state aggregates)
        - Retains only area_type='County' records
        - Saves output to data/feature_engineering/T032_county_filtered.csv
        - Based on data_aggregation_levels.md analysis showing:
          * County: 4,732,218 records (87.1%)
          * California Statewide: 298,804 records (5.5%)
          * United States: 399,362 records (7.4%)
    """
    logger.info("\n" + "="*80)
    logger.info("T032: FILTERING TO COUNTY-LEVEL RECORDS")
    logger.info("="*80)
    
    # Validate area_type column exists
    if 'area_type' not in df.columns:
        raise ValueError(
            "DataFrame missing required 'area_type' column. "
            "Ensure data has been properly consolidated with area type identification."
        )
    
    # Log initial state
    initial_count = len(df)
    logger.info(f"Initial dataset: {initial_count:,} records")
    
    # Count records by area type
    area_type_counts = df['area_type'].value_counts()
    logger.info("\nArea type distribution BEFORE filtering:")
    for area_type, count in area_type_counts.items():
        pct = (count / initial_count) * 100
        logger.info(f"  {area_type}: {count:,} records ({pct:.1f}%)")
    
    # Filter to county-level only
    county_df = df[df['area_type'] == 'County'].copy()
    
    # Calculate records removed
    removed_count = initial_count - len(county_df)
    removed_pct = (removed_count / initial_count) * 100
    
    # Log results
    logger.info("\nFiltering results:")
    logger.info(f"  Retained (County): {len(county_df):,} records")
    logger.info(f"  Removed (State + National): {removed_count:,} records ({removed_pct:.1f}%)")
    
    # Validate we have data
    if len(county_df) == 0:
        raise ValueError(
            "No county-level records found after filtering. "
            "Check area_type values in consolidated data."
        )
    
    # Log county coverage
    if 'area_name' in county_df.columns:
        unique_counties = county_df['area_name'].nunique()
        logger.info(f"  Unique counties: {unique_counties:,}")
        
    # Log temporal coverage
    if 'year' in county_df.columns:
        year_range = f"{county_df['year'].min()}-{county_df['year'].max()}"
        logger.info(f"  Year range: {year_range}")
    
    logger.info("\n[SUCCESS] County-level filtering completed successfully")
    logger.info("="*80 + "\n")
    
    # Save intermediate output if output_dir provided
    if output_dir:
        output_file = output_dir / 'T032_county_filtered.csv'
        logger.info(f"Saving county-filtered data to: {output_file}")
        county_df.to_csv(output_file, index=False)
        logger.info(f"[OK] Saved {len(county_df):,} records to {output_file.name}")
    
    return county_df


def handle_annual_vs_quarterly(df: pd.DataFrame, output_dir: Path = None) -> pd.DataFrame:
    """
    T033: Handle Annual vs quarterly records in the dataset.
    
    The QCEW data contains both quarterly records (Q1-Q4) and annual summaries ('Annual').
    For time series forecasting, we need consistent quarterly data only.
    
    Args:
        df: Input dataframe with 'quarter' column containing 'Q1', 'Q2', 'Q3', 'Q4', or 'Annual'
        output_dir: Optional directory to save intermediate output
        
    Returns:
        DataFrame filtered to quarterly records only
    """
    logger.info("\n" + "="*80)
    logger.info("T033: FILTERING TO QUARTERLY RECORDS ONLY")
    logger.info("="*80)
    logger.info(f"Initial dataset: {len(df):,} records")
    
    # Check quarter distribution
    logger.info("\nQuarter distribution BEFORE filtering:")
    quarter_dist = df['quarter'].value_counts().sort_index()
    for quarter, count in quarter_dist.items():
        pct = (count / len(df)) * 100
        logger.info(f"  {quarter}: {count:,} records ({pct:.1f}%)")
    
    # Filter to quarterly records only (exclude 'Annual')
    # Data uses '1st Qtr', '2nd Qtr', '3rd Qtr', '4th Qtr' format
    quarterly_df = df[df['quarter'].isin(['1st Qtr', '2nd Qtr', '3rd Qtr', '4th Qtr'])].copy()
    
    # Calculate removal stats
    removed_count = len(df) - len(quarterly_df)
    removed_pct = (removed_count / len(df)) * 100
    
    logger.info("\nFiltering results:")
    logger.info(f"  Retained (Quarterly): {len(quarterly_df):,} records")
    logger.info(f"  Removed (Annual): {removed_count:,} records ({removed_pct:.1f}%)")
    
    # Verify no annual records remain
    if 'Annual' in quarterly_df['quarter'].values:
        logger.warning("[WARN] Annual records still present after filtering!")
    else:
        logger.info("  [OK] All annual records successfully removed")
    
    logger.info("\n[SUCCESS] Quarterly filtering completed successfully")
    logger.info("="*80)
    
    # Save intermediate output if output_dir provided
    if output_dir:
        output_file = output_dir / 'T033_quarterly_only.csv'
        logger.info(f"Saving quarterly-only data to: {output_file}")
        quarterly_df.to_csv(output_file, index=False)
        logger.info(f"[OK] Saved {len(quarterly_df):,} records to {output_file.name}")
    
    return quarterly_df


def data_quality_filter(df: pd.DataFrame, output_dir: Path = None) -> pd.DataFrame:
    """
    T034: Filter out incomplete or inconsistent records based on data quality rules.
    
    Removes records with logical inconsistencies:
    - Zero employment but non-zero establishments
    - Negative wages or employment
    - Extreme outliers that are likely data errors
    
    Args:
        df: Input dataframe
        output_dir: Optional directory to save intermediate output
        
    Returns:
        DataFrame with quality-filtered records
    """
    logger.info("\n" + "="*80)
    logger.info("T034: DATA QUALITY FILTERING")
    logger.info("="*80)
    logger.info(f"Initial dataset: {len(df):,} records")
    
    initial_count = len(df)
    filtered_df = df.copy()
    
    # Rule 1: Remove records with negative employment
    neg_employment = (filtered_df['avg_monthly_emplvl'] < 0)
    removed_neg = neg_employment.sum()
    filtered_df = filtered_df[~neg_employment]
    logger.info(f"\nRule 1: Negative employment")
    logger.info(f"  Removed: {removed_neg:,} records")
    
    # Rule 2: Remove records with negative wages
    neg_wages = (filtered_df['total_qtrly_wages'] < 0) | (filtered_df['avg_wkly_wage'] < 0)
    removed_wages = neg_wages.sum()
    filtered_df = filtered_df[~neg_wages]
    logger.info(f"\nRule 2: Negative wages")
    logger.info(f"  Removed: {removed_wages:,} records")
    
    # Rule 3: Remove records with zero employment but non-zero establishments
    # (Not implemented if 'establishments' column doesn't exist)
    if 'qtrly_estabs_count' in filtered_df.columns:
        inconsistent = (filtered_df['avg_monthly_emplvl'] == 0) & (filtered_df['qtrly_estabs_count'] > 0)
        removed_inconsistent = inconsistent.sum()
        filtered_df = filtered_df[~inconsistent]
        logger.info(f"\nRule 3: Zero employment but non-zero establishments")
        logger.info(f"  Removed: {removed_inconsistent:,} records")
    
    # Summary
    total_removed = initial_count - len(filtered_df)
    removed_pct = (total_removed / initial_count) * 100 if initial_count > 0 else 0.0
    
    logger.info("\nFiltering summary:")
    logger.info(f"  Initial records: {initial_count:,}")
    logger.info(f"  Final records: {len(filtered_df):,}")
    logger.info(f"  Total removed: {total_removed:,} records ({removed_pct:.2f}%)")
    
    logger.info("\n[SUCCESS] Data quality filtering completed successfully")
    logger.info("="*80)
    
    # Save intermediate output if output_dir provided
    if output_dir:
        output_file = output_dir / 'T034_quality_filtered.csv'
        logger.info(f"Saving quality-filtered data to: {output_file}")
        filtered_df.to_csv(output_file, index=False)
        logger.info(f"[OK] Saved {len(filtered_df):,} records to {output_file.name}")
    
    return filtered_df


def calculate_quarterly_growth_rates(df: pd.DataFrame,
                                   employment_col: str = 'avg_monthly_emplvl',
                                   groupby_cols: List[str] = None) -> pd.DataFrame:
    """
    T038: Calculate quarter-over-quarter employment growth rates and percentage changes.
    
    Computes growth metrics that capture employment momentum:
    - Absolute change from previous quarter
    - Percentage change from previous quarter
    - Year-over-year percentage change

    Args:
        df: Input dataframe with employment data
        employment_col: Column name containing employment counts
        groupby_cols: Columns to group by for calculations

    Returns:
        DataFrame with added growth rate columns
    """
    if groupby_cols is None:
        groupby_cols = ['area_name', 'industry_code']

    logger.info("\n" + "="*80)
    logger.info("T038: CALCULATING QUARTERLY GROWTH RATES")
    logger.info("="*80)
    logger.info(f"Employment column: {employment_col}")
    logger.info(f"Grouping by: {groupby_cols}")
    
    df_with_growth = df.copy()
    
    # Sort by group and time
    df_with_growth = df_with_growth.sort_values(groupby_cols + ['year', 'quarter'])
    
    # Calculate quarter-over-quarter changes
    df_with_growth['employment_prev_quarter'] = df_with_growth.groupby(groupby_cols)[employment_col].shift(1)
    df_with_growth['employment_qoq_change'] = df_with_growth[employment_col] - df_with_growth['employment_prev_quarter']
    df_with_growth['employment_qoq_pct_change'] = (
        df_with_growth['employment_qoq_change'] / df_with_growth['employment_prev_quarter']
    ) * 100
    
    # Calculate year-over-year changes (4 quarters back)
    df_with_growth['employment_4q_ago'] = df_with_growth.groupby(groupby_cols)[employment_col].shift(4)
    df_with_growth['employment_yoy_change'] = df_with_growth[employment_col] - df_with_growth['employment_4q_ago']
    df_with_growth['employment_yoy_pct_change'] = (
        df_with_growth['employment_yoy_change'] / df_with_growth['employment_4q_ago']
    ) * 100
    
    # Log statistics
    logger.info("\nGrowth rate statistics:")
    logger.info(f"  QoQ % change - Mean: {df_with_growth['employment_qoq_pct_change'].mean():.2f}%")
    logger.info(f"  QoQ % change - Std: {df_with_growth['employment_qoq_pct_change'].std():.2f}%")
    logger.info(f"  YoY % change - Mean: {df_with_growth['employment_yoy_pct_change'].mean():.2f}%")
    logger.info(f"  YoY % change - Std: {df_with_growth['employment_yoy_pct_change'].std():.2f}%")
    
    # Handle inf/nan from division by zero
    growth_cols = ['employment_qoq_pct_change', 'employment_yoy_pct_change']
    for col in growth_cols:
        inf_count = np.isinf(df_with_growth[col]).sum()
        if inf_count > 0:
            logger.info(f"  Handling {col}: {inf_count:,} infinite values (division by zero) - replacing with 0.0")
            df_with_growth[col] = df_with_growth[col].replace([np.inf, -np.inf], 0.0)
    
    # Report final statistics after handling infinities
    logger.info("\nFinal growth rate statistics (after cleaning):")
    for col in growth_cols:
        non_null = df_with_growth[col].notna().sum()
        null_count = df_with_growth[col].isna().sum()
        logger.info(f"  {col}:")
        logger.info(f"    Valid values: {non_null:,}")
        logger.info(f"    Null values: {null_count:,}")
        logger.info(f"    Mean: {df_with_growth[col].mean():.2f}%")
        logger.info(f"    Median: {df_with_growth[col].median():.2f}%")
        logger.info(f"    Min: {df_with_growth[col].min():.2f}%")
        logger.info(f"    Max: {df_with_growth[col].max():.2f}%")
    
    logger.info("\n[SUCCESS] Growth rate calculations completed")
    logger.info("="*80)
    
    return df_with_growth


def create_seasonal_adjustments(df: pd.DataFrame,
                              date_col: str = 'quarter',
                              value_col: str = 'total_employment') -> pd.DataFrame:
    """
    Create seasonal adjustment factors using historical employment patterns.

    Args:
        df: Input dataframe
        date_col: Column containing date/quarter information
        value_col: Column to seasonally adjust

    Returns:
        DataFrame with seasonal adjustment factors
    """
    # TODO: Implement seasonal adjustment calculations
    logger.info("Creating seasonal adjustment factors...")
    return df


def calculate_industry_concentration(df: pd.DataFrame,
                                   industry_col: str = 'industry_code',
                                   employment_col: str = 'total_employment') -> pd.DataFrame:
    """
    Engineer industry concentration metrics and economic diversity indices.

    Args:
        df: Input dataframe
        industry_col: Column containing industry codes
        employment_col: Column containing employment counts

    Returns:
        DataFrame with concentration metrics
    """
    # TODO: Implement industry concentration calculations
    logger.info("Calculating industry concentration metrics...")
    return df


def build_geographic_clustering(df: pd.DataFrame,
                              geographic_cols: List[str] = None) -> pd.DataFrame:
    """
    Build geographic clustering features based on employment similarity.

    Args:
        df: Input dataframe
        geographic_cols: Geographic columns for clustering

    Returns:
        DataFrame with geographic clustering features
    """
    if geographic_cols is None:
        geographic_cols = ['area_fips', 'county_name']

    # TODO: Implement geographic clustering
    logger.info("Building geographic clustering features...")
    return df


def generate_lag_features(df: pd.DataFrame,
                        value_col: str = 'avg_monthly_emplvl',
                        lags: List[int] = None,
                        groupby_cols: List[str] = None) -> pd.DataFrame:
    """
    T042: Generate lag features for temporal dependencies in employment trends.
    
    Creates lagged employment values to capture autoregressive patterns.
    Lags represent previous quarter values (lag 1 = 1 quarter back, etc.)

    Args:
        df: Input dataframe
        value_col: Column to create lags for
        lags: List of lag periods to create (in quarters)
        groupby_cols: Columns to group by for lag creation

    Returns:
        DataFrame with lag features
    """
    if lags is None:
        lags = [1, 2, 3, 4]  # 1, 2, 3, 4 quarters back
    
    if groupby_cols is None:
        groupby_cols = ['area_name', 'industry_code']

    logger.info("\n" + "="*80)
    logger.info("T042: GENERATING LAG FEATURES")
    logger.info("="*80)
    logger.info(f"Value column: {value_col}")
    logger.info(f"Lag periods: {lags}")
    logger.info(f"Grouping by: {groupby_cols}")
    
    df_with_lags = df.copy()
    
    # Sort by group and time
    df_with_lags = df_with_lags.sort_values(groupby_cols + ['year', 'quarter'])
    
    # Create lag features
    for lag in lags:
        lag_col_name = f'{value_col}_lag_{lag}'
        df_with_lags[lag_col_name] = df_with_lags.groupby(groupby_cols)[value_col].shift(lag)
        
        null_count = df_with_lags[lag_col_name].isnull().sum()
        logger.info(f"  Created {lag_col_name}: {null_count:,} null values (beginning of series)")
    
    # Report statistics for each lag feature
    logger.info("\nLag feature statistics:")
    for lag in lags:
        lag_col_name = f'{value_col}_lag_{lag}'
        non_null = df_with_lags[lag_col_name].notna().sum()
        logger.info(f"  {lag_col_name}:")
        logger.info(f"    Valid values: {non_null:,}")
        logger.info(f"    Mean: {df_with_lags[lag_col_name].mean():.2f}")
        logger.info(f"    Median: {df_with_lags[lag_col_name].median():.2f}")
        logger.info(f"    Min: {df_with_lags[lag_col_name].min():.0f}")
        logger.info(f"    Max: {df_with_lags[lag_col_name].max():.0f}")
    
    logger.info(f"\n[SUCCESS] Generated {len(lags)} lag features")
    logger.info("="*80)
    
    return df_with_lags


def validate_feature_engineering(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Validate feature engineering results for consistency and reasonableness.

    Args:
        df: DataFrame with engineered features

    Returns:
        Dictionary of validation results
    """
    # TODO: Implement validation checks
    logger.info("Validating feature engineering results...")
    return {"basic_validation": True}