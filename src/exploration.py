"""
Data Exploration Module

Performs exploratory data analysis and generates visualizations.
"""

import pandas as pd
import numpy as np
# Use non-interactive backend to avoid Tcl/Tk issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict

logger = logging.getLogger(__name__)
sns.set_style("whitegrid")


def perform_eda(df: pd.DataFrame) -> Dict:
    """
    Perform exploratory data analysis on consolidated dataset.
    
    Args:
        df: Consolidated DataFrame
        
    Returns:
        Dictionary with exploration results
    """
    logger.info("\nPerforming exploratory data analysis...")
    
    # Filter to county-level data only for meaningful statistics
    # (State and national aggregates would cause double-counting)
    if 'area_type' in df.columns:
        county_df = df[df['area_type'] == 'County'].copy()
        logger.info(f"  Filtering to county-level records: {len(county_df):,} of {len(df):,} total records")
    else:
        county_df = df.copy()
        logger.info(f"  No area_type filter applied - using all {len(df):,} records")
    
    exploration_results = {
        'shape': df.shape,
        'county_shape': county_df.shape if 'area_type' in df.columns else df.shape,
        'columns': list(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.to_dict()
    }

    # Print key statistics
    logger.info(f"\nDataset Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    if 'area_type' in df.columns:
        logger.info(f"County-Level Records: {county_df.shape[0]:,} rows")
    
    # Check for year and quarter columns
    if 'year' in df.columns:
        exploration_results['date_range'] = (df['year'].min(), df['year'].max())
        logger.info(f"Date Range: {exploration_results['date_range'][0]} - {exploration_results['date_range'][1]}")

    # Analyze employment columns (using county-level data)
    emp_cols = [col for col in county_df.columns if 'emplvl' in col.lower() or 'employment' in col.lower()]
    if emp_cols:
        logger.info(f"\nEmployment Statistics (County-Level Only):")
        for col in emp_cols[:3]:
            if col in county_df.columns:
                stats = county_df[col].describe()
                logger.info(f"  {col}:")
                logger.info(f"    Mean: {stats['mean']:,.0f}")
                logger.info(f"    Median: {stats['50%']:,.0f}")
                logger.info(f"    Min: {stats['min']:,.0f}")
                logger.info(f"    Max: {stats['max']:,.0f}")

    # Analyze wage data (using county-level data)
    wage_cols = [col for col in county_df.columns if 'wage' in col.lower()]
    if wage_cols:
        logger.info(f"\nWage Statistics (County-Level Only):")
        for col in wage_cols[:2]:
            if col in county_df.columns:
                stats = county_df[col].describe()
                logger.info(f"  {col}:")
                logger.info(f"    Mean: ${stats['mean']:,.2f}")
                logger.info(f"    Median: ${stats['50%']:,.2f}")

    return exploration_results


def generate_plots(df: pd.DataFrame, output_dir: Path):
    """
    Generate exploration visualizations.
    
    Args:
        df: Data to visualize
        output_dir: Directory to save plots
    """
    logger.info(f"\nGenerating visualizations in {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter to county-level data to avoid double-counting
    if 'area_type' in df.columns:
        county_df = df[df['area_type'] == 'County'].copy()
        logger.info(f"  Using county-level data for plots ({len(county_df):,} records)")
    else:
        county_df = df.copy()

    try:
        # 1. Employment trends over time
        if 'year' in county_df.columns and 'month1_emplvl' in county_df.columns and 'industry_code' in county_df.columns and 'ownership' in county_df.columns:
            logger.info("  - Creating employment trends plot...")
            fig, ax = plt.subplots(figsize=(12, 6))
            # Filter to "Total, All Industries" (code '10') and "Total Covered" ownership to avoid double-counting
            # Group by year and quarter, sum across counties, then take mean across quarters
            total_industry = county_df[(county_df['industry_code'] == '10') & 
                                      (county_df['ownership'] == 'Total Covered')].copy()
            yearly_quarterly = total_industry.groupby(['year', 'quarter'])['month1_emplvl'].sum()
            yearly_emp = yearly_quarterly.groupby('year').mean() / 1e6
            ax.plot(yearly_emp.index, yearly_emp.values, marker='o', linewidth=2)
            ax.set_xlabel('Year', fontsize=12)
            ax.set_ylabel('Total Employment (Millions)', fontsize=12)
            ax.set_title('California Total Employment Over Time', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'employment_trends.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 2. Quarterly employment distribution
        if 'quarter' in county_df.columns and 'month1_emplvl' in county_df.columns and 'industry_code' in county_df.columns and 'ownership' in county_df.columns:
            logger.info("  - Creating quarterly distribution plot...")
            fig, ax = plt.subplots(figsize=(10, 6))
            # Filter to "Total, All Industries" (code '10') and "Total Covered" ownership
            # Group by year and quarter, sum across counties, then average across years
            total_industry = county_df[(county_df['industry_code'] == '10') & 
                                      (county_df['ownership'] == 'Total Covered')].copy()
            yearly_quarterly = total_industry.groupby(['year', 'quarter'])['month1_emplvl'].sum()
            quarterly_emp = yearly_quarterly.groupby('quarter').mean() / 1e6
            # Sort quarters properly
            quarter_order = ['1st Qtr', '2nd Qtr', '3rd Qtr', '4th Qtr']
            quarterly_emp = quarterly_emp.reindex([q for q in quarter_order if q in quarterly_emp.index])
            ax.bar(range(len(quarterly_emp)), quarterly_emp.values, color='steelblue', alpha=0.7)
            ax.set_xticks(range(len(quarterly_emp)))
            ax.set_xticklabels(quarterly_emp.index)
            ax.set_xlabel('Quarter', fontsize=12)
            ax.set_ylabel('Average Employment (Millions)', fontsize=12)
            ax.set_title('Average Employment Distribution by Quarter', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_dir / 'quarterly_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 3. Average weekly wage trends
        if 'year' in county_df.columns and 'avg_wkly_wage' in county_df.columns and 'industry_code' in county_df.columns and 'ownership' in county_df.columns:
            logger.info("  - Creating wage trends plot...")
            fig, ax = plt.subplots(figsize=(12, 6))
            # Filter to "Total, All Industries" (code '10') and "Total Covered" ownership
            # For wages, we want the average across all counties and quarters (wages are already averages)
            total_industry = county_df[(county_df['industry_code'] == '10') & 
                                      (county_df['ownership'] == 'Total Covered')].copy()
            yearly_wages = total_industry.groupby('year')['avg_wkly_wage'].mean()
            ax.plot(yearly_wages.index, yearly_wages.values, marker='s', linewidth=2, color='green')
            ax.set_xlabel('Year', fontsize=12)
            ax.set_ylabel('Average Weekly Wage ($)', fontsize=12)
            ax.set_title('California Average Weekly Wage Over Time', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'wage_trends.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 4. Top industries by employment
        if 'industry_code' in county_df.columns and 'month1_emplvl' in county_df.columns and 'industry_name' in county_df.columns and 'ownership' in county_df.columns:
            logger.info("  - Creating top industries plot...")
            fig, ax = plt.subplots(figsize=(12, 8))
            # Filter to "Private" ownership (specific industries only exist under Private ownership)
            # Exclude total industry code ('10') and sum across counties/quarters
            industry_data = county_df[(county_df['industry_code'] != '10') & 
                                     (county_df['ownership'] == 'Private')].copy()
            # Group by industry, sum across counties/quarters, then get top 15
            top_industries = industry_data.groupby(['industry_code', 'industry_name'])['month1_emplvl'].sum().sort_values(ascending=True).tail(15)
            # Use industry names for labels (first 40 chars)
            labels = [f"{code}: {name[:40]}" for code, name in top_industries.index]
            ax.barh(range(len(top_industries)), top_industries.values / 1e6, color='coral')
            ax.set_yticks(range(len(top_industries)))
            ax.set_yticklabels(labels, fontsize=9)
            ax.set_xlabel('Total Private Employment (Millions)', fontsize=12)
            ax.set_ylabel('Industry', fontsize=12)
            ax.set_title('Top 15 Private Sector Industries by Employment', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_dir / 'top_industries.png', dpi=300, bbox_inches='tight')
            plt.close()

        logger.info(f"  [OK] All plots saved to {output_dir}")

    except Exception as e:
        logger.error(f"Error generating plots: {e}")
        logger.info("Continuing without plots...")
