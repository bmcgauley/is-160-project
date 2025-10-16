"""
Data Exploration Module

Performs exploratory data analysis and generates visualizations.
"""

import pandas as pd
import numpy as np
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
    
    exploration_results = {
        'shape': df.shape,
        'columns': list(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.to_dict()
    }

    # Print key statistics
    logger.info(f"\nDataset Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    
    # Check for year and quarter columns
    if 'year' in df.columns:
        exploration_results['date_range'] = (df['year'].min(), df['year'].max())
        logger.info(f"Date Range: {exploration_results['date_range'][0]} - {exploration_results['date_range'][1]}")

    # Analyze employment columns
    emp_cols = [col for col in df.columns if 'emplvl' in col.lower() or 'employment' in col.lower()]
    if emp_cols:
        logger.info(f"\nEmployment Statistics:")
        for col in emp_cols[:3]:
            if col in df.columns:
                stats = df[col].describe()
                logger.info(f"  {col}:")
                logger.info(f"    Mean: {stats['mean']:,.0f}")
                logger.info(f"    Median: {stats['50%']:,.0f}")
                logger.info(f"    Min: {stats['min']:,.0f}")
                logger.info(f"    Max: {stats['max']:,.0f}")

    # Analyze wage data
    wage_cols = [col for col in df.columns if 'wage' in col.lower()]
    if wage_cols:
        logger.info(f"\nWage Statistics:")
        for col in wage_cols[:2]:
            if col in df.columns:
                stats = df[col].describe()
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

    try:
        # 1. Employment trends over time
        if 'year' in df.columns and 'month1_emplvl' in df.columns:
            logger.info("  - Creating employment trends plot...")
            fig, ax = plt.subplots(figsize=(12, 6))
            yearly_emp = df.groupby('year')['month1_emplvl'].sum() / 1e6
            ax.plot(yearly_emp.index, yearly_emp.values, marker='o', linewidth=2)
            ax.set_xlabel('Year', fontsize=12)
            ax.set_ylabel('Total Employment (Millions)', fontsize=12)
            ax.set_title('California Total Employment Over Time', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'employment_trends.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 2. Quarterly employment distribution
        if 'quarter' in df.columns and 'month1_emplvl' in df.columns:
            logger.info("  - Creating quarterly distribution plot...")
            fig, ax = plt.subplots(figsize=(10, 6))
            quarterly_emp = df.groupby('quarter')['month1_emplvl'].sum() / 1e6
            ax.bar(quarterly_emp.index, quarterly_emp.values, color='steelblue', alpha=0.7)
            ax.set_xlabel('Quarter', fontsize=12)
            ax.set_ylabel('Total Employment (Millions)', fontsize=12)
            ax.set_title('Employment Distribution by Quarter', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_dir / 'quarterly_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 3. Average weekly wage trends
        if 'year' in df.columns and 'avg_wkly_wage' in df.columns:
            logger.info("  - Creating wage trends plot...")
            fig, ax = plt.subplots(figsize=(12, 6))
            yearly_wages = df.groupby('year')['avg_wkly_wage'].mean()
            ax.plot(yearly_wages.index, yearly_wages.values, marker='s', linewidth=2, color='green')
            ax.set_xlabel('Year', fontsize=12)
            ax.set_ylabel('Average Weekly Wage ($)', fontsize=12)
            ax.set_title('California Average Weekly Wage Over Time', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'wage_trends.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 4. Top industries by employment
        if 'industry_code' in df.columns and 'month1_emplvl' in df.columns:
            logger.info("  - Creating top industries plot...")
            fig, ax = plt.subplots(figsize=(12, 8))
            top_industries = df.groupby('industry_code')['month1_emplvl'].sum().sort_values(ascending=True).tail(15)
            ax.barh(range(len(top_industries)), top_industries.values / 1e6, color='coral')
            ax.set_yticks(range(len(top_industries)))
            ax.set_yticklabels(top_industries.index)
            ax.set_xlabel('Total Employment (Millions)', fontsize=12)
            ax.set_ylabel('Industry Code', fontsize=12)
            ax.set_title('Top 15 Industries by Employment', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_dir / 'top_industries.png', dpi=300, bbox_inches='tight')
            plt.close()

        logger.info(f"  [OK] All plots saved to {output_dir}")

    except Exception as e:
        logger.error(f"Error generating plots: {e}")
        logger.info("Continuing without plots...")
