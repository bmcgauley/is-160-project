"""
Feature Engineering Pipeline Module

This module provides the high-level feature engineering pipeline
that orchestrates all feature engineering tasks (T032-T052).

Calls individual functions from feature_engineering.py, temporal_features.py,
and geographic_features.py modules.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from typing import Optional

from feature_engineering import (
    filter_to_county_level,
    handle_annual_vs_quarterly,
    data_quality_filter,
    calculate_quarterly_growth_rates,
    create_seasonal_adjustments,
    calculate_industry_concentration,
    build_geographic_clustering,
    generate_lag_features
)

logger = logging.getLogger(__name__)

# Set up matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def create_feature_plots(df: pd.DataFrame, plots_dir: Path, stage: str):
    """
    Create visualization plots for feature engineering stages.
    
    Args:
        df: Current dataframe
        plots_dir: Directory to save plots
        stage: Name of the current stage (for filename)
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    if stage == "growth_rates":
        # Plot growth rate distributions
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Growth Rate Feature Distributions', fontsize=16, fontweight='bold')
        
        # QoQ % change
        axes[0, 0].hist(df['employment_qoq_pct_change'].dropna(), bins=100, edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Quarter-over-Quarter % Change')
        axes[0, 0].set_xlabel('% Change')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='No change')
        axes[0, 0].legend()
        
        # YoY % change
        axes[0, 1].hist(df['employment_yoy_pct_change'].dropna(), bins=100, edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('Year-over-Year % Change')
        axes[0, 1].set_xlabel('% Change')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2, label='No change')
        axes[0, 1].legend()
        
        # QoQ absolute change
        axes[1, 0].hist(df['employment_qoq_change'].dropna(), bins=100, edgecolor='black', alpha=0.7)
        axes[1, 0].set_title('Quarter-over-Quarter Absolute Change')
        axes[1, 0].set_xlabel('Change in Employment')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2)
        
        # YoY absolute change  
        axes[1, 1].hist(df['employment_yoy_change'].dropna(), bins=100, edgecolor='black', alpha=0.7)
        axes[1, 1].set_title('Year-over-Year Absolute Change')
        axes[1, 1].set_xlabel('Change in Employment')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'T038_growth_rates_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  [OK] Saved growth rate plots to {plots_dir / 'T038_growth_rates_distribution.png'}")
        
    elif stage == "lag_features":
        # Plot lag feature correlations
        lag_cols = [col for col in df.columns if 'lag_' in col]
        if lag_cols and 'avg_monthly_emplvl' in df.columns:
            corr_data = df[['avg_monthly_emplvl'] + lag_cols].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_data, annot=True, fmt='.3f', cmap='coolwarm', center=0, 
                       square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
            ax.set_title('Correlation: Current Employment vs Lag Features', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(plots_dir / 'T042_lag_features_correlation.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"  [OK] Saved lag correlation heatmap to {plots_dir / 'T042_lag_features_correlation.png'}")


def engineer_features(df: pd.DataFrame, 
                     output_file: Path,
                     feature_eng_dir: Path = None) -> pd.DataFrame:
    """
    Run the complete feature engineering pipeline.
    
    This function orchestrates all feature engineering tasks in sequence:
    - T032: Filter to county-level records
    - T033: Handle Annual vs quarterly records
    - T034: Data quality filtering
    - T035: Central Valley counties subset
    - T036: All California counties with features
    - T038: Calculate quarterly growth rates
    - T039: Create seasonal adjustments
    - T040: Industry concentration metrics
    - T041: Geographic clustering
    - T042: Lag features
    
    Args:
        df: Consolidated QCEW dataframe
        output_file: Path to save final feature-engineered dataset
        feature_eng_dir: Directory for intermediate outputs
        
    Returns:
        DataFrame with all engineered features
    """
    logger.info("\n" + "="*80)
    logger.info("FEATURE ENGINEERING PIPELINE")
    logger.info("="*80)
    
    if feature_eng_dir is None:
        feature_eng_dir = output_file.parent.parent / 'feature_engineering'
        feature_eng_dir.mkdir(parents=True, exist_ok=True)
    
    # T032: Filter to county-level records
    logger.info("\n[STAGE 1/10] T032: Filtering to county-level records...")
    df = filter_to_county_level(df, output_dir=feature_eng_dir)
    logger.info(f"[OK] County filtering complete: {len(df):,} records")
    
    # T033: Handle Annual vs quarterly records
    logger.info("\n[STAGE 2/10] T033: Handling Annual vs quarterly records...")
    df = handle_annual_vs_quarterly(df, output_dir=feature_eng_dir)
    logger.info(f"[OK] Quarterly filtering complete: {len(df):,} records")
    
    # T034: Data quality filtering
    logger.info("\n[STAGE 3/10] T034: Data quality filtering...")
    df = data_quality_filter(df, output_dir=feature_eng_dir)
    logger.info(f"[OK] Quality filtering complete: {len(df):,} records")
    
    # T035: Central Valley counties subset
    logger.info("\n[STAGE 4/10] T035: Central Valley counties subset...")
    logger.info("[PENDING] Not yet implemented - will be added in next iteration")
    
    # T036: All California counties with features
    logger.info("\n[STAGE 5/10] T036: All California counties processing...")
    logger.info("[PENDING] Not yet implemented - will be added in next iteration")
    
    # T038: Calculate quarterly growth rates
    logger.info("\n[STAGE 6/10] T038: Calculating quarterly growth rates...")
    df = calculate_quarterly_growth_rates(df)
    logger.info(f"[OK] Growth rates complete: {len(df):,} records")
    
    # Create visualization for growth rates
    plots_dir = feature_eng_dir / 'plots'
    logger.info("\n[VISUALIZATION] Creating growth rate distribution plots...")
    try:
        create_feature_plots(df, plots_dir, "growth_rates")
        logger.info(f"[OK] Growth rate plots saved to {plots_dir / 'T038_growth_rates_distribution.png'}")
    except Exception as e:
        logger.error(f"[ERROR] Failed to create growth rate plots: {e}")
        logger.exception("Full traceback:")
    
    # T039: Create seasonal adjustments
    logger.info("\n[STAGE 7/10] T039: Creating seasonal adjustments...")
    logger.info("[PENDING] Seasonal adjustments - advanced feature, will be added in next iteration")
    # df = create_seasonal_adjustments(df)
    
    # T040: Industry concentration metrics
    logger.info("\n[STAGE 8/10] T040: Calculating industry concentration...")
    logger.info("[PENDING] Industry concentration - advanced feature, will be added in next iteration")
    # df = calculate_industry_concentration(df)
    
    # T041: Geographic clustering
    logger.info("\n[STAGE 9/10] T041: Building geographic clustering...")
    logger.info("[PENDING] Geographic clustering - advanced feature, will be added in next iteration")
    # df = build_geographic_clustering(df)
    
    # T042: Lag features
    logger.info("\n[STAGE 10/10] T042: Generating lag features...")
    df = generate_lag_features(df)
    logger.info(f"[OK] Lag features complete: {len(df):,} records")
    
    # Create visualization for lag features
    logger.info("\n[VISUALIZATION] Creating lag feature correlation heatmap...")
    try:
        create_feature_plots(df, plots_dir, "lag_features")
        logger.info(f"[OK] Lag correlation heatmap saved to {plots_dir / 'T042_lag_features_correlation.png'}")
    except Exception as e:
        logger.error(f"[ERROR] Failed to create lag feature plots: {e}")
        logger.exception("Full traceback:")
    
    # Save final output
    logger.info(f"\nSaving feature-engineered dataset to: {output_file}")
    df.to_csv(output_file, index=False)
    logger.info(f"[OK] Saved {len(df):,} records to {output_file.name}")
    
    # Summary of features added
    logger.info("\n" + "="*80)
    logger.info("FEATURE ENGINEERING COMPLETE")
    logger.info("="*80)
    logger.info("\nImplemented tasks:")
    logger.info("  [X] T032: County-level filtering (removed state/national aggregates)")
    logger.info("  [X] T033: Quarterly filtering (removed Annual records)")
    logger.info("  [X] T034: Data quality filtering (removed invalid records)")
    logger.info("  [X] T038: Growth rates (added 6 columns: QoQ & YoY metrics)")
    logger.info("  [X] T042: Lag features (added 4 columns: 1-4 quarters back)")
    logger.info("\nDeferred tasks (advanced features):")
    logger.info("  [ ] T039: Seasonal adjustments")
    logger.info("  [ ] T040: Industry concentration")
    logger.info("  [ ] T041: Geographic clustering")
    logger.info("\nTotal new features added: 10 columns")
    logger.info(f"Final dataset: {len(df):,} records × {len(df.columns)} columns")
    logger.info("="*80 + "\n")
    
    return df
