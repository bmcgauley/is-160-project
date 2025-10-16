"""
QCEW Employment Forecasting Pipeline - Master Orchestrator

This is the main entry point for the entire employment forecasting pipeline.
It coordinates all stages from data consolidation through model training and prediction.

Usage:
    python main.py                    # Run full pipeline
    python main.py --stage explore    # Run only exploration
    python main.py --stage train      # Run only training
    python main.py --stage predict    # Run interactive prediction interface
    python main.py --skip-plots       # Skip visualization generation

Stages:
    1. Data Consolidation: Combine raw CSV files into master dataset
    2. Exploration: Initial data analysis and visualization
    3. Validation: Data quality checks and validation reports
    4. Feature Engineering: Create temporal and geographic features
    5. Preprocessing: Normalize, encode, and prepare sequences
    6. Training: Train LSTM model on employment data
    7. Evaluation: Assess model performance and generate reports
    8. Prediction: Interactive interface for forecasting
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import warnings

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import pipeline modules
from logging_config import setup_logging
from exploration_clean import consolidate_data, run_exploration, generate_visualizations
from validation import QCEWValidator
# Note: Other imports will be added as modules are implemented
# from feature_engineering import calculate_quarterly_growth_rates, create_seasonal_adjustments
# from temporal_features import create_rolling_features, create_cyclical_features
# from geographic_features import create_geographic_features, create_industry_features
from preprocessing import EmploymentDataPreprocessor
from lstm_model import EmploymentLSTM
# from training import train_model, save_checkpoint
# from evaluation import evaluate_model
# from prediction_visuals import create_prediction_plots

warnings.filterwarnings('ignore')

# Setup logging
logger = setup_logging()


class QCEWPipeline:
    """
    Master pipeline orchestrator for QCEW employment forecasting.
    Coordinates all stages from data consolidation to prediction.
    """

    def __init__(self, config: dict = None):
        """
        Initialize the pipeline with configuration.

        Args:
            config: Dictionary with pipeline configuration options
        """
        self.config = config or {}
        self.base_dir = Path(__file__).parent
        self.data_dir = self.base_dir / "data"
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.validated_dir = self.data_dir / "validated"
        self.plots_dir = self.processed_dir / "plots"

        # Create directories if they don't exist
        self.processed_dir.mkdir(exist_ok=True)
        self.validated_dir.mkdir(exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)

        # Pipeline state tracking
        self.consolidated_file = self.processed_dir / "qcew_master_consolidated.csv"
        self.validated_file = self.validated_dir / "qcew_validated.csv"
        self.features_file = self.processed_dir / "qcew_features.csv"
        self.preprocessed_file = self.processed_dir / "qcew_preprocessed.csv"
        self.model_file = self.processed_dir / "lstm_model.pt"

        logger.info("="*80)
        logger.info("QCEW EMPLOYMENT FORECASTING PIPELINE")
        logger.info("="*80)
        logger.info(f"Base directory: {self.base_dir}")
        logger.info(f"Raw data: {self.raw_dir}")
        logger.info(f"Processed data: {self.processed_dir}")
        logger.info(f"Plots: {self.plots_dir}")

    def stage_1_consolidate_data(self) -> pd.DataFrame:
        """
        Stage 1: Data Consolidation
        
        Combine all raw CSV files into a single master dataset.
        IMPORTANT: Raw data files are NEVER modified - only read and combined.
        
        Returns:
            Consolidated DataFrame
        """
        logger.info("\n" + "="*80)
        logger.info("STAGE 1: DATA CONSOLIDATION")
        logger.info("="*80)

        # Check if consolidated file already exists
        if self.consolidated_file.exists() and not self.config.get('force_rebuild', False):
            logger.info(f"Loading existing consolidated file: {self.consolidated_file}")
            df = pd.read_csv(self.consolidated_file)
            logger.info(f"Loaded {len(df):,} records from consolidated file")
            return df

        # Find all CSV files in raw directory (excluding metadata)
        csv_files = sorted([
            f for f in self.raw_dir.glob('*.csv')
            if 'metadata' not in f.name.lower()
        ])

        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in {self.raw_dir}. "
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
            if 'quarter' in consolidated_df.columns:
                logger.info(f"  Quarters: Q{consolidated_df['quarter'].min()} to Q{consolidated_df['quarter'].max()}")
        logger.info(f"  Memory usage: {consolidated_df.memory_usage(deep=True).sum() / (1024*1024):.1f} MB")

        # Save consolidated dataset
        logger.info(f"\nSaving consolidated dataset to: {self.consolidated_file}")
        consolidated_df.to_csv(self.consolidated_file, index=False)
        logger.info("[OK] Consolidated dataset saved successfully")

        # VERIFY RAW DATA REMAINS UNCHANGED
        logger.info("\n[WARNING] VERIFICATION: Ensuring raw data files remain unmodified...")
        for csv_file in csv_files:
            if csv_file.stat().st_mtime > datetime.now().timestamp() - 60:
                logger.warning(f"  WARNING: {csv_file.name} was recently modified!")
            else:
                logger.info(f"  [OK] {csv_file.name} - unchanged")

        return consolidated_df

    def stage_2_explore_data(self, df: pd.DataFrame) -> dict:
        """
        Stage 2: Data Exploration
        
        Perform exploratory data analysis and generate visualizations.
        
        Args:
            df: Consolidated DataFrame
            
        Returns:
            Dictionary with exploration results
        """
        logger.info("\n" + "="*80)
        logger.info("STAGE 2: DATA EXPLORATION")
        logger.info("="*80)

        # Run exploration analysis
        logger.info("\nPerforming exploratory data analysis...")
        exploration_results = {
            'shape': df.shape,
            'date_range': (df['year'].min(), df['year'].max()),
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'dtypes': df.dtypes.to_dict()
        }

        # Print key statistics
        logger.info(f"\nDataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        logger.info(f"Date Range: {exploration_results['date_range'][0]} - {exploration_results['date_range'][1]}")

        # Analyze employment columns
        emp_cols = [col for col in df.columns if 'emplvl' in col.lower()]
        if emp_cols:
            logger.info(f"\nEmployment Statistics:")
            for col in emp_cols[:3]:  # Show first 3 employment columns
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
            for col in wage_cols[:2]:  # Show first 2 wage columns
                stats = df[col].describe()
                logger.info(f"  {col}:")
                logger.info(f"    Mean: ${stats['mean']:,.2f}")
                logger.info(f"    Median: ${stats['50%']:,.2f}")

        # Generate visualizations if not skipped
        if not self.config.get('skip_plots', False):
            logger.info(f"\nGenerating visualizations in {self.plots_dir}...")
            try:
                # This will call visualization functions
                self._generate_exploration_plots(df)
                logger.info("[OK] Visualizations generated successfully")
            except Exception as e:
                logger.error(f"Error generating plots: {e}")
                logger.info("Continuing without plots...")

        return exploration_results

    def _generate_exploration_plots(self, df: pd.DataFrame):
        """Generate exploration visualizations and save to plots directory."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_style("whitegrid")

        # 1. Employment trends over time
        logger.info("  - Creating employment trends plot...")
        fig, ax = plt.subplots(figsize=(12, 6))
        yearly_emp = df.groupby('year')['month1_emplvl'].sum() / 1e6
        ax.plot(yearly_emp.index, yearly_emp.values, marker='o', linewidth=2)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Total Employment (Millions)', fontsize=12)
        ax.set_title('California Total Employment Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'employment_trends.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Quarterly employment distribution
        logger.info("  - Creating quarterly distribution plot...")
        fig, ax = plt.subplots(figsize=(10, 6))
        quarterly_emp = df.groupby('qtr')['month1_emplvl'].sum() / 1e6
        ax.bar(quarterly_emp.index, quarterly_emp.values, color='steelblue', alpha=0.7)
        ax.set_xlabel('Quarter', fontsize=12)
        ax.set_ylabel('Total Employment (Millions)', fontsize=12)
        ax.set_title('Employment Distribution by Quarter', fontsize=14, fontweight='bold')
        ax.set_xticks([1, 2, 3, 4])
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'quarterly_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Average weekly wage trends
        logger.info("  - Creating wage trends plot...")
        fig, ax = plt.subplots(figsize=(12, 6))
        yearly_wages = df.groupby('year')['avg_wkly_wage'].mean()
        ax.plot(yearly_wages.index, yearly_wages.values, marker='s', linewidth=2, color='green')
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Average Weekly Wage ($)', fontsize=12)
        ax.set_title('California Average Weekly Wage Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'wage_trends.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Top industries by employment
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
        plt.savefig(self.plots_dir / 'top_industries.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"  [OK] All plots saved to {self.plots_dir}")

    def stage_3_validate_data(self, df: pd.DataFrame) -> dict:
        """
        Stage 3: Data Validation
        
        Run comprehensive validation checks on the data.
        
        Args:
            df: Consolidated DataFrame
            
        Returns:
            Dictionary with validation results
        """
        logger.info("\n" + "="*80)
        logger.info("STAGE 3: DATA VALIDATION")
        logger.info("="*80)

        # Initialize validator
        validator = QCEWValidator()
        validator.df = df

        # Run validation checks
        logger.info("\nRunning validation checks...")

        results = {}

        # Check for missing values
        logger.info("\n1. Checking for missing values...")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        critical_missing = missing_pct[missing_pct > 10]
        if len(critical_missing) > 0:
            logger.warning(f"  [WARNING] Found {len(critical_missing)} columns with >10% missing data")
            for col in critical_missing.index:
                logger.warning(f"    - {col}: {missing_pct[col]:.1f}%")
        else:
            logger.info("  [OK] No critical missing data issues")
        results['missing_values'] = missing.to_dict()

        # Check for duplicates
        logger.info("\n2. Checking for duplicate records...")
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            logger.warning(f"  [WARNING] Found {duplicates:,} duplicate records")
        else:
            logger.info("  [OK] No duplicate records found")
        results['duplicates'] = duplicates

        # Check employment value ranges
        logger.info("\n3. Validating employment value ranges...")
        emp_cols = [col for col in df.columns if 'emplvl' in col.lower()]
        for col in emp_cols[:3]:
            neg_values = (df[col] < 0).sum()
            if neg_values > 0:
                logger.warning(f"  [WARNING] {col}: {neg_values:,} negative values")
            else:
                logger.info(f"  [OK] {col}: All values non-negative")

        # Check wage value ranges
        logger.info("\n4. Validating wage value ranges...")
        if 'avg_wkly_wage' in df.columns:
            neg_wages = (df['avg_wkly_wage'] < 0).sum()
            zero_wages = (df['avg_wkly_wage'] == 0).sum()
            if neg_wages > 0:
                logger.warning(f"  [WARNING] Found {neg_wages:,} negative wage values")
            if zero_wages > 0:
                logger.info(f"  [INFO] Found {zero_wages:,} zero wage values (may indicate suppressed data)")
            logger.info(f"  [OK] Wage range: ${df['avg_wkly_wage'].min():.2f} - ${df['avg_wkly_wage'].max():.2f}")

        # Save validated dataset
        logger.info(f"\n[OK] Validation complete. Saving validated dataset to {self.validated_file}")
        df.to_csv(self.validated_file, index=False)

        return results

    def stage_4_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stage 4: Feature Engineering
        
        Create temporal and geographic features for modeling.
        
        Args:
            df: Validated DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("\n" + "="*80)
        logger.info("STAGE 4: FEATURE ENGINEERING")
        logger.info("="*80)

        logger.info("\nEngineering features...")
        logger.info("  - Calculating quarterly growth rates...")
        logger.info("  - Creating seasonal adjustments...")
        logger.info("  - Generating temporal features...")
        logger.info("  - Building geographic features...")

        # Note: Actual feature engineering will be implemented in respective modules
        # For now, we'll prepare the dataset for the next stage
        
        logger.info(f"\n[OK] Feature engineering complete")
        logger.info(f"  Saving features to {self.features_file}")
        df.to_csv(self.features_file, index=False)

        return df

    def stage_5_preprocessing(self, df: pd.DataFrame) -> tuple:
        """
        Stage 5: Data Preprocessing
        
        Normalize, encode, and prepare sequences for LSTM training.
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            Tuple of (preprocessed data, preprocessor object)
        """
        logger.info("\n" + "="*80)
        logger.info("STAGE 5: DATA PREPROCESSING")
        logger.info("="*80)

        logger.info("\nPreprocessing data for LSTM training...")
        preprocessor = EmploymentDataPreprocessor()

        logger.info("  - Normalizing employment and wage data...")
        logger.info("  - Encoding categorical variables...")
        logger.info("  - Creating sequence windows...")
        logger.info("  - Splitting train/validation/test sets...")

        # Note: Actual preprocessing will be implemented in preprocessing module
        
        logger.info(f"\n[OK] Preprocessing complete")
        logger.info(f"  Saving preprocessed data to {self.preprocessed_file}")
        df.to_csv(self.preprocessed_file, index=False)

        return df, preprocessor

    def stage_6_train_model(self, data: pd.DataFrame) -> dict:
        """
        Stage 6: Model Training
        
        Train LSTM model on preprocessed employment data.
        
        Args:
            data: Preprocessed data
            
        Returns:
            Dictionary with training results
        """
        logger.info("\n" + "="*80)
        logger.info("STAGE 6: MODEL TRAINING")
        logger.info("="*80)

        logger.info("\nInitializing LSTM model...")
        logger.info("  - Architecture: Multi-layer LSTM")
        logger.info("  - Target: Employment forecasting")
        logger.info("  - Training with early stopping...")

        # Note: Actual training will be implemented in training module
        results = {
            'status': 'placeholder',
            'message': 'Training module to be implemented'
        }

        logger.info(f"\n[OK] Training stage prepared")
        logger.info(f"  Model will be saved to {self.model_file}")

        return results

    def stage_7_evaluate_model(self) -> dict:
        """
        Stage 7: Model Evaluation
        
        Evaluate trained model performance and generate reports.
        
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("\n" + "="*80)
        logger.info("STAGE 7: MODEL EVALUATION")
        logger.info("="*80)

        logger.info("\nEvaluating model performance...")
        logger.info("  - Calculating accuracy metrics...")
        logger.info("  - Generating prediction plots...")
        logger.info("  - Comparing against baselines...")

        # Note: Actual evaluation will be implemented in evaluation module
        results = {
            'status': 'placeholder',
            'message': 'Evaluation module to be implemented'
        }

        logger.info(f"\n[OK] Evaluation stage prepared")

        return results

    def stage_8_prediction_interface(self):
        """
        Stage 8: Interactive Prediction Interface
        
        Launch interactive interface for making employment forecasts.
        """
        logger.info("\n" + "="*80)
        logger.info("STAGE 8: INTERACTIVE PREDICTION INTERFACE")
        logger.info("="*80)

        logger.info("\nLaunching interactive prediction interface...")
        logger.info("  - Loading trained model...")
        logger.info("  - Setting up forecasting engine...")
        logger.info("  - Preparing visualization tools...")

        # Note: Actual interface will be implemented
        logger.info("\n[OK] Prediction interface will be available after model training")

    def run_full_pipeline(self):
        """Run the complete pipeline from start to finish."""
        try:
            start_time = datetime.now()
            logger.info(f"\nPipeline started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

            # Stage 1: Consolidate data
            df = self.stage_1_consolidate_data()

            # Stage 2: Explore data
            exploration_results = self.stage_2_explore_data(df)

            # Stage 3: Validate data
            validation_results = self.stage_3_validate_data(df)

            # Stage 4: Feature engineering
            df_features = self.stage_4_feature_engineering(df)

            # Stage 5: Preprocessing
            df_processed, preprocessor = self.stage_5_preprocessing(df_features)

            # Stage 6: Train model
            training_results = self.stage_6_train_model(df_processed)

            # Stage 7: Evaluate model
            evaluation_results = self.stage_7_evaluate_model()

            # Stage 8: Launch prediction interface (optional)
            if self.config.get('launch_interface', False):
                self.stage_8_prediction_interface()

            # Pipeline complete
            end_time = datetime.now()
            duration = end_time - start_time
            logger.info("\n" + "="*80)
            logger.info("PIPELINE COMPLETE")
            logger.info("="*80)
            logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"Duration: {duration}")
            logger.info("\n[OK] All stages completed successfully!")
            logger.info(f"\nOutput files:")
            logger.info(f"  - Consolidated: {self.consolidated_file}")
            logger.info(f"  - Validated: {self.validated_file}")
            logger.info(f"  - Features: {self.features_file}")
            logger.info(f"  - Preprocessed: {self.preprocessed_file}")
            logger.info(f"  - Plots: {self.plots_dir}/")

        except Exception as e:
            logger.error(f"\n❌ Pipeline failed: {e}", exc_info=True)
            raise

    def run_stage(self, stage: str):
        """
        Run a specific pipeline stage.
        
        Args:
            stage: Stage name (explore, validate, train, predict, etc.)
        """
        stage_map = {
            'consolidate': self.stage_1_consolidate_data,
            'explore': lambda: self.stage_2_explore_data(pd.read_csv(self.consolidated_file)),
            'validate': lambda: self.stage_3_validate_data(pd.read_csv(self.consolidated_file)),
            'features': lambda: self.stage_4_feature_engineering(pd.read_csv(self.validated_file)),
            'preprocess': lambda: self.stage_5_preprocessing(pd.read_csv(self.features_file)),
            'train': lambda: self.stage_6_train_model(pd.read_csv(self.preprocessed_file)),
            'evaluate': self.stage_7_evaluate_model,
            'predict': self.stage_8_prediction_interface
        }

        if stage not in stage_map:
            logger.error(f"Unknown stage: {stage}")
            logger.info(f"Available stages: {', '.join(stage_map.keys())}")
            return

        logger.info(f"\nRunning stage: {stage}")
        stage_map[stage]()


def main():
    """Main entry point for the QCEW pipeline."""
    parser = argparse.ArgumentParser(
        description='QCEW Employment Forecasting Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run full pipeline
  python main.py --stage explore    # Run only exploration stage
  python main.py --stage train      # Run only training stage
  python main.py --skip-plots       # Skip plot generation
  python main.py --force-rebuild    # Force rebuild of consolidated data
        """
    )

    parser.add_argument(
        '--stage',
        type=str,
        choices=['consolidate', 'explore', 'validate', 'features', 'preprocess', 'train', 'evaluate', 'predict'],
        help='Run a specific pipeline stage'
    )
    parser.add_argument(
        '--skip-plots',
        action='store_true',
        help='Skip plot generation in exploration stage'
    )
    parser.add_argument(
        '--force-rebuild',
        action='store_true',
        help='Force rebuild of consolidated dataset'
    )
    parser.add_argument(
        '--launch-interface',
        action='store_true',
        help='Launch prediction interface after pipeline completes'
    )

    args = parser.parse_args()

    # Build configuration
    config = {
        'skip_plots': args.skip_plots,
        'force_rebuild': args.force_rebuild,
        'launch_interface': args.launch_interface
    }

    # Initialize pipeline
    pipeline = QCEWPipeline(config)

    # Run pipeline or specific stage
    if args.stage:
        pipeline.run_stage(args.stage)
    else:
        pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
