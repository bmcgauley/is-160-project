"""
Pipeline Orchestrator Module

This module contains the core QCEWPipeline class that coordinates
all stages of the employment forecasting pipeline.
"""

import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
import warnings

from validation import QCEWValidator
from preprocessing import EmploymentDataPreprocessor
from lstm_model import EmploymentLSTM

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


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
        self.base_dir = Path(__file__).parent.parent
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
        """Stage 1: Data Consolidation - Combine all raw CSV files"""
        from consolidation import consolidate_raw_data
        return consolidate_raw_data(
            self.raw_dir,
            self.consolidated_file,
            force_rebuild=self.config.get('force_rebuild', False)
        )

    def stage_2_explore_data(self, df: pd.DataFrame) -> dict:
        """Stage 2: Data Exploration - EDA and visualizations"""
        from exploration import perform_eda, generate_plots
        
        logger.info("\n" + "="*80)
        logger.info("STAGE 2: DATA EXPLORATION")
        logger.info("="*80)
        
        results = perform_eda(df)
        
        if not self.config.get('skip_plots', False):
            generate_plots(df, self.plots_dir)
        
        return results

    def stage_3_validate_data(self, df: pd.DataFrame) -> dict:
        """Stage 3: Data Validation - Quality checks"""
        from validation import validate_data_quality
        
        logger.info("\n" + "="*80)
        logger.info("STAGE 3: DATA VALIDATION")
        logger.info("="*80)
        
        results = validate_data_quality(df, self.validated_file)
        return results

    def stage_4_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stage 4: Feature Engineering"""
        from feature_pipeline import engineer_features
        
        logger.info("\n" + "="*80)
        logger.info("STAGE 4: FEATURE ENGINEERING")
        logger.info("="*80)
        
        df_features = engineer_features(df, self.features_file)
        return df_features

    def stage_5_preprocessing(self, df: pd.DataFrame) -> tuple:
        """Stage 5: Data Preprocessing"""
        from preprocessing_pipeline import preprocess_for_lstm
        
        logger.info("\n" + "="*80)
        logger.info("STAGE 5: DATA PREPROCESSING")
        logger.info("="*80)
        
        result = preprocess_for_lstm(df, self.preprocessed_file)
        return result

    def stage_6_train_model(self, data: pd.DataFrame) -> dict:
        """Stage 6: Model Training"""
        from training_pipeline import train_lstm_model
        
        logger.info("\n" + "="*80)
        logger.info("STAGE 6: MODEL TRAINING")
        logger.info("="*80)
        
        results = train_lstm_model(data, self.model_file)
        return results

    def stage_7_evaluate_model(self) -> dict:
        """Stage 7: Model Evaluation"""
        from evaluation_pipeline import evaluate_lstm_model
        
        logger.info("\n" + "="*80)
        logger.info("STAGE 7: MODEL EVALUATION")
        logger.info("="*80)
        
        results = evaluate_lstm_model(self.model_file, self.preprocessed_file)
        return results

    def stage_8_prediction_interface(self):
        """Stage 8: Interactive Prediction Interface"""
        from prediction_interface import launch_interface
        
        logger.info("\n" + "="*80)
        logger.info("STAGE 8: INTERACTIVE PREDICTION INTERFACE")
        logger.info("="*80)
        
        launch_interface(self.model_file, self.preprocessed_file)

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
            logger.error(f"\n[ERROR] Pipeline failed: {e}", exc_info=True)
            raise

    def run_stage(self, stage: str):
        """Run a specific pipeline stage."""
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
