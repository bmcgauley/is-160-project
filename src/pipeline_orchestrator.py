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

# Only import modules that currently exist
# Other modules will be imported locally when their stages are called
try:
    from validation import QCEWValidator
except ImportError:
    QCEWValidator = None
    
try:
    from preprocessing import EmploymentDataPreprocessor
except ImportError:
    EmploymentDataPreprocessor = None
    
try:
    from lstm_model import EmploymentLSTM
except ImportError:
    EmploymentLSTM = None

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
        self.feature_eng_dir = self.data_dir / "feature_engineering"
        self.plots_dir = self.processed_dir / "plots"

        # Create directories if they don't exist
        self.processed_dir.mkdir(exist_ok=True)
        self.validated_dir.mkdir(exist_ok=True)
        self.feature_eng_dir.mkdir(exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)

        # Pipeline state tracking
        self.consolidated_file = self.processed_dir / "qcew_master_consolidated.csv"
        self.validated_file = self.validated_dir / "qcew_validated.csv"
        self.features_file = self.feature_eng_dir / "final_features.csv"
        self.preprocessed_file = self.processed_dir / "qcew_preprocessed.csv"
        self.model_file = self.processed_dir / "lstm_model.pt"

        logger.info("="*80)
        logger.info("QCEW EMPLOYMENT FORECASTING PIPELINE")
        logger.info("="*80)
        logger.info(f"Base directory: {self.base_dir}")
        logger.info(f"Raw data: {self.raw_dir}")
        logger.info(f"Processed data: {self.processed_dir}")
        logger.info(f"Feature engineering: {self.feature_eng_dir}")
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
        
        df_features = engineer_features(
            df, 
            output_file=self.features_file,
            feature_eng_dir=self.feature_eng_dir
        )
        return df_features

    def stage_5_preprocessing(self, df: pd.DataFrame) -> tuple:
        """Stage 5: Data Preprocessing"""
        from preprocessing_pipeline import preprocess_for_lstm
        
        logger.info("\n" + "="*80)
        logger.info("STAGE 5: DATA PREPROCESSING")
        logger.info("="*80)
        
        X_tensor, y_tensor, preprocessor = preprocess_for_lstm(
            df, 
            output_file=self.preprocessed_file,
            sequence_length=12  # 12 quarters = 3 years
        )
        return X_tensor, y_tensor, preprocessor

    def stage_6_train_model(self, X_tensor, y_tensor, preprocessor) -> dict:
        """Stage 6: Model Training"""
        logger.info("\n" + "="*80)
        logger.info("STAGE 6: MODEL TRAINING")
        logger.info("="*80)
        
        logger.error("[ERROR] Stage 6: Model Training not yet implemented")
        logger.error("[ERROR] This stage will be implemented after preprocessing is complete")
        raise NotImplementedError("Model training stage (T065-T074) not yet implemented")

    def stage_7_evaluate_model(self) -> dict:
        """Stage 7: Model Evaluation"""
        logger.info("\n" + "="*80)
        logger.info("STAGE 7: MODEL EVALUATION")
        logger.info("="*80)
        
        logger.error("[ERROR] Stage 7: Model Evaluation not yet implemented")
        logger.error("[ERROR] This stage will be implemented after model training is complete")
        raise NotImplementedError("Model evaluation stage (T076-T085) not yet implemented")

    def stage_8_prediction_interface(self):
        """Stage 8: Interactive Prediction Interface"""
        logger.info("\n" + "="*80)
        logger.info("STAGE 8: INTERACTIVE PREDICTION INTERFACE")
        logger.info("="*80)
        
        logger.error("[ERROR] Stage 8: Prediction Interface not yet implemented")
        logger.error("[ERROR] This stage will be implemented after model evaluation is complete")
        raise NotImplementedError("Prediction interface stage (T117-T119) not yet implemented")

    def run_full_pipeline(self):
        """Run the complete pipeline from start to finish."""
        try:
            start_time = datetime.now()
            logger.info(f"\nPipeline started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

            # Stage 1: Consolidate data
            df_consolidated = self.stage_1_consolidate_data()

            # Stage 2: Explore data
            exploration_results = self.stage_2_explore_data(df_consolidated)

            # Stage 3: Validate data (uses consolidated data)
            validation_results = self.stage_3_validate_data(df_consolidated)

            # Stage 4: Feature engineering (uses validated data from file)
            df_validated = pd.read_csv(self.validated_file)
            df_features = self.stage_4_feature_engineering(df_validated)

            # Stage 5: Preprocessing (uses features data)
            X_tensor, y_tensor, preprocessor = self.stage_5_preprocessing(df_features)

            # Stage 6: Train model (uses preprocessed sequences)
            if X_tensor is None or len(X_tensor) == 0:
                logger.error("[ERROR] No preprocessed sequences available for training")
                return
            training_results = self.stage_6_train_model(X_tensor, y_tensor, preprocessor)

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
