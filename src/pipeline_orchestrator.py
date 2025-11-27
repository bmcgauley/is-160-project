"""
Pipeline Orchestrator Module

This module contains the core QCEWPipeline class that coordinates
all stages of the employment forecasting pipeline.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import warnings
import torch
import joblib
import sys

# Add config directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'config'))
from hyperparameters import ModelConfig, TrainingConfig, DataConfig, ExperimentConfig

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
        self.preprocessor_file = self.processed_dir / "qcew_preprocessed_preprocessor.pkl"
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
            sequence_length=DataConfig.SEQUENCE_LENGTH
        )
        return X_tensor, y_tensor, preprocessor

    def load_preprocessed_data_for_training(self) -> tuple:
        """
        Load preprocessed sequences and preprocessor from disk.

        This helper method is used when running the training stage individually
        without having run the preprocessing stage in the same session.

        Returns:
            Tuple of (X_tensor, y_tensor, preprocessor)
        """
        logger.info("Loading preprocessed data from disk...")

        # Load sequences from .npz file
        sequences_file = self.preprocessed_file.parent / (self.preprocessed_file.stem + '_sequences.npz')
        if not sequences_file.exists():
            raise FileNotFoundError(
                f"Sequences file not found: {sequences_file}\n"
                f"Please run preprocessing stage first: python main.py --stage preprocess"
            )

        logger.info(f"  Loading sequences from: {sequences_file.name}")
        data = np.load(sequences_file)
        X_sequences = data['X']
        y_targets = data['y']
        logger.info(f"  Loaded {len(X_sequences):,} sequences")

        # Load preprocessor from .pkl file
        preprocessor_file = self.preprocessed_file.parent / (self.preprocessed_file.stem + '_preprocessor.pkl')
        if not preprocessor_file.exists():
            raise FileNotFoundError(
                f"Preprocessor file not found: {preprocessor_file}\n"
                f"Please run preprocessing stage first: python main.py --stage preprocess"
            )

        logger.info(f"  Loading preprocessor from: {preprocessor_file.name}")
        preprocessor = joblib.load(preprocessor_file)
        logger.info("  Preprocessor loaded successfully")

        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_sequences)
        y_tensor = torch.FloatTensor(y_targets)

        logger.info("[OK] Preprocessed data loaded from disk\n")
        return X_tensor, y_tensor, preprocessor

    def stage_6_train_model(self, X_tensor, y_tensor, preprocessor) -> dict:
        """Stage 6: Model Training"""
        import torch
        from dataset import create_train_val_splits, build_data_loader
        from lstm_model import EmploymentLSTM, validate_lstm_architecture
        from training import EmploymentTrainer

        logger.info("\n" + "="*80)
        logger.info("STAGE 6: MODEL TRAINING")
        logger.info("="*80)

        # Get input dimensions
        num_sequences, seq_length, num_features = X_tensor.shape
        logger.info(f"  Input shape: {X_tensor.shape}")
        logger.info(f"  Target shape: {y_tensor.shape}")
        logger.info(f"  Features: {num_features}, Sequence length: {seq_length}")

        # Create train/val/test splits
        logger.info("\nCreating data splits...")
        train_dataset, val_dataset, test_dataset = create_train_val_splits(
            X_tensor.numpy(),
            y_tensor.numpy(),
            val_size=DataConfig.VAL_SIZE,
            test_size=DataConfig.TEST_SIZE,
            shuffle=False  # Preserve temporal order
        )

        # Create data loaders
        logger.info("\nCreating data loaders...")
        train_loader = build_data_loader(train_dataset, batch_size=DataConfig.BATCH_SIZE, shuffle=DataConfig.SHUFFLE_TRAIN)
        val_loader = build_data_loader(val_dataset, batch_size=DataConfig.BATCH_SIZE, shuffle=DataConfig.SHUFFLE_VAL)
        test_loader = build_data_loader(test_dataset, batch_size=DataConfig.BATCH_SIZE, shuffle=DataConfig.SHUFFLE_VAL)

        # Initialize model
        logger.info("\nInitializing LSTM model...")
        model = EmploymentLSTM(
            input_size=num_features,
            hidden_size=ModelConfig.HIDDEN_SIZE,
            num_layers=ModelConfig.NUM_LAYERS,
            output_size=ModelConfig.OUTPUT_SIZE,
            dropout=ModelConfig.DROPOUT
        )

        # Validate architecture
        logger.info("\nValidating model architecture...")
        validation_results = validate_lstm_architecture(
            model,
            input_shape=(32, seq_length, num_features),
            expected_output_size=1
        )

        if not all(validation_results.values()):
            logger.error("[ERROR] Model architecture validation failed!")
            return {"success": False, "validation_results": validation_results}

        # Set device - smart selection based on config and availability
        if ExperimentConfig.DEVICE == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif ExperimentConfig.DEVICE == 'cuda':
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                logger.warning("CUDA requested but not available, falling back to CPU")
                device = 'cpu'
        else:
            device = ExperimentConfig.DEVICE

        # Log detailed device information
        logger.info(f"\nDevice Selection:")
        logger.info(f"  Config setting: {ExperimentConfig.DEVICE}")
        logger.info(f"  Selected device: {device}")
        if device == 'cuda':
            logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"  CUDA version: {torch.version.cuda}")
            logger.info(f"  Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            if torch.cuda.is_available():
                logger.info(f"  Note: CUDA is available but not selected")
            else:
                logger.info(f"  Note: CUDA not available (CPU-only PyTorch or no GPU detected)")

        # Create TensorBoard writer for live monitoring (optional)
        writer = None
        tensorboard_dir = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            tensorboard_dir = self.base_dir / 'runs' / f'employment_lstm_{timestamp}'
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(str(tensorboard_dir))
            logger.info(f"\nTensorBoard logging enabled: {tensorboard_dir}")
            logger.info("  To view live training: tensorboard --logdir=runs")
        except ImportError:
            logger.info("\nTensorBoard not available (install with: pip install tensorboard)")
            logger.info("  Training will proceed without live monitoring")

        # Create trainer
        logger.info("\nCreating trainer...")
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=TrainingConfig.LEARNING_RATE,
            weight_decay=TrainingConfig.WEIGHT_DECAY
        )

        trainer = EmploymentTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=torch.nn.MSELoss(),
            optimizer=optimizer,
            device=device,
            tensorboard_writer=writer
        )

        # Add learning rate scheduler
        from training import build_learning_rate_scheduler
        trainer.scheduler = build_learning_rate_scheduler(
            optimizer,
            scheduler_type=TrainingConfig.LR_SCHEDULER_TYPE,
            factor=TrainingConfig.LR_FACTOR,
            patience=TrainingConfig.LR_PATIENCE
        )

        # Train model
        logger.info("\nStarting training...")
        history = trainer.train_model(
            num_epochs=TrainingConfig.NUM_EPOCHS,
            patience=TrainingConfig.PATIENCE,
            save_path=str(self.model_file)
        )

        # Evaluate on test set
        logger.info("\nEvaluating on test set...")
        model.eval()
        test_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for sequences, targets in test_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)

                outputs = model(sequences).squeeze()
                loss = torch.nn.functional.mse_loss(outputs, targets)
                test_loss += loss.item()

                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        test_loss /= len(test_loader)
        logger.info(f"  Test Loss (MSE): {test_loss:.6f}")

        # Calculate additional metrics
        from loss_metrics import (mean_absolute_percentage_error,
                                  directional_accuracy,
                                  root_mean_squared_error)

        import numpy as np
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)

        rmse = root_mean_squared_error(targets, predictions)
        mape = mean_absolute_percentage_error(targets, predictions)
        dir_acc = directional_accuracy(targets, predictions)

        logger.info(f"  Test RMSE: {rmse:.6f}")
        logger.info(f"  Test MAPE: {mape:.2f}%")
        logger.info(f"  Directional Accuracy: {dir_acc:.2f}%")

        # Log test metrics to TensorBoard (if available)
        if writer is not None:
            writer.add_scalar('Test/loss', test_loss, 0)
            writer.add_scalar('Test/rmse', rmse, 0)
            writer.add_scalar('Test/mape', mape, 0)
            writer.add_scalar('Test/directional_accuracy', dir_acc, 0)
            writer.close()
            logger.info("\n[OK] TensorBoard logs saved")

        logger.info("\n[OK] Model training complete!")

        results = {
            "success": True,
            "history": history,
            "test_loss": test_loss,
            "test_rmse": rmse,
            "test_mape": mape,
            "directional_accuracy": dir_acc,
            "model_path": str(self.model_file),
            "num_epochs": len(history['train_loss']),
            "best_val_loss": min(history['val_loss']),
            "test_predictions": predictions,  # Add for visualization
            "test_targets": targets,  # Add for visualization
            "tensorboard_dir": str(tensorboard_dir) if tensorboard_dir else None  # Add for user reference
        }

        return results

    def stage_7_evaluate_model(self, training_results: dict) -> dict:
        """Stage 7: Model Evaluation"""
        import torch
        import matplotlib.pyplot as plt
        import numpy as np

        logger.info("\n" + "="*80)
        logger.info("STAGE 7: MODEL EVALUATION")
        logger.info("="*80)

        if not training_results.get("success", False):
            logger.error("[ERROR] Cannot evaluate - training failed")
            return {"success": False, "error": "Training failed"}

        # Create evaluation plots directory
        eval_plots_dir = self.plots_dir / "evaluation"
        eval_plots_dir.mkdir(exist_ok=True)

        # Plot training history using enhanced visualization
        logger.info("\nGenerating enhanced training history plots...")
        from visualization import plot_enhanced_training_history, plot_prediction_analysis

        history = training_results['history']

        # Determine best epoch (where validation loss was lowest)
        # Handle case where history might be empty (when loading from old checkpoints)
        if history.get('val_loss') and len(history['val_loss']) > 0:
            best_epoch = np.argmin(history['val_loss']) + 1  # +1 because epochs are 1-indexed
        else:
            # Fallback: use num_epochs or 1
            best_epoch = training_results.get('num_epochs', 1)
            logger.warning("No validation loss history found; using fallback for best_epoch")

        # Create enhanced training history plot
        loss_plot_path = eval_plots_dir / "training_history.png"
        plot_enhanced_training_history(history, loss_plot_path, best_epoch=best_epoch)

        # Create prediction analysis plots if we have test predictions
        if 'test_predictions' in training_results and 'test_targets' in training_results:
            logger.info("\nGenerating prediction analysis plots...")
            plot_prediction_analysis(
                y_true=training_results['test_targets'],
                y_pred=training_results['test_predictions'],
                save_dir=eval_plots_dir,
                sample_size=5000  # Sample 5000 points for visualization
            )

        # Log summary statistics
        logger.info("\nModel Performance Summary:")
        logger.info("="*80)
        logger.info(f"  Training epochs: {training_results['num_epochs']}")
        logger.info(f"  Best validation loss: {training_results['best_val_loss']:.6f}")
        logger.info(f"  Test loss (MSE): {training_results['test_loss']:.6f}")
        logger.info(f"  Test RMSE: {training_results['test_rmse']:.6f}")
        logger.info(f"  Test MAPE: {training_results['test_mape']:.2f}%")
        logger.info(f"  Directional Accuracy: {training_results['directional_accuracy']:.2f}%")
        logger.info(f"  Model saved at: {training_results['model_path']}")
        logger.info("="*80)

        # Determine model quality
        mape = training_results['test_mape']
        dir_acc = training_results['directional_accuracy']

        if mape < 10 and dir_acc > 70:
            quality = "EXCELLENT"
        elif mape < 20 and dir_acc > 60:
            quality = "GOOD"
        elif mape < 30 and dir_acc > 50:
            quality = "FAIR"
        else:
            quality = "NEEDS IMPROVEMENT"

        logger.info(f"\n  Overall Model Quality: {quality}")

        # Comprehensive evaluation with baselines
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE MODEL EVALUATION WITH BASELINES")
        logger.info("="*80)

        from comprehensive_evaluation import (
            evaluate_against_baselines,
            generate_naivebaselines,
            generate_evaluation_report,
            plot_model_comparison
        )

        y_true = training_results['test_targets']
        y_pred = training_results['test_predictions']

        # Generate naive baselines for comparison
        # Use full dataset for historical context (if available)
        historical_data = None
        if hasattr(self, 'preprocessed_data') and self.preprocessed_data is not None:
            # Try to get historical employment data for baselines
            try:
                historical_data = self.preprocessed_data['AvgAnnualEmployment'].values
                logger.info(f"Using {len(historical_data)} historical data points for baseline generation")
            except:
                logger.warning("Could not extract historical data; using test set for baselines")

        baseline_predictions = generate_naivebaselines(y_true, historical_data)

        # Evaluate LSTM against baselines
        eval_results = evaluate_against_baselines(y_true, y_pred, baseline_predictions)

        # Generate comprehensive evaluation report
        report_path = eval_plots_dir / "comprehensive_evaluation_report.txt"
        model_info = {
            "Model Type": "LSTM",
            "Training Epochs": training_results['num_epochs'],
            "Best Val Loss": f"{training_results['best_val_loss']:.6f}",
            "Test Samples": len(y_true),
            "Model Path": training_results['model_path']
        }
        generate_evaluation_report(eval_results, report_path, model_info)

        # Generate model comparison plot
        comparison_plot_path = eval_plots_dir / "model_comparison.png"
        plot_model_comparison(eval_results, comparison_plot_path)

        logger.info("\n[OK] Comprehensive model evaluation complete!")

        results = {
            "success": True,
            "quality": quality,
            "plots_dir": str(eval_plots_dir),
            "training_results": training_results,
            "eval_results": eval_results,
            "report_path": str(report_path)
        }

        return results

    def stage_8_prediction_interface(self, training_results: dict = None):
        """Stage 8: Interactive Prediction Interface"""
        logger.info("\n" + "="*80)
        logger.info("STAGE 8: INTERACTIVE PREDICTION INTERFACE")
        logger.info("="*80)

        # Check if model exists
        if not self.model_file.exists():
            logger.error("[ERROR] Model file not found. Please run training first.")
            return {"success": False, "error": "Model not found"}

        # Check if preprocessor exists
        if not self.preprocessor_file.exists():
            logger.error("[ERROR] Preprocessor file not found. Please run preprocessing first.")
            return {"success": False, "error": "Preprocessor not found"}

        logger.info(f"Model path: {self.model_file}")
        logger.info(f"Preprocessor path: {self.preprocessor_file}")

        # Ask user if they want to run interactive mode
        print("\n" + "="*80)
        print("INTERACTIVE PREDICTION INTERFACE")
        print("="*80)
        print("\nThe prediction interface allows you to:")
        print("  - Make predictions for specific counties/industries")
        print("  - Run batch predictions from CSV files")
        print("  - Export prediction results")
        print("\nWould you like to launch the interactive prediction interface?")
        response = input("Enter 'y' to launch, or any other key to skip: ").strip().lower()

        if response == 'y':
            from prediction_interface import run_prediction_interface
            result = run_prediction_interface(
                model_path=str(self.model_file),
                preprocessor_path=str(self.preprocessor_file)
            )
            return result
        else:
            logger.info("\n[INFO] Skipping interactive mode")
            logger.info(f"[INFO] You can launch it later with:")
            logger.info(f"[INFO]   python src/prediction_interface.py {self.model_file} {self.preprocessor_file}")

            return {
                "success": True,
                "status": "skipped",
                "model_path": str(self.model_file),
                "preprocessor_path": str(self.preprocessor_file)
            }

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
            evaluation_results = self.stage_7_evaluate_model(training_results)

            # Stage 8: Launch prediction interface (optional)
            prediction_results = self.stage_8_prediction_interface(training_results)

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
            'train': lambda: self.stage_6_train_model(*self.load_preprocessed_data_for_training()),
            'evaluate': self.stage_7_evaluate_model,
            'predict': self.stage_8_prediction_interface
        }

        if stage not in stage_map:
            logger.error(f"Unknown stage: {stage}")
            logger.info(f"Available stages: {', '.join(stage_map.keys())}")
            return

        logger.info(f"\nRunning stage: {stage}")
        stage_map[stage]()
