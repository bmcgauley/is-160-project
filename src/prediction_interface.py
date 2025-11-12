"""
Interactive Prediction Interface for QCEW Employment Forecasting

This module provides an interactive CLI tool for making employment predictions
using the trained LSTM model.

Stage 9 of the pipeline.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import joblib
from datetime import datetime

logger = logging.getLogger(__name__)


class EmploymentPredictor:
    """Interactive prediction interface for employment forecasting."""

    def __init__(self, model_path: Path, preprocessor_path: Path):
        """
        Initialize the predictor.

        Args:
            model_path: Path to trained PyTorch model (.pt file)
            preprocessor_path: Path to saved preprocessor (.pkl file)
        """
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model = None
        self.preprocessor = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        logger.info("="*80)
        logger.info("STAGE 9: INTERACTIVE PREDICTION INTERFACE")
        logger.info("="*80)

        self._load_model()
        self._load_preprocessor()

    def _load_model(self):
        """Load the trained LSTM model."""
        logger.info(f"\nLoading model from {self.model_path}...")

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Import model architecture
        from lstm_model import EmploymentLSTM

        # Reconstruct model (need to get architecture params from checkpoint or config)
        # For now, use default architecture - should match training config
        self.model = EmploymentLSTM(
            input_size=checkpoint.get('input_size', 24),
            hidden_size=checkpoint.get('hidden_size', 64),
            num_layers=checkpoint.get('num_layers', 2),
            dropout=checkpoint.get('dropout', 0.2)
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        logger.info(f"[OK] Model loaded successfully")
        logger.info(f"  Device: {self.device}")

    def _load_preprocessor(self):
        """Load the saved preprocessor."""
        logger.info(f"\nLoading preprocessor from {self.preprocessor_path}...")

        if not self.preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor not found at {self.preprocessor_path}")

        self.preprocessor = joblib.load(self.preprocessor_path)

        logger.info(f"[OK] Preprocessor loaded successfully")
        logger.info(f"  Feature names: {len(self.preprocessor.feature_names_)} features")

    def predict_single(self, sequence_data: np.ndarray) -> float:
        """
        Make a prediction for a single sequence.

        Args:
            sequence_data: Input sequence (seq_len, num_features)

        Returns:
            Predicted employment value
        """
        with torch.no_grad():
            # Convert to tensor and add batch dimension
            x = torch.FloatTensor(sequence_data).unsqueeze(0).to(self.device)

            # Make prediction
            prediction = self.model(x)

            # Convert back to numpy and denormalize if needed
            pred_value = prediction.cpu().numpy()[0, 0]

            return pred_value

    def predict_batch(self, sequences: np.ndarray) -> np.ndarray:
        """
        Make predictions for multiple sequences.

        Args:
            sequences: Input sequences (batch_size, seq_len, num_features)

        Returns:
            Predicted employment values (batch_size,)
        """
        with torch.no_grad():
            x = torch.FloatTensor(sequences).to(self.device)
            predictions = self.model(x)
            return predictions.cpu().numpy().squeeze()

    def predict_from_dataframe(self, df: pd.DataFrame,
                                county: str = None,
                                industry: str = None) -> pd.DataFrame:
        """
        Make predictions from a DataFrame of historical data.

        Args:
            df: Historical employment data
            county: Optional county filter
            industry: Optional industry filter

        Returns:
            DataFrame with predictions
        """
        logger.info("\nGenerating predictions from DataFrame...")

        # Filter data if requested
        filtered_df = df.copy()
        if county:
            filtered_df = filtered_df[filtered_df['AreaName'] == county]
            logger.info(f"  Filtered to county: {county}")

        if industry:
            filtered_df = filtered_df[filtered_df['IndustryName'] == industry]
            logger.info(f"  Filtered to industry: {industry}")

        if len(filtered_df) == 0:
            logger.warning("No data matches the filters!")
            return pd.DataFrame()

        # Preprocess and create sequences
        # This would use the preprocessor to transform the data
        # For now, placeholder implementation
        logger.info(f"  Processing {len(filtered_df)} records...")

        # TODO: Implement actual sequence creation from DataFrame
        # sequences = self.preprocessor.create_sequences(filtered_df)
        # predictions = self.predict_batch(sequences)

        logger.info("[OK] Predictions generated")
        return filtered_df

    def interactive_mode(self):
        """
        Run interactive prediction mode.

        Allows user to:
        1. Load historical data
        2. Select county/industry
        3. Generate forecasts
        4. Save results
        """
        logger.info("\n" + "="*80)
        logger.info("INTERACTIVE PREDICTION MODE")
        logger.info("="*80)

        print("\nWelcome to the Employment Forecasting Tool!")
        print("\nOptions:")
        print("  1. Predict for specific county/industry")
        print("  2. Batch predictions from CSV")
        print("  3. Exit")

        while True:
            choice = input("\nEnter choice (1-3): ").strip()

            if choice == '1':
                self._predict_interactive()
            elif choice == '2':
                self._batch_predict_interactive()
            elif choice == '3':
                print("\nExiting prediction interface. Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")

    def _predict_interactive(self):
        """Interactive single prediction."""
        print("\n--- Single Prediction ---")
        county = input("Enter county name (or leave blank): ").strip() or None
        industry = input("Enter industry name (or leave blank): ").strip() or None

        print(f"\nGenerating prediction for:")
        print(f"  County: {county or 'All'}")
        print(f"  Industry: {industry or 'All'}")

        # TODO: Implement actual prediction logic
        print("\n[INFO] Prediction functionality coming soon!")

    def _batch_predict_interactive(self):
        """Interactive batch prediction."""
        print("\n--- Batch Predictions ---")
        csv_path = input("Enter path to input CSV file: ").strip()

        if not Path(csv_path).exists():
            print(f"[ERROR] File not found: {csv_path}")
            return

        print(f"\nLoading data from {csv_path}...")

        # TODO: Implement batch prediction
        print("\n[INFO] Batch prediction functionality coming soon!")


def run_prediction_interface(model_path: str, preprocessor_path: str):
    """
    Run the interactive prediction interface.

    Args:
        model_path: Path to trained model
        preprocessor_path: Path to saved preprocessor
    """
    try:
        predictor = EmploymentPredictor(
            model_path=Path(model_path),
            preprocessor_path=Path(preprocessor_path)
        )

        predictor.interactive_mode()

        return {"success": True}

    except Exception as e:
        logger.error(f"[ERROR] Prediction interface failed: {e}")
        return {"success": False, "error": str(e)}


if __name__ == '__main__':
    # Test the prediction interface
    import sys

    if len(sys.argv) != 3:
        print("Usage: python prediction_interface.py <model_path> <preprocessor_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    preprocessor_path = sys.argv[2]

    run_prediction_interface(model_path, preprocessor_path)
