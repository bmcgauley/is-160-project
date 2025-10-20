"""
Preprocessing Pipeline Module

This module provides the high-level preprocessing pipeline
that orchestrates all data preprocessing tasks (T054-T058).

Calls individual functions from preprocessing.py module.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional
import torch

from preprocessing import EmploymentDataPreprocessor

logger = logging.getLogger(__name__)


def preprocess_for_lstm(df: pd.DataFrame, 
                       output_file: Path,
                       sequence_length: int = 12) -> Tuple[torch.Tensor, torch.Tensor, EmploymentDataPreprocessor]:
    """
    Run the complete preprocessing pipeline for LSTM model.
    
    This function orchestrates all preprocessing tasks in sequence:
    - T054: Normalize employment counts and wage data
    - T055: Handle missing values with domain-appropriate imputation
    - T056: Create categorical encodings for industry codes and geographic identifiers
    - T057: Transform tabular data into sequence format suitable for RNN/LSTM
    - T058: Validate preprocessing steps maintain data distribution properties
    
    Args:
        df: Feature-engineered QCEW dataframe
        output_file: Path to save preprocessed dataset
        sequence_length: Length of sequences for LSTM (default: 12 quarters = 3 years)
        
    Returns:
        Tuple of (X_sequences, y_targets, preprocessor)
    """
    logger.info("\n" + "="*80)
    logger.info("PREPROCESSING PIPELINE FOR LSTM")
    logger.info("="*80)
    
    # Initialize preprocessor
    preprocessor = EmploymentDataPreprocessor(scaler_type='robust')
    
    # T054: Normalize employment counts and wage data
    logger.info("\n[STAGE 1/5] T054: Normalizing employment and wage data...")
    employment_cols = [col for col in df.columns if 'emplvl' in col.lower() or 'employment' in col.lower()]
    wage_cols = [col for col in df.columns if 'wage' in col.lower()]
    normalize_cols = employment_cols + wage_cols
    
    if normalize_cols:
        logger.info(f"  Normalizing {len(normalize_cols)} columns using robust scaling")
        df = preprocessor.normalize_employment_data(df, employment_cols=normalize_cols)
        logger.info("[OK] Normalization complete")
    else:
        logger.warning("[WARN] No employment/wage columns found for normalization")
    
    # T055: Handle missing values
    logger.info("\n[STAGE 2/5] T055: Handling missing values...")
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        logger.info(f"  Found {missing_count:,} missing values")
        df = preprocessor.handle_missing_values(df, strategy='median')
        logger.info("[OK] Missing values handled")
    else:
        logger.info("[OK] No missing values found")
    
    # T056: Create categorical encodings
    logger.info("\n[STAGE 3/5] T056: Creating categorical encodings...")
    categorical_cols = ['industry_code', 'area_name', 'ownership']
    existing_categorical = [col for col in categorical_cols if col in df.columns]
    
    if existing_categorical:
        logger.info(f"  Encoding {len(existing_categorical)} categorical columns")
        df = preprocessor.create_categorical_encodings(df, categorical_cols=existing_categorical)
        logger.info("[OK] Categorical encoding complete")
    else:
        logger.warning("[WARN] No categorical columns found for encoding")
    
    # T057: Transform to sequences
    logger.info("\n[STAGE 4/5] T057: Transforming to sequence format...")
    logger.info(f"  Creating sequences of length {sequence_length}")
    
    # Determine target column
    target_col = 'month1_emplvl' if 'month1_emplvl' in df.columns else 'avg_monthly_emplvl'
    if target_col not in df.columns:
        raise ValueError(f"Target column not found. Available columns: {df.columns.tolist()}")
    
    logger.info(f"  Target column: {target_col}")
    X_sequences, y_targets = preprocessor.transform_to_sequences(
        df, 
        sequence_length=sequence_length,
        target_col=target_col
    )
    logger.info(f"[OK] Created {len(X_sequences):,} sequences")
    
    # T058: Validate preprocessing
    logger.info("\n[STAGE 5/5] T058: Validating preprocessing...")
    is_valid = preprocessor.validate_preprocessing(df)
    if is_valid:
        logger.info("[OK] Preprocessing validation passed")
    else:
        logger.warning("[WARN] Preprocessing validation found issues")
    
    # Save preprocessed data
    logger.info(f"\nSaving preprocessed data to: {output_file}")
    df.to_csv(output_file, index=False)
    logger.info(f"[OK] Saved preprocessed data: {output_file.name}")
    
    # Save sequences as numpy arrays
    sequences_file = output_file.parent / (output_file.stem + '_sequences.npz')
    np.savez(sequences_file, X=X_sequences, y=y_targets)
    logger.info(f"[OK] Saved sequences: {sequences_file.name}")
    
    logger.info("\n" + "="*80)
    logger.info("PREPROCESSING COMPLETE")
    logger.info(f"Sequences: {X_sequences.shape}")
    logger.info(f"Targets: {y_targets.shape}")
    logger.info("="*80 + "\n")
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X_sequences)
    y_tensor = torch.FloatTensor(y_targets)
    
    return X_tensor, y_tensor, preprocessor
