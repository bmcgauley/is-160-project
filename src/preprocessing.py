"""
Data Preprocessing Module for QCEW Employment Data Analysis

This module contains functions for normalizing data, handling missing values,
creating categorical encodings, and transforming data for RNN/LSTM processing.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from typing import Dict, List, Tuple, Optional, Union
import logging
import torch

logger = logging.getLogger(__name__)


class EmploymentDataPreprocessor:
    """Class for preprocessing employment data for LSTM modeling."""

    def __init__(self, scaler_type: str = 'robust'):
        """
        Initialize the preprocessor.

        Args:
            scaler_type: Type of scaler to use ('standard' or 'robust')
        """
        self.scaler_type = scaler_type
        self.scalers = {}
        self.imputers = {}
        self.encoders = {}

    def normalize_employment_data(self, df: pd.DataFrame,
                                employment_cols: List[str] = None) -> pd.DataFrame:
        """
        Normalize employment counts and wage data using robust scaling techniques.

        Args:
            df: Input dataframe
            employment_cols: Columns to normalize

        Returns:
            DataFrame with normalized columns
        """
        from sklearn.preprocessing import RobustScaler, StandardScaler
        
        if employment_cols is None:
            employment_cols = ['avg_monthly_emplvl', 'total_qtrly_wages', 'avg_wkly_wage']
        
        # Filter to columns that exist
        employment_cols = [col for col in employment_cols if col in df.columns]
        
        if len(employment_cols) == 0:
            logger.info("  No employment columns to normalize")
            return df

        logger.info(f"Normalizing columns: {employment_cols}")
        
        # Choose scaler
        if self.scaler_type == 'robust':
            scaler = RobustScaler()
            logger.info("  Using RobustScaler (resistant to outliers)")
        else:
            scaler = StandardScaler()
            logger.info("  Using StandardScaler")
        
        # Apply normalization
        df[employment_cols] = scaler.fit_transform(df[employment_cols])
        self.scalers['employment'] = scaler
        
        logger.info(f"  Normalized {len(employment_cols)} columns")
        for col in employment_cols[:3]:  # Show stats for first 3 columns
            logger.info(f"    {col}: mean={df[col].mean():.3f}, std={df[col].std():.3f}")
        
        return df
        return df

    def handle_missing_values(self, df: pd.DataFrame,
                            strategy: str = 'median') -> pd.DataFrame:
        """
        Handle missing values with domain-appropriate imputation strategies.

        Args:
            df: Input dataframe
            strategy: Imputation strategy

        Returns:
            DataFrame with imputed values
        """
        from sklearn.impute import SimpleImputer
        
        logger.info(f"Handling missing values with {strategy} strategy")
        
        # Identify numeric columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        if len(numeric_cols) == 0:
            logger.info("  No numeric columns to impute")
            return df
        
        # Count missing values before imputation
        missing_before = df[numeric_cols].isnull().sum().sum()
        logger.info(f"  Missing values before imputation: {missing_before:,}")
        
        # Apply imputation
        imputer = SimpleImputer(strategy=strategy)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        
        # Store imputer for later use
        self.imputers['numeric'] = imputer
        
        # Count missing values after imputation
        missing_after = df[numeric_cols].isnull().sum().sum()
        logger.info(f"  Missing values after imputation: {missing_after:,}")
        logger.info(f"  Imputed {missing_before - missing_after:,} values")
        
        return df

    def create_categorical_encodings(self, df: pd.DataFrame,
                                   categorical_cols: List[str] = None) -> pd.DataFrame:
        """
        Create categorical encodings for industry codes and geographic identifiers.

        Args:
            df: Input dataframe
            categorical_cols: Columns to encode

        Returns:
            DataFrame with encoded categorical columns
        """
        from sklearn.preprocessing import LabelEncoder
        
        if categorical_cols is None:
            categorical_cols = ['industry_code', 'area_name', 'ownership']
        
        # Filter to columns that exist in dataframe
        categorical_cols = [col for col in categorical_cols if col in df.columns]
        
        if len(categorical_cols) == 0:
            logger.info("  No categorical columns to encode")
            return df

        logger.info(f"Creating categorical encodings for: {categorical_cols}")
        
        for col in categorical_cols:
            le = LabelEncoder()
            # Handle NaN values by converting to string first
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])
            self.encoders[col] = le
            logger.info(f"  Encoded {col}: {len(le.classes_)} unique classes")
        
        return df
        return df

    def transform_to_sequences(self, df: pd.DataFrame,
                             sequence_length: int = 12,
                             target_col: str = 'month1_emplvl') -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform tabular data into sequence format suitable for RNN/LSTM processing.
        
        Creates sliding window sequences grouped by county and industry.
        Each sequence contains `sequence_length` consecutive quarters of data.

        Args:
            df: Input dataframe with columns: year, quarter, area_name, industry_code, employment metrics
            sequence_length: Length of sequences to create (default 12 = 3 years)
            target_col: Target column for prediction (default 'month1_emplvl')

        Returns:
            Tuple of (sequences, targets) as numpy arrays
            - sequences: Shape (num_sequences, sequence_length, num_features)
            - targets: Shape (num_sequences,)
        """
        logger.info(f"Transforming data to sequences of length {sequence_length}")
        logger.info(f"  Target column: {target_col}")
        
        # Identify feature columns (numeric columns excluding identifiers)
        exclude_cols = ['year', 'quarter', 'area_name', 'industry_code', 'ownership', 'area_type']
        feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
        
        logger.info(f"  Feature columns: {len(feature_cols)} columns")
        logger.info(f"  Grouping by: area_name, industry_code")
        
        # Create sequences grouped by county and industry
        sequences = []
        targets = []
        
        # Group by county and industry
        grouped = df.groupby(['area_name', 'industry_code'])
        
        for (area, industry), group in grouped:
            # Sort by year and quarter
            group = group.sort_values(['year', 'quarter'])
            
            # Skip if group too small
            if len(group) < sequence_length + 1:
                continue
            
            # Extract feature values
            values = group[feature_cols].values
            
            # Create sliding window sequences
            for i in range(len(values) - sequence_length):
                seq = values[i:i+sequence_length]
                target = group[target_col].iloc[i+sequence_length]
                
                sequences.append(seq)
                targets.append(target)
        
        logger.info(f"  Created {len(sequences)} sequences from {len(grouped)} groups")
        
        if len(sequences) == 0:
            logger.warning("[WARN] No sequences created! Check data has sufficient time series length")
            return np.array([]), np.array([])
        
        # Convert to numpy arrays
        X_sequences = np.array(sequences)
        y_targets = np.array(targets)
        
        logger.info(f"  Sequences shape: {X_sequences.shape}")
        logger.info(f"  Targets shape: {y_targets.shape}")
        
        return X_sequences, y_targets

    def validate_preprocessing(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate preprocessing steps maintain data distribution properties.

        Args:
            df: Preprocessed dataframe

        Returns:
            Dictionary of validation results
        """
        # TODO: Implement preprocessing validation
        logger.info("Validating preprocessing steps...")
        return {"preprocessing_validation": True}