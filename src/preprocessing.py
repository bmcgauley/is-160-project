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
                             sequence_length: int = 8,
                             target_col: str = 'avg_monthly_emplvl') -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform tabular data into sequence format suitable for RNN/LSTM processing.

        Creates sliding window sequences grouped by county and industry.
        Each sequence contains `sequence_length` consecutive quarters of data.

        Args:
            df: Input dataframe with columns: year, quarter, area_name, industry_code, employment metrics
            sequence_length: Length of sequences to create (default 8 = 2 years, reduced from 12)
            target_col: Target column for prediction (default 'avg_monthly_emplvl')

        Returns:
            Tuple of (sequences, targets) as numpy arrays
            - sequences: Shape (num_sequences, sequence_length, num_features)
            - targets: Shape (num_sequences,)
        """
        logger.info(f"Transforming data to sequences of length {sequence_length}")
        logger.info(f"  Target column: {target_col}")
        logger.info(f"  Available columns: {sorted(df.columns.tolist())[:10]}...")  # Show first 10 columns

        # Check for lag features - they should be included in sequences
        lag_cols = [col for col in df.columns if 'lag' in col.lower() or col.endswith('_lag')]
        growth_cols = [col for col in df.columns if 'growth' in col.lower() or 'chg' in col.lower()]
        logger.info(f"  Lag features found: {len(lag_cols)} ({lag_cols[:5]}...)")
        logger.info(f"  Growth features found: {len(growth_cols)} ({growth_cols[:5]}...)")

        # Identify feature columns (numeric columns excluding identifiers)
        # CRITICAL FIX: Include lag and growth features explicitly
        exclude_cols = ['year', 'quarter', 'area_name', 'industry_code', 'ownership', 'area_type']
        feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]

        # Prioritize lag and growth features
        temporal_features = [col for col in feature_cols if 'lag' in col.lower() or 'growth' in col.lower() or 'chg' in col.lower()]
        other_features = [col for col in feature_cols if col not in temporal_features]

        # Sort features to ensure consistent ordering
        feature_cols = sorted(temporal_features) + sorted(other_features)

        logger.info(f"  Feature columns: {len(feature_cols)} total")
        logger.info(f"    - Temporal features: {len(temporal_features)} ({temporal_features[:3]}...)")
        logger.info(f"    - Other features: {len(other_features)} ({other_features[:3]}...)")
        logger.info(f"  Grouping by: area_name, industry_code")

        # Verify target column exists
        if target_col not in df.columns:
            available_targets = [col for col in df.columns if 'emplvl' in col.lower() or 'employment' in col.lower()]
            logger.warning(f"Target column '{target_col}' not found. Available: {available_targets}")
            if available_targets:
                target_col = available_targets[0]
                logger.info(f"Using target column: {target_col}")
            else:
                logger.error("No suitable target column found!")
                return np.array([]), np.array([])

        # Create sequences grouped by county and industry
        sequences = []
        targets = []
        skipped_groups = 0
        total_groups = 0

        # Group by county and industry
        grouped = df.groupby(['area_name', 'industry_code'])

        for (area, industry), group in grouped:
            total_groups += 1

            # Sort by year and quarter
            group = group.sort_values(['year', 'quarter'])

            # Skip if group too small
            if len(group) < sequence_length + 1:
                skipped_groups += 1
                continue

            # Extract feature values - ensure all feature columns exist in this group
            group_features = [col for col in feature_cols if col in group.columns]
            if len(group_features) != len(feature_cols):
                logger.warning(f"  Group {area}-{industry}: missing {len(feature_cols) - len(group_features)} features")
                continue

            values = group[group_features].values

            # Create sliding window sequences
            for i in range(len(values) - sequence_length):
                seq = values[i:i+sequence_length]
                target = group[target_col].iloc[i+sequence_length]

                # Validate sequence is not all NaN
                if not np.isnan(seq).all() and not np.isnan(target):
                    sequences.append(seq)
                    targets.append(target)

        logger.info(f"  Processed {total_groups} groups, skipped {skipped_groups} (insufficient data)")
        logger.info(f"  Created {len(sequences)} sequences from {total_groups - skipped_groups} valid groups")

        if len(sequences) == 0:
            logger.error("[ERROR] No sequences created! Possible issues:")
            logger.error("  - Target column missing or all NaN")
            logger.error("  - Feature columns missing from groups")
            logger.error("  - Sequence length too long for available data")
            logger.error("  - All sequences contain NaN values")
            return np.array([]), np.array([])

        # Convert to numpy arrays
        X_sequences = np.array(sequences)
        y_targets = np.array(targets)

        logger.info(f"  Sequences shape: {X_sequences.shape}")
        logger.info(f"  Targets shape: {y_targets.shape}")
        logger.info(f"  Target range: [{y_targets.min():.2f}, {y_targets.max():.2f}]")

        return X_sequences, y_targets

    def validate_preprocessing(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate preprocessing steps maintain data distribution properties.

        Args:
            df: Preprocessed dataframe

        Returns:
            Dictionary of validation results
        """
        logger.info("Validating preprocessing steps...")

        results = {}

        # Check 1: No infinite values
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        inf_count = np.isinf(df[numeric_cols]).sum().sum()
        results['no_infinites'] = inf_count == 0
        logger.info(f"  Infinite values check: {inf_count} found - {'PASS' if results['no_infinites'] else 'FAIL'}")

        # Check 2: No NaN values after imputation
        nan_count = df[numeric_cols].isnull().sum().sum()
        results['no_nans'] = nan_count == 0
        logger.info(f"  NaN values check: {nan_count} found - {'PASS' if results['no_nans'] else 'FAIL'}")

        # Check 3: Normalized columns have reasonable ranges (within -10 to 10 std)
        if 'employment' in self.scalers:
            # Check that normalized data is within reasonable bounds
            scaled_cols = ['avg_monthly_emplvl', 'total_qtrly_wages', 'avg_wkly_wage']
            scaled_cols = [col for col in scaled_cols if col in df.columns]
            if scaled_cols:
                max_abs_val = df[scaled_cols].abs().max().max()
                results['normalized_range'] = max_abs_val < 10.0
                logger.info(f"  Normalization range check: max |value| = {max_abs_val:.2f} - {'PASS' if results['normalized_range'] else 'FAIL'}")

        # Check 4: Categorical encodings are integers
        if self.encoders:
            encoded_cols = list(self.encoders.keys())
            encoded_cols = [col for col in encoded_cols if col in df.columns]
            if encoded_cols:
                all_int = all(df[col].dtype in ['int32', 'int64'] for col in encoded_cols)
                results['categorical_encoded'] = all_int
                logger.info(f"  Categorical encoding check: All integer types - {'PASS' if all_int else 'FAIL'}")

        # Check 5: Data shape preservation (rows should not disappear)
        results['shape_preserved'] = len(df) > 0
        logger.info(f"  Shape preservation: {len(df):,} rows - {'PASS' if results['shape_preserved'] else 'FAIL'}")

        # Overall validation
        all_passed = all(results.values())
        logger.info(f"Overall validation: {'PASS' if all_passed else 'FAIL'} ({sum(results.values())}/{len(results)} checks passed)")

        return results