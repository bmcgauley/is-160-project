"""
Dataset Module for QCEW Employment Data Analysis

This module contains PyTorch Dataset classes for efficient data loading
and batching of employment time series data.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class EmploymentDataset(Dataset):
    """PyTorch Dataset class for QCEW employment data."""

    def __init__(self,
                 sequences: np.ndarray,
                 targets: np.ndarray,
                 augment: bool = False):
        """
        Initialize the dataset.

        Args:
            sequences: Input sequences array
            targets: Target values array
            augment: Whether to apply data augmentation
        """
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        self.augment = augment

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single data sample.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (sequence, target)
        """
        sequence = self.sequences[idx]
        target = self.targets[idx]

        if self.augment:
            # Apply small Gaussian noise for augmentation (1% noise)
            noise = torch.randn_like(sequence) * 0.01
            sequence = sequence + noise

        return sequence, target


def create_data_augmentation(sequences: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
    """
    Implement data augmentation techniques appropriate for employment time series.

    Applies small Gaussian noise to create variation while maintaining temporal patterns.

    Args:
        sequences: Input sequences to augment (shape: num_sequences, seq_len, num_features)
        noise_level: Standard deviation of Gaussian noise (default 0.01 = 1%)

    Returns:
        Augmented sequences with added noise
    """
    logger.info(f"Applying data augmentation with noise_level={noise_level}...")

    # Add small Gaussian noise to sequences
    noise = np.random.normal(0, noise_level, sequences.shape)
    augmented = sequences + noise

    logger.info(f"  Augmented {len(sequences)} sequences")

    return augmented


def build_data_loader(dataset: EmploymentDataset,
                     batch_size: int = 32,
                     shuffle: bool = True,
                     num_workers: int = 0) -> DataLoader:
    """
    Build DataLoader with proper batch sizes for employment tensor processing.

    Args:
        dataset: EmploymentDataset instance
        batch_size: Batch size for loading
        shuffle: Whether to shuffle the data
        num_workers: Number of parallel workers for data loading

    Returns:
        PyTorch DataLoader
    """
    logger.info(f"Building DataLoader with batch_size={batch_size}, shuffle={shuffle}")

    # Configure DataLoader with appropriate settings
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False  # Keep all data, don't drop incomplete batches
    )

    logger.info(f"  Created DataLoader with {len(dataset)} samples in {len(data_loader)} batches")

    return data_loader


def create_train_val_splits(sequences: np.ndarray,
                          targets: np.ndarray,
                          val_size: float = 0.2,
                          test_size: float = 0.1,
                          random_state: int = 42,
                          shuffle: bool = False) -> Tuple[EmploymentDataset, EmploymentDataset, EmploymentDataset]:
    """
    Create train/validation/test data splits preserving temporal order.

    For time series data, we typically don't shuffle to preserve temporal ordering.
    The splits are done sequentially: train (earliest) -> val (middle) -> test (latest).

    Args:
        sequences: Input sequences (shape: num_sequences, seq_len, num_features)
        targets: Target values (shape: num_sequences,)
        val_size: Validation set size ratio (default 0.2)
        test_size: Test set size ratio (default 0.1)
        random_state: Random seed for reproducibility
        shuffle: Whether to shuffle before split (False preserves temporal order)

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    logger.info("Creating train/validation/test splits...")
    logger.info(f"  Total samples: {len(sequences):,}")
    logger.info(f"  Val size: {val_size:.1%}, Test size: {test_size:.1%}")
    logger.info(f"  Shuffle: {shuffle} (recommended: False for time series)")

    # Set random seed for reproducibility
    np.random.seed(random_state)

    # Calculate split sizes
    n_samples = len(sequences)
    test_samples = int(n_samples * test_size)
    val_samples = int(n_samples * val_size)
    train_samples = n_samples - val_samples - test_samples

    logger.info(f"  Train samples: {train_samples:,}")
    logger.info(f"  Val samples: {val_samples:,}")
    logger.info(f"  Test samples: {test_samples:,}")

    if shuffle:
        # Shuffle indices if requested (not recommended for time series)
        indices = np.random.permutation(n_samples)
        sequences = sequences[indices]
        targets = targets[indices]
        logger.warning("  [WARNING] Data shuffled - temporal order lost!")

    # Sequential split (preserves temporal order)
    # Train: [0 : train_samples]
    # Val:   [train_samples : train_samples + val_samples]
    # Test:  [train_samples + val_samples : end]

    train_seq = sequences[:train_samples]
    train_tgt = targets[:train_samples]

    val_seq = sequences[train_samples:train_samples + val_samples]
    val_tgt = targets[train_samples:train_samples + val_samples]

    test_seq = sequences[train_samples + val_samples:]
    test_tgt = targets[train_samples + val_samples:]

    # Create datasets
    train_dataset = EmploymentDataset(train_seq, train_tgt, augment=True)
    val_dataset = EmploymentDataset(val_seq, val_tgt, augment=False)
    test_dataset = EmploymentDataset(test_seq, test_tgt, augment=False)

    logger.info("  [OK] Datasets created successfully")

    return train_dataset, val_dataset, test_dataset


def validate_batch_processing(dataset: EmploymentDataset,
                            batch_size: int = 32) -> Dict[str, bool]:
    """
    Validate batch processing maintains employment data integrity and relationships.

    Args:
        dataset: Dataset to validate
        batch_size: Batch size for validation

    Returns:
        Dictionary of validation results
    """
    logger.info("Validating batch processing...")
    logger.info(f"  Dataset size: {len(dataset)}")
    logger.info(f"  Batch size: {batch_size}")

    results = {}

    try:
        # Create a dataloader
        loader = build_data_loader(dataset, batch_size=batch_size, shuffle=False)

        # Test 1: Can iterate through all batches
        total_samples = 0
        batch_count = 0
        for batch_seq, batch_tgt in loader:
            total_samples += len(batch_seq)
            batch_count += 1

        results['iteration_complete'] = True
        logger.info(f"  [OK] Iterated through {batch_count} batches")

        # Test 2: All samples accounted for
        results['all_samples_present'] = (total_samples == len(dataset))
        logger.info(f"  Samples: {total_samples}/{len(dataset)} - {'[PASS]' if results['all_samples_present'] else '[FAIL]'}")

        # Test 3: Batch shapes are consistent
        batch_seq, batch_tgt = next(iter(loader))
        results['valid_batch_shape'] = (len(batch_seq.shape) == 3 and  # (batch, seq, features)
                                       len(batch_tgt.shape) == 1)       # (batch,)
        logger.info(f"  Batch sequence shape: {tuple(batch_seq.shape)}")
        logger.info(f"  Batch target shape: {tuple(batch_tgt.shape)}")
        logger.info(f"  Shape validation: {'[PASS]' if results['valid_batch_shape'] else '[FAIL]'}")

        # Test 4: No NaN or Inf in batches
        has_nan_seq = torch.isnan(batch_seq).any().item()
        has_inf_seq = torch.isinf(batch_seq).any().item()
        has_nan_tgt = torch.isnan(batch_tgt).any().item()
        has_inf_tgt = torch.isinf(batch_tgt).any().item()

        results['no_nan_inf'] = not (has_nan_seq or has_inf_seq or has_nan_tgt or has_inf_tgt)
        logger.info(f"  No NaN/Inf: {'[PASS]' if results['no_nan_inf'] else '[FAIL]'}")

        # Test 5: Data types are correct
        results['correct_dtypes'] = (batch_seq.dtype == torch.float32 and
                                    batch_tgt.dtype == torch.float32)
        logger.info(f"  Data types: seq={batch_seq.dtype}, tgt={batch_tgt.dtype} - {'[PASS]' if results['correct_dtypes'] else '[FAIL]'}")

    except Exception as e:
        logger.error(f"  [FAIL] Validation error: {str(e)}")
        results['iteration_complete'] = False
        results['all_samples_present'] = False
        results['valid_batch_shape'] = False
        results['no_nan_inf'] = False
        results['correct_dtypes'] = False

    # Overall validation
    all_passed = all(results.values())
    logger.info(f"Overall validation: {'[PASS]' if all_passed else '[FAIL]'} ({sum(results.values())}/{len(results)} checks passed)")

    return results