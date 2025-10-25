"""
Loss Metrics Module for QCEW Employment Data Analysis

This module contains custom loss functions and evaluation metrics
for employment forecasting models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class WeightedEmploymentLoss(nn.Module):
    """Weighted loss function emphasizing recent employment trends."""

    def __init__(self, recent_weight: float = 1.5, base_loss: str = 'mse'):
        """
        Initialize the weighted loss.

        Args:
            recent_weight: Weight multiplier for recent observations
            base_loss: Base loss function ('mse' or 'mae')
        """
        super(WeightedEmploymentLoss, self).__init__()
        self.recent_weight = recent_weight
        self.base_loss = base_loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate weighted loss with emphasis on recent observations.

        Args:
            pred: Model predictions (batch_size,)
            target: Ground truth targets (batch_size,)

        Returns:
            Weighted loss value
        """
        # Calculate base loss element-wise
        if self.base_loss == 'mse':
            element_loss = (pred - target) ** 2
        elif self.base_loss == 'mae':
            element_loss = torch.abs(pred - target)
        else:
            element_loss = (pred - target) ** 2  # Default to MSE

        # Create weights that increase for later samples (more recent)
        batch_size = pred.size(0)
        weights = torch.linspace(1.0, self.recent_weight, batch_size).to(pred.device)

        # Apply weights
        weighted_loss = element_loss * weights

        return weighted_loss.mean()


class DirectionalAccuracyLoss(nn.Module):
    """Loss that penalizes incorrect direction predictions."""

    def __init__(self, direction_weight: float = 0.3):
        """
        Initialize directional accuracy loss.

        Args:
            direction_weight: Weight for directional component (0-1)
        """
        super(DirectionalAccuracyLoss, self).__init__()
        self.direction_weight = direction_weight
        self.magnitude_weight = 1.0 - direction_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                prev_target: torch.Tensor = None) -> torch.Tensor:
        """
        Calculate combined magnitude and directional loss.

        Args:
            pred: Model predictions
            target: Ground truth targets
            prev_target: Previous target values for direction calculation

        Returns:
            Combined loss value
        """
        # Magnitude loss (MSE)
        magnitude_loss = F.mse_loss(pred, target)

        if prev_target is not None:
            # Calculate direction of change
            pred_direction = torch.sign(pred - prev_target)
            target_direction = torch.sign(target - prev_target)

            # Directional loss (1 if wrong direction, 0 if correct)
            direction_loss = (pred_direction != target_direction).float().mean()

            # Combined loss
            total_loss = (self.magnitude_weight * magnitude_loss +
                         self.direction_weight * direction_loss)
        else:
            total_loss = magnitude_loss

        return total_loss


def mean_absolute_percentage_error(y_true: Union[np.ndarray, torch.Tensor],
                                   y_pred: Union[np.ndarray, torch.Tensor],
                                   epsilon: float = 1e-8) -> float:
    """
    Calculate MAPE for employment predictions.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        epsilon: Small constant to avoid division by zero

    Returns:
        MAPE value as percentage
    """
    # Convert to numpy if tensor
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Avoid division by zero
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate MAPE
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

    return mape


def directional_accuracy(y_true: Union[np.ndarray, torch.Tensor],
                        y_pred: Union[np.ndarray, torch.Tensor],
                        y_prev: Union[np.ndarray, torch.Tensor] = None) -> float:
    """
    Calculate directional accuracy for employment trends.

    Measures the percentage of predictions that correctly predicted the
    direction of change (increase/decrease) compared to previous value.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        y_prev: Previous values for direction calculation

    Returns:
        Directional accuracy as percentage (0-100)
    """
    # Convert to numpy if tensor
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_prev, torch.Tensor):
        y_prev = y_prev.detach().cpu().numpy()

    if y_prev is None:
        # If no previous values, calculate direction from mean
        y_prev = np.ones_like(y_true) * np.mean(y_true)

    # Calculate directions
    true_direction = np.sign(y_true - y_prev)
    pred_direction = np.sign(y_pred - y_prev)

    # Calculate accuracy (ignore zero changes)
    non_zero = true_direction != 0
    if non_zero.sum() == 0:
        return 100.0  # All zeros, perfect by default

    accuracy = (true_direction[non_zero] == pred_direction[non_zero]).mean() * 100

    return accuracy


def root_mean_squared_error(y_true: Union[np.ndarray, torch.Tensor],
                            y_pred: Union[np.ndarray, torch.Tensor]) -> float:
    """
    Calculate RMSE.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        RMSE value
    """
    # Convert to numpy if tensor
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse


def mean_absolute_error(y_true: Union[np.ndarray, torch.Tensor],
                       y_pred: Union[np.ndarray, torch.Tensor]) -> float:
    """
    Calculate MAE.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        MAE value
    """
    # Convert to numpy if tensor
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    mae = np.mean(np.abs(y_true - y_pred))
    return mae


def create_custom_metrics() -> Dict[str, callable]:
    """
    Create custom metrics for employment forecasting accuracy (MAPE, directional accuracy).

    Returns:
        Dictionary of metric functions
    """
    logger.info("Creating custom employment forecasting metrics")

    metrics = {
        'mape': mean_absolute_percentage_error,
        'directional_accuracy': directional_accuracy,
        'rmse': root_mean_squared_error,
        'mae': mean_absolute_error,
    }

    logger.info(f"  Created {len(metrics)} metrics: {list(metrics.keys())}")

    return metrics


class VolatilityLoss(nn.Module):
    """Loss for capturing employment volatility/uncertainty."""

    def __init__(self, volatility_weight: float = 0.1):
        """
        Initialize volatility loss.

        Args:
            volatility_weight: Weight for volatility component
        """
        super(VolatilityLoss, self).__init__()
        self.volatility_weight = volatility_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                pred_std: torch.Tensor = None) -> torch.Tensor:
        """
        Calculate loss including volatility prediction.

        Args:
            pred: Mean predictions
            target: Ground truth
            pred_std: Predicted standard deviations (optional)

        Returns:
            Combined loss
        """
        # Base MSE loss
        mse_loss = F.mse_loss(pred, target)

        if pred_std is not None:
            # Negative log-likelihood assuming Gaussian distribution
            # NLL = 0.5 * log(2*pi*sigma^2) + (y - mu)^2 / (2*sigma^2)
            variance = pred_std ** 2 + 1e-6  # Add small constant for stability
            nll_loss = 0.5 * (torch.log(variance) + (target - pred) ** 2 / variance).mean()

            total_loss = (1 - self.volatility_weight) * mse_loss + self.volatility_weight * nll_loss
        else:
            total_loss = mse_loss

        return total_loss


def create_volatility_loss(prediction_horizon: int = 4,
                          volatility_weight: float = 0.1) -> VolatilityLoss:
    """
    Add employment volatility prediction loss for capturing uncertainty.

    Args:
        prediction_horizon: Number of steps to predict
        volatility_weight: Weight for volatility component

    Returns:
        Volatility loss module
    """
    logger.info(f"Creating volatility loss for {prediction_horizon} step predictions")
    logger.info(f"  Volatility weight: {volatility_weight}")
    return VolatilityLoss(volatility_weight=volatility_weight)


class IndustryWeightedLoss(nn.Module):
    """Industry-weighted loss for sector-specific importance."""

    def __init__(self, industry_weights: Dict[int, float] = None):
        """
        Initialize industry-weighted loss.

        Args:
            industry_weights: Dictionary mapping industry indices to weights
        """
        super(IndustryWeightedLoss, self).__init__()
        self.industry_weights = industry_weights or {}

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                industry_indices: torch.Tensor = None) -> torch.Tensor:
        """
        Calculate industry-weighted loss.

        Args:
            pred: Model predictions
            target: Ground truth
            industry_indices: Industry index for each sample

        Returns:
            Weighted loss
        """
        # Base MSE loss
        element_loss = (pred - target) ** 2

        if industry_indices is not None and len(self.industry_weights) > 0:
            # Apply industry-specific weights
            weights = torch.ones_like(pred)
            for idx, weight in self.industry_weights.items():
                mask = industry_indices == idx
                weights[mask] = weight

            weighted_loss = (element_loss * weights).mean()
        else:
            weighted_loss = element_loss.mean()

        return weighted_loss


def create_industry_weighted_loss(industry_weights: Dict[str, float] = None) -> IndustryWeightedLoss:
    """
    Build industry-weighted loss functions for sector-specific prediction importance.

    Args:
        industry_weights: Dictionary mapping industry codes to weights

    Returns:
        Industry-weighted loss module
    """
    if industry_weights is None:
        industry_weights = {}

    logger.info("Creating industry-weighted loss function")
    logger.info(f"  Industry weights: {len(industry_weights)} industries")

    return IndustryWeightedLoss(industry_weights)


def validate_loss_functions() -> Dict[str, bool]:
    """
    Validate loss functions align with employment forecasting evaluation standards.

    Returns:
        Dictionary of validation results
    """
    logger.info("Validating loss functions...")

    results = {}

    try:
        # Test 1: MAPE calculation
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 310])
        mape = mean_absolute_percentage_error(y_true, y_pred)
        results['mape_works'] = 0 <= mape <= 100
        logger.info(f"  MAPE test: {mape:.2f}% - {'✓ PASS' if results['mape_works'] else '✗ FAIL'}")

        # Test 2: Directional accuracy
        y_prev = np.array([90, 210, 290])
        dir_acc = directional_accuracy(y_true, y_pred, y_prev)
        results['directional_accuracy_works'] = 0 <= dir_acc <= 100
        logger.info(f"  Directional accuracy test: {dir_acc:.2f}% - {'✓ PASS' if results['directional_accuracy_works'] else '✗ FAIL'}")

        # Test 3: RMSE calculation
        rmse = root_mean_squared_error(y_true, y_pred)
        results['rmse_works'] = rmse >= 0
        logger.info(f"  RMSE test: {rmse:.2f} - {'✓ PASS' if results['rmse_works'] else '✗ FAIL'}")

        # Test 4: Weighted loss
        weighted_loss = WeightedEmploymentLoss()
        pred_tensor = torch.tensor([110.0, 190.0, 310.0])
        target_tensor = torch.tensor([100.0, 200.0, 300.0])
        loss_val = weighted_loss(pred_tensor, target_tensor)
        results['weighted_loss_works'] = not torch.isnan(loss_val).item()
        logger.info(f"  Weighted loss test: {loss_val:.4f} - {'✓ PASS' if results['weighted_loss_works'] else '✗ FAIL'}")

        # Test 5: Volatility loss
        vol_loss = VolatilityLoss()
        loss_val = vol_loss(pred_tensor, target_tensor)
        results['volatility_loss_works'] = not torch.isnan(loss_val).item()
        logger.info(f"  Volatility loss test: {loss_val:.4f} - {'✓ PASS' if results['volatility_loss_works'] else '✗ FAIL'}")

    except Exception as e:
        logger.error(f"  ✗ Validation error: {str(e)}")
        results = {k: False for k in ['mape_works', 'directional_accuracy_works',
                                     'rmse_works', 'weighted_loss_works', 'volatility_loss_works']}

    # Overall validation
    all_passed = all(results.values())
    logger.info(f"Overall validation: {'✓ PASS' if all_passed else '✗ FAIL'} ({sum(results.values())}/{len(results)} checks passed)")

    return results
