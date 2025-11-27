"""
Visualization Module for QCEW Employment Data Analysis

This module contains functions for visualizing LSTM patterns,
feature importance, and employment prediction results.

Enhanced with comprehensive training history and prediction analysis plots.
"""

# Configure matplotlib to use non-interactive backend (must be before importing pyplot)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend that doesn't require Tcl/Tk

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Set style for all plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def implement_feature_attribution(model, data: pd.DataFrame) -> Dict[str, float]:
    """
    Implement feature attribution techniques for employment factor importance.

    Args:
        model: Trained model
        data: Input data for attribution

    Returns:
        Dictionary of feature importance scores
    """
    # TODO: Implement feature attribution
    logger.info("Implementing feature attribution techniques...")
    return {"feature_importance": {}}


def visualize_lstm_patterns(model, data: pd.DataFrame) -> plt.Figure:
    """
    Visualize LSTM learned patterns and their relationship to employment sequences.

    Args:
        model: Trained LSTM model
        data: Employment sequence data

    Returns:
        Matplotlib figure
    """
    # TODO: Implement LSTM pattern visualization
    logger.info("Visualizing LSTM learned patterns...")
    fig, ax = plt.subplots(figsize=(12, 8))
    return fig


def create_employment_trend_visualizations(predictions: pd.DataFrame,
                                         actuals: pd.DataFrame) -> plt.Figure:
    """
    Create employment trend visualizations showing model predictions vs reality.

    Args:
        predictions: Model predictions
        actuals: Actual employment data

    Returns:
        Matplotlib figure
    """
    # TODO: Implement trend visualization
    logger.info("Creating employment trend visualizations...")
    fig, ax = plt.subplots(figsize=(14, 8))
    return fig


def generate_geographic_heat_maps(accuracy_data: pd.DataFrame) -> plt.Figure:
    """
    Generate geographic heat maps of employment prediction accuracy.

    Args:
        accuracy_data: DataFrame with geographic accuracy metrics

    Returns:
        Matplotlib figure
    """
    # TODO: Implement geographic heat maps
    logger.info("Generating geographic heat maps...")
    fig, ax = plt.subplots(figsize=(10, 8))
    return fig


def validate_feature_importance(feature_importance: Dict[str, float]) -> Dict[str, bool]:
    """
    Validate feature importance aligns with known employment economic factors.

    Args:
        feature_importance: Feature importance scores

    Returns:
        Dictionary of validation results
    """
    # TODO: Implement feature importance validation
    logger.info("Validating feature importance...")
    return {"importance_validation": True}


# ============================================================================
# ENHANCED TRAINING AND EVALUATION VISUALIZATIONS
# ============================================================================

def plot_enhanced_training_history(history: Dict[str, List[float]],
                                   save_path: Path,
                                   best_epoch: int = None) -> None:
    """
    Create comprehensive training history visualization with detailed annotations.

    This creates a 2x3 grid of plots showing:
    1. Training/Val Loss (Linear): Shows raw loss values over time
    2. Training/Val Loss (Log): Better for seeing trends across large value ranges
    3. Learning Rate Schedule: Shows how LR adapts during training
    4. Overfitting Monitor: Train/Val ratio indicates model fit quality

    Args:
        history: Dictionary containing 'train_loss', 'val_loss', 'learning_rates'
        save_path: Path to save the figure
        best_epoch: Epoch number with best validation loss (for marking)
    """
    # Validate that we have training history data
    if not history or 'train_loss' not in history or len(history.get('train_loss', [])) == 0:
        logger.warning("No training history available - skipping training history plot")
        # Create a simple placeholder plot
        fig = plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 'No Training History Available\n\nModel may have been loaded from checkpoint without history',
                ha='center', va='center', fontsize=16, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.axis('off')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"[OK] Saved placeholder training history plot: {save_path}")
        return

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    epochs = range(1, len(history['train_loss']) + 1)

    # Plot 1: Training and Validation Loss (Linear Scale)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(epochs, history['train_loss'], label='Training Loss',
             linewidth=2, marker='o', markersize=4, alpha=0.8, color='#3498db')

    # Only plot validation loss if it exists and has data
    has_val_loss = 'val_loss' in history and len(history['val_loss']) > 0
    if has_val_loss:
        ax1.plot(epochs, history['val_loss'], label='Validation Loss',
                 linewidth=2, marker='s', markersize=4, alpha=0.8, color='#e74c3c')

    # Only mark best epoch if we have validation loss data
    if best_epoch is not None and has_val_loss and best_epoch <= len(history['val_loss']):
        ax1.axvline(x=best_epoch, color='green', linestyle='--',
                    linewidth=2, alpha=0.7, label=f'Best Epoch ({best_epoch})')
        ax1.plot(best_epoch, history['val_loss'][best_epoch-1],
                marker='*', markersize=20, color='gold', zorder=5)

    ax1.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Loss (MSE)', fontsize=14, fontweight='bold')
    ax1.set_title('Training Progress: Loss Curves\n' +
                  'Lower is better. Gap indicates overfitting if training << validation',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle='--')

    final_train = history['train_loss'][-1]
    if has_val_loss:
        final_val = history['val_loss'][-1]
        ax1.text(0.02, 0.98, f'Final Train Loss: {final_train:.2f}\nFinal Val Loss: {final_val:.2f}',
                 transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax1.text(0.02, 0.98, f'Final Train Loss: {final_train:.2f}',
                 transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Loss Curves (Log Scale)
    ax2 = fig.add_subplot(gs[1, :2])
    ax2.plot(epochs, history['train_loss'], label='Training Loss',
             linewidth=2, alpha=0.8, color='#3498db')

    if has_val_loss:
        ax2.plot(epochs, history['val_loss'], label='Validation Loss',
                 linewidth=2, alpha=0.8, color='#e74c3c')

    if best_epoch is not None and has_val_loss:
        ax2.axvline(x=best_epoch, color='green', linestyle='--',
                    linewidth=2, alpha=0.7)

    ax2.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Loss (MSE, log scale)', fontsize=14, fontweight='bold')
    ax2.set_title('Training Progress: Loss Curves (Log Scale)\n' +
                  'Better for seeing trends when loss values span large ranges',
                  fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle='--', which='both')

    # Plot 3: Learning Rate Schedule
    ax3 = fig.add_subplot(gs[0, 2])
    has_lr = 'learning_rates' in history and len(history['learning_rates']) > 0

    if has_lr:
        ax3.plot(epochs, history['learning_rates'], linewidth=2.5,
                 color='#2ecc71', marker='o', markersize=5)

        # Mark LR changes
        lrs = history['learning_rates']
        for i in range(1, len(lrs)):
            if lrs[i] != lrs[i-1]:
                ax3.axvline(x=i+1, color='red', linestyle=':', alpha=0.5, linewidth=1)
                ax3.text(i+1, lrs[i], f'  {lrs[i]:.2e}', fontsize=8, color='red')
        ax3.set_yscale('log')
    else:
        ax3.text(0.5, 0.5, 'No Learning Rate Data Available',
                transform=ax3.transAxes, fontsize=14, ha='center', va='center')

    ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    ax3.set_title('Learning Rate Schedule\n' +
                  'Adaptive: Decreases when\nvalidation loss plateaus',
                  fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')

    # Plot 4: Overfitting Monitor
    ax4 = fig.add_subplot(gs[1, 2])

    if has_val_loss:
        train_val_ratio = [t/v if v > 0 else 1.0
                           for t, v in zip(history['train_loss'], history['val_loss'])]
        ax4.plot(epochs, train_val_ratio, linewidth=2, color='#9b59b6', marker='d', markersize=4)
        ax4.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Perfect Fit')
        ax4.fill_between(epochs, 0.8, 1.2, alpha=0.2, color='green', label='Good Range')
        ax4.set_ylim([0, min(3, max(train_val_ratio) * 1.1)])
    else:
        # No validation data - just show a message
        ax4.text(0.5, 0.5, 'No Validation Data Available',
                transform=ax4.transAxes, fontsize=14, ha='center', va='center')

    ax4.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Train/Val Loss Ratio', fontsize=12, fontweight='bold')
    ax4.set_title('Overfitting Monitor\n' +
                  'Ratio < 1: Underfitting\nRatio > 1: Overfitting',
                  fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9, loc='upper right')
    ax4.grid(True, alpha=0.3, linestyle='--')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"[OK] Saved enhanced training history: {save_path}")


def plot_prediction_analysis(y_true: np.ndarray,
                             y_pred: np.ndarray,
                             save_dir: Path,
                             sample_size: int = 1000) -> None:
    """
    Create comprehensive prediction analysis plots.

    Creates a 2x2 grid showing:
    1. Prediction vs Actual: Scatter plot with perfect prediction line
    2. Residual Plot: Shows error patterns
    3. Error Distribution: Histogram of prediction errors
    4. Absolute Error vs Magnitude: Shows if error scales with value

    Args:
        y_true: True target values
        y_pred: Predicted values
        save_dir: Directory to save plots
        sample_size: Number of samples to plot (for performance with large datasets)
    """
    # Sample data if too large
    if len(y_true) > sample_size:
        indices = np.random.choice(len(y_true), sample_size, replace=False)
        y_true_sample = y_true[indices]
        y_pred_sample = y_pred[indices]
    else:
        y_true_sample = y_true
        y_pred_sample = y_pred

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Plot 1: Prediction vs Actual Scatter
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_true_sample, y_pred_sample, alpha=0.5, s=20, c='#3498db')

    min_val = min(y_true_sample.min(), y_pred_sample.min())
    max_val = max(y_true_sample.max(), y_pred_sample.max())
    ax1.plot([min_val, max_val], [min_val, max_val],
             'r--', linewidth=2, label='Perfect Prediction')

    ax1.set_xlabel('Actual Values', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
    ax1.set_title('Predictions vs Actual Values\n' +
                  'Points on red line = perfect predictions',
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    ax1.text(0.05, 0.95, f'R² Score: {r2:.4f}',
             transform=ax1.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # Plot 2: Residual Plot
    ax2 = fig.add_subplot(gs[0, 1])
    residuals = y_true_sample - y_pred_sample
    ax2.scatter(y_pred_sample, residuals, alpha=0.5, s=20, c='#e74c3c')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=2)

    std_residual = np.std(residuals)
    ax2.axhline(y=2*std_residual, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.axhline(y=-2*std_residual, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)

    ax2.set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Residuals (Actual - Predicted)', fontsize=12, fontweight='bold')
    ax2.set_title('Residual Plot\n' +
                  'Random scatter around 0 = good model',
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Error Distribution
    ax3 = fig.add_subplot(gs[1, 0])
    errors = y_true - y_pred
    ax3.hist(errors, bins=50, alpha=0.7, color='#9b59b6', edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax3.axvline(x=np.mean(errors), color='green', linestyle='--',
                linewidth=2, label=f'Mean Error: {np.mean(errors):.2f}')

    ax3.set_xlabel('Prediction Error', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax3.set_title('Error Distribution\n' +
                  'Should be centered near 0 (unbiased)',
                  fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Absolute Error vs Actual Value
    ax4 = fig.add_subplot(gs[1, 1])
    abs_errors = np.abs(y_true_sample - y_pred_sample)
    ax4.scatter(y_true_sample, abs_errors, alpha=0.5, s=20, c='#f39c12')

    z = np.polyfit(y_true_sample, abs_errors, 1)
    p = np.poly1d(z)
    ax4.plot(np.sort(y_true_sample), p(np.sort(y_true_sample)),
             "r--", linewidth=2, label='Trend')

    ax4.set_xlabel('Actual Values', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Absolute Error', fontsize=12, fontweight='bold')
    ax4.set_title('Absolute Error vs Actual Value\n' +
                  'Shows if error varies with magnitude',
                  fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    ax4.text(0.05, 0.95, f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}',
             transform=ax4.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    save_path = save_dir / "prediction_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"[OK] Saved prediction analysis: {save_path}")


def plot_live_training_update(history: Dict[str, List[float]],
                               epoch: int,
                               save_path: Path) -> None:
    """
    Create a quick training progress plot for live updates during training.

    This is optimized for speed (lower DPI) to enable frequent updates.

    Args:
        history: Training history up to current epoch
        epoch: Current epoch number
        save_path: Path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], label='Train', linewidth=2, color='blue')

    # Only plot validation loss if it exists and has data
    if 'val_loss' in history and len(history['val_loss']) > 0:
        ax1.plot(epochs, history['val_loss'], label='Val', linewidth=2, color='red')

    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title(f'Training Progress (Epoch {epoch})', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['learning_rates'], linewidth=2, color='green')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Learning Rate', fontsize=11)
    ax2.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()