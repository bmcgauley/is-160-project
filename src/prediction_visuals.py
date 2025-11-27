"""
Prediction Visuals Module for QCEW Employment Data Analysis

This module contains functions for creating visual predictions vs actuals plots.
"""

# Configure matplotlib to use non-interactive backend (must be before importing pyplot)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


def create_predictions_vs_actuals_plot(predictions: pd.DataFrame,
                                      actuals: pd.DataFrame,
                                      confidence_intervals: Optional[pd.DataFrame] = None) -> plt.Figure:
    """
    Create visual predictions vs actuals plots showing predicted employment alongside actual values.

    Args:
        predictions: Model predictions
        actuals: Actual employment data
        confidence_intervals: Optional confidence intervals

    Returns:
        Matplotlib figure
    """
    # TODO: Implement predictions vs actuals plotting
    logger.info("Creating predictions vs actuals plots...")
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot actuals
    ax.plot(actuals.index, actuals.values, label='Actual', color='blue', linewidth=2)

    # Plot predictions
    ax.plot(predictions.index, predictions.values, label='Predicted', color='red', linestyle='--', linewidth=2)

    # Add confidence intervals if provided
    if confidence_intervals is not None:
        ax.fill_between(predictions.index,
                       confidence_intervals['lower'],
                       confidence_intervals['upper'],
                       alpha=0.3, color='red', label='Confidence Interval')

    ax.set_title('Employment Predictions vs Actual Values')
    ax.set_xlabel('Time')
    ax.set_ylabel('Employment Count')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig