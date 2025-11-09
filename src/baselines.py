"""
Baselines Module for QCEW Employment Data Analysis

This module contains traditional employment forecasting models
for comparison with LSTM performance.

Key Functions (T092-T095):
- implement_arima_model: ARIMA forecasting model with configurable parameters
- implement_exponential_smoothing: Exponential smoothing with seasonal components
- compare_lstm_performance: Performance comparison with RMSE, MAPE, directional accuracy
- benchmark_computational_efficiency: Computational efficiency benchmarking with timing
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from typing import Dict, Tuple, Union
import logging
import time
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

logger = logging.getLogger(__name__)


def implement_arima_model(data: pd.Series,
                         order: Tuple[int, int, int] = (1, 1, 1),
                         forecast_steps: int = 4) -> Dict[str, Union[object, np.ndarray, Dict]]:
    """
    T092: Implement ARIMA model for employment forecasting with configurable parameters.

    ARIMA (AutoRegressive Integrated Moving Average) is a traditional time-series
    forecasting method that captures trends, seasonality, and autocorrelation.

    Args:
        data: Time series data (employment values)
        order: ARIMA order parameters (p, d, q)
               p: AR order (autoregressive terms)
               d: Integration order (differencing)
               q: MA order (moving average terms)
        forecast_steps: Number of steps ahead to forecast

    Returns:
        Dictionary containing:
        - model: Fitted ARIMA model object
        - predictions: Out-of-sample predictions
        - fitted_values: In-sample fitted values
        - diagnostics: Model fit statistics (AIC, BIC, HQIC)
        - residuals: Model residuals for diagnostics
    """
    logger.info(f"T092: Implementing ARIMA({order[0]}, {order[1]}, {order[2]}) model...")

    try:
        # Fit ARIMA model
        model = ARIMA(data, order=order)
        fitted_model = model.fit()

        # Generate forecasts
        forecast = fitted_model.forecast(steps=forecast_steps)

        # Extract diagnostics
        diagnostics = {
            'aic': fitted_model.aic,
            'bic': fitted_model.bic,
            'hqic': fitted_model.hqic,
            'params': fitted_model.params.to_dict() if hasattr(fitted_model.params, 'to_dict') else dict(fitted_model.params)
        }

        logger.info(f"[OK] ARIMA model fitted successfully (AIC: {diagnostics['aic']:.2f}, BIC: {diagnostics['bic']:.2f})")

        return {
            "model": fitted_model,
            "predictions": forecast.values if hasattr(forecast, 'values') else forecast,
            "fitted_values": fitted_model.fittedvalues.values,
            "diagnostics": diagnostics,
            "residuals": fitted_model.resid.values
        }

    except Exception as e:
        logger.error(f"Error fitting ARIMA model: {str(e)}")
        return {
            "model": None,
            "predictions": np.array([]),
            "fitted_values": np.array([]),
            "diagnostics": {},
            "residuals": np.array([])
        }


def implement_exponential_smoothing(data: pd.Series,
                                  seasonal_periods: int = 4,
                                  trend: str = 'add',
                                  seasonal: str = 'add',
                                  forecast_steps: int = 4) -> Dict[str, Union[object, np.ndarray]]:
    """
    T093: Implement exponential smoothing model with seasonal components.

    Exponential smoothing (Holt-Winters) captures level, trend, and seasonality
    in time-series data. Optimized for quarterly employment data (seasonal_periods=4).

    Args:
        data: Time series data (employment values)
        seasonal_periods: Number of seasonal periods (4 for quarterly data)
        trend: Type of trend component ('add' or 'mul' or None)
        seasonal: Type of seasonal component ('add' or 'mul' or None)
        forecast_steps: Number of steps ahead to forecast

    Returns:
        Dictionary containing:
        - model: Fitted ExponentialSmoothing model object
        - predictions: Out-of-sample predictions
        - fitted_values: In-sample fitted values
        - smoothing_params: Smoothing parameters (alpha, beta, gamma)
    """
    logger.info(f"T093: Implementing exponential smoothing with {seasonal_periods} seasonal periods...")

    try:
        # Ensure sufficient data for seasonal model
        min_data_points = 2 * seasonal_periods
        if len(data) < min_data_points:
            logger.warning(f"Insufficient data points ({len(data)}) for seasonal model. Need at least {min_data_points}.")
            # Fall back to simple exponential smoothing
            trend = None
            seasonal = None

        # Fit Exponential Smoothing model
        model = ExponentialSmoothing(
            data,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods if seasonal else None
        )
        fitted_model = model.fit()

        # Generate forecasts
        forecast = fitted_model.forecast(steps=forecast_steps)

        # Extract smoothing parameters
        smoothing_params = {
            'alpha': fitted_model.params.get('smoothing_level', None),
            'beta': fitted_model.params.get('smoothing_trend', None),
            'gamma': fitted_model.params.get('smoothing_seasonal', None)
        }

        logger.info(f"[OK] Exponential smoothing fitted (alpha: {smoothing_params['alpha']:.4f})")

        return {
            "model": fitted_model,
            "predictions": forecast.values if hasattr(forecast, 'values') else forecast,
            "fitted_values": fitted_model.fittedvalues.values,
            "smoothing_params": smoothing_params
        }

    except Exception as e:
        logger.error(f"Error fitting exponential smoothing model: {str(e)}")
        return {
            "model": None,
            "predictions": np.array([]),
            "fitted_values": np.array([]),
            "smoothing_params": {}
        }


def compare_lstm_performance(lstm_predictions: np.ndarray,
                           baseline_predictions: Dict[str, np.ndarray],
                           actuals: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    T094: Compare LSTM performance against econometric employment prediction models.

    Calculates comprehensive performance metrics including:
    - RMSE (Root Mean Squared Error)
    - MAPE (Mean Absolute Percentage Error)
    - Directional Accuracy (percentage of correctly predicted directions)

    Args:
        lstm_predictions: LSTM model predictions
        baseline_predictions: Dictionary of baseline model predictions (e.g., {'arima': preds, 'exp_smoothing': preds})
        actuals: Actual employment values

    Returns:
        Dictionary with metrics for each model and relative performance ratios
    """
    logger.info("T094: Comparing LSTM performance against baselines...")

    try:
        results = {}

        # Calculate LSTM metrics
        lstm_rmse = np.sqrt(mean_squared_error(actuals, lstm_predictions))
        lstm_mape = mean_absolute_percentage_error(actuals, lstm_predictions) * 100  # Convert to percentage
        lstm_directional = _calculate_directional_accuracy(actuals, lstm_predictions)

        results['lstm'] = {
            'rmse': lstm_rmse,
            'mape': lstm_mape,
            'directional_accuracy': lstm_directional
        }

        # Calculate baseline metrics and relative performance
        for model_name, predictions in baseline_predictions.items():
            if len(predictions) == 0:
                logger.warning(f"Skipping {model_name}: no predictions available")
                continue

            # Ensure same length
            min_len = min(len(actuals), len(predictions))
            actuals_trimmed = actuals[:min_len]
            preds_trimmed = predictions[:min_len]

            rmse = np.sqrt(mean_squared_error(actuals_trimmed, preds_trimmed))
            mape = mean_absolute_percentage_error(actuals_trimmed, preds_trimmed) * 100
            directional = _calculate_directional_accuracy(actuals_trimmed, preds_trimmed)

            results[model_name] = {
                'rmse': rmse,
                'mape': mape,
                'directional_accuracy': directional
            }

            # Calculate relative performance (LSTM vs baseline)
            results[f'{model_name}_vs_lstm'] = {
                'rmse_improvement': ((rmse - lstm_rmse) / rmse) * 100,  # Positive = LSTM better
                'mape_improvement': ((mape - lstm_mape) / mape) * 100,
                'directional_improvement': lstm_directional - directional
            }

        logger.info(f"[OK] Performance comparison completed for {len(baseline_predictions)} baseline models")
        return results

    except Exception as e:
        logger.error(f"Error comparing performance: {str(e)}")
        return {}


def _calculate_directional_accuracy(actuals: np.ndarray, predictions: np.ndarray) -> float:
    """
    Calculate directional accuracy (percentage of correctly predicted directions).

    Args:
        actuals: Actual values
        predictions: Predicted values

    Returns:
        Directional accuracy as percentage (0-100)
    """
    if len(actuals) < 2:
        return 0.0

    # Calculate direction changes
    actual_directions = np.diff(actuals) > 0
    pred_directions = np.diff(predictions) > 0

    # Calculate accuracy
    correct = np.sum(actual_directions == pred_directions)
    total = len(actual_directions)

    return (correct / total) * 100


def create_ensemble_methods(lstm_model, baseline_models: Dict[str, object]) -> object:
    """
    Create ensemble methods combining LSTM with traditional employment forecasting.

    Args:
        lstm_model: Trained LSTM model
        baseline_models: Dictionary of baseline models

    Returns:
        Ensemble model object
    """
    # TODO: Implement ensemble methods
    logger.info("Creating ensemble methods...")
    return None


def benchmark_computational_efficiency(models: Dict[str, Dict],
                                      data: pd.Series,
                                      num_runs: int = 5,
                                      forecast_steps: int = 4) -> Dict[str, Dict[str, float]]:
    """
    T095: Benchmark computational efficiency for large-scale employment data processing.

    Measures training time, prediction time, and overall efficiency for each model.
    Performs multiple runs for robust timing measurements.

    Args:
        models: Dictionary of model configurations {'model_name': {'type': 'arima'|'exp_smoothing', 'params': {}}}
        data: Time series data for training/testing
        num_runs: Number of runs for averaging (default: 5)
        forecast_steps: Number of forecast steps

    Returns:
        Dictionary with timing metrics and efficiency analysis:
        - training_time: Average model fitting time (seconds)
        - prediction_time: Average forecasting time (seconds)
        - total_time: Total time per run (seconds)
        - throughput: Predictions per second
    """
    logger.info(f"T095: Benchmarking computational efficiency ({num_runs} runs)...")

    results = {}

    try:
        for model_name, config in models.items():
            logger.info(f"Benchmarking {model_name}...")

            training_times = []
            prediction_times = []

            for run in range(num_runs):
                # Training phase
                train_start = time.time()

                if config['type'] == 'arima':
                    order = config.get('params', {}).get('order', (1, 1, 1))
                    model = ARIMA(data, order=order)
                    fitted_model = model.fit()
                elif config['type'] == 'exp_smoothing':
                    seasonal_periods = config.get('params', {}).get('seasonal_periods', 4)
                    trend = config.get('params', {}).get('trend', 'add')
                    seasonal = config.get('params', {}).get('seasonal', 'add')
                    model = ExponentialSmoothing(
                        data,
                        trend=trend,
                        seasonal=seasonal,
                        seasonal_periods=seasonal_periods if seasonal else None
                    )
                    fitted_model = model.fit()
                else:
                    logger.warning(f"Unknown model type: {config['type']}")
                    continue

                train_end = time.time()
                training_times.append(train_end - train_start)

                # Prediction phase
                pred_start = time.time()
                _ = fitted_model.forecast(steps=forecast_steps)
                pred_end = time.time()
                prediction_times.append(pred_end - pred_start)

            # Calculate statistics
            avg_training = np.mean(training_times)
            avg_prediction = np.mean(prediction_times)
            std_training = np.std(training_times)
            std_prediction = np.std(prediction_times)
            total_time = avg_training + avg_prediction

            results[model_name] = {
                'training_time_mean': avg_training,
                'training_time_std': std_training,
                'prediction_time_mean': avg_prediction,
                'prediction_time_std': std_prediction,
                'total_time': total_time,
                'throughput': forecast_steps / total_time if total_time > 0 else 0,
                'num_runs': num_runs
            }

            logger.info(f"[OK] {model_name}: {total_time:.4f}s total ({avg_training:.4f}s train, {avg_prediction:.4f}s predict)")

        return results

    except Exception as e:
        logger.error(f"Error during benchmarking: {str(e)}")
        return {}


def validate_lstm_improvement(lstm_metrics: Dict[str, float],
                            baseline_metrics: Dict[str, Dict[str, float]],
                            improvement_threshold: float = 5.0) -> Dict[str, bool]:
    """
    Validate LSTM provides meaningful improvement over employment forecasting baselines.

    Checks if LSTM shows statistically significant improvement (threshold %)
    over baseline models in key metrics.

    Args:
        lstm_metrics: LSTM model performance metrics {'rmse': X, 'mape': Y, 'directional_accuracy': Z}
        baseline_metrics: Baseline model performance metrics (from compare_lstm_performance)
        improvement_threshold: Minimum improvement percentage to consider significant (default: 5%)

    Returns:
        Dictionary of validation results for each baseline model
    """
    logger.info("Validating LSTM improvement over baselines...")

    validation_results = {}

    try:
        lstm_rmse = lstm_metrics.get('rmse', float('inf'))
        lstm_mape = lstm_metrics.get('mape', float('inf'))
        lstm_directional = lstm_metrics.get('directional_accuracy', 0.0)

        for model_name, metrics in baseline_metrics.items():
            # Skip LSTM and comparison entries
            if model_name == 'lstm' or '_vs_lstm' in model_name:
                continue

            baseline_rmse = metrics.get('rmse', float('inf'))
            baseline_mape = metrics.get('mape', float('inf'))
            baseline_directional = metrics.get('directional_accuracy', 0.0)

            # Calculate improvements
            rmse_improvement = ((baseline_rmse - lstm_rmse) / baseline_rmse) * 100 if baseline_rmse > 0 else 0
            mape_improvement = ((baseline_mape - lstm_mape) / baseline_mape) * 100 if baseline_mape > 0 else 0
            directional_improvement = lstm_directional - baseline_directional

            # Validate improvements
            validation_results[model_name] = {
                'rmse_improved': rmse_improvement >= improvement_threshold,
                'mape_improved': mape_improvement >= improvement_threshold,
                'directional_improved': directional_improvement >= improvement_threshold,
                'overall_improved': (
                    rmse_improvement >= improvement_threshold or
                    mape_improvement >= improvement_threshold
                ),
                'rmse_improvement_pct': rmse_improvement,
                'mape_improvement_pct': mape_improvement,
                'directional_improvement_pct': directional_improvement
            }

            logger.info(f"[OK] {model_name} validation: RMSE={rmse_improvement:.2f}%, MAPE={mape_improvement:.2f}%")

        return validation_results

    except Exception as e:
        logger.error(f"Error validating improvements: {str(e)}")
        return {}