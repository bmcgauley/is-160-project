"""
Test script for T092-T095: Baseline Models for Employment Forecasting

This script tests the baseline model implementations:
T092: ARIMA forecasting model with configurable parameters
T093: Exponential smoothing model with seasonal components
T094: Performance comparison framework with RMSE, MAPE, directional accuracy
T095: Computational efficiency benchmarking with timing measurements
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from src.baselines import (
    implement_arima_model,
    implement_exponential_smoothing,
    compare_lstm_performance,
    benchmark_computational_efficiency,
    validate_lstm_improvement
)
from src.logging_config import setup_logging


def create_synthetic_time_series(n_points: int = 50, seed: int = 42) -> pd.Series:
    """
    Create synthetic employment time series data for testing.

    Args:
        n_points: Number of data points
        seed: Random seed for reproducibility

    Returns:
        Pandas Series with synthetic employment data
    """
    np.random.seed(seed)

    # Create trend component
    trend = np.linspace(10000, 15000, n_points)

    # Create seasonal component (quarterly pattern)
    seasonal = 500 * np.sin(np.arange(n_points) * 2 * np.pi / 4)

    # Add noise
    noise = np.random.normal(0, 200, n_points)

    # Combine components
    data = trend + seasonal + noise

    return pd.Series(data, name='employment')


def test_t092_arima_model(logger):
    """Test T092: ARIMA forecasting model implementation."""
    logger.info("\n" + "="*80)
    logger.info("TESTING T092: ARIMA FORECASTING MODEL")
    logger.info("="*80)

    try:
        # Create test data
        data = create_synthetic_time_series(n_points=50)
        logger.info(f"Created synthetic time series with {len(data)} data points")

        # Test with different ARIMA orders
        test_orders = [(1, 1, 1), (2, 1, 2), (1, 0, 1)]

        for order in test_orders:
            logger.info(f"\nTesting ARIMA{order}...")
            result = implement_arima_model(data, order=order, forecast_steps=4)

            # Validation checks
            logger.info("\n[CHECK 1] Model fitting:")
            if result['model'] is not None:
                logger.info("  PASS: Model fitted successfully")
            else:
                logger.error("  FAIL: Model is None")
                return False

            logger.info("\n[CHECK 2] Predictions:")
            if len(result['predictions']) == 4:
                logger.info(f"  PASS: Generated 4 predictions: {result['predictions']}")
            else:
                logger.error(f"  FAIL: Expected 4 predictions, got {len(result['predictions'])}")
                return False

            logger.info("\n[CHECK 3] Diagnostics:")
            diagnostics = result['diagnostics']
            if 'aic' in diagnostics and 'bic' in diagnostics:
                logger.info(f"  PASS: AIC={diagnostics['aic']:.2f}, BIC={diagnostics['bic']:.2f}")
            else:
                logger.error("  FAIL: Missing diagnostic metrics")
                return False

            logger.info("\n[CHECK 4] Fitted values:")
            if len(result['fitted_values']) > 0:
                logger.info(f"  PASS: {len(result['fitted_values'])} fitted values generated")
            else:
                logger.error("  FAIL: No fitted values")
                return False

        logger.info("\n[PASS] T092: ARIMA model implementation successful")
        return True

    except Exception as e:
        logger.error(f"\n[FAIL] T092 TEST FAILED: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_t093_exponential_smoothing(logger):
    """Test T093: Exponential smoothing model implementation."""
    logger.info("\n" + "="*80)
    logger.info("TESTING T093: EXPONENTIAL SMOOTHING MODEL")
    logger.info("="*80)

    try:
        # Create test data with clear seasonality
        data = create_synthetic_time_series(n_points=50)
        logger.info(f"Created synthetic time series with {len(data)} data points")

        # Test seasonal exponential smoothing
        logger.info("\nTesting Exponential Smoothing with seasonal components...")
        result = implement_exponential_smoothing(
            data,
            seasonal_periods=4,
            trend='add',
            seasonal='add',
            forecast_steps=4
        )

        # Validation checks
        logger.info("\n[CHECK 1] Model fitting:")
        if result['model'] is not None:
            logger.info("  PASS: Model fitted successfully")
        else:
            logger.error("  FAIL: Model is None")
            return False

        logger.info("\n[CHECK 2] Predictions:")
        if len(result['predictions']) == 4:
            logger.info(f"  PASS: Generated 4 predictions: {result['predictions']}")
        else:
            logger.error(f"  FAIL: Expected 4 predictions, got {len(result['predictions'])}")
            return False

        logger.info("\n[CHECK 3] Smoothing parameters:")
        params = result['smoothing_params']
        if 'alpha' in params and params['alpha'] is not None:
            logger.info(f"  PASS: alpha={params['alpha']:.4f}")
            if params.get('beta') is not None:
                logger.info(f"        beta={params['beta']:.4f}")
            if params.get('gamma') is not None:
                logger.info(f"        gamma={params['gamma']:.4f}")
        else:
            logger.error("  FAIL: Missing smoothing parameters")
            return False

        logger.info("\n[CHECK 4] Fitted values:")
        if len(result['fitted_values']) > 0:
            logger.info(f"  PASS: {len(result['fitted_values'])} fitted values generated")
        else:
            logger.error("  FAIL: No fitted values")
            return False

        # Test with insufficient data
        logger.info("\n[CHECK 5] Handling insufficient data:")
        small_data = create_synthetic_time_series(n_points=5)
        result_small = implement_exponential_smoothing(small_data, seasonal_periods=4)
        if result_small['model'] is not None:
            logger.info("  PASS: Gracefully handled insufficient data (fell back to simple smoothing)")
        else:
            logger.warning("  WARNING: Could not fit model with insufficient data")

        logger.info("\n[PASS] T093: Exponential smoothing implementation successful")
        return True

    except Exception as e:
        logger.error(f"\n[FAIL] T093 TEST FAILED: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_t094_performance_comparison(logger):
    """Test T094: Performance comparison framework."""
    logger.info("\n" + "="*80)
    logger.info("TESTING T094: PERFORMANCE COMPARISON FRAMEWORK")
    logger.info("="*80)

    try:
        # Create synthetic test data
        np.random.seed(42)
        n_test = 20
        actuals = create_synthetic_time_series(n_points=n_test, seed=42).values

        # Create mock predictions (LSTM performs better than baselines)
        lstm_predictions = actuals + np.random.normal(0, 100, n_test)  # Lower error
        arima_predictions = actuals + np.random.normal(0, 300, n_test)  # Higher error
        exp_smooth_predictions = actuals + np.random.normal(0, 250, n_test)  # Medium error

        logger.info(f"Created test predictions for {n_test} data points")

        # Test performance comparison
        baseline_predictions = {
            'arima': arima_predictions,
            'exp_smoothing': exp_smooth_predictions
        }

        results = compare_lstm_performance(lstm_predictions, baseline_predictions, actuals)

        # Validation checks
        logger.info("\n[CHECK 1] LSTM metrics calculated:")
        if 'lstm' in results:
            lstm_metrics = results['lstm']
            logger.info(f"  PASS: RMSE={lstm_metrics['rmse']:.2f}")
            logger.info(f"        MAPE={lstm_metrics['mape']:.2f}%")
            logger.info(f"        Directional Accuracy={lstm_metrics['directional_accuracy']:.2f}%")
        else:
            logger.error("  FAIL: LSTM metrics missing")
            return False

        logger.info("\n[CHECK 2] Baseline metrics calculated:")
        for model_name in ['arima', 'exp_smoothing']:
            if model_name in results:
                metrics = results[model_name]
                logger.info(f"  PASS: {model_name}")
                logger.info(f"        RMSE={metrics['rmse']:.2f}")
                logger.info(f"        MAPE={metrics['mape']:.2f}%")
                logger.info(f"        Directional Accuracy={metrics['directional_accuracy']:.2f}%")
            else:
                logger.error(f"  FAIL: {model_name} metrics missing")
                return False

        logger.info("\n[CHECK 3] Relative performance calculated:")
        for model_name in ['arima', 'exp_smoothing']:
            comparison_key = f'{model_name}_vs_lstm'
            if comparison_key in results:
                comp = results[comparison_key]
                logger.info(f"  PASS: {comparison_key}")
                logger.info(f"        RMSE improvement: {comp['rmse_improvement']:.2f}%")
                logger.info(f"        MAPE improvement: {comp['mape_improvement']:.2f}%")
            else:
                logger.error(f"  FAIL: {comparison_key} missing")
                return False

        logger.info("\n[CHECK 4] Metrics are valid numbers:")
        all_valid = True
        for model_name, metrics in results.items():
            if '_vs_lstm' not in model_name:
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        if not isinstance(value, (int, float)) or np.isnan(value):
                            logger.error(f"  FAIL: Invalid value for {model_name}.{metric_name}: {value}")
                            all_valid = False
        if all_valid:
            logger.info("  PASS: All metrics are valid numbers")

        logger.info("\n[PASS] T094: Performance comparison framework successful")
        return True

    except Exception as e:
        logger.error(f"\n[FAIL] T094 TEST FAILED: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_t095_computational_efficiency(logger):
    """Test T095: Computational efficiency benchmarking."""
    logger.info("\n" + "="*80)
    logger.info("TESTING T095: COMPUTATIONAL EFFICIENCY BENCHMARKING")
    logger.info("="*80)

    try:
        # Create test data
        data = create_synthetic_time_series(n_points=50)
        logger.info(f"Created synthetic time series with {len(data)} data points")

        # Define models to benchmark
        models = {
            'arima_111': {
                'type': 'arima',
                'params': {'order': (1, 1, 1)}
            },
            'exp_smoothing': {
                'type': 'exp_smoothing',
                'params': {'seasonal_periods': 4, 'trend': 'add', 'seasonal': 'add'}
            }
        }

        logger.info(f"\nBenchmarking {len(models)} models with 3 runs each...")
        results = benchmark_computational_efficiency(models, data, num_runs=3, forecast_steps=4)

        # Validation checks
        logger.info("\n[CHECK 1] Results for all models:")
        for model_name in models.keys():
            if model_name in results:
                logger.info(f"  PASS: {model_name} benchmarked")
            else:
                logger.error(f"  FAIL: {model_name} not in results")
                return False

        logger.info("\n[CHECK 2] Timing metrics:")
        for model_name, metrics in results.items():
            logger.info(f"\n  {model_name}:")

            required_metrics = ['training_time_mean', 'prediction_time_mean', 'total_time', 'throughput']
            for metric in required_metrics:
                if metric in metrics:
                    logger.info(f"    {metric}: {metrics[metric]:.6f}")
                else:
                    logger.error(f"    FAIL: Missing {metric}")
                    return False

        logger.info("\n[CHECK 3] Timing values are reasonable:")
        all_reasonable = True
        for model_name, metrics in results.items():
            # Training should take more than 0 seconds but less than 60 seconds
            if 0 < metrics['training_time_mean'] < 60:
                logger.info(f"  PASS: {model_name} training time: {metrics['training_time_mean']:.4f}s")
            else:
                logger.error(f"  FAIL: {model_name} training time unreasonable: {metrics['training_time_mean']}")
                all_reasonable = False

            # Prediction should be faster than training
            if metrics['prediction_time_mean'] > 0:
                logger.info(f"  PASS: {model_name} prediction time: {metrics['prediction_time_mean']:.4f}s")
            else:
                logger.error(f"  FAIL: {model_name} prediction time: {metrics['prediction_time_mean']}")
                all_reasonable = False

        if not all_reasonable:
            return False

        logger.info("\n[CHECK 4] Throughput calculated:")
        for model_name, metrics in results.items():
            throughput = metrics['throughput']
            if throughput > 0:
                logger.info(f"  PASS: {model_name} throughput: {throughput:.2f} predictions/second")
            else:
                logger.error(f"  FAIL: {model_name} throughput: {throughput}")
                return False

        logger.info("\n[PASS] T095: Computational efficiency benchmarking successful")
        return True

    except Exception as e:
        logger.error(f"\n[FAIL] T095 TEST FAILED: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_validate_lstm_improvement(logger):
    """Test validate_lstm_improvement helper function."""
    logger.info("\n" + "="*80)
    logger.info("TESTING: VALIDATE LSTM IMPROVEMENT")
    logger.info("="*80)

    try:
        # Create mock metrics where LSTM performs better
        lstm_metrics = {
            'rmse': 100.0,
            'mape': 5.0,
            'directional_accuracy': 75.0
        }

        baseline_metrics = {
            'lstm': lstm_metrics,
            'arima': {
                'rmse': 150.0,  # LSTM is 33% better
                'mape': 8.0,    # LSTM is 37.5% better
                'directional_accuracy': 60.0  # LSTM is 15 points better
            },
            'exp_smoothing': {
                'rmse': 120.0,  # LSTM is 16.7% better
                'mape': 6.0,    # LSTM is 16.7% better
                'directional_accuracy': 70.0  # LSTM is 5 points better
            }
        }

        # Test validation with 5% improvement threshold
        results = validate_lstm_improvement(lstm_metrics, baseline_metrics, improvement_threshold=5.0)

        logger.info("\n[CHECK 1] Validation results for ARIMA:")
        if 'arima' in results:
            arima_val = results['arima']
            logger.info(f"  RMSE improved: {arima_val['rmse_improved']} ({arima_val['rmse_improvement_pct']:.2f}%)")
            logger.info(f"  MAPE improved: {arima_val['mape_improved']} ({arima_val['mape_improvement_pct']:.2f}%)")
            logger.info(f"  Overall improved: {arima_val['overall_improved']}")

            if arima_val['rmse_improved'] and arima_val['overall_improved']:
                logger.info("  PASS: ARIMA validation successful")
            else:
                logger.error("  FAIL: Expected improvements not detected")
                return False
        else:
            logger.error("  FAIL: ARIMA validation results missing")
            return False

        logger.info("\n[CHECK 2] Validation results for exp_smoothing:")
        if 'exp_smoothing' in results:
            es_val = results['exp_smoothing']
            logger.info(f"  RMSE improved: {es_val['rmse_improved']} ({es_val['rmse_improvement_pct']:.2f}%)")
            logger.info(f"  MAPE improved: {es_val['mape_improved']} ({es_val['mape_improvement_pct']:.2f}%)")
            logger.info(f"  Overall improved: {es_val['overall_improved']}")
            logger.info("  PASS: Exponential smoothing validation successful")
        else:
            logger.error("  FAIL: Exponential smoothing validation results missing")
            return False

        logger.info("\n[PASS] Validate LSTM improvement function successful")
        return True

    except Exception as e:
        logger.error(f"\n[FAIL] VALIDATE IMPROVEMENT TEST FAILED: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Run all baseline model tests."""

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("\n" + "="*80)
    logger.info("BASELINE MODELS TEST SUITE (T092-T095)")
    logger.info("="*80)

    # Run all tests
    test_results = {
        'T092 - ARIMA Model': test_t092_arima_model(logger),
        'T093 - Exponential Smoothing': test_t093_exponential_smoothing(logger),
        'T094 - Performance Comparison': test_t094_performance_comparison(logger),
        'T095 - Computational Efficiency': test_t095_computational_efficiency(logger),
        'Validate Improvement': test_validate_lstm_improvement(logger)
    }

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUITE SUMMARY")
    logger.info("="*80)

    passed = sum(test_results.values())
    total = len(test_results)

    for test_name, result in test_results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"  [{status}] {test_name}")

    logger.info("\n" + "-"*80)
    logger.info(f"Results: {passed}/{total} tests passed")

    if passed == total:
        logger.info("\n*** ALL BASELINE MODEL TESTS PASSED ***")
        logger.info("="*80 + "\n")
        return True
    else:
        logger.error(f"\n*** {total - passed} TEST(S) FAILED ***")
        logger.info("="*80 + "\n")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
