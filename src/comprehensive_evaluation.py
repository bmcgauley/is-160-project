"""
Comprehensive Model Evaluation Module

This module provides advanced evaluation capabilities including:
- Baseline model comparisons
- Statistical analysis
- Per-segment performance analysis
- Automated evaluation reporting

Stage 8 of the pipeline.
"""

import numpy as np
import pandas as pd

# Configure matplotlib to use non-interactive backend (must be before importing pyplot)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


def evaluate_against_baselines(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               baseline_predictions: Dict[str, np.ndarray] = None) -> Dict:
    """
    Compare LSTM model performance against baseline models.

    Args:
        y_true: True values
        y_pred: LSTM predictions
        baseline_predictions: Dict of baseline model predictions

    Returns:
        Dictionary with comparative metrics
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from loss_metrics import mean_absolute_percentage_error, directional_accuracy

    logger.info("="*80)
    logger.info("STAGE 8: COMPREHENSIVE MODEL EVALUATION")
    logger.info("="*80)

    results = {}

    # LSTM metrics
    lstm_metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'directional_accuracy': directional_accuracy(y_true, y_pred)
    }
    results['lstm'] = lstm_metrics

    logger.info("\nLSTM Model Performance:")
    for metric, value in lstm_metrics.items():
        logger.info(f"  {metric.upper()}: {value:.4f}")

    # Baseline comparisons (if provided)
    if baseline_predictions:
        logger.info("\nBaseline Model Comparisons:")
        for model_name, preds in baseline_predictions.items():
            baseline_metrics = {
                'rmse': np.sqrt(mean_squared_error(y_true, preds)),
                'mae': mean_absolute_error(y_true, preds),
                'mape': mean_absolute_percentage_error(y_true, preds),
                'r2': r2_score(y_true, preds),
                'directional_accuracy': directional_accuracy(y_true, preds)
            }
            results[model_name] = baseline_metrics

            logger.info(f"\n{model_name.upper()} Model:")
            for metric, value in baseline_metrics.items():
                lstm_val = lstm_metrics[metric]
                improvement = ((baseline_metrics[metric] - lstm_val) / baseline_metrics[metric] * 100
                              if baseline_metrics[metric] != 0 else 0)
                symbol = "[+]" if improvement > 0 else "[-]"
                logger.info(f"  {metric.upper()}: {value:.4f} ({symbol} {improvement:+.2f}% vs LSTM)")

    return results


def generate_naivebaselines(y_true: np.ndarray,
                            historical_data: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """
    Generate naive baseline predictions for comparison.

    Baselines:
    - Last Value: Predict last known value
    - Mean: Predict historical mean
    - Median: Predict historical median
    - Seasonal Naive: Predict same quarter from last year (if available)

    Args:
        y_true: True values (for shape)
        historical_data: Historical data for computing baselines

    Returns:
        Dictionary of baseline predictions
    """
    logger.info("\nGenerating naive baseline predictions...")

    baselines = {}

    if historical_data is not None and len(historical_data) > 0:
        # Last value baseline
        last_value = historical_data[-1]
        baselines['last_value'] = np.full_like(y_true, last_value)

        # Mean baseline
        mean_value = np.mean(historical_data)
        baselines['mean'] = np.full_like(y_true, mean_value)

        # Median baseline
        median_value = np.median(historical_data)
        baselines['median'] = np.full_like(y_true, median_value)

        logger.info(f"  Last Value Baseline: {last_value:.2f}")
        logger.info(f"  Mean Baseline: {mean_value:.2f}")
        logger.info(f"  Median Baseline: {median_value:.2f}")
    else:
        # Fallback: use mean of true values (not ideal but functional)
        mean_value = np.mean(y_true)
        baselines['mean'] = np.full_like(y_true, mean_value)
        logger.warning("  No historical data provided; using test set mean as baseline")

    return baselines


def analyze_error_segments(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           segment_labels: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Analyze prediction errors across different segments (e.g., by county, industry).

    Args:
        y_true: True values
        y_pred: Predictions
        segment_labels: Optional segment identifiers

    Returns:
        DataFrame with per-segment metrics
    """
    logger.info("\nAnalyzing error distribution across segments...")

    errors = y_true - y_pred
    abs_errors = np.abs(errors)

    if segment_labels is not None and len(segment_labels) == len(y_true):
        df = pd.DataFrame({
            'segment': segment_labels,
            'true': y_true,
            'pred': y_pred,
            'error': errors,
            'abs_error': abs_errors
        })

        segment_stats = df.groupby('segment').agg({
            'error': ['mean', 'std'],
            'abs_error': ['mean', 'median', 'max'],
            'true': 'count'
        }).round(2)

        logger.info(f"\nPer-Segment Performance (top 10 by count):")
        top_segments = segment_stats.nlargest(10, ('true', 'count'))
        logger.info(f"\n{top_segments}")

        return segment_stats
    else:
        logger.info("  No segment labels provided; skipping segment analysis")
        return pd.DataFrame()


def generate_evaluation_report(results: Dict,
                               save_path: Path,
                               model_info: Dict = None) -> None:
    """
    Generate comprehensive evaluation report.

    Args:
        results: Evaluation results dictionary
        save_path: Path to save report
        model_info: Additional model information
    """
    logger.info(f"\nGenerating comprehensive evaluation report...")

    report_lines = []
    report_lines.append("="*80)
    report_lines.append("COMPREHENSIVE MODEL EVALUATION REPORT")
    report_lines.append("="*80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    if model_info:
        report_lines.append("Model Information:")
        for key, value in model_info.items():
            report_lines.append(f"  {key}: {value}")
        report_lines.append("")

    report_lines.append("="*80)
    report_lines.append("MODEL PERFORMANCE METRICS")
    report_lines.append("="*80)

    # LSTM results
    if 'lstm' in results:
        report_lines.append("\nLSTM Model:")
        report_lines.append("-" * 40)
        for metric, value in results['lstm'].items():
            report_lines.append(f"  {metric.upper():25s}: {value:10.4f}")

    # Baseline comparisons
    baseline_models = [k for k in results.keys() if k != 'lstm']
    if baseline_models:
        report_lines.append("\n" + "="*80)
        report_lines.append("BASELINE MODEL COMPARISONS")
        report_lines.append("="*80)

        for model_name in baseline_models:
            report_lines.append(f"\n{model_name.upper()} Baseline:")
            report_lines.append("-" * 40)
            for metric, value in results[model_name].items():
                lstm_val = results['lstm'][metric]
                improvement = ((value - lstm_val) / value * 100 if value != 0 else 0)
                symbol = "[+]" if improvement > 0 else "[-]"
                report_lines.append(f"  {metric.upper():25s}: {value:10.4f} ({symbol} {improvement:+6.2f}% vs LSTM)")

    # Performance Summary
    report_lines.append("\n" + "="*80)
    report_lines.append("PERFORMANCE SUMMARY")
    report_lines.append("="*80)

    lstm_metrics = results.get('lstm', {})
    mape = lstm_metrics.get('mape', 0)
    dir_acc = lstm_metrics.get('directional_accuracy', 0)
    r2 = lstm_metrics.get('r2', 0)

    if mape < 10 and dir_acc > 75 and r2 > 0.7:
        quality = "EXCELLENT"
        emoji = "🌟"
    elif mape < 20 and dir_acc > 60 and r2 > 0.5:
        quality = "GOOD"
        emoji = "✅"
    elif mape < 30 and dir_acc > 50 and r2 > 0.3:
        quality = "ACCEPTABLE"
        emoji = "⚠️"
    else:
        quality = "NEEDS IMPROVEMENT"
        emoji = "❌"

    report_lines.append(f"\nOverall Model Quality: {emoji} {quality}")
    report_lines.append(f"  MAPE: {mape:.2f}% (target: <20%)")
    report_lines.append(f"  Directional Accuracy: {dir_acc:.2f}% (target: >60%)")
    report_lines.append(f"  R² Score: {r2:.4f} (target: >0.5)")

    # Recommendations
    report_lines.append("\n" + "="*80)
    report_lines.append("RECOMMENDATIONS")
    report_lines.append("="*80)

    if mape > 20:
        report_lines.append("  • High MAPE indicates large prediction errors")
        report_lines.append("    - Consider: more training data, feature engineering, hyperparameter tuning")

    if dir_acc < 60:
        report_lines.append("  • Low directional accuracy indicates trend prediction issues")
        report_lines.append("    - Consider: increasing sequence length, adding trend features")

    if r2 < 0.5:
        report_lines.append("  • Low R² indicates poor overall fit")
        report_lines.append("    - Consider: more complex model, better feature selection")

    if quality in ["EXCELLENT", "GOOD"]:
        report_lines.append("  • Model performs well - ready for deployment consideration")
        report_lines.append("  • Monitor performance on new data and retrain periodically")

    report_lines.append("\n" + "="*80)
    report_lines.append("END OF REPORT")
    report_lines.append("="*80)

    # Save report
    report_text = "\n".join(report_lines)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    logger.info(f"[OK] Evaluation report saved: {save_path}")

    # Also print to console
    print("\n" + report_text)


def plot_model_comparison(results: Dict, save_path: Path) -> None:
    """
    Create visual comparison of model performance.

    Args:
        results: Evaluation results
        save_path: Path to save plot
    """
    logger.info("\nCreating model comparison visualizations...")

    metrics = ['rmse', 'mae', 'mape', 'r2', 'directional_accuracy']
    model_names = list(results.keys())

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = [results[model].get(metric, 0) for model in model_names]
        colors = ['#2ecc71' if model == 'lstm' else '#95a5a6' for model in model_names]

        bars = ax.bar(model_names, values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_title(f'{metric.upper()}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45)

        # Annotate bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.2f}',
                   ha='center', va='bottom', fontsize=10)

    # Remove extra subplot
    fig.delaxes(axes[5])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"[OK] Model comparison plot saved: {save_path}")


if __name__ == '__main__':
    # Test the evaluation module
    print("Testing comprehensive evaluation module...")

    # Generate dummy data
    np.random.seed(42)
    y_true = np.random.randn(1000) * 50 + 100
    y_pred_lstm = y_true + np.random.randn(1000) * 10
    y_pred_baseline = np.full_like(y_true, np.mean(y_true))

    # Test evaluation
    baselines = {'mean_baseline': y_pred_baseline}
    results = evaluate_against_baselines(y_true, y_pred_lstm, baselines)

    # Test report generation
    generate_evaluation_report(results, Path("test_evaluation_report.txt"))

    # Test plot
    plot_model_comparison(results, Path("test_model_comparison.png"))

    print("\n[OK] All tests passed!")
