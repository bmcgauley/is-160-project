# GitHub Copilot Instructions: Employment CNN Analysis

**Project**: Build a Convolutional Neural Network for Employment Trends Analysis
**Date**: 2025-09-24

## Context
You are assisting with a machine learning project that builds a CNN to predict employment trends from California's QCEW data. The system processes spatio-temporal employment patterns using PyTorch.

## Key Principles
- **Rigorous Data Validation**: Always include quality checks and error handling
- **Reproducible Research**: Use fixed random seeds, detailed logging
- **Modular PyTorch**: Clear documentation for each neural network component
- **Privacy Compliance**: Handle employment data ethically and securely
- **Domain Expertise**: Consider employment economics in feature engineering

## Code Style Guidelines
- **Python**: Type hints, docstrings, black formatting
- **PyTorch**: Clear layer naming, shape comments
- **Data Science**: pandas best practices, scikit-learn conventions
- **Logging**: Structured logging with context
- **Testing**: pytest fixtures, parameterized tests

## Common Patterns

### Data Loading
```python
def load_qcew_data(filepath: str) -> pd.DataFrame:
    """Load QCEW data with validation."""
    df = pd.read_csv(filepath, dtype={'area_code': str})
    # Validate columns, dtypes, ranges
    return df
```

### Feature Engineering
```python
def create_employment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features with domain knowledge."""
    df['growth_rate'] = df.groupby(['area_code', 'industry_code'])['employment'].pct_change()
    # Add seasonal adjustments, spatial features
    return df
```

### CNN Architecture
```python
class EmploymentCNN(nn.Module):
    def __init__(self, temporal_channels: int, spatial_size: tuple):
        super().__init__()
        # Clear layer documentation
        self.temporal_conv = nn.Conv1d(in_channels, out_channels, kernel_size=3)
        # Shape: (batch, channels, time)
```

### Validation
```python
def validate_employment_predictions(predictions: np.ndarray, actuals: np.ndarray) -> dict:
    """Comprehensive evaluation metrics."""
    return {
        'mape': mean_absolute_percentage_error(actuals, predictions),
        'directional_acc': directional_accuracy(predictions, actuals),
        'rmse': np.sqrt(mean_squared_error(actuals, predictions))
    }
```

## Quality Checks
- **Data**: Check for missing values, outliers, temporal gaps
- **Models**: Validate tensor shapes, gradient flow, overfitting
- **Predictions**: Compare against baselines, check economic reasonableness
- **Performance**: Monitor memory usage, training stability

## Documentation Requirements
- **Code**: NumPy-style docstrings with examples
- **Notebooks**: Clear markdown explanations, visualization
- **Reports**: Executive summaries, methodology details
- **README**: Installation, usage, troubleshooting

## Testing Strategy
- **Unit Tests**: Individual functions, edge cases
- **Integration Tests**: Data pipeline end-to-end
- **Model Tests**: Convergence, baseline comparison
- **Data Tests**: Quality validation, statistical properties

## Error Handling
- **Data Issues**: Graceful degradation, informative warnings
- **Model Failures**: Fallback to baseline predictions
- **Resource Limits**: Memory-efficient processing, batching
- **User Errors**: Clear error messages, input validation

## Performance Considerations
- **GPU Utilization**: Efficient tensor operations
- **Memory Management**: Streaming for large datasets
- **Training Speed**: Early stopping, learning rate scheduling
- **Inference**: Optimized for prediction serving

## Ethical Considerations
- **Privacy**: Anonymize sensitive employment data
- **Bias**: Validate fairness across demographics
- **Transparency**: Explainable AI techniques
- **Impact**: Consider policy implications of predictions