# Contract: Employment Prediction API

**Date**: 2025-09-24
**Purpose**: Define interface for employment trend prediction service

## Overview
The employment prediction service provides CNN-based forecasting of quarterly employment changes using QCEW data.

## Endpoint: /predict

### Request
```json
{
  "quarter": "2025-Q1",
  "area_code": "06001",
  "horizon": 1,
  "include_uncertainty": true
}
```

### Response
```json
{
  "prediction": {
    "quarter": "2025-Q1",
    "area_code": "06001",
    "employment_change_pct": 2.3,
    "confidence_interval": {
      "lower": 1.8,
      "upper": 2.8
    }
  },
  "metadata": {
    "model_version": "v1.0.0",
    "prediction_date": "2025-09-24",
    "baseline_comparison": {
      "arima": 1.9,
      "linear_regression": 2.1
    }
  }
}
```

### Error Responses
- `400 Bad Request`: Invalid quarter/area code format
- `404 Not Found`: Area code not in training data
- `500 Internal Server Error`: Model loading or prediction failure

## Data Validation Rules

### Input Validation
- Quarter format: YYYY-Q[1-4]
- Area code: Valid California FIPS code
- Horizon: 1-4 quarters
- All fields required

### Output Validation
- Employment change: -50% to +50% range
- Confidence interval: Symmetric around prediction
- Metadata: All fields present

## Performance Requirements
- Response time: <500ms median
- Availability: 99.9% uptime
- Accuracy: MAPE < 15% on validation set