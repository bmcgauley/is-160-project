# Quick Start: Testing with 2 Epochs

This guide shows you how to quickly modify hyperparameters for testing.

## Problem Solved

Previously, when you changed `num_epochs=2` in the function call, it was being ignored because [pipeline_orchestrator.py:266](../src/pipeline_orchestrator.py#L266) was hardcoded to use 100 epochs. Now all hyperparameters are centralized!

## For Quick Testing

Open [config/hyperparameters.py](hyperparameters.py) and change:

```python
class TrainingConfig:
    """Training loop hyperparameters"""

    # Training iterations
    NUM_EPOCHS: int = 2             # Changed from 100 to 2 for testing
    PATIENCE: int = 2               # Changed from 10 to 2 for testing
```

That's it! Now when you run:

```bash
python main.py --stage train
```

It will use 2 epochs instead of 100.

## All Hyperparameters in One Place

Everything is now in [config/hyperparameters.py](hyperparameters.py):

- **Model Architecture** (ModelConfig): hidden_size, num_layers, dropout
- **Training** (TrainingConfig): num_epochs, patience, learning_rate
- **Data Loading** (DataConfig): batch_size, val_size, test_size
- **Experiments** (ExperimentConfig): random_seed, device, checkpointing

## View Current Settings

```bash
python config/hyperparameters.py
```

## Example: Common Testing Configurations

### Quick Test (2 epochs)
```python
class TrainingConfig:
    NUM_EPOCHS: int = 2
    PATIENCE: int = 2
```

### Medium Test (10 epochs)
```python
class TrainingConfig:
    NUM_EPOCHS: int = 10
    PATIENCE: int = 3
```

### Full Training (100 epochs)
```python
class TrainingConfig:
    NUM_EPOCHS: int = 100
    PATIENCE: int = 10
```

## Files Modified

The following files now use the centralized config:

1. [src/pipeline_orchestrator.py](../src/pipeline_orchestrator.py#L20) - Imports ModelConfig, TrainingConfig, DataConfig
2. [src/training.py](../src/training.py#L20) - Imports TrainingConfig for gradient clipping

All hardcoded values have been replaced with config references.
