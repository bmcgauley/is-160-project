# Hyperparameters Configuration

This directory contains centralized configuration for all model hyperparameters and training settings.

## Quick Start

**To modify hyperparameters**, simply edit the values in [hyperparameters.py](hyperparameters.py). No need to hunt through multiple files!

```python
# Example: Change number of training epochs from 100 to 50
class TrainingConfig:
    NUM_EPOCHS: int = 50  # Changed from 100
```

## Configuration Structure

All hyperparameters are organized into 4 main classes:

### 1. ModelConfig - Model Architecture
Controls LSTM model architecture:
- `HIDDEN_SIZE`: Hidden layer size (default: 64)
- `NUM_LAYERS`: Number of LSTM layers (default: 2)
- `DROPOUT`: Dropout probability (default: 0.2)
- `OUTPUT_SIZE`: Output dimension (default: 1)
- `SEQUENCE_LENGTH`: Time steps per sequence (default: 12)

### 2. TrainingConfig - Training Loop
Controls training behavior:
- `NUM_EPOCHS`: Maximum training epochs (default: 100)
- `PATIENCE`: Early stopping patience (default: 10)
- `LEARNING_RATE`: Initial learning rate (default: 0.001)
- `WEIGHT_DECAY`: L2 regularization (default: 0.0)
- `GRADIENT_CLIP_NORM`: Max gradient norm (default: 1.0)
- `LR_SCHEDULER_TYPE`: Learning rate scheduler type (default: 'reduce_on_plateau')
- `LR_FACTOR`: LR reduction factor (default: 0.5)
- `LR_PATIENCE`: Epochs before reducing LR (default: 5)

### 3. DataConfig - Data Loading
Controls data splits and loading:
- `VAL_SIZE`: Validation set ratio (default: 0.2)
- `TEST_SIZE`: Test set ratio (default: 0.1)
- `BATCH_SIZE`: Batch size for training (default: 32)
- `SHUFFLE_TRAIN`: Shuffle training data (default: True)
- `SHUFFLE_VAL`: Shuffle validation data (default: False)
- `SEQUENCE_LENGTH`: Time steps per sequence (default: 12)

### 4. ExperimentConfig - Reproducibility
Controls experiment settings:
- `RANDOM_SEED`: Random seed for reproducibility (default: 42)
- `DEVICE`: Compute device (default: 'auto')
- `SAVE_CHECKPOINTS`: Save model checkpoints (default: True)
- `CHECKPOINT_DIR`: Checkpoint directory (default: 'checkpoints')

## Usage in Code

The configuration is automatically imported and used by:
- [src/pipeline_orchestrator.py](../src/pipeline_orchestrator.py) - Uses ModelConfig, TrainingConfig, DataConfig
- [src/training.py](../src/training.py) - Uses TrainingConfig for gradient clipping and defaults

### Example Usage

```python
from config.hyperparameters import ModelConfig, TrainingConfig, DataConfig

# Use in model initialization
model = EmploymentLSTM(
    input_size=num_features,
    hidden_size=ModelConfig.HIDDEN_SIZE,
    num_layers=ModelConfig.NUM_LAYERS,
    dropout=ModelConfig.DROPOUT
)

# Use in training
history = trainer.train_model(
    num_epochs=TrainingConfig.NUM_EPOCHS,
    patience=TrainingConfig.PATIENCE
)

# Use in data loading
train_loader = build_data_loader(
    train_dataset,
    batch_size=DataConfig.BATCH_SIZE,
    shuffle=DataConfig.SHUFFLE_TRAIN
)
```

## Viewing Current Configuration

Run the configuration file directly to see all current settings:

```bash
python config/hyperparameters.py
```

This will print all hyperparameters in a readable format:

```
================================================================================
HYPERPARAMETERS CONFIGURATION
================================================================================

[MODEL ARCHITECTURE]
  Hidden Size:        64
  Num Layers:         2
  Dropout:            0.2
  Output Size:        1
  Sequence Length:    12

[TRAINING]
  Num Epochs:         100
  Patience:           10
  Learning Rate:      0.001
  ...
```

## Common Modifications

### Quick Testing (Fast Iterations)
For rapid testing with fewer epochs:
```python
class TrainingConfig:
    NUM_EPOCHS: int = 2         # Fast testing
    PATIENCE: int = 2           # Stop early
```

### Production Training (Full Run)
For full training runs:
```python
class TrainingConfig:
    NUM_EPOCHS: int = 100       # Full training
    PATIENCE: int = 15          # More patient
    LEARNING_RATE: float = 0.0005  # Lower LR for stability
```

### Memory Optimization
If running into memory issues:
```python
class DataConfig:
    BATCH_SIZE: int = 16        # Reduce batch size
```

### Larger Model
For more capacity:
```python
class ModelConfig:
    HIDDEN_SIZE: int = 128      # Double the size
    NUM_LAYERS: int = 3         # Add another layer
```

## Files in this Directory

- **hyperparameters.py** - Main configuration file with all hyperparameters
- **__init__.py** - Package initialization for easy imports
- **README.md** - This file

## Notes

- All hyperparameters are class variables for easy discovery and modification
- The configuration uses type hints for clarity
- Default values are carefully chosen based on experimentation
- Changes to this file affect the entire pipeline automatically
- No need to pass config files around - import and use directly
