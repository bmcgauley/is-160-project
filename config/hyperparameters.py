"""
Centralized Hyperparameters Configuration for QCEW Employment Forecasting

This module contains all model hyperparameters and training configuration
in one place for easy experimentation and tuning.

To modify hyperparameters, simply edit the values in this file - no need
to hunt through multiple files!
"""

from typing import Dict, Any


class ModelConfig:
    """Model architecture hyperparameters"""

    # LSTM Architecture
    HIDDEN_SIZE: int = 64           # Hidden layer size
    NUM_LAYERS: int = 2             # Number of LSTM layers
    DROPOUT: float = 0.2            # Dropout probability
    OUTPUT_SIZE: int = 1            # Output dimension (1 for regression)

    # Input dimensions (set dynamically from data)
    INPUT_SIZE: int = None          # Will be set based on preprocessed features
    SEQUENCE_LENGTH: int = 12       # Time steps per sequence (3 years quarterly)


class TrainingConfig:
    """Training loop hyperparameters"""

    # Training iterations
    NUM_EPOCHS: int = 10           # Maximum number of training epochs
    PATIENCE: int = 1              # Early stopping patience (epochs)

    # Optimizer settings
    LEARNING_RATE: float = 0.001    # Initial learning rate
    WEIGHT_DECAY: float = 0.0       # L2 regularization

    # Gradient settings
    GRADIENT_CLIP_NORM: float = 1.0 # Max gradient norm for clipping

    # Learning rate scheduler
    LR_SCHEDULER_TYPE: str = 'reduce_on_plateau'  # 'reduce_on_plateau', 'step', 'cosine', 'exponential'
    LR_FACTOR: float = 0.5          # LR reduction factor
    LR_PATIENCE: int = 5            # Epochs to wait before reducing LR


class DataConfig:
    """Data loading and preprocessing hyperparameters"""

    # Data splits
    VAL_SIZE: float = 0.2           # Validation set ratio
    TEST_SIZE: float = 0.1          # Test set ratio
    TRAIN_SIZE: float = 0.7         # Training set ratio (derived: 1 - val - test)

    # DataLoader settings
    BATCH_SIZE: int = 32            # Batch size for training
    SHUFFLE_TRAIN: bool = True      # Shuffle training data
    SHUFFLE_VAL: bool = False       # Don't shuffle validation/test
    NUM_WORKERS: int = 0            # DataLoader workers (0 = main process)

    # Sequence creation
    SEQUENCE_LENGTH: int = 12       # Time steps per sequence
    TARGET_OFFSET: int = 1          # Predict N steps ahead


class ExperimentConfig:
    """Experiment tracking and reproducibility"""

    # Random seeds
    RANDOM_SEED: int = 42           # For reproducibility

    # Device
    DEVICE: str = 'auto'            # 'auto', 'cuda', 'cpu'

    # Checkpointing
    SAVE_CHECKPOINTS: bool = True   # Save model checkpoints
    CHECKPOINT_DIR: str = 'checkpoints'

    # Logging
    LOG_INTERVAL: int = 1000        # Log every N batches
    LOG_LEVEL: str = 'INFO'         # Logging level


def get_config_dict() -> Dict[str, Any]:
    """
    Get all configurations as a dictionary.

    Returns:
        Dictionary containing all hyperparameters
    """
    config = {}

    # Model config
    config['model'] = {
        'hidden_size': ModelConfig.HIDDEN_SIZE,
        'num_layers': ModelConfig.NUM_LAYERS,
        'dropout': ModelConfig.DROPOUT,
        'output_size': ModelConfig.OUTPUT_SIZE,
        'sequence_length': ModelConfig.SEQUENCE_LENGTH,
    }

    # Training config
    config['training'] = {
        'num_epochs': TrainingConfig.NUM_EPOCHS,
        'patience': TrainingConfig.PATIENCE,
        'learning_rate': TrainingConfig.LEARNING_RATE,
        'weight_decay': TrainingConfig.WEIGHT_DECAY,
        'gradient_clip_norm': TrainingConfig.GRADIENT_CLIP_NORM,
        'lr_scheduler_type': TrainingConfig.LR_SCHEDULER_TYPE,
        'lr_factor': TrainingConfig.LR_FACTOR,
        'lr_patience': TrainingConfig.LR_PATIENCE,
    }

    # Data config
    config['data'] = {
        'val_size': DataConfig.VAL_SIZE,
        'test_size': DataConfig.TEST_SIZE,
        'batch_size': DataConfig.BATCH_SIZE,
        'shuffle_train': DataConfig.SHUFFLE_TRAIN,
        'shuffle_val': DataConfig.SHUFFLE_VAL,
        'num_workers': DataConfig.NUM_WORKERS,
        'sequence_length': DataConfig.SEQUENCE_LENGTH,
        'target_offset': DataConfig.TARGET_OFFSET,
    }

    # Experiment config
    config['experiment'] = {
        'random_seed': ExperimentConfig.RANDOM_SEED,
        'device': ExperimentConfig.DEVICE,
        'save_checkpoints': ExperimentConfig.SAVE_CHECKPOINTS,
        'checkpoint_dir': ExperimentConfig.CHECKPOINT_DIR,
        'log_interval': ExperimentConfig.LOG_INTERVAL,
        'log_level': ExperimentConfig.LOG_LEVEL,
    }

    return config


def print_config():
    """Print all hyperparameters in a readable format."""
    print("="*80)
    print("HYPERPARAMETERS CONFIGURATION")
    print("="*80)

    print("\n[MODEL ARCHITECTURE]")
    print(f"  Hidden Size:        {ModelConfig.HIDDEN_SIZE}")
    print(f"  Num Layers:         {ModelConfig.NUM_LAYERS}")
    print(f"  Dropout:            {ModelConfig.DROPOUT}")
    print(f"  Output Size:        {ModelConfig.OUTPUT_SIZE}")
    print(f"  Sequence Length:    {ModelConfig.SEQUENCE_LENGTH}")

    print("\n[TRAINING]")
    print(f"  Num Epochs:         {TrainingConfig.NUM_EPOCHS}")
    print(f"  Patience:           {TrainingConfig.PATIENCE}")
    print(f"  Learning Rate:      {TrainingConfig.LEARNING_RATE}")
    print(f"  Weight Decay:       {TrainingConfig.WEIGHT_DECAY}")
    print(f"  Gradient Clip:      {TrainingConfig.GRADIENT_CLIP_NORM}")
    print(f"  LR Scheduler:       {TrainingConfig.LR_SCHEDULER_TYPE}")
    print(f"  LR Factor:          {TrainingConfig.LR_FACTOR}")
    print(f"  LR Patience:        {TrainingConfig.LR_PATIENCE}")

    print("\n[DATA]")
    print(f"  Val Size:           {DataConfig.VAL_SIZE:.1%}")
    print(f"  Test Size:          {DataConfig.TEST_SIZE:.1%}")
    print(f"  Batch Size:         {DataConfig.BATCH_SIZE}")
    print(f"  Shuffle Train:      {DataConfig.SHUFFLE_TRAIN}")
    print(f"  Sequence Length:    {DataConfig.SEQUENCE_LENGTH}")

    print("\n[EXPERIMENT]")
    print(f"  Random Seed:        {ExperimentConfig.RANDOM_SEED}")
    print(f"  Device:             {ExperimentConfig.DEVICE}")
    print(f"  Save Checkpoints:   {ExperimentConfig.SAVE_CHECKPOINTS}")
    print(f"  Checkpoint Dir:     {ExperimentConfig.CHECKPOINT_DIR}")

    print("="*80)


if __name__ == '__main__':
    # Print configuration when run directly
    print_config()

    # Example: Get as dictionary
    config = get_config_dict()
    print("\nAvailable config sections:", list(config.keys()))
