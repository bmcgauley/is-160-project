"""
Example: How to use the centralized hyperparameters configuration

This file demonstrates how easy it is to access and use the configuration.
"""

from hyperparameters import (
    ModelConfig,
    TrainingConfig,
    DataConfig,
    ExperimentConfig,
    print_config,
    get_config_dict
)


def example_1_print_all_config():
    """Example 1: Print all configuration settings"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Print all configuration")
    print("="*80)
    print_config()


def example_2_access_individual_values():
    """Example 2: Access individual configuration values"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Access individual values")
    print("="*80)

    print(f"Training will run for {TrainingConfig.NUM_EPOCHS} epochs")
    print(f"Model has {ModelConfig.NUM_LAYERS} LSTM layers")
    print(f"Hidden size is {ModelConfig.HIDDEN_SIZE}")
    print(f"Batch size is {DataConfig.BATCH_SIZE}")
    print(f"Learning rate is {TrainingConfig.LEARNING_RATE}")


def example_3_use_in_model():
    """Example 3: How to use config when creating a model"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Use in model initialization")
    print("="*80)

    # Pseudo-code showing how to use config
    print("Example code:")
    print("""
    from config.hyperparameters import ModelConfig
    from lstm_model import EmploymentLSTM

    model = EmploymentLSTM(
        input_size=num_features,
        hidden_size=ModelConfig.HIDDEN_SIZE,
        num_layers=ModelConfig.NUM_LAYERS,
        output_size=ModelConfig.OUTPUT_SIZE,
        dropout=ModelConfig.DROPOUT
    )
    """)


def example_4_use_in_training():
    """Example 4: How to use config in training loop"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Use in training loop")
    print("="*80)

    print("Example code:")
    print("""
    from config.hyperparameters import TrainingConfig
    from training import EmploymentTrainer

    trainer = EmploymentTrainer(model, train_loader, val_loader)
    history = trainer.train_model(
        num_epochs=TrainingConfig.NUM_EPOCHS,
        patience=TrainingConfig.PATIENCE
    )
    """)


def example_5_get_as_dict():
    """Example 5: Get configuration as dictionary"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Get configuration as dictionary")
    print("="*80)

    config_dict = get_config_dict()

    print("Config sections:", list(config_dict.keys()))
    print("\nModel config:")
    for key, value in config_dict['model'].items():
        print(f"  {key}: {value}")

    print("\nTraining config:")
    for key, value in config_dict['training'].items():
        print(f"  {key}: {value}")


def example_6_quick_modifications():
    """Example 6: Common quick modifications"""
    print("\n" + "="*80)
    print("EXAMPLE 6: Common quick modifications")
    print("="*80)

    print("""
To quickly test with 2 epochs, edit config/hyperparameters.py:

    class TrainingConfig:
        NUM_EPOCHS: int = 2        # Change from 100 to 2
        PATIENCE: int = 2          # Change from 10 to 2

To increase model capacity:

    class ModelConfig:
        HIDDEN_SIZE: int = 128     # Change from 64 to 128
        NUM_LAYERS: int = 3        # Change from 2 to 3

To reduce memory usage:

    class DataConfig:
        BATCH_SIZE: int = 16       # Change from 32 to 16
    """)


if __name__ == '__main__':
    # Run all examples
    example_1_print_all_config()
    example_2_access_individual_values()
    example_3_use_in_model()
    example_4_use_in_training()
    example_5_get_as_dict()
    example_6_quick_modifications()

    print("\n" + "="*80)
    print("All examples complete!")
    print("="*80)
    print("\nTo modify hyperparameters, edit: config/hyperparameters.py")
    print("To view current config, run:    python config/hyperparameters.py")
