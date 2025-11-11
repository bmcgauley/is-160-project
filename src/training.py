"""
Training Module for QCEW Employment Data Analysis

This module contains training loops, validation functions,
and training utilities for LSTM models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Union
import logging
import numpy as np
from pathlib import Path
import sys

# Add config directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'config'))
from hyperparameters import TrainingConfig

logger = logging.getLogger(__name__)


class EmploymentTrainer:
    """Trainer class for employment forecasting models."""

    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 criterion: nn.Module = None,
                 optimizer: optim.Optimizer = None,
                 device: str = 'cpu'):
        """
        Initialize the trainer.

        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        if criterion is None:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = criterion

        if optimizer is None:
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=TrainingConfig.LEARNING_RATE,
                weight_decay=TrainingConfig.WEIGHT_DECAY
            )
        else:
            self.optimizer = optimizer

        self.scheduler = None

    def train_epoch(self, epoch_num: int = 0) -> float:
        """
        Run one training epoch.

        Args:
            epoch_num: Current epoch number (for logging)

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Calculate logging interval (log every 5% of batches, or at least every 1000 batches)
        total_batches = len(self.train_loader)
        log_interval = max(1000, total_batches // 20)  # Log 20 times per epoch

        for batch_idx, (sequences, targets) in enumerate(self.train_loader):
            # Move data to device
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(sequences)

            # Calculate loss
            # outputs shape: (batch_size, 1), targets shape: (batch_size,)
            # Squeeze outputs to match targets shape
            outputs = outputs.squeeze()
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=TrainingConfig.GRADIENT_CLIP_NORM)

            # Update weights
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Log progress periodically
            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == total_batches:
                avg_loss_so_far = total_loss / num_batches
                progress = (batch_idx + 1) / total_batches * 100
                logger.info(f"  Epoch {epoch_num} | Batch {batch_idx + 1}/{total_batches} ({progress:.1f}%) | "
                           f"Loss: {avg_loss_so_far:.6f}")

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate_epoch(self, epoch_num: int = 0) -> float:
        """
        Run one validation epoch.

        Args:
            epoch_num: Current epoch number (for logging)

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        # Calculate logging interval
        total_batches = len(self.val_loader)
        log_interval = max(500, total_batches // 10)  # Log 10 times during validation

        with torch.no_grad():  # No gradient computation during validation
            for batch_idx, (sequences, targets) in enumerate(self.val_loader):
                # Move data to device
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                outputs = self.model(sequences)

                # Calculate loss
                outputs = outputs.squeeze()
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                num_batches += 1

                # Log progress periodically
                if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == total_batches:
                    avg_loss_so_far = total_loss / num_batches
                    progress = (batch_idx + 1) / total_batches * 100
                    logger.info(f"  Validation | Batch {batch_idx + 1}/{total_batches} ({progress:.1f}%) | "
                               f"Loss: {avg_loss_so_far:.6f}")

        avg_loss = total_loss / num_batches
        return avg_loss

    def train_model(self,
                   num_epochs: int = None,
                   patience: int = None,
                   save_path: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Train the model with early stopping and checkpointing.

        Args:
            num_epochs: Maximum number of epochs (defaults to TrainingConfig.NUM_EPOCHS)
            patience: Early stopping patience (defaults to TrainingConfig.PATIENCE)
            save_path: Path to save model checkpoints

        Returns:
            Dictionary with training history
        """
        # Use config defaults if not provided
        if num_epochs is None:
            num_epochs = TrainingConfig.NUM_EPOCHS
        if patience is None:
            patience = TrainingConfig.PATIENCE

        logger.info("="*80)
        logger.info("TRAINING EMPLOYMENT FORECASTING MODEL")
        logger.info("="*80)
        logger.info(f"  Max epochs: {num_epochs}")
        logger.info(f"  Early stopping patience: {patience}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Optimizer: {self.optimizer.__class__.__name__}")
        logger.info(f"  Loss function: {self.criterion.__class__.__name__}")

        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }

        best_val_loss = float('inf')
        epochs_without_improvement = 0
        best_model_state = None

        for epoch in range(num_epochs):
            logger.info("\n" + "-"*80)
            logger.info(f"EPOCH {epoch+1}/{num_epochs}")
            logger.info("-"*80)

            # Train for one epoch
            logger.info("Training...")
            train_loss = self.train_epoch(epoch_num=epoch+1)
            history['train_loss'].append(train_loss)

            # Validate
            logger.info("\nValidating...")
            val_loss = self.validate_epoch(epoch_num=epoch+1)
            history['val_loss'].append(val_loss)

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)

            # Update learning rate scheduler if exists
            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            # Log epoch summary
            logger.info("\n" + "="*80)
            logger.info(f"EPOCH {epoch+1}/{num_epochs} SUMMARY")
            logger.info(f"  Train Loss: {train_loss:.6f}")
            logger.info(f"  Val Loss:   {val_loss:.6f}")
            logger.info(f"  LR:         {current_lr:.6f}")
            logger.info("="*80)

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_model_state = self.model.state_dict().copy()

                # Save checkpoint if path provided
                if save_path:
                    self.save_checkpoint(save_path, epoch, val_loss)
                    logger.info(f"  [OK] New best model saved (val_loss: {val_loss:.6f})")
            else:
                epochs_without_improvement += 1

            # Early stopping check
            if epochs_without_improvement >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                logger.info(f"  Best val_loss: {best_val_loss:.6f}")
                break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info("Restored best model weights")

        logger.info("="*80)
        logger.info("TRAINING COMPLETE")
        logger.info(f"  Total epochs: {len(history['train_loss'])}")
        logger.info(f"  Best val loss: {best_val_loss:.6f}")
        logger.info("="*80)

        return history

    def save_checkpoint(self, save_path: str, epoch: int, val_loss: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Ensure directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, save_path)


def create_checkpointing(save_dir: str = 'checkpoints') -> Dict[str, callable]:
    """
    Create model checkpointing for best employment prediction performance.

    Args:
        save_dir: Directory to save checkpoints

    Returns:
        Dictionary of checkpointing functions
    """
    logger.info(f"Setting up checkpointing in {save_dir}")

    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(model, optimizer, epoch, val_loss, filename='best_model.pt'):
        """Save checkpoint to disk."""
        filepath = save_path / filename
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }
        torch.save(checkpoint, filepath)
        logger.info(f"  Checkpoint saved: {filepath}")

    def load_checkpoint(model, optimizer, filename='best_model.pt'):
        """Load checkpoint from disk."""
        filepath = save_path / filename
        if filepath.exists():
            checkpoint = torch.load(filepath)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"  Checkpoint loaded: {filepath}")
            return checkpoint['epoch'], checkpoint['val_loss']
        else:
            logger.warning(f"  No checkpoint found at {filepath}")
            return 0, float('inf')

    return {
        "save_checkpoint": save_checkpoint,
        "load_checkpoint": load_checkpoint,
        "save_dir": save_path
    }


class EarlyStopping:
    """Early stopping handler."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.

        Args:
            val_loss: Current validation loss

        Returns:
            True if should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def implement_early_stopping(patience: int = 10) -> EarlyStopping:
    """
    Implement early stopping based on employment prediction validation loss.

    Args:
        patience: Number of epochs to wait before stopping

    Returns:
        EarlyStopping instance
    """
    logger.info(f"Setting up early stopping with patience={patience}")
    return EarlyStopping(patience=patience)


def build_learning_rate_scheduler(optimizer: optim.Optimizer,
                                scheduler_type: str = 'reduce_on_plateau',
                                **kwargs) -> optim.lr_scheduler._LRScheduler:
    """
    Build learning rate scheduling appropriate for employment data convergence.

    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler ('reduce_on_plateau', 'step', 'cosine', 'exponential')
        **kwargs: Additional arguments for the scheduler

    Returns:
        Learning rate scheduler
    """
    logger.info(f"Building {scheduler_type} learning rate scheduler")

    if scheduler_type == 'reduce_on_plateau':
        # Reduces LR when validation loss plateaus
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 5),
            verbose=True
        )
        logger.info(f"  ReduceLROnPlateau: factor={kwargs.get('factor', 0.5)}, patience={kwargs.get('patience', 5)}")

    elif scheduler_type == 'step':
        # Step decay
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 10),
            gamma=kwargs.get('gamma', 0.1)
        )
        logger.info(f"  StepLR: step_size={kwargs.get('step_size', 10)}, gamma={kwargs.get('gamma', 0.1)}")

    elif scheduler_type == 'cosine':
        # Cosine annealing
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 50),
            eta_min=kwargs.get('eta_min', 0)
        )
        logger.info(f"  CosineAnnealingLR: T_max={kwargs.get('T_max', 50)}")

    elif scheduler_type == 'exponential':
        # Exponential decay
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=kwargs.get('gamma', 0.95)
        )
        logger.info(f"  ExponentialLR: gamma={kwargs.get('gamma', 0.95)}")

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return scheduler