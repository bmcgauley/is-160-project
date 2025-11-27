"""
LSTM Model Architecture Module for QCEW Employment Data Analysis

This module contains LSTM model definitions, RNN architectures,
and model validation functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class EmploymentLSTM(nn.Module):
    """LSTM model for employment time series forecasting."""

    def __init__(self,
                 input_size: int,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 output_size: int = 1,
                 dropout: float = 0.2):
        """
        Initialize the LSTM model.

        Args:
            input_size: Number of input features
            hidden_size: Hidden layer size
            num_layers: Number of LSTM layers
            output_size: Number of output features
            dropout: Dropout probability
        """
        super(EmploymentLSTM, self).__init__()

        # Store architecture parameters for checkpoint saving/loading
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout_prob = dropout

        # TODO: Define LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout)

        # TODO: Define output layers
        self.fc = nn.Linear(hidden_size, output_size)

        # TODO: Add batch normalization and dropout
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Initialize hidden state
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # LSTM forward pass
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))

        # Take the last time step output
        # last_out shape: (batch_size, hidden_size)
        last_out = lstm_out[:, -1, :]

        # Apply batch normalization
        # Note: BatchNorm1d expects (batch_size, features)
        last_out = self.batch_norm(last_out)

        # Apply dropout
        last_out = self.dropout(last_out)

        # Final linear layer
        # output shape: (batch_size, output_size)
        output = self.fc(last_out)

        return output


class EmploymentRNN(nn.Module):
    """RNN model for sequential employment pattern recognition."""

    def __init__(self,
                 input_size: int,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 output_size: int = 1):
        """
        Initialize the RNN model.

        Args:
            input_size: Number of input features
            hidden_size: Hidden layer size
            num_layers: Number of RNN layers
            output_size: Number of output features
        """
        super(EmploymentRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define RNN layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Initialize hidden state
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # RNN forward pass
        # rnn_out shape: (batch_size, seq_len, hidden_size)
        rnn_out, hn = self.rnn(x, h0)

        # Take the last time step output
        # last_out shape: (batch_size, hidden_size)
        last_out = rnn_out[:, -1, :]

        # Final linear layer
        # output shape: (batch_size, output_size)
        output = self.fc(last_out)

        return output


class CustomLSTM(nn.Module):
    """Custom LSTM architecture combining temporal dependencies and spatial features."""

    def __init__(self,
                 temporal_input_size: int,
                 spatial_input_size: int,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 output_size: int = 1,
                 dropout: float = 0.2):
        """
        Initialize the custom LSTM model.

        Args:
            temporal_input_size: Size of temporal features
            spatial_input_size: Size of spatial features
            hidden_size: Hidden layer size
            num_layers: Number of layers
            output_size: Output size
            dropout: Dropout probability
        """
        super(CustomLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define temporal and spatial processing layers
        self.temporal_lstm = nn.LSTM(temporal_input_size, hidden_size, num_layers,
                                     batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.spatial_fc = nn.Linear(spatial_input_size, hidden_size)
        self.spatial_activation = nn.ReLU()

        # Define fusion and output layers
        self.fusion = nn.Linear(hidden_size * 2, hidden_size)
        self.fusion_activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, temporal_x: torch.Tensor, spatial_x: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass combining temporal and spatial features.

        Args:
            temporal_x: Temporal input tensor of shape (batch_size, seq_len, temporal_input_size)
            spatial_x: Spatial input tensor of shape (batch_size, spatial_input_size) [optional]

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Process temporal features with LSTM
        batch_size = temporal_x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(temporal_x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(temporal_x.device)

        # LSTM forward pass
        # temporal_out shape: (batch_size, seq_len, hidden_size)
        temporal_out, (hn, cn) = self.temporal_lstm(temporal_x, (h0, c0))

        # Take the last time step output
        # temporal_features shape: (batch_size, hidden_size)
        temporal_features = temporal_out[:, -1, :]

        # Process spatial features if provided
        if spatial_x is not None:
            # spatial_features shape: (batch_size, hidden_size)
            spatial_features = self.spatial_activation(self.spatial_fc(spatial_x))

            # Concatenate temporal and spatial features
            # combined shape: (batch_size, hidden_size * 2)
            combined = torch.cat([temporal_features, spatial_features], dim=1)

            # Fusion layer
            # fused shape: (batch_size, hidden_size)
            fused = self.fusion_activation(self.fusion(combined))
        else:
            # If no spatial features, just use temporal
            fused = temporal_features

        # Apply batch normalization and dropout
        fused = self.batch_norm(fused)
        fused = self.dropout(fused)

        # Final output layer
        # output shape: (batch_size, output_size)
        output = self.output(fused)

        return output


def validate_lstm_architecture(model: nn.Module,
                              input_shape: Tuple[int, ...],
                              expected_output_size: int = 1) -> Dict[str, bool]:
    """
    Validate LSTM architecture dimensions match processed employment sequence shapes.

    Args:
        model: PyTorch model to validate
        input_shape: Expected input shape (batch_size, seq_len, input_size) or (batch_size, seq_len, input_size) for EmploymentLSTM/RNN
        expected_output_size: Expected output dimension

    Returns:
        Dictionary of validation results
    """
    logger.info("Validating LSTM architecture dimensions...")
    logger.info(f"  Input shape: {input_shape}")
    logger.info(f"  Expected output size: {expected_output_size}")

    results = {}

    try:
        # Test 1: Model can process input of expected shape
        model.eval()
        with torch.no_grad():
            # Create dummy input tensor
            if isinstance(model, CustomLSTM):
                # CustomLSTM expects temporal and optional spatial inputs
                batch_size, seq_len, input_size = input_shape
                temporal_x = torch.randn(batch_size, seq_len, input_size)
                output = model(temporal_x)
            else:
                # EmploymentLSTM or EmploymentRNN
                dummy_input = torch.randn(*input_shape)
                output = model(dummy_input)

            results['forward_pass'] = True
            logger.info(f"  [OK] Forward pass successful")

        # Test 2: Output shape is correct
        expected_batch_size = input_shape[0]
        output_shape_correct = (output.shape[0] == expected_batch_size and
                               output.shape[1] == expected_output_size)
        results['output_shape'] = output_shape_correct
        logger.info(f"  Output shape: {tuple(output.shape)} - {'[PASS]' if output_shape_correct else '[FAIL]'}")

        # Test 3: No NaN or Inf in output
        has_nan = torch.isnan(output).any().item()
        has_inf = torch.isinf(output).any().item()
        results['no_nan_inf'] = not (has_nan or has_inf)
        logger.info(f"  No NaN/Inf: {'[PASS]' if results['no_nan_inf'] else '[FAIL]'}")

        # Test 4: Model has trainable parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        results['has_parameters'] = num_params > 0
        logger.info(f"  Trainable parameters: {num_params:,} - {'[PASS]' if results['has_parameters'] else '[FAIL]'}")

        # Test 5: Model architecture layers exist
        has_lstm_or_rnn = (hasattr(model, 'lstm') or hasattr(model, 'rnn') or
                          hasattr(model, 'temporal_lstm'))
        results['has_recurrent_layer'] = has_lstm_or_rnn
        logger.info(f"  Has recurrent layer: {'[PASS]' if has_lstm_or_rnn else '[FAIL]'}")

    except Exception as e:
        logger.error(f"  [FAIL] Validation error: {str(e)}")
        results['forward_pass'] = False
        results['output_shape'] = False
        results['no_nan_inf'] = False
        results['has_parameters'] = False
        results['has_recurrent_layer'] = False

    # Overall validation
    all_passed = all(results.values())
    logger.info(f"Overall validation: {'[PASS]' if all_passed else '[FAIL]'} ({sum(results.values())}/{len(results)} checks passed)")

    return results