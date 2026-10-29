"""Deep Learning model architectures for Remaining Useful Life (RUL) estimation.

Includes:
1. LSTMRegressor: Multi-layer bidirectional/unidirectional LSTM with dropout.
2. CNN1DRegressor: Temporal 1D Convolutional Neural Network with residual connections.
3. TransformerRegressor: Multi-head self-attention encoder with positional encoding.
"""

import math
from typing import Optional
import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):
    """Multi-layer LSTM network for multivariate time-series RUL regression."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        direction_factor = 2 if bidirectional else 1
        fc_in_dim = hidden_dim * direction_factor

        self.regressor = nn.Sequential(
            nn.Linear(fc_in_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            torch.Tensor: Predicted RUL of shape (batch_size,).
        """
        # lstm_out shape: (batch_size, seq_len, hidden_dim * directions)
        lstm_out, _ = self.lstm(x)
        # Use final time-step representation
        last_step = lstm_out[:, -1, :]
        out = self.regressor(last_step)
        return out.squeeze(-1)


class CNN1DRegressor(nn.Module):
    """1D Temporal Convolutional Neural Network for sensor sequences."""

    def __init__(
        self,
        input_dim: int,
        window_length: int = 30,
        filters: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        # Conv1d expects (batch_size, channels, seq_len)
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(filters),
            nn.ReLU(),
            nn.Conv1d(in_channels=filters, out_channels=filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(filters * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=filters * 2, out_channels=filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(filters * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.regressor = nn.Sequential(
            nn.Linear(filters * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            torch.Tensor: Predicted RUL of shape (batch_size,).
        """
        # Permute to (batch_size, input_dim, seq_len)
        x_perm = x.permute(0, 2, 1)
        feat = self.features(x_perm).squeeze(-1)
        out = self.regressor(feat)
        return out.squeeze(-1)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer input."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class TransformerRegressor(nn.Module):
    """Transformer Encoder architecture for predictive maintenance time-series."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.regressor = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        proj = self.input_proj(x)
        encoded = self.pos_encoder(proj)
        trans_out = self.transformer_encoder(encoded)
        # Average pooling across sequence length
        pooled = trans_out.mean(dim=1)
        out = self.regressor(pooled)
        return out.squeeze(-1)


class TemporalAttention(nn.Module):
    """Additive Temporal Self-Attention over sequential hidden states."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False),
        )

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        # lstm_out: (batch_size, seq_len, hidden_dim)
        scores = self.projection(lstm_out)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(lstm_out * weights, dim=1)
        return context


class BiLSTMAttentionRegressor(nn.Module):
    """Bidirectional LSTM with Temporal Attention Pooling for enhanced prognostic accuracy."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_layer_norm = nn.LayerNorm(input_dim)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        total_hidden = hidden_dim * 2
        self.attention = TemporalAttention(total_hidden)

        self.regressor = nn.Sequential(
            nn.Linear(total_hidden, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        x_norm = self.input_layer_norm(x)
        lstm_out, _ = self.lstm(x_norm)
        attended = self.attention(lstm_out)
        out = self.regressor(attended)
        return out.squeeze(-1)


def build_model(
    model_type: str,
    input_dim: int,
    window_length: int = 30,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
) -> nn.Module:
    """Factory function to instantiate models by name.

    Args:
        model_type: One of 'lstm', 'bilstm_attn', 'cnn_1d', 'transformer'.
        input_dim: Number of input sensor channels.
        window_length: Length of sequence window.
        hidden_dim: Hidden dimension / filter count.
        num_layers: Layer count.
        dropout: Dropout probability.

    Returns:
        nn.Module: Instantiated PyTorch model.
    """
    model_type_norm = model_type.lower()
    if model_type_norm in ("bilstm_attn", "bilstm_attention", "attention_lstm", "bilstm"):
        return BiLSTMAttentionRegressor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    elif model_type_norm == "lstm":
        return LSTMRegressor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    elif model_type_norm in ("cnn_1d", "cnn"):
        return CNN1DRegressor(
            input_dim=input_dim,
            window_length=window_length,
            filters=hidden_dim,
            dropout=dropout,
        )
    elif model_type_norm == "transformer":
        return TransformerRegressor(
            input_dim=input_dim,
            d_model=hidden_dim,
            nhead=4,
            num_layers=num_layers,
            dropout=dropout,
        )
    else:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. Choose 'bilstm_attn', 'lstm', 'cnn_1d', or 'transformer'."
        )

