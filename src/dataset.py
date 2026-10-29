"""PyTorch Dataset and DataLoader wrappers for turbofan sequence data."""

from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TurbofanSequenceDataset(Dataset):
    """PyTorch Dataset for Multivariate Turbofan Time-Series Windows.

    Accepts 3D array of windows (N, L, F) and 1D target array (N,).
    """

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]


def get_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 64,
    shuffle_train: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Creates training and validation PyTorch DataLoaders.

    Args:
        X_train: Training sequences (N_train, window_length, num_features).
        y_train: Training RUL targets (N_train,).
        X_val: Validation sequences (N_val, window_length, num_features).
        y_val: Validation RUL targets (N_val,).
        batch_size: Mini-batch size.
        shuffle_train: Whether to shuffle training batches.
        num_workers: DataLoader worker processes (0 for Windows compatibility).

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = TurbofanSequenceDataset(X_train, y_train)
    val_dataset = TurbofanSequenceDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader

