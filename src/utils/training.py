import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

import numpy as np
import math

from multiprocessing import cpu_count
from typing import Tuple


def get_loaders(
        dataset: Dataset,
        batch_size: int,
        train_pro: float,
        drop_last: bool=False,
        offset: int = 0,
        step: int = 1
    ) -> Tuple[DataLoader, DataLoader, int]:
    '''
    Splits the dataset into training and validation sets based on the given proportion,
    creates DataLoaders for each, and returns the DataLoaders along with an updated offset 
    for cyclic iteration through splits.

    Args:
        dataset (Dataset): The dataset to be split and loaded.
        batch_size (int): Number of samples per batch.
        train_pro (float): Proportion of data to use for training (between 0 and 1).
        drop_last (bool, optional): Whether to drop the last incomplete batch if the dataset 
                                    size is not divisible by batch size. Default is False.
        offset (int, optional): Starting point for the train/validation split to allow for 
                                cyclic shifting of the split. Default is 0.
        step (int, optional): Amount to increment the offset after each function call, 
                              useful for iterating through different splits. Default is 1.

    Returns:
        Tuple[DataLoader, DataLoader, int]:
            - `train_loader`: DataLoader for the training set.
            - `valid_loader`: DataLoader for the validation set.
            - `offset`: The updated offset value for the next split.
    '''
    # Use NumPy for efficient index handling
    indices = np.arange(len(dataset))
    # num_splits = len(dataset) // batch_size + (1 if len(dataset) % batch_size != 0 else 0)
    num_splits = len(dataset) // batch_size
    splits = np.array_split(indices, num_splits)

    train_splits_size = int(len(splits) * train_pro)
    valid_splits_size = len(splits) - train_splits_size

    # Create train splits with wrapping around if necessary
    if train_splits_size + offset > len(splits):
        train_splits = splits[offset:] + splits[: (train_splits_size + offset) % len(splits)]
    else:
        train_splits = splits[offset: train_splits_size + offset]

    # Create valid splits with wrapping around if necessary
    valid_offset = (train_splits_size + offset) % len(splits)
    if valid_offset + valid_splits_size > len(splits):
        valid_splits = splits[valid_offset:] + splits[: (valid_offset + valid_splits_size) % len(splits)]
    else:
        valid_splits = splits[valid_offset: valid_offset + valid_splits_size]

    # Flatten the lists of batches to lists of indices
    train_indices = np.concatenate(train_splits)
    valid_indices = np.concatenate(valid_splits)

    # Creating DataLoaders
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=cpu_count(), pin_memory=True)
    valid_loader = DataLoader(Subset(dataset, valid_indices), batch_size=batch_size, shuffle=False, drop_last=drop_last, num_workers=cpu_count(), pin_memory=True)

    # Update the offset
    offset = (offset + step) % len(splits)

    return train_loader, valid_loader, offset


class PositionalEncoding(nn.Module):
    def __init__(self, block_size: int, d_model: int) -> None:
        super().__init__()

        # Positional encoding matrix
        pe = torch.zeros(block_size, d_model)

        # Position indices
        position = torch.arange(0, block_size, dtype=torch.float).unsqueeze(1)

        # Scaling factors for positions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Sine and cosine positional encodings
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe) # buffers are tensors that are not updated during training

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, :x.size(1), :]
