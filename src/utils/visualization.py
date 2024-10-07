from src.utils.log import configure_logger

import matplotlib.pyplot as plt

import os
from typing import List, Tuple, Optional


# Get the logger for this module
logger = configure_logger(__name__)


def plot_loss(
        loss_list: List[float],
        type: str = 'eval',
        c: str = 'g',
        figsize: Tuple[int, int] = (6, 4),
        fontsize: int = 14,
        save_path: Optional[str] = None
    ) -> None:
    '''
    Plots the training or evaluation loss over epochs and optionally saves the plot to a specified path.

    Args:
        loss_list (List[float]): List of loss values to be plotted.
        type (str, optional): Type of loss. Default is 'eval'. Accepted values are 'train' or 'eval'.
        c (str, optional): Color of the plot. Default is 'g' (green).
        figsize (Tuple, optional): Size of the figure (width, height) in inches. Default is (6, 4).
        fontsize (int, optional): Font size of the title. Default is 14.
        save_path (Optional[str], optional): Path to save the plot image. If None, the plot is not saved. Default is None.

    Raises:
        AssertionError: If an invalid loss type is provided.

    Returns:
        None
    '''

    if type not in ['train', 'eval']:
        logger.error(f'Invalid loss type: Got `{type}`. Only `train`, `eval` are accepted')
        return

    plt.figure(figsize=figsize)

    if type == 'eval':
        plt.plot(range(len(loss_list)), loss_list, c=c, label="Validation Lost")
        plt.title(f"Validation Loss", fontsize=fontsize)
    else:
        plt.plot(range(len(loss_list)), loss_list, c=c, label="Training Lost")
        plt.title(f"Training Loss", fontsize=fontsize)

    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    if save_path:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Plot saved to `{save_path}`.")

    plt.grid(color='gray', linestyle='--', linewidth=0.5)

    plt.show()


def plot_losses(
        train_loss: List[float],
        eval_loss: List[float],
        c: List[str] = ['g', 'b'],
        fig_size: Tuple[int, int] = (6, 4),
        font_size: int = 11,
        save_path: Optional[str] = None
    ) -> None:
    '''
    Plots both training and evaluation losses over epochs and optionally saves the plot to a specified path.

    Args:
        train_loss (List[float]): List of training loss values to be plotted.
        eval_loss (List[float]): List of evaluation loss values to be plotted.
        c (List[str], optional): List of colors for the plots. Default is ['g', 'b'] (green for validation loss, blue for training loss).
        fig_size (Tuple[int, int], optional): Size of the figure (width, height) in inches. Default is (6, 4).
        font_size (int, optional): Font size of the legend and title. Default is 11.
        save_path (Optional[str], optional): Path to save the plot image. If None, the plot is not saved. Default is None.

    Returns:
        None
    '''

    plt.figure(figsize=fig_size)

    plt.plot(range(len(train_loss)), train_loss, c=c[0], label="Training Loss")
    plt.plot(range(len(eval_loss)), eval_loss, c=c[1], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Loss Curves", fontsize=14)
    plt.legend(fontsize=font_size)

    if save_path:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Plot saved to `{save_path}`.")

    plt.grid(color='gray', linestyle='--', linewidth=0.5)

    plt.show()
