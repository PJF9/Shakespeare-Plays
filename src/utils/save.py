from src.utils.log import configure_logger

import torch
from torch import nn
from pathlib import Path
from os import remove


# Get the logger for this module
logger = configure_logger(__name__)


def save_model(model: nn.Module, path: str, stops=False) -> None:
    '''
    Save a PyTorch model to a specified path.

    Args:
        model (torch.Module): The PyTorch model to be saved.
        path (str): The path where the model will be saved.
        stops (bool, optional): If True, stops the function execution if the model file already exists at the given path. Defaults to False.

    Raises:
        AssertionError: If the file extension of the specified path is not `.pt` or `.pth`.

    Returns:
        None
    '''
    target_path = Path('/'.join(path.split('/')[:-1]))
    model_name = path.split('/')[-1]

    if not (model_name.endswith('.pth') or model_name.endswith('.pt')):
        logger.error('Wrong extension: Expecting `.pt` or `.pth`.')
        return
    
    # Creating the directory that the model is going to be saved if not exists
    if not target_path.exists():
        target_path.mkdir(parents=True, exist_ok=True)

    # If path already exists
    if Path(path).is_file():
        logger.info(f'Model `{model_name}` already exists on `{target_path}`.')
        if stops:
            return
        logger.info(f'Deleting `{path}`.')
        remove(path)

    # Saving the Model to the given path
    logger.info(f'Saving Model `{model_name}` to `{target_path}`.')
    torch.save(model.state_dict(), path)

    logger.info(f'Model Successfully Saved to `{path}`.')


def load_model(model_class: nn.Module, model_path: str, device: torch.device=torch.device('cpu'), model_device: bool=False, **kwargs) -> nn.Module:
    '''
    Loads a PyTorch model from a specified file.
    
    Parameters:
        model_path (str): Path to the saved model file (e.g., 'model.pth').
        model_class (nn.Module): The class of the model to be loaded.
        device (torch.device): The device that the model will be load on. Default is CPU.
        model_device (bool): If True the model needs device in its arguments. Default is False
        **kwargs: Additional arguments required to initialize the model class.

    Returns:
        The loaded model.
    '''
    # Initialize the model
    if model_device:
        model = model_class(device=device, **kwargs)
    else:
        model = model_class(**kwargs)

    # Load the state dict (parameters)
    state_dict = torch.load(model_path, map_location=torch.device(device))
    
    # Load the parameters into the model
    model.load_state_dict(state_dict)
    
    # Set the model to evaluation mode
    model.eval()

    logger.info('Model succesfully loaded.')
    
    return model
