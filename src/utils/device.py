import torch


def get_device() -> torch.device:
    '''
    Return CUDA device if cuda is available
    '''
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
