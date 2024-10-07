from src import config
from src.dataset import LSTMShakespeareDataset
from src.models import LSTMCharModel
from src.utils import configure_logger, get_device, load_model

import torch

import os
from typing import List, Tuple


# Get the logger for this module
logger = configure_logger(__name__)

# Get default device
device = get_device()


def process_input(dataset: LSTMShakespeareDataset, message: str) -> torch.Tensor:
    '''
    Encode the given input and get it ready to pass it to the model

    Args:
        dataset (LSTMShakespeareDataset): The dataset object in which we get the encode method
        message (str): The message to be encoded and converted to tensor

    Returns:
        torch.Tensor: The input to the model (initial seed for predictions)
    '''
    encoded_message = dataset.encode(message)[:config.BLOCK_SIZE]
    model_input = torch.tensor(encoded_message, dtype=torch.int32).unsqueeze(dim=0).to(device)
    
    return model_input


def _temperature_sampling(model_logits: torch.Tensor, temperature:float=1.0) -> int:
    '''
    Perform temperature sampling to generate the next token based on model logits.

    Args:
    - model_logits (torch.Tensor): Logits (raw predictions) from the model, typically for the next
        token prediction. Should have shape [sequence_length, vocab_size].
    - temperature (float): Temperature parameter to scale the logits before applying softmax.
        Higher values make the probability distribution flatter (more random), while lower values
        make it sharper (more deterministic). Default is 1.0.

    Returns:
    - int: The sampled token index from the probability distribution.
    '''
    # Get the logits for the last character in the sequence
    logits = model_logits[-1, :]
    # Scale them using temperature to affect the steepness of the underling distrubution
    logits = logits / temperature
    # Generate probabilities
    probs = torch.softmax(logits, dim=-1)
    # Sample from the probability distribution
    next_token = torch.multinomial(probs, num_samples=1).item()

    return next_token


def _predict_next(model: LSTMCharModel, sequence: torch.Tensor, temperature: float = 1.0) -> Tuple[int, torch.Tensor]:
    '''
    Generating the next token of the sequence

    Args:
        model (LSTMCharModel): The model that is used to make predictions
        sequence (torch.Tensor): The sequence that is used to generate the next token
        temperature (float): The temperature value to control randomness

    Returns
        Tuple[int, torch.Tensor]: First returns the next token of the sequence and then
            the new sequence.
    '''
    model_logits = model(sequence)

    # Generate next token
    next_token = _temperature_sampling(model_logits, temperature=temperature)

    # Reshape the next_token to be 2-dimensional to concatenate
    next_token_tensor = torch.tensor([[next_token]], dtype=torch.int32).to(device)
    
    # Slide the input sequence left by one position and add the next token
    new_sequence = torch.cat((sequence[:, 1:], next_token_tensor), dim=1)

    return next_token, new_sequence


def predict(model: LSTMCharModel, initial_sequence: torch.Tensor, L: int, temperature: float=1.0) -> List[int]:
    '''
    Generate next token of the sequence L times

    Args:
        model (LSTMCharModel): The model that is used to make predictions
        sequence (torch.Tensor): The initial sequence that will used to seed the predictions
        temperature (float): The temperature value to control randomness
        
    Returns:
        List[int]: The total encoded tokens
    '''
    tokens = initial_sequence.squeeze(dim=0).tolist()

    model.eval()
    with torch.inference_mode():
        model_input = initial_sequence
        for _ in range(L):
            next_token, model_input = _predict_next(model, model_input, temperature)
            tokens.append(next_token)

    return tokens


def main() -> None:
    # Initialize the dataset object to get encoder-decoder and vocab_size
    dataset = LSTMShakespeareDataset(
        dataset_path=config.TRAIN_DATASETS_PATH,
        norm_dataset_path=config.NORMALIZE_TRAIN_DATA_PATH,
        constractions_path=config.CONTRACTIONS_PATH,
        block_size=config.BLOCK_SIZE,
        to_tensors=True,
        device=device
    )

    os.makedirs(config.GENERATE_PATH, exist_ok=True)

    # Create the initial seed for the prediction
    message = '''Peter, the Great Emperor

**** ACT I ****
**** SCENE I. Venice. A street. ****
     Enter Peter with his sword
'''
    initial_sequence = process_input(dataset, message)
    logger.info(f'Input has been encoded and moved to device: {device.type}')

    # Load the pre-trained model
    model_name = 'LSTMCharModel_checkpoint_20.pth'
    model = load_model(LSTMCharModel, os.path.join(config.MODELS_PATH, model_name), block_size=config.BLOCK_SIZE, vocab_size=dataset.vocab_size, **config.LSTM_CONFIGS).to(device)
    logger.info(f'The model is loaded and moved to device: {device.type}')

    # Generate predictions
    tokens = predict(model, initial_sequence, L=1000, temperature=1.0)
    decoded_preds = dataset.decode(tokens)
    logger.info(f'Predictions have been made and decoded succesfully, saving them into {os.path.join(config.GENERATE_PATH, "lstm_generation.txt")}')

    # Saving predictions
    with open(os.path.join(config.GENERATE_PATH, 'lstm_generation.txt'), 'w') as f:
        f.write(decoded_preds)


if __name__ == '__main__':
    main()
