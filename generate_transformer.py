from src import config
from src.dataset import TransformerShakespeareDataset
from src.models import TransformerCharModel
from src.utils import configure_logger, get_device, load_model

import torch

import os
from typing import Tuple


# Get the logger for this module
logger = configure_logger(__name__)

# Get default device
device = get_device()


def process_input(dataset: TransformerShakespeareDataset, message: str) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Encode the given input and get it ready to pass it to the model

    Args:
        dataset (TransformerShakespeareDataset): The dataset object in which we get the encode method
        message (str): The message to be encoded and converted to tensor

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The input to the model (initial seed for predictions)
    '''
    encoded_message = dataset.encode(message)[:config.BLOCK_SIZE + 1]
    prev_token = torch.tensor(encoded_message[0], dtype=torch.int32).unsqueeze(dim=0).to(device)
    model_input = torch.tensor(encoded_message[1:], dtype=torch.int32).unsqueeze(dim=0).to(device)
    
    return prev_token, model_input


def main() -> None:
    # Initialize the dataset object to get encoder-decoder and vocab_size
    dataset = TransformerShakespeareDataset(
        dataset_path=config.TRAIN_DATASETS_PATH,
        norm_dataset_path=config.NORMALIZE_TRAIN_DATA_PATH,
        constractions_path=config.CONTRACTIONS_PATH,
        block_size=config.BLOCK_SIZE,
        to_tensors=True,
        write_norm=False,
        device=device
    )

    os.makedirs(config.GENERATE_PATH, exist_ok=True)

    # Create the initial seed for the prediction
    message = '''Peter, the Great Emperor

**** ACT I ****
**** SCENE I. France. In the battlefield. ****
     Enter Peter with his sword.
Peter
 '''
    first_token, initial_tokens = process_input(dataset, message)
    logger.info(f'Input has been encoded and moved to device: {device.type}')

    # Load the pre-trained model
    model_name = 'checkpoints/TransformerCharModel_checkpoint_20.pth'
    model = load_model(
        model_class=TransformerCharModel,
        model_path=model_name,
        device=device,
        model_device=True,
        block_size=config.BLOCK_SIZE,
        vocab_size=dataset.vocab_size,
        **config.TRANSFORMER_CONFIGS
    ).to(device)

    logger.info(f'The model is loaded and moved to device: {device.type}')

    # Generate predictions
    tokens = model.generate(first_token, initial_tokens, max_length=1000, temperature=1.0)
    decoded_preds = dataset.decode(tokens)
    logger.info(f'Predictions have been made and decoded succesfully, saving them into {os.path.join(config.GENERATE_PATH, 'transformer_generation.txt')}')

    # Saving predictions
    with open(os.path.join(config.GENERATE_PATH, 'generation.txt'), 'w') as f:
        f.write(decoded_preds)


if __name__ == '__main__':
    main()
