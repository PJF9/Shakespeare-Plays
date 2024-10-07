from src import config
from src.dataset import LSTMShakespeareDataset
from src.models import LSTMCharModel
from src.evaluator import LSTMEvaluator
from src.utils.save import load_model
from src.utils.log import configure_logger
from src.utils import get_device

from torch import nn

import os
from typing import Dict


# Get the logger for this module
logger = configure_logger(__name__)

# Get default device
device = get_device()

# The vocab_size from training the model
VOCAB_SIZE = 71


def evaluate(model: nn.Module, dataset: LSTMShakespeareDataset, loss_fn: nn.Module, file: str) -> Dict[str, int]:
    '''
    Evaluate the model on the given dataset.

    Args:
        model (nn.Module): The model to evaluate.
        dataset (Dataset): The dataset to evaluate on.
        loss_fn (nn.Module): The loss function to use.
        file (str): Filename to save the evaluation results.

    Returns:
        Dict[str, float]: Evaluation results.
    '''
    evaluator = LSTMEvaluator(
        model = model,
        test_ds = dataset,
        cretirion = loss_fn,
        device = device
    )
    logger.info('The evaluator is created.')

    eval_res = evaluator.evaluate()

    os.makedirs(config.LOGS_PATH, exist_ok=True)

    with open(os.path.join(config.LOGS_PATH, file), 'w') as f:
        f.write(str(eval_res))

    return eval_res


def main() -> None:
    '''Main function to evaluate the deep learning model.'''

    # Initialize the dataset object
    dataset = LSTMShakespeareDataset(
        dataset_path=config.TEST_DATASETS_PATH,
        norm_dataset_path=config.NORMALIZE_TEST_DATA_PATH,
        constractions_path=config.CONTRACTIONS_PATH,
        block_size=config.BLOCK_SIZE,
        to_tensors=True,
        device=device
    )

    # Instanciate the Model and load the pre-trained
    model_name = 'LSTMCharModel_checkpoint_20.pth'
    model = load_model(
        model_class=LSTMCharModel,
        model_path=os.path.join(config.MODELS_PATH, model_name),
        block_size=config.BLOCK_SIZE,
        vocab_size=VOCAB_SIZE,
        device=device,
        **config.LSTM_CONFIGS
    ).to(device, non_blocking=True)

    logger.info(f'The model is loaded and moved to device: {device.type}')

    loss_fn = nn.CrossEntropyLoss()
    
    # Evaluate the model
    results = evaluate(model, dataset, loss_fn, file='lstm_evaluate.txt')

    logger.info(f'Results: {results}')


if __name__ == '__main__':
    main()
