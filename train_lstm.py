from src import config
from src.dataset import LSTMShakespeareDataset
from src.models import LSTMCharModel
from src.training import LSTMTrainer
from src.utils import (
    configure_logger,
    accuracy_fn,
    plot_losses,
    get_device
)

from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR

import os


# Get the logger for this module
logger = configure_logger(__name__)

# Get default device
device = get_device()


def main() -> None:
    '''Main function to train the deep learning model.'''

    # Initialize the dataset object
    dataset = LSTMShakespeareDataset(
        dataset_path=config.TRAIN_DATASETS_PATH,
        norm_dataset_path=config.NORMALIZE_TRAIN_DATA_PATH,
        constractions_path=config.CONTRACTIONS_PATH,
        block_size=config.BLOCK_SIZE,
        to_tensors=True,
        write_norm=False,
        device=device
    )

    # Saving the vocab size because we'll use it for evaluating the model
    vocab_size = dataset.vocab_size

    # Instanciate the Model
    model = LSTMCharModel(block_size=config.BLOCK_SIZE, vocab_size=vocab_size, **config.LSTM_CONFIGS).to(device, non_blocking=True)

    logger.info(f'The model is created and placed on the device: {device.type}')

    loss_fn = nn.CrossEntropyLoss()

    opt = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    # scheduler = ExponentialLR(opt, gamma=config.GAMMA)

    # Instanciate the Trainer
    trainer = LSTMTrainer(
        model = model,
        dataset = dataset,
        batch_size = config.BATCH_SIZE,
        criterion = loss_fn,
        eval_fn = accuracy_fn,
        opt = opt,
        # scheduler=scheduler,
        device = device,
    )

    logger.info('The trainer is created.')

    # Train the model
    train_res = trainer.fit(
        epochs = config.EPOCHS,
        save_per = config.EPOCHS,
        save_path = config.MODELS_PATH,
        cross_validate = False,
        save_best=True
    )

    logger.info(f'Training Results: {train_res}')

    # Plot the losses and save them
    plot_losses(train_res['train_loss'], train_res['valid_loss'], save_path=os.path.join(config.PLOTS_PATH, 'lstm_losses.png'))


if __name__ == '__main__':
    main()
