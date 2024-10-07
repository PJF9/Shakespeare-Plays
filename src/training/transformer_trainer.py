from src.utils.save import save_model
from src.utils.log import configure_logger
from src.utils.training import get_loaders
from src.utils.models import create_tgt

import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset, DataLoader, random_split

from multiprocessing import cpu_count

from tqdm import tqdm
from timeit import default_timer as timer
from typing import Callable, Tuple, Dict, Union, List


class TransformerTrainer:
    '''
    A class to handle the training of a PyTorch model.
    '''

    # Initialize the logger as a class attribute
    logger = configure_logger(__name__)

    def __init__(self,
            model: nn.Module,
            dataset: Dataset,
            batch_size: int,
            criterion : Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            eval_fn: Callable[[torch.Tensor, torch.Tensor], float],
            opt: torch.optim.Optimizer,
            scheduler: Union[LRScheduler, None]=None,
            train_prop: float=0.8,
            step: int=1,
            device: torch.device=torch.device('cpu'),
        ) -> None:
        '''
        Initializes the TransformerTrainer with the model, data loaders, loss function, evaluation function, and optimizer.

        Args:
            model (nn.Module): The neural network model to be trained.
            dataset (Dataset): The dataset the will be train the model.
            batch_size (int): The batch_size of the model's DataLoaders.
            criterion (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): Loss function.
            eval_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): Evaluation function.
            opt (torch.optim.Optimizer): Optimizer for updating the model parameters.
            train_prop (int): The propotion of the dataset that will be use to train the model. Default is 80%.
            step (int): The step of the cross validational batches.
            device (torch.device): The device that the model will be trained on. Default is cpu.
        '''
        self.model = model.to(device, non_blocking=True)
        self.dataset = dataset
        self.batch_size = batch_size
        self.criterion  = criterion
        self.eval_fn = eval_fn
        self.opt = opt
        self.scheduler = scheduler
        self.train_prop = train_prop
        self.step = step
        self.device = device

    def _get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        # Get the training and validation Loaders for the epoch
        train_ds, valid_ds = random_split(self.dataset, [self.train_prop, 1 - self.train_prop])

        train_dl = DataLoader(train_ds, self.batch_size, shuffle=True, num_workers=cpu_count(), pin_memory=True)
        valid_dl = DataLoader(valid_ds, self.batch_size, shuffle=False, num_workers=cpu_count(), pin_memory=True)

        return train_dl, valid_dl

    def _get_loaders_cv(self, offset: Union[List[int], int]) -> Tuple[DataLoader, DataLoader, int]:
        train_dl, valid_dl, offset = get_loaders(
            dataset=self.dataset,
            batch_size=self.batch_size,
            train_pro=self.train_prop,
            drop_last=True,
            offset=offset,
            step=self.step
        )

        return train_dl, valid_dl, offset
    
    def _process_data_loaders(self, dl: DataLoader) -> Tuple[float, float]:
        '''
        Process batches from a DataLoader for either training or validation.

        This method iterates over batches of data from a given DataLoader (`dl`), computes
        the loss and evaluation metrics for each batch, and optionally performs gradient 
        descent (backpropagation) if the model is in training mode.

        Args:
            dl (DataLoader): DataLoader containing batches of data.

        Returns:
            Tuple[float, float]: A tuple containing the average batch loss and evaluation 
                score across all batches in the DataLoader.
        '''
        # Initialize batch loss and accuracy
        batch_loss, batch_eval = 0.0, 0.0

        phase = 'Training Step' if self.model.training else 'Validation Step'

        for prev_batch, x_batch, y_batch in tqdm(dl, ascii=True, desc=f'             {phase}'):
            # Moving batches to device
            prev_batch = prev_batch.to(self.device, non_blocking=True) # batch containing all the previous tokens of the sequence
            x_batch = x_batch.to(self.device, non_blocking=True)       # batch containing the sequence to be passed into the model
            y_batch = y_batch.to(self.device, non_blocking=True)       # batch containing the next token of the sequence

            # Creating the inputs of the embedding layer of the decoder, which consist of the previous tokens in the sequence
            tgt = create_tgt(prev_batch, x_batch)

            # Generating predictions (forward pass)
            model_logits = self.model(x_batch, tgt)

            # Calculate loss
            loss = self.criterion(model_logits, y_batch)
            batch_loss += loss.item()
            batch_eval += self.eval_fn(model_logits, y_batch)

            # Backward pass and optimizer step (only for training)
            if self.model.training:
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

        batch_loss /= len(dl)
        batch_eval /= len(dl)

        return batch_loss, batch_eval
    
    def _training_step(self, train_dl: Union[DataLoader, List[DataLoader]]) -> Tuple[float, float]:
        '''
        Performs a single training step over the training DataLoader.

        Args:
            train_dl (DataLoader): The training dataloader that will fit the model.

        Returns:
            Tuple[float, float]: The average training loss and evaluation score for the epoch.
        '''
        self.model.train()

        train_loss, train_eval = self._process_data_loaders(train_dl)

        self.model.eval()
        
        return train_loss, train_eval
    
    def _validation_step(self, valid_dl: Union[DataLoader, List[DataLoader]]) -> Tuple[float, float]:
        '''
        Performs a single validation step over the validation DataLoader.

        Args:
            valid_dl (Dataloader): The validation dataloader to evaluate the model.

        Returns:
            Tuple[float, float]: The average validation loss and evaluation score for the epoch.
        '''
        self.model.eval()

        with torch.inference_mode():
            valid_loss, valid_eval = self._process_data_loaders(valid_dl)

        return valid_loss, valid_eval

    def fit(self,
            epochs: int,
            save_per: Union[int, None]=None,
            save_path: Union[str, None]=None,
            save_best: bool=False,
            cross_validate: bool=False,
        ) -> Dict[str, Union[List[float], str, int]]:
        '''
        Trains the model for a specified number of epochs and optionally saves checkpoints.

        Args:
            epochs (int): The number of epochs to train the model for.
            save_per (Union[int, None], optional): Frequency (in epochs) to save model checkpoints. Defaults to None.
            save_path (Union[str, None], optional): The path that the checkpoints will be saved on. Defaults to None.
            cross_validate (bool, optional): Whether to use cross-validation. Default is False.

        Returns:
            Dict[str, Union[List[float], str, int]]: A dictionary containing training statistics and metadata.
                It includes:
                - 'train_loss': List of training losses for each epoch.
                - 'train_eval': List of training evaluation scores for each epoch.
                - 'valid_loss': List of validation losses for each epoch.
                - 'valid_eval': List of validation evaluation scores for each epoch.
                - 'model_name': Name of the model class.
                - 'loss_fn': Name of the loss function class.
                - 'eval_fn': Name of the evaluation function.
                - 'optimizer': Name of the optimizer class.
                - 'device': Type of device the model is on.
                - 'epochs': Total number of epochs trained.
                - 'total_time': Total time taken for training and evaluation.
                - 'save_path': Saved path of the checkpoints.
                - 'cross_validate': Wheather cross-validation is being used.
        '''
        start_time = timer()
        train_losses, train_evals = [], []
        valid_losses, valid_evals = [], []
        best_valid_loss = float('inf')

        # the variable of cross validation
        offset = 0

        TransformerTrainer.logger.info('Start Training Process.')

        # Get the loaders if the user doesn't want to use cross validation
        if not cross_validate:
            TransformerTrainer.logger.info('Creating training and validation DataLoaders.')
            train_dl, valid_dl = self._get_loaders()
            TransformerTrainer.logger.info('Dataloaders created succesfully.')

        for epoch in range(1, epochs + 1):
            TransformerTrainer.logger.info(f'-> Epoch: {epoch}/{epochs}')

            # Get the loaders if the user want to use cross validation
            if cross_validate:
                TransformerTrainer.logger.info('    Creating training and validation DataLoaders (cross-validation step)')
                train_dl, valid_dl, offset = self._get_loaders_cv(offset)
                TransformerTrainer.logger.info('    Dataloaders created succesfully.')

            # Training and Evaluating the Model
            train_loss, train_eval = self._training_step(train_dl)
            valid_loss, valid_eval = self._validation_step(valid_dl)

            # Log the results
            TransformerTrainer.logger.info('')
            TransformerTrainer.logger.info(f'    Results (lr={self.opt.param_groups[0]["lr"]:.6f}):')
            TransformerTrainer.logger.info(f'    Train Loss:       {train_loss:.4f}')
            TransformerTrainer.logger.info(f'    Train Eval Score: {train_eval:.4f}')
            TransformerTrainer.logger.info(f'    Valid Loss:       {valid_loss:.4f}')
            TransformerTrainer.logger.info(f'    Valid Eval Score: {valid_eval:.4f}')

            train_losses.append(train_loss)
            train_evals.append(train_eval)
            valid_losses.append(valid_loss)
            valid_evals.append(valid_eval)

            # Step the scheduler
            if self.scheduler is not None:
                self.scheduler.step()
                TransformerTrainer.logger.info('')
                TransformerTrainer.logger.info(f'    Scheduling step executed succesfully (new lr={self.opt.param_groups[0]["lr"]:.6f})')

            # Save the best generator model based on loss
            if save_best and valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                save_model(self.model, f'{save_path}/{self.model.__class__.__name__}_best.pth')

            # Saving the model
            if save_per and save_path and (epoch % save_per == 0):
                save_model(self.model, f'{save_path}/{self.model.__class__.__name__}_checkpoint_{epoch}.pth')

            TransformerTrainer.logger.info(('-' * 100))

        # After training, clear the CUDA cache
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        TransformerTrainer.logger.info('Training Process Completed Successfully.')

        return {
            'train_loss': train_losses,
            'train_eval': train_evals,
            'valid_loss': valid_losses,
            'valid_eval': valid_evals,
            'model_name': self.model.__class__.__name__,
            'loss_fn': self.criterion.__class__.__name__,
            'eval_fn': self.eval_fn.__name__,
            'optimizer': self.opt.__class__.__name__,
            'scheduler': self.scheduler.__class__.__name__,
            'device': self.device.type,
            'epochs': epochs,
            'total_time': timer() - start_time,
            'save_path': save_path,
            'cross_validate': cross_validate
        }
