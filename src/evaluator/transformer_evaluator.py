from src.utils.evaluation import (
    accuracy_fn,
    get_precision,
    get_recall,
    get_specificity,
    get_f1_score,
    get_perplexity
)
from src.utils.log import configure_logger
from src.utils.models import create_tgt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import multiprocessing
from multiprocessing.managers import BaseManager
from tqdm import tqdm
from typing import Callable, Union, List, Dict


class TransformerEvaluator:
    '''
    A class to evaluate a PyTorch model on a test dataset using various metrics.
    '''

    # Initialize the logger as a class attribute
    logger = configure_logger(__name__)

    def __init__(self,
            model: nn.Module,
            test_ds: Dataset,
            batch_size: int,
            cretirion: nn.Module,
            device: torch.device=torch.device('cpu')
        ) -> None:
        '''
        Initializes the TransformerEvaluator with the model, test dataset, loss function, and device.

        Args:
            model (nn.Module): The neural network model to be evaluated.
            test_ds (Dataset): Dataset containing the test data.
            batch_size (int): The batch size for creating the DataLoader.
            criterion (nn.Module): Loss function used for evaluation.
            device (torch.device, optional): Device to run the evaluation on (CPU or GPU). Defaults to CPU.
        '''
        self.model = model.to(device, non_blocking=True)
        self.test_ds = test_ds
        self.batch_size = batch_size
        self.cretirion = cretirion
        self.device = device

    def _create_DataLoader(self) -> DataLoader:
        '''
        Create a DataLoader from the dataset
        '''
        return DataLoader(
            dataset=self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=multiprocessing.cpu_count(),
            pin_memory=True
        )
    
    def evaluate(self) -> Dict[str, Union[float, List[List[float]]]]:
        '''
        Evaluates the model on the test dataset using various metrics.

        Returns:
            Dict[str, Union[float, List[List[float]]]]: A dictionary containing evaluation metrics.
                - "Loss": The loss value.
                - "perplexity": The perplexity value.
                - "accuracy": Accuracy of the model.
                - "precision": Precision of the model.
                - "recall": Recall of the model.
                - "specificity": Specificity of the model.
                - "f1_score": F1 score of the model.
        '''
        def _initialize_results(manager: BaseManager) -> Dict[str, Union[float, List[List[float]]]]:
            return manager.dict({
                'Loss': 0.0,
                'perplexity': 0.0,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'specificity': 0.0,
                'f1_score': 0.0,
            })

        def _define_metrics() -> Dict[str, Callable[[torch.Tensor, torch.Tensor], float]]:
            return {
                'Loss': self.cretirion,
                'perplexity': get_perplexity,
                'accuracy': accuracy_fn,
                'precision': get_precision,
                'recall': get_recall,
                'specificity': get_specificity,
                'f1_score': get_f1_score,
            }

        def _calculate_and_update(
                y_pred: torch.Tensor,
                y_true: torch.Tensor,
                key: str,
                metric: Callable[[torch.Tensor, torch.Tensor], Union[List[float], float]],
            ) -> None:
            '''
            Calculate the specified metric and update the results dictionary.

            Args:
                y_pred (torch.Tensor): The predicted labels.
                y_true (torch.Tensor): The ground truth labels.
                key (str): The metric name.
                metric (Callable[[torch.Tensor, torch.Tensor], Union[List[float], float]]): The metric function.
            '''
            nonlocal results
            if key == 'Loss':
                results[key] = metric(y_pred, y_true.to(torch.long)).item()
            else:
                results[key] = metric(y_pred, y_true)

        manager = multiprocessing.Manager()
        results = _initialize_results(manager)
        metrics = _define_metrics()

        TransformerEvaluator.logger.info('Start Evaluation Process.')

        test_dl = self._create_DataLoader()
        y_pred = []
        y_true = []

        self.model.eval()
        with torch.inference_mode():
            for prev_batch, x_batch, y_batch in tqdm(test_dl, ascii=True, desc='Producing Predictions'):
                # Move samples to the same device as the model
                x_batch = x_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)

                # Produce the target tensor and generate logits
                tgt = create_tgt(prev_batch, x_batch)
                y_logits = self.model(x_batch, tgt)

                for batch in y_logits:
                    y_pred.append(batch.tolist())
                y_true.extend(y_batch.tolist())

        # Start multiprocessing for metric calculations
        processes = []
        for metric_name, metric_fn in tqdm(metrics.items(), ascii=True, desc='Calculating Metrics'):
            process = multiprocessing.Process(target=_calculate_and_update, args=(torch.tensor(y_pred), torch.tensor(y_true, dtype=torch.float32), metric_name, metric_fn))
            processes.append(process)
            process.start()

        # Ensure all processes have completed
        for process in processes:
            process.join()

        TransformerEvaluator.logger.info("Evaluation Process Completed Successfully.")

        return dict(results)
