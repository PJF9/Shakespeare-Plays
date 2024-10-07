from src.utils import configure_logger
from src.utils.data import load_contractions, expand_contractions

import torch
from torch.utils.data import Dataset

import os
import re
import string
from tqdm import tqdm
from typing import Dict, List, Tuple, Union, Iterable


class TransformerShakespeareDataset(Dataset):
    '''
    Dataset class for text generation using Shakespeare's works.
    '''

    # Get the logger as a class attribute
    logger = configure_logger(__name__)

    def __init__(self,
            dataset_path: str,
            norm_dataset_path: str,
            constractions_path: str,
            block_size: int,
            to_tensors: bool=False,
            write_norm: bool=True,
            device: torch.device = torch.device('cpu')
        ) -> None:
        '''
        Initializes a TransformerShakespeareDataset object.

        Args:
            dataset_path (str): Path to the directory containing the raw dataset files.
            norm_dataset_path (str): Path to save the normalized dataset files.
            contractions_path (str): Path to the file containing contractions for expansion.
            block_size (int): Size of the sequence block for creating training samples.
            to_tensors (bool, optional): If True, returns samples as PyTorch tensors (default: False).
            write_norm (bool, optional): If True, writes normlized files to memory (default: True).
            device (torch.device, optional): Device to store tensors on (default: 'cpu').
        '''
        super().__init__()

        os.makedirs(dataset_path, exist_ok=True)
        os.makedirs(norm_dataset_path, exist_ok=True)

        self.dataset_path = dataset_path
        self.norm_dataset_path = norm_dataset_path
        self.constractions_path = constractions_path
        self.block_size = block_size
        self.to_tensors = to_tensors
        self.device = device

        if write_norm:
            self._normalize_data()

        self.vocab = self._create_vocab()
        self.vocab_size = len(self.vocab)

        # Create mapping from characters to integers
        self.char_to_int = {char: i for i, char in enumerate(self.vocab)}
        self.int_to_char = {i: char for i, char in enumerate(self.vocab)}

        self.samples = self._create_samples()
    
    def __getitem__(self, index: slice) -> List[Tuple[int, Union[torch.Tensor, Iterable], int]]:
        '''
        Retrieves a sample from the dataset by index or slice.

        Args:
            index (int or slice): Index or slice to retrieve from the dataset.

        Returns:
            List[Tuple[Union[torch.Tensor, Iterable], int]]: List of samples, each sample is a tuple of 
                (input_sequence, target_char). Input_sequence can be a torch.Tensor if to_tensors is True, 
                otherwise an iterable (list of integers).
        '''
        if isinstance(index, slice):
            if self.to_tensors:
                return [(sample[0], torch.tensor(sample[1], dtype=torch.int64), sample[2]) for sample in self.samples[index]]
            return [(sample[0], sample[1], sample[2]) for sample in self.samples[index]]
        else:
            if self.to_tensors:
                return (self.samples[index][0], torch.tensor(self.samples[index][1], dtype=torch.int64), self.samples[index][2])
            return (self.samples[index][0], self.samples[index][1], self.samples[index][2])
        
    def __len__(self) -> int:
        '''
        Returns the total number of samples in the dataset.

        Returns:
            int: Total number of samples in the dataset.
        '''
        return len(self.samples)
    
    def __str__(self) -> str:
        '''
        Returns a string representation of the dataset object.

        Returns:
            str: String representation of the dataset object.
        '''
        return f'TransformerShakespeareDataset(vocab_size: {self.vocab_size}, block_size: {self.block_size}, to_tensors: {self.to_tensors}, device: {self.device})'
    
    def _encode(self, s: str) -> List[int]:
        '''
        Encode a string into a list of integers using the vocabulary.
        '''
        return [self.char_to_int[c] for c in s]

    def _decode(self, l: List[int]) -> str:
        '''
        Decode a list of integers into a string using the vocabulary.
        '''
        return ''.join([self.int_to_char[i] for i in l])

    @staticmethod
    def _create_vocab() -> List[str]:
        '''
        Geerate the vocabulary of the dataset, containing all lowercase english letters, all
            digits (0-9) and the white space characters (' ', '\n', '\t'). It doen't contain
            any punctuations for now.

        Returns:
            List[str]: The vocabulary as a list of strings.
        '''
        vocab = list(string.ascii_lowercase)
        vocab += list(string.digits)
        vocab += list(string.punctuation)
        vocab += [' ', '\n', '\t']
        vocab += '\0' # <sos> token (goes at index 0)

        return sorted(set(vocab))
    
    @staticmethod
    def _normalize(content: str, contractions_dict: Dict[str, str]) -> str:
        '''
        Normilize the given text.

        Args:
            content (str): The string to be normilized
            contractions_dict (Dict[str, str]): Dictionary of contractions for expansion.

        Returns:
            str: The normilized string.
        '''
        content = content.lower()
        content = re.sub(r' +', ' ', content) # Replace multiple spaces
        content = expand_contractions(content, contractions_dict=contractions_dict)

        return content

    def _proc_file(self, file: str, contractions_dict: Dict[str, str]) -> str:
        '''
        Processes a file by normalizing its content.

        Args:
            file (str): File to be processed.
            contractions_dict (Dict[str, str]): Dictionary of contractions for expansion.

        Returns:
            str: Normalized content of the file.
        '''
        # Get the content of the file
        file_path = os.path.join(self.dataset_path, file)

        # Only identifying .txt files as datasets
        if not file_path.endswith('.txt'):
            return ''

        with open(os.path.join(self.dataset_path, file)) as f:
            content = f.read()

        content = TransformerShakespeareDataset._normalize(content, contractions_dict)

        return content

    def _write_norm_file(self, text: str, file_name: str) -> None:
        '''
        Writes normalized text to a file.

        Args:
            text (str): Normalized text to be written.
            file_name (str): Name of the file to write the text to.
        '''
        with open(os.path.join(self.norm_dataset_path, file_name), 'w') as f:
            f.write(text)

    def _normalize_data(self) -> None:
        '''
        Normilize the dataset files.
        '''
        contractions_dict = load_contractions(self.constractions_path)

        for file_name in os.listdir(self.dataset_path):
            TransformerShakespeareDataset.logger.info(f'Processing dataset: {os.path.join(self.dataset_path, file_name)}.')
            # Normalize the content of the file
            norm_text = self._proc_file(file_name, contractions_dict)

            if norm_text == '':
                TransformerShakespeareDataset.logger.info(f'File {file_name} is not a .txt file, so it\'s been ignored.')
                continue

            self._write_norm_file(norm_text, file_name=file_name)
            TransformerShakespeareDataset.logger.info(f'Normalized dataset succesfully saved to: {os.path.join(self.norm_dataset_path, file_name)}')

    def _load_texts(self) -> str:
        '''
        Load and concatenate the text content of all files in the normalized data path.

        Returns:
            str: A single string containing the concatenated text content of all files.
        '''
        texts = ''
        for file_name in os.listdir(self.norm_dataset_path):
            with open(os.path.join(self.norm_dataset_path, file_name), 'r') as file:
                texts += file.read()

        return texts

    def _create_samples(self) -> List[Tuple[int, List[int], int]]:
        '''
        Creates training samples from the concatenated text data.

        Returns:
            List[int, Tuple[List[int], int]]: List of training samples, where each sample is a tuple of 
                (input_sequence, target_char). Each input_sequence is a list of integers (encoded characters),
                and target_char is the next character in the sequence.
        '''
        # Load the content of those normalized datasets
        text = self._load_texts()

        # Tokenizing the entire dataset
        data = self._encode(text)

        samples: List[Tuple[int, List[int], int]] = []
        for i in tqdm(range(1, len(data) - self.block_size), ascii=True, desc='Creating Samples'):
            prev_token = data[i-1]                     # previous token of the sequence
            input_sequence = data[i:i+self.block_size] # the sequence that will be passed into the model
            target_token = data[i+self.block_size]     # next token of the sequence
            samples.append((prev_token, input_sequence, target_token))

        TransformerShakespeareDataset.logger.info('Samples created succesfully.')

        return samples

    def encode(self, input_string: str) -> List[int]:
        '''
        Encode the given string.

        Args:
            input_string (str): The string to be encoded

        Returns:
            List[int]: The encoded version of the string
        '''
        contractions_dict = load_contractions(self.constractions_path)

        norm_input = self._normalize(input_string, contractions_dict)

        return self._encode(norm_input)
    
    def decode(self, output_list: List[int]) -> str:
        '''
        Decode a list of intagers and convert them into string

        Args:
            output_list (List[int]): The encoded list of intagers

        Returns:
            str: The decoded string
        '''
        return self._decode(output_list)
