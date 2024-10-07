# Shakespeare Play Generation using LSTM and Transformer Models

## Overview
This project explores the use of deep learning models, specifically Long Short-Term Memory (LSTM) networks and Transformer architectures, to generate text in the style of William Shakespeare. Leveraging a dataset of Shakespeare's plays, the project aims to experiment with these models to understand their effectiveness in generating coherent and stylistically appropriate text that mimics the playwright’s language.

## Dataset
The dataset used for this project contains the complete works of William Shakespeare, including his plays, sonnets, and poems. The dataset has been sourced from [this repository](https://github.com/ravexina/shakespeare-plays-dataset-scraper), which scrapes and organizes Shakespeare’s plays from public online sources.

### Dataset Preprocessing
Before training the models, the raw text was preprocessed by:
* Converting all characters to lowercase.
* Removing multiple spaces.
* Expanding contractions.
* Tokenizing the text into characters.
* Creating vocabulary mappings (for tokenizing text into integer sequences) and reversing mappings for decoding.

## Model Architecture

### 1. LSTM Model.
The LSTM model is a type of recurrent neural network (RNN) that is well-suited for processing sequences of data, such as text. LSTMs handle the vanishing gradient problem better than traditional RNNs, which allows them to capture long-term dependencies in text data.

**Hyperparameters:**

* Embedding dimension: 64
* LSTM units: 2
* Number of LSTM layers: 2
* Dropout: 0.4
* Sequence length: 100 tokens
* Optimizer: Adam
* Loss function: Sparse categorical crossentropy

### 2. Transformer Model
The Transformer model, introduced by Vaswani et al. (2017), is based on the self-attention mechanism, which allows it to focus on different parts of the input sequence. Unlike LSTMs, Transformers do not require sequential data processing and can be more computationally efficient for training.

**Hyperparameters:**

* Embedding dimension: 64
* Number of heads in the multi-head attention: 2
* Number of transformer layers: 2
* Feedforward dimension: 64
* Dropout: 0.2
* Sequence length: 100 tokens
* Optimizer: Adam
* Loss function: Sparse categorical crossentropy

## Installation and Setup
To run the project on your local machine, follow these steps:
### Prerequisites
Ensure that you have Python 3.8 or higher installed on your machine. You also need to install the necessary Python packages.

### Step-by-Step Setup

**1. Clone the repository:**
```bash
git clone https://github.com/yourusername/shakespeare-generation.git
cd shakespeare-generation
```

**2. Install required dependencies:**
You can install the required packages using the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

Then you can customize the model parameters to further train and evaluate the models.

## Conclusion
This project demonstrates the potential of deep learning models, particularly LSTM and Transformer architectures, to generate text in the style of William Shakespeare. With better hyperparameter tunning we can get even better results.
