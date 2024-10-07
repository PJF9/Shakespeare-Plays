from src.utils.training import PositionalEncoding
from src.utils.models import temperature_sampling, create_tgt

import torch
import torch.nn as nn

from typing import List


class LSTMCharModel(nn.Module):
    def __init__(self,
            block_size: int,
            vocab_size: int,
            embedding_dim: int,
            hidden_dim: int,
            num_layers: int,
            dropout: int
        ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2*hidden_dim*block_size, vocab_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.dropout(x)

        return self.fc(x)


class TransformerCharModel(nn.Module):
    def __init__(self,
            vocab_size: int,
            block_size: int,
            embedding_dim: int,
            n_head: int,
            n_encoders: int,
            n_decoders: int,
            dim_feedforward: int,
            dropout: float,
            device: torch.device
        ) -> None:
        super().__init__()
        self.block_size = block_size
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.device = device

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.positional_embeddings = PositionalEncoding(block_size, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            norm_first=True,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoders, enable_nested_tensor=False)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            norm_first=True,
            activation='gelu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoders)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embedding_dim)

        self.classification_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=embedding_dim*block_size, out_features=128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=128, out_features=vocab_size)
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_square_subsequent_mask(self, size: int) -> torch.Tensor:
        '''
        Create a mask for the target sequence to prevent the model from looking ahead.
        '''
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        '''
        - The `src` provides contextual information to the encoder, which the decoder uses
            along with the partially generated `tgt` sequence to produce the next token.
        - Teacher Forcing: During training, the tgt sequence is used to guide the model in
            generating the correct sequence. This technique, known as teacher forcing, helps
            the model learn to generate sequences efficiently.
        '''
        # Get the embeddings for the input tokens
        src = self.embeddings(src) + self.positional_embeddings(src)
        tgt = self.embeddings(tgt) + self.positional_embeddings(tgt)

        # Normalization and dropout
        src = self.norm(self.dropout(src))
        tgt = self.norm(self.dropout(tgt))

        # Prevents the model from attending to future tokens in the target sequence (look-ahead mask).
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(self.device)

        # Transformer encoder and decoder
        memory = self.transformer_encoder(src)
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)

        # Normalization, dropout, and classification head
        output = self.norm(self.dropout(output))
        return self.classification_head(output)

    def generate(self, first_token: torch.Tensor, initial_tokens: torch.Tensor, max_length: int, temperature: float) -> List[int]:
        '''
        Generate a sequence of tokens using the model, starting from an initial set of tokens and generating up to a specified maximum length.

        Args:
            first_token (torch.Tensor): A tensor containing the first token to start the sequence generation.
            initial_tokens (torch.Tensor): A tensor containing the initial sequence of tokens.
            max_length (int): The maximum length of the sequence to be generated.
            temperature (float): A temperature parameter used for controlling the randomness of predictions during sampling.

        Returns:
            List[int]: A list of generated token IDs.
        '''
        # generated_tokens = [first_token.item()] + initial_tokens.tolist()
        generated_tokens = first_token.tolist()
        generated_tokens.extend(initial_tokens.squeeze(dim=0).tolist())

        initial_idx = 0
        for _ in range(max_length):
            # Get the current input sequence
            tokens_cond = initial_tokens[:, -self.block_size:]
            # tgt = create_tgt(first_token, initial_tokens)
            tgt = create_tgt(first_token, tokens_cond)

            # Forward pass through the model
            model_logits = self(tokens_cond, tgt)
            next_token = temperature_sampling(model_logits.squeeze(dim=0), temperature)

            # Append the generated token to the sequence
            generated_tokens.append(next_token)

            # Resize the next token to concatenate it to the `initial_tokens`
            next_token = torch.tensor([[next_token]]).to(self.device, non_blocking=True)
            
            # Update the sequence for the next prediction
            initial_tokens = torch.cat((initial_tokens, next_token), dim=1)
            first_token[0] = initial_tokens[:, initial_idx]
            initial_idx += 1

        return generated_tokens
