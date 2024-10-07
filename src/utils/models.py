import torch


def temperature_sampling(model_logits: torch.Tensor, temperature:float=1.0) -> int:
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
    # logits = model_logits[-1, :]
    # Scale them using temperature to affect the steepness of the underling distrubution
    # logits = logits / temperature
    logits = model_logits / temperature
    # Generate probabilities
    probs = torch.softmax(logits, dim=-1)
    # Sample from the probability distribution
    next_token = torch.multinomial(probs, num_samples=1).item()

    return next_token

def create_tgt(prev_batch: torch.Tensor, x_batch: torch.Tensor) -> torch.Tensor:
    '''
    Generate a target tensor for sequence models by shifting the input sequence to the
    right and prepending a previous batch tensor.

    Args:
        prev_batch (torch.Tensor): A tensor containing the previous batch data, typically the start-of-sequence tokens.
        x_batch (torch.Tensor): A tensor containing the current batch of input sequences.

    Returns:
        torch.Tensor: A tensor where the `x_batch` is shifted to the right, and the `prev_batch` is prepended as the first element in the sequence.
    '''
    # Initialize tgt with zeros (assuming 0 is the start-of-sequence token)
    tgt = torch.zeros_like(x_batch)

    # Shift x_batch to the right and append y_batch as the last element
    tgt[:, 0] = prev_batch
    tgt[:, 1:] = x_batch[:, :-1]

    return tgt
