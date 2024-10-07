import torch
from torch import nn

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def accuracy_fn(model_logits: torch.Tensor, labels: torch.Tensor) -> float:
    '''
    Compute the accuracy of a classification model.

    Args:
      	model_logits (torch.Tensor): The logits or outputs from the model, of shape (N, C),
			where N is the number of samples and C is the number of classes.
    	labels (torch.Tensor): The true labels, of shape (N,), where each value is in the range [0, C-1].

    Returns:
    	float: The accuracy of the model on the provided batch of data, in percentage (%).
    '''
    preds = torch.softmax(model_logits, dim=1).argmax(dim=1)

    return (torch.sum(preds == labels).item() / len(labels))

def get_perplexity(model_logits: torch.Tensor, labels: torch.Tensor) -> float:
    '''
    Calculate the perplexity of a language model.

    Args:
    	model_logits (torch.Tensor): The logits or outputs from the model.
    	labels (torch.Tensor): The true labels.

    Returns:
    	float: The perplexity of the model.
    '''
    criterion = nn.CrossEntropyLoss()

    loss = criterion(model_logits, labels.to(torch.long))

    perplexity = torch.exp(loss)

    return perplexity.item()

def get_precision(model_logits: torch.Tensor, labels: torch.Tensor) -> float:
    '''
    Compute the precision of a classification model.

    Args:
    	model_logits (torch.Tensor): The logits or outputs from the model.
    	labels (torch.Tensor): The true labels.

    Returns:
    	float: The precision of the model.
    '''
    preds = torch.softmax(model_logits, dim=1).argmax(dim=1)

    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy().astype('int64')

    return precision_score(labels, preds, average='weighted')

def get_recall(model_logits: torch.Tensor, labels: torch.Tensor) -> float:
    '''
    Compute the recall of a classification model.

    Args:
    	model_logits (torch.Tensor): The logits or outputs from the model.
    	labels (torch.Tensor): The true labels.

    Returns:
		float: The recall of the model.
    '''
    preds = torch.softmax(model_logits, dim=1).argmax(dim=1)

    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy().astype('int64')

    return recall_score(labels, preds, average='weighted')

def get_f1_score(model_logits: torch.Tensor, labels: torch.Tensor) -> float:
    '''
    Compute the F1 score of a classification model.

    Args:
    	model_logits (torch.Tensor): The logits or outputs from the model.
    	labels (torch.Tensor): The true labels.

    Returns:
    	float: The F1 score of the model.
    '''
    preds = torch.softmax(model_logits, dim=1).argmax(dim=1)

    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy().astype('int64')

    return f1_score(labels, preds, average='weighted')

def get_specificity(model_logits: torch.Tensor, labels: torch.Tensor) -> float:
    '''
    Compute the specificity of a classification model.

    Args:
    	model_logits (torch.Tensor): The logits or outputs from the model.
    	labels (torch.Tensor): The true labels.

    Returns:
    	float: The specificity of the model.
    '''
    preds = torch.softmax(model_logits, dim=1).argmax(dim=1)

    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    # Get the confusion matrix
    cm = confusion_matrix(labels, preds)

    # Calculate specificity for each class
    average_specificity = 0
    for i in range(len(cm)):
        # True Positives, False Positives, False Negatives, and True Negatives
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        
        # Calculate specificity
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

        average_specificity += specificity
    
    average_specificity /= len(cm)

    return average_specificity
