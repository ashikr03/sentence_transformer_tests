import transformers
import sentence_transformers
import torch
import torch.nn as nn
import numpy as np
import datasets

class MultiTaskLearning_ClassificationandSentiment(nn.Module):
    """
    MTL model for joint classification and sentiment analysis.

    This model uses a shared SentenceTransformer encoder backbone to extract sentence-level
    embeddings, with two task-specific heads:

    - A classification head for predicting categorical platform types
    - A sentiment analysis head for predicting sentiment categories

    Architecture:
    -------------
    - Shared Encoder: A pre-trained SentenceTransformer model ('nlptown/bert-base-multilingual-uncased-sentiment')
                     used to compute fixed-size embeddings.
    - Classification Head: A two-layer feedforward neural network with ReLU activation.
    - Sentiment Head: A two-layer feedforward neural network with ReLU activation.

    Loss Computation:
    -----------------
    - Computes task-specific cross-entropy loss for both classification and sentiment.
    - If only one task is active (i.e., its labels are provided), only that loss is used.
    - If both tasks are active, the final loss is the average of the two.

    Parameters:
    -----------
    a_class_num : int
        Number of classes for the classification task.
    b_class_num : int
        Number of classes for the sentiment task.

    Inputs:
    -------
    input_ids : torch.Tensor
        Tokenized input IDs for the transformer encoder.
    attention_mask : torch.Tensor, optional
        Attention mask for the encoder input.
    sentiment_labels : torch.Tensor, optional
        Ground-truth sentiment labels for computing sentiment task loss.
    classification_labels : torch.Tensor, optional
        Ground-truth classification labels for computing classification task loss.

    Returns:
    --------
    dict with keys:
        'classification' : torch.Tensor
            Logits for the classification task.
        'sentiment' : torch.Tensor
            Logits for the sentiment task.
        'loss' : torch.Tensor
            Combined or individual task loss, depending on available labels.
    """
    def __init__(self, a_class_num, b_class_num):
        super(MultiTaskLearning_ClassificationandSentiment, self).__init__()
        # shared base encoder for both tasks
        self.shared_encoder = SentenceTransformerModel_custom('nlptown/bert-base-multilingual-uncased-sentiment')
        # output dim of the shared encoder
        hidden_size = self.shared_encoder.pooling.out_features

        # head for the classification task
        self.classification_task = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, a_class_num))

        # head for the sentiment task
        self.sentiment_task = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, b_class_num))

    def forward(self, input_ids, attention_mask = None, sentiment_labels = None, classification_labels = None):
        shared_embeddings = self.shared_encoder(input_ids, attention_mask)

        classification_logits = self.classification_task(shared_embeddings)
        sentiment_logits = self.sentiment_task(shared_embeddings)

        loss = 0
        num_losses = 0

        # If either of the task specific heads are frozen (so the task_labels are None)
        # then train normally for only one of them -> adjust loss fn accordingly.

        if classification_labels is not None:
            classification_loss = nn.CrossEntropyLoss()(classification_logits, classification_labels)
            loss = loss + classification_loss
            num_losses += 1

        if sentiment_labels is not None:
            sentiment_loss = nn.CrossEntropyLoss()(sentiment_logits, sentiment_labels)
            loss = loss + sentiment_loss
            num_losses += 1

        # LOSS = 1/2(class_loss + sent_loss), adjusted appropriately if either is None
        # for task-specific training.
        if num_losses > 0:
            loss = loss / num_losses


        outputs = {
        'classification': classification_logits,
        'sentiment': sentiment_logits,
        'loss': loss}

        return outputs
