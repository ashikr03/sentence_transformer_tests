import transformers
import sentence_transformers
import torch
import torch.nn as nn
import numpy as np
import datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SentenceTransformerModel_custom(nn.Module):
    """
        Custom model for generating sentence embeddings using a pretrained transformer.

        Architecture:
        -------------
        - Encoder: Pretrained transformer model (e.g., BERT) from Hugging Face.
        - Attention-Weighted Pooling: Softmax-weighted sum over token embeddings (excluding [CLS]).
        - Projection: Linear layer to reduce to output_dim.
        - Normalization: L2-normalizes the output embeddings.

        Parameters:
        -----------
        model_name : str
            Pretrained transformer model name or path.
        output_dim : int, optional (default=384)
            Output embedding dimension.

        Inputs:
        -------
        input_ids : torch.Tensor
            Tokenized input IDs (batch_size, seq_len).
        attention_mask : torch.Tensor
            Attention mask (batch_size, seq_len).

        Returns:
        --------
        torch.Tensor
            L2-normalized sentence embeddings (batch_size, output_dim).
    """

    def __init__(self, model_name, output_dim=384):
        super(SentenceTransformerModel_custom, self).__init__()
        self.model = transformers.AutoModel.from_pretrained(model_name)
        hidden_size = self.model.config.hidden_size
        self.pooling = nn.Linear(hidden_size, output_dim)
        self.normalize = nn.functional.normalize
        self.weights = nn.Parameter(torch.randn(hidden_size))

    def forward(self, input_ids, attention_mask):

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state[:, 1:, :]
        softmaxed_embeddings = torch.nn.functional.softmax(token_embeddings, dim=1)
        weighted_sum = torch.sum(token_embeddings * softmaxed_embeddings, dim=1)
        pooled_embeddings = self.pooling(weighted_sum)
        normalized_embeddings = nn.functional.normalize(pooled_embeddings, p=2, dim=1)
        return normalized_embeddings

custom_model = SentenceTransformerModel_custom('nlptown/bert-base-multilingual-uncased-sentiment')
custom_model.to(device)
tokenizer = transformers.AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

#tokenizing the sample sentences such that they can be processed by the transformer
inputs = tokenizer(input_sentences, padding=True, truncation=True, return_tensors="pt")
inputs = inputs.to(device)
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

#encoding
with torch.no_grad():
    custom_embeddings = custom_model(input_ids, attention_mask)
    print(custom_embeddings)
    print(custom_embeddings.shape)

#comparing similarities between my custom transformer and a basic one:
from sklearn.metrics.pairwise import cosine_similarity
generic_model = sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")
generic_model.to(device)

input_sentences = ["I hate this app.",
                   "I love how it works omg",
                   "it's alright, but i wish it was blue",
                   "i never will download this app.",
                   "i might have something else.",
                   "This is the best thing I've ever worked with."]
generic_embeddings = generic_model.encode(input_sentences)

print(generic_embeddings.shape)
print(custom_embeddings.shape)

if generic_embeddings.device != 'cpu':
    generic_embeddings = generic_embeddings.cpu().numpy()

if custom_embeddings.device != 'cpu':
    custom_embeddings = custom_embeddings.cpu().numpy()

# Compute cosine similarity, comparing the "basic" model to the custom model
generic_similarity = cosine_similarity([generic_embeddings[1]], [generic_embeddings[5]])
custom_similarity = cosine_similarity([custom_embeddings[1]], [custom_embeddings[5]])

print(f"Generic Model Similarity for first sentence: {generic_similarity[0][0]}")
print(f"Custom Model Similarity for first sentence: {custom_similarity[0][0]}")
