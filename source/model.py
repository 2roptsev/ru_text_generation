from typing import Optional, Dict

import numpy as np
import torch
import torch.nn as nn


class PretrainedSpacyEmbedding(nn.Module):
    def __init__(self, nlp, index2word: Dict[int, str], trainable: bool = False):
        super(PretrainedSpacyEmbedding, self).__init__()
        self.trainable = trainable
        embeddings = []
        for index, word in sorted(index2word.items(), key=lambda item: item[0]):
            vector = nlp(word)[0].vector
            embeddings.append(vector)
        
        embeddings = torch.tensor(np.array(embeddings))
        self.embedding = nn.Embedding.from_pretrained(embeddings)
        self.num_embeddings = self.embedding.num_embeddings
        self.embedding_dim = self.embedding.embedding_dim

    def enable_train(self):
        self.trainable = True

    def freeze(self):
        self.trainable = False

    def __call__(self, x):
        if self.trainable:
            return self.embedding(x)
        else:
            with torch.no_grad():
                return self.embedding(x)


class RNNModel(nn.Module):
    """Recurrent neural network model based on LSTM layer. 
        
    For input of shape (batch_size, sequence_size) 
    returns (batch_size, sequence_size, vocab_size) log-probabilities.
    """
    
    def __init__(self,
                 embedding_layer: nn.Module,
                 hidden_dim: int = 128,
                 lstm_layers: int = 2,
                 dropout: Optional[float] = 0.1):
        """Args:
            embedding_layer (nn.Module): embedding layer 
                which has num_embeddings and embedding_dim attributes
            hidden_dim (int): size of LSTM hidden dimension
            lstm_layers (int): number of LSRM layers
            dropout (float): dropout ratio
        """
        super(RNNModel, self).__init__()
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.num_embeddings = embedding_layer.num_embeddings
        self.embedding_dim = embedding_layer.embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding = embedding_layer
        
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True
        )
        self.dropout = nn.Dropout(p=0.1)
        self.index_projection = nn.Linear(
            in_features=self.hidden_dim,
            out_features=self.num_embeddings
        )
        self.softmax = nn.LogSoftmax(dim=-1)

    def __call__(self, x):
        embs = self.embedding(x)
        lstm_processed, _ = self.lstm(embs)
        lstm_processed = self.dropout(lstm_processed)
        out_word_indices = self.index_projection(lstm_processed)
        out_word_probas = self.softmax(out_word_indices)
        return out_word_probas
    
    def step(self, x, state=None):
        """Class method similar to __call__ but handling hidden LSTM state."""
        embs = self.embedding(x)
        lstm_processed, state = self.lstm(embs, state)         
        out_word_indices = self.index_projection(lstm_processed)
        out_word_probas = self.softmax(out_word_indices)
        return out_word_probas, state
