import torch
from torch import nn
from typing import List
import os
import json


class LSTM(nn.Module):
    """
    Bidirectional LSTM running over word embeddings.
    """
    def __init__(self, embedding_matrix, word_embedding_dimension: int, hidden_dim: int, num_layers: int = 1, num_classes: int = 13, vocab_size: int = 0, dropout: float = 0, bidirectional: bool = True):
        nn.Module.__init__(self)
        self.config_keys = ['word_embedding_dimension', 'hidden_dim', 'num_layers', 'dropout', 'bidirectional']
        self.word_embedding_dimension = word_embedding_dimension
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.vocab_size = vocab_size

        self.embeddings_dimension = hidden_dim
        if self.bidirectional:
            self.embeddings_dimension *= 2

        self.embeddings = nn.Embedding(vocab_size, word_embedding_dimension, padding_idx=0)
        self.embeddings.weight=nn.Parameter(torch.tensor(embedding_matrix,dtype=torch.float32))
        self.embeddings.weight.requires_grad=False
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.LSTM(word_embedding_dimension, int(word_embedding_dimension/2), num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(int(word_embedding_dimension/2), num_classes)

    def forward(self, features, sentence_length):
        x = self.embeddings(features)
        x = self.dropout(x)
        out_pack, (ht, ct) = self.encoder(x)
        ht[-1] = self.dropout(ht[-1])
        out = self.linear(ht[-1])
        return out

    def get_word_embedding_dimension(self) -> int:
        return self.embeddings_dimension

    def tokenize(self, text: str) -> List[int]:
        raise NotImplementedError()

    def save(self, output_path: str):
        with open(os.path.join(output_path, 'lstm_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

        torch.save(self.state_dict(), os.path.join(output_path, 'pytorch_model.bin'))

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    @staticmethod
    def load(input_path: str):
        with open(os.path.join(input_path, 'lstm_config.json'), 'r') as fIn:
            config = json.load(fIn)

        weights = torch.load(os.path.join(input_path, 'pytorch_model.bin'))
        model = LSTM(**config)
        model.load_state_dict(weights)
        return model
