import numpy as np
import torch

class load_data(torch.utils.data.Dataset):
    """
    Classe auxiliar para utilização do DataLoader
    """
    def __init__(self, X, Y):
        self.X = X
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # word_embedding, label, document length, city, doc_id
        return torch.from_numpy(self.X[idx][-1][0].astype(np.int32)), self.y[idx], self.X[idx][-1][1], self.X[idx][0], self.X[idx][1]


class load_text_data(torch.utils.data.Dataset):
    """
    Classe auxiliar para utilização do DataLoader
    """
    def __init__(self, X, Y):
        self.X = X
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # document, label, city, doc_id
        return self.X[idx][-1], self.y[idx], self.X[idx][0], self.X[idx][1]
