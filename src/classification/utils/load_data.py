import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import spacy
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split

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
