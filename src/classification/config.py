import json


class Config:

    def __init__(self, embedding_dim=600, num_layers=3, dropout=0.2, lr=0.001,
                 num_terms=1000, batch_size=24, num_epochs=10, num_classe=None,
                 vocab_size=None, num_classes=None, patience=5,
                 artifacts_path='../data/output/'):

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.num_terms = num_terms
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_classe = num_classe
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.patience = patience
        self.artifacts_path = artifacts_path

    def get_config_dict(self):

        config_dict = {
            'embedding_dim': self.embedding_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'lr': self.lr,
            'num_terms': self.num_terms,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'num_classe': self.num_classe,
            'vocab_size': self.vocab_size,
            'num_classes': self.num_classes,
            'patience': self.patience
        }

        return config_dict

    def load_config(self, path):

        self.artifacts_path = path

        with open(self.artifacts_path + 'config.json', 'r') as config_file:
            config_dict = json.load(config_file)

        self.embedding_dim = config_dict['embedding_dim']
        self.num_layers = config_dict['num_layers']
        self.dropout = config_dict['dropout']
        self.lr = config_dict['lr']
        self.num_terms = config_dict['num_terms']
        self.batch_size = config_dict['batch_size']
        self.num_epochs = config_dict['num_epochs']
        self.num_classe = config_dict['num_classe']
        self.vocab_size = config_dict['vocab_size']
        self.num_classes = config_dict['num_classes']
        self.patience = config_dict['patience']

    def save_config(self, path):

        config_dict = self.get_config_dict()

        with open(self.artifacts_path + 'config.json', 'w') as config_file:
            json.dump(config_dict, config_file)
