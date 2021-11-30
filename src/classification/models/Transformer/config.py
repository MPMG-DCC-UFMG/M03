import json


class Config:

    def __init__(self, model_name_or_path="neuralmind/bert-base-portuguese-cased",
                 max_seq_length=1000, num_classes=13, model_args={}, tokenizer_args={},
                 do_lower_case=False, pooling_mode=None, lr=0.001, batch_size=24,
                 num_epochs=10, num_classes=None, patience=5,
                 artifacts_path='../data/output/'):

        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.num_classes = num_classes
        self.model_args = model_args
        self.tokenizer_args = tokenizer_args
        self.do_lower_case = do_lower_case
        self.pooling_mode = pooling_mode
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.artifacts_path = artifacts_path

    def get_config_dict(self):

        config_dict = {
            'model_name_or_path': self.model_name_or_path,
            'max_seq_length': self.max_seq_length,
            'num_classes': self.num_classes,
            'model_args': self.model_args,
            'tokenizer_args': self.tokenizer_args,
            'do_lower_case': self.do_lower_case,
            'pooling_mode': self.pooling_mode,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'patience': self.patience
        }

        return config_dict

    def load_config(self, path):

        self.artifacts_path = path

        with open(self.artifacts_path + 'config.json', 'r') as config_file:
            config_dict = json.load(config_file)

        self.model_name_or_path = config_dict['model_name_or_path']
        self.max_seq_length = config_dict['max_seq_length']
        self.num_classes = config_dict['num_classes']
        self.model_args = config_dict['model_args']
        self.tokenizer_args = config_dict['tokenizer_args']
        self.do_lower_case = config_dict['do_lower_case']
        self.pooling_mode = config_dict['pooling_mode']
        self.lr = config_dict['lr']
        self.batch_size = config_dict['batch_size']
        self.num_epochs = config_dict['num_epochs']
        self.patience = config_dict['patience']

    def save_config(self, path):

        config_dict = self.get_config_dict()

        with open(self.artifacts_path + 'config.json', 'w') as config_file:
            json.dump(config_dict, config_file)
