from collections import Counter
from functools import partial
import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import spacy
import seaborn as sns
from tqdm.autonotebook import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split

from .config import Config
from .transformer import Transformer
from classification.utils.utils import (
    set_seed,
    split_data_by_city_and_class
)
from classification.utils.load_data import (
    load_text_data
)
from classification.evaluate.evaluate import (
    calculate_metrics
)

sns.set(rc={'figure.figsize': (10, 10)})


class DocumentClassification:

    def __init__(self, input_data, config=None, seed=42, device=None):

        if config is not None:
            self.config = Config(**config)
        else:
            self.config = Config()

        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"

        self.seed = seed
        set_seed(self.seed)

        self.config.num_classes = input_data['label'].nunique()
        self.best_model = None

        self.model = Transformer(
            model_name_or_path = self.config.model_name_or_path,
            max_seq_length = self.config.max_seq_length,
            num_classes = self.config.num_classes,
            model_args = self.config.model_args,
            tokenizer_args = self.config.tokenizer_args,
            do_lower_case = self.config.do_lower_case,
            pooling_mode = self.config.pooling_mode
        )

        print("Encoding data...")
        self.encode_data(input_data)

        print("Creating data loaders...")
        self.data_loaders = self.data_load()

    def encode_sentence(self, text):

        tokenized = self.model.tokenize([text])
        tokenized['input_ids'] = torch.Tensor(tokenized['input_ids'].tolist()[0])
        tokenized['token_type_ids'] = torch.Tensor(tokenized['token_type_ids'].tolist()[0])
        tokenized['attention_mask'] = torch.Tensor(tokenized['attention_mask'].tolist()[0])
        return tokenized

    def split_data(self, fold):
        df_data = self.df_data
        X = df_data.loc[df_data['fold'] == fold, [
            "city", "doc_id", "four_pages_encoded"]].values
        y = df_data.loc[df_data['fold'] == fold, 'label_int'].values

        return X, y

    def encode_data(self, df_data):

        # encoding
        df_data['four_pages_encoded'] = None
        df_data.loc[df_data['fold'] == 'train', 'four_pages_encoded'] = \
            df_data.loc[df_data['fold'] == 'train', 'four_pages_processed'].apply(
                lambda x: np.array(self.encode_sentence(x)))
        df_data.loc[df_data['fold'] == 'val', 'four_pages_encoded'] = \
            df_data.loc[df_data['fold'] == 'val', 'four_pages_processed'].apply(
                lambda x: np.array(self.encode_sentence(x)))
        df_data.loc[df_data['fold'] == 'test', 'four_pages_encoded'] = \
            df_data.loc[df_data['fold'] == 'test', 'four_pages_processed'].apply(
                lambda x: np.array(self.encode_sentence(x)))

        self.df_data = df_data

    def data_load(self):

        # Split data
        X_train, y_train = self.split_data("train")
        X_val, y_val = self.split_data("val")
        X_test, y_test = self.split_data("test")

        # Load dataset
        train_set = load_text_data(X_train, y_train)
        val_set = load_text_data(X_val, y_val)
        test_set = load_text_data(X_test, y_test)

        data_loaders = dict()

        data_loaders["train"] = torch.utils.data.DataLoader(
            train_set,
            batch_size=int(self.config.batch_size),
            shuffle=True,
            num_workers=8)
        data_loaders["val"] = torch.utils.data.DataLoader(
            val_set,
            batch_size=int(self.config.batch_size),
            shuffle=False,
            num_workers=8)
        data_loaders["test"] = torch.utils.data.DataLoader(
            test_set,
            batch_size=int(self.config.batch_size),
            shuffle=False,
            num_workers=8)

        return data_loaders

    def train_model(self, show_progress_bar=True):

        checkpoint_dir = self.config.artifacts_path
        data_loaders = self.data_loaders

        if not self.config.patience:
            self.config.patience = self.config.num_epochs
        patience_counter = 0

        train_loader = data_loaders["train"]
        val_loader = data_loaders["val"]

        self.model.to(self.device)

        self.best_model = self.model
        best_loss = float("inf")
        #best_macro = 0.0

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)

        """if checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(checkpoint_dir, "checkpoint"))
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)"""

        steps_per_epoch = len(train_loader)
        data_iterator = iter(train_loader)
        # loop over the dataset multiple times
        for epoch in trange(self.config.num_epochs, desc="Epoch", disable=not show_progress_bar):
            print("=" * 20, "Epoch: {}".format(epoch + 1), "=" * 20)
            self.model.train()
            training_loss = 0.0
            epoch_steps = 0
            y_pred = []
            y_true = []
            for i in trange(steps_per_epoch, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):

                try:
                    data = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(train_loader)
                    data = next(data_iterator)

                # get the inputs; data is a list of [inputs, labels]
                inputs, labels, _, _ = data
                labels = labels.long().to(self.device)
                inputs['input_ids'] = inputs['input_ids'].long().to(self.device)
                inputs['token_type_ids'] = inputs['token_type_ids'].long().to(self.device)
                inputs['attention_mask'] = inputs['attention_mask'].long().to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                epoch_steps += 1
                training_loss += loss.cpu().detach().numpy()

                _, predicted = torch.max(outputs.data, 1)

                y_pred.extend(predicted.cpu().tolist())
                y_true.extend(labels.cpu().tolist())

            train_metrics = [(training_loss / epoch_steps)]
            train_metrics.extend(calculate_metrics(y_true, y_pred))

            print("Train:\n loss %.3f, accuracy %.3f, F1-Macro %.3f, F1-Weighted %.3f" % (
                train_metrics[0], train_metrics[1], train_metrics[2], train_metrics[3]))

            _, val_metrics = self.eval_model(self.model, "val", probability=True)
            print("Val:\n loss %.3f, accuracy %.3f, F1-Macro %.3f, F1-Weighted %.3f \n" % (
                val_metrics[0], val_metrics[1], val_metrics[2], val_metrics[3]))

            if val_metrics[0] < best_loss - 0.001:
                best_loss = val_metrics[0]
                best_model = copy.deepcopy(self.model)
                best_macro = val_metrics[2]
                patience_counter = 0
                # path = os.path.join(checkpoint_dir, "model_setup_1-2.pth")
                # torch.save((model.state_dict(), optimizer.state_dict()), path)
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    print("Model training was stopped early")
                    break

        self.best_model = best_model
        print("Finished Training")
        return best_model

    def eval_model(self, model=None, fold=None, labels_dict=None, probability=False):

        if model is None:
            self.model = self.best_model
        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        loader = self.data_loaders[fold]
        steps = 0
        sum_loss = 0.0
        y_true = []
        y_pred = []
        cities = []
        docs = []
        probabilities = []
        with torch.no_grad():
            for data in loader:
                if fold:
                    inputs, labels, city, doc_id = data
                    cities.extend(city)
                    docs.extend(doc_id)
                else:
                    inputs, labels, _, _ = data
                inputs, labels = inputs, labels.long().to(self.device)
                inputs['input_ids'] = inputs['input_ids'].long().to(self.device)
                inputs['token_type_ids'] = inputs['token_type_ids'].long().to(self.device)
                inputs['attention_mask'] = inputs['attention_mask'].long().to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)

                steps += 1
                sum_loss += loss.cpu().detach().numpy()

                y_pred.extend(predicted.cpu().tolist())
                y_true.extend(labels.cpu().tolist())

                prob = F.softmax(outputs, dim=1).cpu().detach().numpy()
                probabilities.extend(prob)

        metrics = [(sum_loss / steps)]
        metrics.extend(calculate_metrics(y_true, y_pred))
        df_predictions = pd.DataFrame()
        if fold:
            if probability:
                df_predictions = pd.DataFrame(probabilities)
                df_predictions['doc_id'] = docs
                df_predictions['city'] = cities
                df_predictions['label'] = y_true
                df_predictions['y_pred'] = y_pred
                df_predictions['fold'] = fold
            else:
                df_predictions = pd.DataFrame(
                    {"doc_id": docs, "city": cities, "label": y_true, "pred": y_pred})
                df_predictions['fold'] = fold
            # df_predictions.to_csv("./transformer_data/results/setup_1-2/setup_1-2_{}.csv".format(fold, index=False))
            # Plot confusion matrix
            # plot_confusion_matrix(y_pred, y_true, fold, labels_dict)
        return df_predictions, metrics
