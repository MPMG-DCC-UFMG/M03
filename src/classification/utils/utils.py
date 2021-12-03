import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from wordcloud import WordCloud
from tqdm.notebook import tqdm
import spacy
from .input_example import InputExample


def split_stratified_into_train_val_test(df_input, stratify_colname='label',
                                         frac_train=0.7, frac_val=0.2, frac_test=0.1,
                                         random_state=42):
    '''
    Source: https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test
    Splits a Pandas dataframe into three subsets (train, val, and test)
    following fractional ratios provided by the user, where each subset is
    stratified by the values in a specific column (that is, each subset has
    the same relative frequency of the values in the column). It performs this
    splitting by running train_test_split() twice.

    Parameters
    ----------
    df_input : Pandas dataframe
        Input dataframe to be split.
    stratify_colname : str
        The name of the column that will be used for stratification. Usually
        this column would be for the label.
    frac_train : float
    frac_val   : float
    frac_test  : float
        The ratios with which the dataframe will be split into train, val, and
        test data. The values should be expressed as float fractions and should
        sum to 1.0.
    random_state : int, None, or RandomStateInstance
        Value to be passed to train_test_split().

    Returns
    -------
    df_train, df_val, df_test :
        Dataframes containing the three splits.
    '''
    print(round(frac_train + frac_val + frac_test))
    if round(frac_train + frac_val + frac_test) != 1:
        raise ValueError('fractions %f, %f, %f do not add up to 1.0' %
                         (frac_train, frac_val, frac_test))

    if stratify_colname not in df_input.columns:
        raise ValueError('%s is not a column in the dataframe' %
                         (stratify_colname))

    X = df_input  # Contains all columns.
    # Dataframe of just the column on which to stratify.
    y = df_input[[stratify_colname]]

    # Split original dataframe into train and temp dataframes.
    df_train, df_temp, y_train, y_temp = train_test_split(X,
                                                          y,
                                                          stratify=y,
                                                          test_size=(
                                                              1.0 - frac_train),
                                                          random_state=random_state)

    # Split the temp dataframe into val and test dataframes.
    relative_frac_test = frac_test / (frac_val + frac_test)
    df_val, df_test, y_val, y_test = train_test_split(df_temp,
                                                      y_temp,
                                                      stratify=y_temp,
                                                      test_size=relative_frac_test,
                                                      random_state=random_state)

    assert len(df_input) == len(df_train) + len(df_val) + len(df_test)

    return df_train, df_val, df_test, y_train, y_val, y_test


def split_data_by_city_and_class(df):
    # criamos uma nova classe cidade+label
    df["multi_label"] = df["label"] + "_" + df["city"]
    # verificamos se existem classes que aparecem apenas 1 vez em algum município
    df_multilabel_size = df.groupby(
        "multi_label").size().reset_index(name="size")
    # selecionamos esses casos
    unique_classes = df_multilabel_size.loc[df_multilabel_size["size"]
                                            == 1, 'multi_label'].values
    # definimos todos como treino
    df["fold"] = "train"
    # "classes unicas" vão pro conjunto de validação
    df.loc[(df["multi_label"].isin(unique_classes)) &
           (df["fold"] == "train"), "fold"] = "val"

    num_docs = df.shape[0]
    test_size = round(0.1 * num_docs)
    val_size = round(0.2 * num_docs)

    # fazemos o split considerando a multi_label
    train, test, _, _ = train_test_split(
        df.loc[df["fold"] == "train"],
        df.loc[df["fold"] == "train", "multi_label"],
        stratify=df.loc[df["fold"] == "train", "multi_label"],
        test_size=test_size,
        random_state=SEED)
    # define os docs que vão pra validação
    df.loc[test.index, "fold"] = "test"

    # fazemos o split considerando a multi_label
    train, val, _, _ = train_test_split(
        df.loc[df["fold"] == "train"],
        df.loc[df["fold"] == "train", "multi_label"],
        stratify=df.loc[df["fold"] == "train", "multi_label"],
        test_size=val_size,
        random_state=SEED)
    # define os docs que vão pra validação
    df.loc[val.index, "fold"] = "val"

    return df


def confunsion_matrix_chart(y_true_str, y_pred_str, fold, eval_type, normalize):

    class_labels = list(set(itertools.chain(y_true_str["label"], y_pred_str)))
    cm = confusion_matrix(
        y_true_str["label"], y_pred_str, normalize="true", labels=class_labels)
    df_cm = pd.DataFrame(cm, columns=class_labels, index=class_labels)
    df_cm.index.name = 'Classe Verdadeira'
    df_cm.columns.name = 'Classe Estimada'

    plt.figure(figsize=(15, 10))
    plt.title(f'Matriz de Confusão Normalizada - {eval_type} ({fold})', fontsize=20) if normalize == 'true' else plt.title(
        f'Matriz de Confusão - {eval_type} ({fold})', fontsize=20)
    sns_plot = sns.heatmap(df_cm, cmap='flare', annot=True, annot_kws={
                           "size": 16}, linewidths=.5, fmt='.2f')
    # x-axis label with fontsize 15
    sns_plot.set_xlabel('Classe Predita', fontsize=15)
    # y-axis label with fontsize 15
    sns_plot.set_ylabel('Classe Verdadeira', fontsize=15)
    plt.show()


def evaluation_models(y_true, y_pred, fold='', eval_type='Completo', normalize='true'):
    print('=' * 80)
    print(f"{fold}:")
    print('_' * 80)
    print("F1_score macro:", f1_score(y_true, y_pred, average='macro'))
    print("F1_score weighted:", f1_score(y_true, y_pred, average='weighted'))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print('_' * 80)

    confunsion_matrix_chart(y_true, y_pred, fold, eval_type, normalize)


def plot_confusion_matrix(y_pred, y_true, fold, labels_dict):
    ticks = copy.deepcopy(y_pred)
    ticks.extend(y_true)
    print(set(ticks))
    print(labels_dict)
    ticks = [labels_dict[tick] for tick in set(ticks)]

    cm = sns.heatmap(
        confusion_matrix(y_true, y_pred, normalize="true"),
        annot=True,
        center=0,
        vmin=0,
        vmax=1,
        square=True,
        fmt=".2f",
        xticklabels=ticks,
        yticklabels=ticks,
    )
    plt.yticks(rotation=0)
    img_name = "Confusion Matrix - {}".format(fold)
    plt.title(img_name)
    plt.savefig(
        "./lstm_data/results/setup_1-2/img/confusion_matrix_{}.png".format(fold))
    plt.show()


def set_seed(seed):
    """
    Fixa semente aleatória para garantir que os resultados possam ser reproduzidos
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
