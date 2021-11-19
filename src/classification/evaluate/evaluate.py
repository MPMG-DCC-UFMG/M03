import os
import glob
import json
import re
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.facecolor':'white'})

from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import f1_score, accuracy_score, classification_report
from tqdm.notebook import tqdm


def create_res_diresctory(setup='teste'):
    # Directory
    directory = "model_results"
    # Parent Directory path
    parent_dir = f"./lstm_data/results/setup_{setup}/"
    # Final path
    path = os.path.join(parent_dir, directory)
    # Create the directory
    try:
        os.mkdir(path)
    except OSError as error:
        print(error)

    return path

def grouper(n, iterable, fillvalue=None):
    args = [iter(iterable)] * n
    return itertools.zip_longest(fillvalue=fillvalue, *args)

def convert_fold(fold='train'):
    fold = fold.lower()
    if fold == 'train':
        return 'Treino'
    elif fold == 'test':
        return 'Teste'
    elif fold == 'val':
        return 'Validação'
    return fold

def evaluation_metrics(y_true, y_pred, fold='', eval_type='Geral', city_t=''):
    results = {
        'Metric': ["F1_score macro", "F1_score weighted", "Accuracy"],
        "Score": [f1_score(y_true, y_pred, average='macro'), f1_score(y_true, y_pred, average='weighted'), accuracy_score(y_true, y_pred)],
        "Fold": [fold] * 3,
        "Avaliação": [eval_type] * 3,
        "Município": [city_t] * 3
    }
    df_results = pd.DataFrame(data=results)
    return df_results

def plot_cm(y_true, y_pred, figsize=(15,10), normalize=None, labels=None, cmap='flare', fold='', eval_type='Geral', result_path='./'):
    cm = confusion_matrix(y_true, y_pred, normalize=normalize, labels=labels)
    df_cm = pd.DataFrame(cm, columns=labels, index=labels)
#     display(df_cm)
    df_cm.index.name = 'Classe Verdadeira'
    df_cm.columns.name = 'Classe Estimada'

    fig = plt.figure(figsize = figsize)
    plt.title(f'Matriz de Confusão Normalizada - {eval_type} ({fold})', fontsize = 20) if normalize=='true' else  plt.title(f'Matriz de Confusão - {eval_type} ({fold})', fontsize = 20)
    sns_plot = sns.heatmap(df_cm, cmap=cmap, annot=True, annot_kws={"size": 16}, linewidths=.5, fmt='.2f') # font size
    sns_plot.set_xlabel('Classe Predita', fontsize = 15) # x-axis label with fontsize 15
    sns_plot.set_ylabel('Classe Verdadeira', fontsize = 15) # y-axis label with fontsize 15
    plt.tight_layout()
    if normalize == 'true':
        plt.savefig(f'{result_path}Confusion_Matrix_{fold}_{eval_type}.png')
    else:
        plt.savefig(f'{result_path}Confusion_Matrix_{fold}_{eval_type}_regular.png')

    plt.close(fig)
#     plt.show()

def map_label(df, label_dict, lbl_column_name='label', pred_column_name='pred'):
    labels_name = [label_dict[l] for l in df[lbl_column_name]]
    preds_name = [label_dict[l] for l in df[pred_column_name]]
    return [labels_name, preds_name]

def read_result_files(setup_root='teste', setup_leaf='teste'):
    # Gerando o caminho para os arquivos de resultados
    result_files_path = f"./lstm_data/results/setup_{setup_root}/setup_{setup_leaf}*.csv"
    all_filenames = [i for i in glob.glob(result_files_path)]
#     print(all_filenames)

    # Lendo os arquivos de resultados
    return [pd.read_csv(f) for f in all_filenames]

def evaluation_models(df_1, df_2, df_3, eval_type='general', result_path='./', normalize='true'):
    # Análise Geral
    if eval_type == 'general':
        # labels
        df_1_labels = set(itertools.chain(df_1['label_name'], df_1['pred_name']))
        df_2_labels = set(itertools.chain(df_2['label_name'], df_2['pred_name']))
        df_3_labels = set(itertools.chain(df_3['label_name'], df_3['pred_name']))
        #print(df_1_labels, df_2_labels, df_3_labels)

        # Métricas de avaliação
        df1 = evaluation_metrics(y_true=df_1['label'], y_pred=df_1['pred'], fold=convert_fold(df_1['fold'].iloc[0]), eval_type='Geral')
        df2 = evaluation_metrics(y_true=df_2['label'], y_pred=df_2['pred'], fold=convert_fold(df_2['fold'].iloc[0]), eval_type='Geral')
        df3 = evaluation_metrics(y_true=df_3['label'], y_pred=df_3['pred'], fold=convert_fold(df_3['fold'].iloc[0]), eval_type='Geral')
        pd.concat([df1, df2, df3], ignore_index=True).to_csv(f"{result_path}evaluation_metrics_general.csv", index=False)

        # Matriz de confusão
        plot_cm(y_true=df_1['label_name'], y_pred=df_1['pred_name'], figsize=(15,10), normalize=normalize, labels=list(df_1_labels), eval_type='Geral', fold=convert_fold(df_1['fold'].iloc[0]), result_path=result_path)
        plot_cm(y_true=df_2['label_name'], y_pred=df_2['pred_name'], figsize=(15,10), normalize=normalize, labels=list(df_2_labels), eval_type='Geral', fold=convert_fold(df_2['fold'].iloc[0]), result_path=result_path)
        plot_cm(y_true=df_3['label_name'], y_pred=df_3['pred_name'], figsize=(15,10), normalize=normalize, labels=list(df_3_labels), eval_type='Geral', fold=convert_fold(df_3['fold'].iloc[0]), result_path=result_path)

    # Análise do setup 4
    elif eval_type == 'setup_4':
        # labels
        df_1_labels = set(itertools.chain(df_1['label_name'], df_1['pred_name']))
        df_2_labels = set(itertools.chain(df_2['label_name'], df_2['pred_name']))
        df_3_labels = set(itertools.chain(df_3['label_name'], df_3['pred_name']))
        #print(df_1_labels, df_2_labels, df_3_labels)

        # Métricas de avaliação
        df1 = evaluation_metrics(y_true=df_1['label'], y_pred=df_1['pred'], fold=convert_fold(df_1['fold'].iloc[0]), eval_type=df_1['partition'].iloc[0], city_t=df_1['city_t'].iloc[0])
        df2 = evaluation_metrics(y_true=df_2['label'], y_pred=df_2['pred'], fold=convert_fold(df_2['fold'].iloc[0]), eval_type=df_2['partition'].iloc[0], city_t=df_2['city_t'].iloc[0])
        df3 = evaluation_metrics(y_true=df_3['label'], y_pred=df_3['pred'], fold=convert_fold(df_3['fold'].iloc[0]), eval_type=df_3['partition'].iloc[0], city_t=df_3['city_t'].iloc[0])
        pd.concat([df1, df2, df3], ignore_index=True).to_csv(f"{result_path}evaluation_metrics_general.csv", index=False)

        # Matriz de confusão
        plot_cm(y_true=df_1['label_name'], y_pred=df_1['pred_name'], figsize=(15,10), normalize=normalize, labels=list(df_1_labels), eval_type='Geral', fold=convert_fold(df_1['fold'].iloc[0]), result_path=result_path)
        plot_cm(y_true=df_2['label_name'], y_pred=df_2['pred_name'], figsize=(15,10), normalize=normalize, labels=list(df_2_labels), eval_type='Geral', fold=convert_fold(df_2['fold'].iloc[0]), result_path=result_path)
        plot_cm(y_true=df_3['label_name'], y_pred=df_3['pred_name'], figsize=(15,10), normalize=normalize, labels=list(df_3_labels), eval_type='Geral', fold=convert_fold(df_3['fold'].iloc[0]), result_path=result_path)

    # Análise por município
    elif eval_type == 'by_city':
        # pegar o conjunto de municípios
        cities = list(df_1['city'].unique())
        for city in cities:
            print(city)
            # filtrar por município
            df_1_filtred = df_1[df_1['city'] == city]
            df_2_filtred = df_2[df_2['city'] == city]
            df_3_filtred = df_3[df_3['city'] == city]

            # labels
            df_1_labels = set(itertools.chain(df_1_filtred['label_name'], df_1_filtred['pred_name']))
            df_2_labels = set(itertools.chain(df_2_filtred['label_name'], df_2_filtred['pred_name']))
            df_3_labels = set(itertools.chain(df_3_filtred['label_name'], df_3_filtred['pred_name']))
            #print(df_1_labels, df_2_labels, df_3_labels)

            # Métricas de avaliação
            df1 = evaluation_metrics(y_true=df_1_filtred['label'], y_pred=df_1_filtred['pred'], fold=convert_fold(df_1_filtred['fold'].iloc[0]), eval_type=city)
            df2 = evaluation_metrics(y_true=df_2_filtred['label'], y_pred=df_2_filtred['pred'], fold=convert_fold(df_2_filtred['fold'].iloc[0]), eval_type=city)
            df3 = evaluation_metrics(y_true=df_3_filtred['label'], y_pred=df_3_filtred['pred'], fold=convert_fold(df_3_filtred['fold'].iloc[0]), eval_type=city)
            pd.concat([df1, df2, df3], ignore_index=True).to_csv(f"{result_path}evaluation_metrics_{city}.csv", index=False)

            # Matriz de confusão
            plot_cm(y_true=df_1_filtred['label_name'], y_pred=df_1_filtred['pred_name'], figsize=(15,10), normalize=normalize, labels=list(df_1_labels), eval_type=city, fold=convert_fold(df_1_filtred['fold'].iloc[0]), result_path=result_path)
            plot_cm(y_true=df_2_filtred['label_name'], y_pred=df_2_filtred['pred_name'], figsize=(15,10), normalize=normalize, labels=list(df_2_labels), eval_type=city, fold=convert_fold(df_2_filtred['fold'].iloc[0]), result_path=result_path)
            plot_cm(y_true=df_3_filtred['label_name'], y_pred=df_3_filtred['pred_name'], figsize=(15,10), normalize=normalize, labels=list(df_3_labels), eval_type=city, fold=convert_fold(df_3_filtred['fold'].iloc[0]), result_path=result_path)

            # Classification report
            report_1 = classification_report(df_1_filtred['label_name'], y_pred=df_1_filtred['pred_name'], output_dict=True)
            report_2 = classification_report(df_2_filtred['label_name'], y_pred=df_2_filtred['pred_name'], output_dict=True)
            report_3 = classification_report(df_3_filtred['label_name'], y_pred=df_3_filtred['pred_name'], output_dict=True)
            df_report_1 = pd.DataFrame(report_1).transpose()
            df_report_2 = pd.DataFrame(report_2).transpose()
            df_report_3 = pd.DataFrame(report_3).transpose()
            df_report_1['fold'], df_report_2['fold'], df_report_3['fold'] = convert_fold(df_1_filtred['fold'].iloc[0]), convert_fold(df_2_filtred['fold'].iloc[0]), convert_fold(df_3_filtred['fold'].iloc[0])
            df_report_1['city'], df_report_2['city'], df_report_3['city'] = city, city, city
            pd.concat([df_report_1, df_report_2, df_report_3]).to_csv(f"{result_path}classification_report_{city}.csv")

            #display(df_report_1.head())
            #display(df_report_2.head())
            #display(df_report_3.head())

            print()
    else:
        print('Tipo de avaliação incorreto! Opções válidas: general e by_type')
