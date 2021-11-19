import json
import os
import re
import os.path as osp
import pandas as pd
import glob
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import time
import multiprocessing as mp

##Classe com vários métodos de pre-processamento de texto em português criado pelo grupo F03
import utils.preprocessing_portuguese as preprossPT
import utils.preprocessing_module as preprocess

SEED = 42


def limpeza_texto(page_text, city_name):
    txt_process = preprossPT.TextPreProcessing()
    city_name = city_name.replace("_", " ")

    page_text = txt_process.remove_person_names(page_text)

    page_text = page_text.lower()

    page_text = txt_process.remove_emails(page_text)

    page_text = txt_process.remove_urls(page_text)

    page_text = txt_process.remove_pronouns(page_text)

    page_text = txt_process.remove_adverbs(page_text)

    page_text = txt_process.remove_special_characters(page_text)

    page_text = txt_process.remove_accents(page_text)

    page_text = txt_process.remove_stopwords(page_text)

    page_text = txt_process.remove_hour(page_text)
    # split numbers from letters
    page_text = ' '.join(re.split('(\d+)',page_text))

    page_text = txt_process.remove_symbols_from_numbers(page_text)

    page_text = txt_process.remove_numbers(page_text)

    page_text = txt_process.remove_reduced_or_contracted_words(page_text)

    #Removendo letras sozinhas no texto
    #page_text = re.sub(r'(?:^| )\w(?:$| )', ' ', page_text).strip()
    page_text = re.sub(r"\b[a-zA-Z]\b", "", page_text)

    page_text = page_text.replace("_","")

    # remove nome do municipio e estado
    page_text = page_text.replace(city_name,"")

    words_to_remove = ["tel", "fax", "cnpj", "cpf", "mail", "cep", "estado", "minas gerais", "prefeitura", "municipal", "municipio"]
    for word in words_to_remove:
        page_text = page_text.replace(word,"")

    page_text = txt_process.remove_excessive_spaces(page_text)

    return page_text


def limpeza_texto_unit_entity(page_text):
    txt_process = preprossPT.TextPreProcessing()

    page_text = page_text.lower()

    page_text = txt_process.remove_entities(page_text)

    page_text = txt_process.remove_units(page_text)

    #Remover palavras com digitos
    page_text = ' '.join(w for w in page_text.split() if not any(x.isdigit() for x in w))

    page_text = txt_process.remove_person_names(page_text)

    page_text = txt_process.remove_emails(page_text)

    page_text = txt_process.remove_urls(page_text)

    page_text = txt_process.remove_pronouns(page_text)

    page_text = txt_process.remove_adverbs(page_text)

    page_text = txt_process.remove_special_characters(page_text)

    page_text = txt_process.remove_accents(page_text)

    page_text = txt_process.remove_stopwords(page_text)

    page_text = txt_process.remove_hour(page_text)

    # split numbers from letters
    #page_text = ' '.join(re.split('(\d+)',page_text))

    page_text = txt_process.remove_symbols_from_numbers(page_text)

    page_text = txt_process.remove_numbers(page_text)

    page_text = txt_process.remove_reduced_or_contracted_words(page_text)

    #Removendo letras sozinhas no texto
    page_text = re.sub(r"\b[a-zA-Z]\b", "", page_text)

    page_text = page_text.replace("_","")


    page_text = txt_process.remove_excessive_spaces(page_text)

    return page_text


def get_name(directory):
    return re.search("licitacoes-(.*)/", directory)[1].replace("-", "_")


def list_json_files_dir(city_dir, city_name):
    if city_name != 'itamarati':
        return glob.glob(os.path.join(city_dir, 'data', 'files_json', '*'))
    else:
        return glob.glob(os.path.join(city_dir, '*', 'data', 'files_json', '*'))


def read_files(file_dir):
    with open(file_dir) as f:
        lines = f.read() # lê o conteúdo (pode ser lido em um stream, se achar necessário)
        return json.loads(lines)


def preprocess_text(document, num_pages, city_name):
    return [preprocess.limpeza_texto(page_content, city_name, [], "complete-unity") for page_content in document['text_content'][:num_pages]]


def merge_pages(document, num_pages):
    # retorna lista onde a cada posicao uma nova pagina e concatenada ao texto
    num_pages+=1
    return [" ".join(document['text_preprocessed'][0:num_pages]) for num_pages in range(1,num_pages)]


def calc_real_fold_size(labels_df, folds_size=None):
    """
    Calculates the atual size of each fold.
    """
    if folds_size is None:
        folds_size = {
            'train' : [0.7],
            'val' : [0.2],
            'test' : [0.1],
        }
    num_docs = labels_df.shape[0]
    folds_size['train'].append(round(folds_size['train'][0] * num_docs))
    folds_size['test'].append(round(folds_size['test'][0] * num_docs))
    folds_size['val'].append(round(folds_size['val'][0] * num_docs))

    # check if number os docs match
    assert num_docs == (folds_size['train'][1] + folds_size['test'][1] + folds_size['val'][1]), \
    "Folds size sum ({}) are different from num_docs ({})".format((folds_size['train'][1] + folds_size['test'][1] + folds_size['val'][1]), num_docs)

    return folds_size


def split_data(labels_df, folds_size=None):
    """
    Split data into train/val/test folds in a stratified way.
    Input:
        - args: parsed arguments
        - labels_df: dataframe containing images code and their labels
    Output:
        - labels_df: same as input with a new flag porounding images fold
    """

    if folds_size is None:
        folds_size = {
            'train' : [0.7],
            'val' : [0.2],
            'test' : [0.1],
        }

    test_size = folds_size['test'][1]
    val_size = folds_size['val'] [1]
    # First we split test fold
    train, test, _, _ = train_test_split(
        labels_df,
        labels_df['label_int'],
        test_size=test_size,
        random_state=SEED,
        stratify=labels_df['label_int'],
    )
    # Now we split train and val folds
    train, val, _, _ = train_test_split(
        train,
        train['label_int'],
        test_size=val_size,
        random_state=SEED,
        stratify=train['label_int'],
    )

    # Set folds
    labels_df["fold"] = "train"
    labels_df.loc[test.index, "fold"] = "test"
    labels_df.loc[val.index, "fold"] = "val"

    return labels_df


def split_by_city(labels_df, val_city, test_city):
    # Set folds
    labels_df["fold"] = "train"
    labels_df.loc[labels_df["city"] == test_city, "fold"] = "test"
    labels_df.loc[labels_df["city"] == val_city, "fold"] = "val"

    return labels_df


def filter_data(document_text):
    document_text = document_text.split(" ")
    document_text = [term for term in document_text if term in idf_terms_to_keep]
    return " ".join(document_text)


def preprocess_files(file_dir):
    document = read_files(file_dir)
    # Verifica se foi possível extrair texto do documento
    if document['status'] == 'SUCCESS':
        # conta numero de paginas
        doc_size = len(document['text_content'])
        # conta termos antes preprocessamento
        num_termos_antes_preproc = count_terms(document['text_content'])
        # preprocessamento
        document['text_preprocessed'] = preprocess_text(document, num_pages, city_name)
        # conta termos depois preprocessamento
        #num_termos_depois_preproc = count_terms(document['text_preprocessed'])
        # gera variacoes do texto concatenando 1 a num_pages páginas
        page_content = merge_pages(document, num_pages)
        # gera linha a ser inserida no dataframe
        new_row = [document['file_id'], city_name, file_dir, doc_size, num_termos_antes_preproc]#, num_termos_depois_preproc]
        new_row.extend(page_content)
        # insere nova linha no dataframe
        return new_row
