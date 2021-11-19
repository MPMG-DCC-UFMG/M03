import os
import re
import glob
import json
import pandas as pd

from .preprocessing_portuguese import TextPreProcessing

import spacy
nlp = spacy.load('pt_core_news_sm')

from tqdm.notebook import tqdm

def limpeza_texto(page_text, city_name, filter_class, module):
    txt_process = TextPreProcessing()
    city_name = city_name.replace("_", " ")
    doc = nlp(page_text)
    doc.text.split()

    # lowercasing
    page_text = page_text.lower()

    # Punctuation removal
    page_text = page_text.replace("_"," ")
    page_text = txt_process.remove_special_characters(page_text)

    # PoS tagging
    [(token.orth_, token.pos_) for token in doc]

    for token in doc:
        # Numeral normalization
        if token.pos_ == 'NUM':
            page_text.replace(token.text, '0')

    # Custom processing
    # remove all single characters
    page_text = re.sub(r'\s+[a-zA-Z]\s+', " ", page_text)
    # remove single characters from the start
    page_text = re.sub(r'\^[a-zA-Z]\s+', " ", page_text)
    # remove nome do municipio e estado
    page_text = page_text.replace(city_name," ")
    page_text = page_text.replace('minas gerais'," ")
    page_text = page_text.replace('prefeitura municipal'," ")
    page_text = page_text.replace('prefeitura'," ")

    if module == 'simple':
        # Replacing URL and email structures into the tokens
        page_text = re.sub(r'[-a-zA-Z0-9@:%_\+.~#?&//=]{2,256}\.[a-z]{2,4}\b(\/[-a-zA-Z0-9@:%_\+.~#?&//=]*)?', "URL", page_text)
        page_text = re.sub(r'\S+@\S+', "EMAIL", page_text)

    elif module == 'simple-unity':
        # Replacing URL and email structures into the tokens
        page_text = re.sub(r'[-a-zA-Z0-9@:%_\+.~#?&//=]{2,256}\.[a-z]{2,4}\b(\/[-a-zA-Z0-9@:%_\+.~#?&//=]*)?', "URL", page_text)
        page_text = re.sub(r'\S+@\S+', "EMAIL", page_text)

        page_text = txt_process.remove_entities(page_text)

        page_text = txt_process.remove_units(page_text)

    elif module == 'complete' or module == 'complete-unity':

        # Name normalization
        page_text = txt_process.normalize_person_names(page_text)

        if module == 'complete-unity':
            page_text = txt_process.remove_entities(page_text)

            page_text = txt_process.remove_units(page_text)

        # Stop-words Removal
        page_text = txt_process.remove_emails(page_text)
        page_text = txt_process.remove_urls(page_text)
        page_text = txt_process.remove_stopwords(page_text)
        page_text = txt_process.remove_hour(page_text)
        # split numbers from letters
        page_text = ' '.join(re.split('(\d+)',page_text))
        page_text = txt_process.remove_symbols_from_numbers(page_text)
        page_text = txt_process.remove_numbers(page_text)
        # page_text = txt_process.remove_reduced_or_contracted_words(page_text)

        for token in doc:
            # Lemmatization
            if token.pos_ == 'VERB':
                page_text.replace(token.text, token.lemma_)

            # filter_class = ['NOUN', 'VERB', 'ADJ', 'ADP']
            if token.pos_ not in filter_class:
                page_text.replace(token.text, ' ')

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

def preprocess_text(document, num_pages, city_name, filter_class, module):
    return [limpeza_texto(page_content, city_name, filter_class, module) for page_content in document['text_content'][:num_pages]]

def merge_pages(document, num_pages):
    # retorna lista onde a cada posicao uma nova pagina e concatenada ao texto
    num_pages+=1
    return [" ".join(document['text_preprocessed'][0:num_pages]) for num_pages in range(1,num_pages)]

def read_data(cities_dir, num_pages, filter_class, module='complete'):
    cities_docs = {}
    df_document_content = pd.DataFrame(columns=['doc_id', 'city', 'file_dir', 'one_page', 'two_pages', 'three_pages', 'four_pages'])

    for city_dir in cities_dir:

        # Pega o nome da cidade
        city_name = get_name(city_dir)
        if city_name == "bh":
            continue

        print("-"*100)
        print(city_name)

        # Lista os arquivos a serem lidos
        files_dir = list_json_files_dir(city_dir, city_name)

        # Faz a leitura dos arquivos
        for file_dir in tqdm(files_dir):
            document = read_files(file_dir)

            # Verifica se foi possível extrair texto do documento
            if document['status'] == 'SUCCESS':

                # preprocessamento
                document['text_preprocessed'] = preprocess_text(document, num_pages, city_name, filter_class, module)

                # gera variacoes do texto concatenando 1 a num_pages páginas
                page_content = merge_pages(document, num_pages)

                # gera linha a ser inserida no dataframe
                new_row = [document['file_id'], city_name, file_dir]
                new_row.extend(page_content)

                # insere nova linha no dataframe
                df_document_content.loc[len(df_document_content)] = new_row

    return df_document_content
