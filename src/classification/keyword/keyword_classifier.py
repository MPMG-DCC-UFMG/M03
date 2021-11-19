import glob
import json
import os
import random
import re
import shutil
import copy
from collections import defaultdict
import pandas as pd
import pdfplumber
import seaborn as sns
import preprocessing.preprocessing_portuguese as preprossPT
from keywords import keywords
from collections import defaultdict
from .utils import (
    get_pages,
    extract_title,
    extract_content,
    title_extraction_bold,
    title_extraction_upper,
    title_extraction_upper_bold,
    title_extraction_breaklines,
    get_first_tokens,
    table_extraction
)

def KeywordClassifier:

    def __init__(self, limit):

        self.classes = ["ATA", "HOMOLOG", "EDITAL", "OUTROS"]
        self.title_keys = ["ata_title_count",
                           "homolog_title_count",
                           "edital_title_count",
                           "outros_title_count"]
        self.content_keys = ["ata_content_count",
                             "homolog_content_count",
                             "edital_content_count",
                             "outros_content_count"]
        self.keywords_of_interest = [key_word for key_word in keywords if key_word["class"] in classes_of_interest]
        self.limit = limit
        self.vocabulary = set()
        self.idfs = defaultdict(lambda: defaultdict(int))

        self.cities = ["pirapetinga", "coqueiral", "cristais", "olaria",
                       "passa vinte", "arantina", "ijaci", "sao bento abade"]
        self.default_features_dict = {'contrato_administrativo_t': 0,
                                      'contrato_administrativo_c': 0,
                                      'retificacao_t': 0,
                                      'retificacao_c': 0,
                                      'aviso_de_t': 0,
                                      'aviso_de_c': 0,
                                      'aditamento_t': 0,
                                      'aditamento_c': 0,
                                      'extrato_t': 0,
                                      'extrato_c': 0,
                                      'ordem_de_servico_t': 0,
                                      'ordem_de_servico_c': 0,
                                      'sessao_publica_t': 0,
                                      'sessao_publica_c': 0,
                                      'adjudicacao_t': 0,
                                      'adjudicacao_c': 0,
                                      'ata_t': 0,
                                      'ata_c': 0,
                                      'homologacao_t': 0,
                                      'homologacao_c': 0,
                                      'convite_t': 0,
                                      'convite_c': 0,
                                      'resposta_t': 0,
                                      'resposta_c': 0,
                                      'edital_t': 0,
                                      'edital_c': 0,
                                      'cronograma_t': 0,
                                      'cronograma_c': 0,
                                      'diario_oficial_t': 0,
                                      'diario_oficial_c': 0}


    def get_pdfs_info(self, foldes_paths):

        cities = set()
        for folder_path in folder_paths:
            for _, filepath in zip(range(self.limit), glob.iglob(folder_path)):
                try:
                    pdf_id = filepath.split("/")[-1]
                    pdf_city = (
                        filepath.split("/")[2]
                        .split("licitacoes")[-1]
                        .replace("-", "", 1)
                        .replace("-", " ")
                    )
                    cities.add(pdf_city)
                    pdfs.append({"id": pdf_id, "city": pdf_city, "path": filepath})
                except Exception as e:
                    print("**", e)

        return pdfs


    def get_content_matches(self, pdf_id, title, content):

        features_dict = {'doc_id': pdf_id, 'title': title, 'city': city}
        features_dict.update(copy.copy(self.default_features_dict))

        for word_dict in keywords:
            word = word_dict["word"]
            title_regex = word_dict["title_regex"]
            content_regex = word_dict["title_regex"]
            doc_class = word_dict["class"].lower()
            title_matches = []

            for index in range(len(title)):
                line = title[index]
                title_matches = re.findall(title_regex, line.lower())

                if bool(title_matches) and len(title_matches) > 0:
                    for match in title_matches:
                        match = match.replace(' ', '_')
                        match = text_preprocessing.remove_accents(match)
                        features_dict[f'{match}_t'] += 1
                        self.vocabulary.add(match)
                        idfs['']

            content_matches = re.findall(content_regex, content.lower())

            for match in content_matches:
                match = match.replace(' ', '_')
                match = text_preprocessing.remove_accents(match)
                features_dict[f'{match}_c'] += 1
                self.vocabulary.add(match)

        return features_dict


    def get_city_pdfs(self, folder_paths, cities=None):

        if cities is None:
            cities = self.cities

        cities_data = defaultdict(list)
        empty = 0

        pdfs = self.get_pdfs_info(folder_paths)

        for city in cities:
            city_pdfs = [info for info in pdfs if info["city"] == city]
            ata_count, homolog_count, edital_count, others_count = (0,)*4

            pdf_i = 0
            print('-'*80)
            print(f'## {city} ##')

            while pdf_i < len(city_pdfs):
                try:
                    pdf = city_pdfs[pdf_i]
                    pdf_file = pdfplumber.open(pdf['path'])
                    pdf_id = pdf["id"].split(".")[0]

                    pages = get_pages(pdf_file)
                    content = get_first_tokens(pdf_file)
                    title = title_extraction_breaklines(pdf_file)

                    if (bool(title) and len(title)) == 0 or not (content):
                        empty += 1
                        print("ERRO! DOCUMENTO ESCANEADO")
                    else:
                        tokens_matches_dict = get_content_matches(pdf_id, title, content)
                        tokens_matches_dict.update({'num_pages': len(pages)})
                        cities_data[city].append(tokens_matches_dict)
                        pdf_file.close()
                except Exception as e:
                    print("*-*-*", e)
                pdf_i += 1
