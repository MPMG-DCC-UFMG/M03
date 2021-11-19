import os

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from operator import itemgetter
import re

from pyhive import hive

# Classe com vários métodos de pre-processamento de texto em português criado pelo grupo F03
import utils.preprocessing_portuguese as preprossPT

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import word_tokenize
import xgboost as xgb
from wordcloud import WordCloud
from tqdm.notebook import tqdm
import itertools


class DocumentCleaning:

    def __init__(self, dir_cids,
                 path_data="/dados01/workspace/ufmg.f01dcc/m03/business_understanding/data/",
                 path_res="/dados01/workspace/ufmg.f01dcc/m03/business_understanding/data/resultados_classificacao/",
                 path_classes_doc="/dados01/workspace/ufmg.f01dcc/m03/business_understanding/notebooks/resultado_m03_meta_classes_extraction/relacao_documentos_label_v2.csv",
                 path_itens="/dados01/workspace/ufmg.f01dcc/m03/business_understanding/notebooks/resultado_m03_meta_classes_extraction/sicom_20210109_licitacao_item_202109091726.csv"):

        self.dir_cids = dir_cids
        self.path_data = path_data
        self.path_res = path_res
        self.path_classes_doc = path_classes_doc
        self.path_itens = path_itens
        self.stop_words = ["municipal", "mg", "minas", "prefeitura", "pirapetinga",
                           "ijaci", "itamarati", "itamarati de minas", "cristais",
                           "olaria", "passa-vinte", "arantina", "ribeirao vermelho",
                           "sao bento abade", "coqueiral", "estabelecimentolicitante",
                           "gerais", "rc", "tc", "hr", "rx"]
        self.df_itens = pd.read_csv(path_itens)
        self.left_itens = df_itens["dsc_item_raw"].to_list()

    def read_data(self, dir_cids):

        all_documents = []

        for cidade in dir_cids:
            files = os.listdir(path_data + cidade + "/data/files_json/")

            for filename in files:
                # print(filename)

                f = open(path_data + cidade + "/data/files_json/" + filename,)

                # returns JSON object as
                # a dictionary
                data = json.load(f)

                if data['status'] != 'FAILED' or data['text_content']:
                    # Adicionar a cidade ao dicionário
                    if cidade in ["351-licitacoes-itamarati/licitacoes_itamarati_2017", "351-licitacoes-itamarati/licitacoes_itamarati_2018", "351-licitacoes-itamarati/licitacoes_itamarati_2019", "351-licitacoes-itamarati/licitacoes_itamarati_2020"]:
                        data["city"] = "Itamarati de Minas"
                    elif cidade == "304-licitacoes-passa-vinte":
                        data["city"] = "Passa vinte"
                    elif cidade == "381-licitacoes-sao-bento-abade":
                        data["city"] = "São Bendo Abade"
                    elif cidade == "385-licitacoes-ribeirao-vermelho":
                        data["city"] = "Ribeirão Vermelho"
                    else:
                        data["city"] = cidade.split("-")[-1].title()

                    all_documents.append(data)

        print(len(all_documents))
        print(len(all_documents[5968]['text_content']))

        return all_documents

    def limpeza_texto(self, page_text):
        txt_process = preprossPT.TextPreProcessing()
        #city_name = city_name.replace("_", " ")

        page_text = ' '.join(re.split('(\d+)', page_text))

        page_text = txt_process.remove_person_names(page_text)

        page_text = page_text.lower()

        page_text = txt_process.remove_emails(page_text)

        page_text = txt_process.remove_urls(page_text)

        page_text = txt_process.remove_pronouns(page_text)

        page_text = txt_process.remove_adverbs(page_text)

        page_text = txt_process.remove_special_characters(page_text)

        page_text = txt_process.remove_accents(page_text)

        page_text = txt_process.remove_stopwords(page_text)

        page_text = txt_process.remove_symbols_from_numbers(page_text)

        page_text = txt_process.remove_numbers(page_text)

        page_text = txt_process.remove_reduced_or_contracted_words(page_text)

        # Removendo letras sozinhas no texto
        page_text = re.sub(r'(?:^| )\w(?:$| )', ' ', page_text).strip()
        page_text = re.sub(r"\b[a-zA-Z]\b", "", page_text)

        # Remove words with 1 to 3 letters
        '''shortword = re.compile(r'\W*\b\w{1,3}\b')
        page_text = shortword.sub('', page_text)'''

        page_text = page_text.replace("_", "")

        page_text = page_text.replace('minas gerais', "")
        page_text = page_text.replace('prefeitura municipal', "")
        page_text = page_text.replace('prefeitura', "")

        page_text = txt_process.remove_excessive_spaces(page_text)

        page_text = page_text.replace("_", "")

        page_text = page_text.replace("xx", "")

        return page_text

    def tokenize(self, text):

        words = [token for token in word_tokenize(text)]
        words_doc = ' '.join(words[:1000])
        return words_doc
