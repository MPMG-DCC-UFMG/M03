import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import time
import multiprocessing as mp
import preprocessing.preprocessing_module as preprocess
from operator import itemgetter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


def tf_idf_data(corpus_doc, fold):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus_doc)

    print("_"*30, fold)
    tf_idf_features = vectorizer.get_feature_names()
    dense = X.todense()
    lst1 = dense.tolist()

    df = pd.DataFrame(lst1, columns=tf_idf_features)
    df = df.T.sum(axis=1)
    df_aux = df.to_frame().rename(columns={'term':'tf-idf'}).reset_index()
    df_tfidf = pd.DataFrame(data=X.toarray(), columns=tf_idf_features)
    print(">> Matrix TF-IDF")
    display(df_tfidf.head())
    print(type(df_tfidf))


    return df_tfidf[sorted(df_tfidf.columns)]
