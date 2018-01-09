#--articles-list ./LeMonde2003.csv

import argparse
import logging
import time
import sys
import pickle

import clf as clf
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from PIL import Image, ImageFilter
from sklearn.cluster import KMeans
from sklearn import svm, metrics, neighbors
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sys import stdin

import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract features, train a classifier on text and test the classifier')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--articles-list',help='file containing the text path and text class, one per line, comma separated')
    args = parser.parse_args()

    #quest 1
    df = pd.read_csv(args.articles_list, header=0, sep='\t')
    df.dropna(axis=0, how='all')
    print(df.size)
    categories = ['ENT','INT','ART','SOC','FRA','SPO','LIV','TEL','UNE']
    df_isin = df["category"].isin(categories)
    df=df[df_isin]
    print(df.size)
    sns.countplot(df['category'])
    plt.show()

    #quest 2
    X_train, X_temp, Y_train, Y_temp= train_test_split(df["text"], df["category"], test_size=0.4)
    X_test, X_dev, Y_test, Y_dev = train_test_split(X_temp, Y_temp, test_size= 0.5)
    vectorizer = CountVectorizer(max_features=1000)
    vectorizer.fit(X_train)
    X_train_counts = vectorizer.transform(X_train)
    X_test_counts = vectorizer.transform(X_test)
    print(X_train.size)
    print(X_dev.size)
    print(X_test.size)
    print(Y_train.size)
    print(X_train_counts)

else:
    sys.exit()