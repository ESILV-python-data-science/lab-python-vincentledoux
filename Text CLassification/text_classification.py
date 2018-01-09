#--articles-list ./LeMonde2003.csv

import argparse
import logging
import time
import sys
import pickle

import clf as clf
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
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
from sklearn.naive_bayes import MultinomialNB

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

    features = [1000,3000,5000,7000,10000,15000]
    alpha = [0.1,0.4,0.7,1]
    X_train_count = []
    X_test_count = []
    X_dev_count = []
    Tab = []
    count = 0
    for i in features:
        for a in alpha:
            vectorizer = CountVectorizer(max_features=i)
            vectorizer.fit(X_train)
            X_train_counts = vectorizer.transform(X_train)
            X_test_counts = vectorizer.transform(X_test)
            X_dev_counts = vectorizer.transform(X_dev)

            print(X_train.size)
            print(X_dev.size)
            print(X_test.size)

            # quest 3
            mmb = MultinomialNB(alpha=a)
            mmb.fit(X_train_counts, Y_train)
            y_pred_train = mmb.predict(X_train_counts)
            y_pred_test = mmb.predict(X_test_counts)
            y_pred_dev = mmb.predict(X_dev_counts)
            print("alpha : " + str(a))
            print("feature : " + str(i))
            print("train_score " + str(metrics.accuracy_score(Y_train, y_pred_train)))
            print("test_score " + str(metrics.accuracy_score(Y_test, y_pred_test)))
            print("dev_score " + str(metrics.accuracy_score(Y_dev, y_pred_dev)))

            if Tab == []:
                Tab.append(metrics.accuracy_score(Y_test, y_pred_test))

            mymax = max(Tab, key=float)
            if mymax < metrics.accuracy_score(Y_test, y_pred_test):
                X_train_count = X_train_counts
                X_test_count = X_test_counts
                X_dev_count = X_dev_counts


            Tab.append(metrics.accuracy_score(Y_test, y_pred_test))
            count += 1



    # quest4

    tf_transformer = TfidfTransformer().fit(X_train_count)
    X_train_tf = tf_transformer.transform(X_train_count)
    X_test_tf = tf_transformer.transform(X_test_count)
    X_dev_tf = tf_transformer.transform(X_dev_count)


    mmb.fit(X_train_tf, Y_train)
    y_pred_train = mmb.predict(X_train_tf)
    y_pred_test = mmb.predict(X_test_tf)
    y_pred_dev = mmb.predict(X_dev_tf)
    print("train_score " + str(metrics.accuracy_score(Y_train, y_pred_train)))
    print("test_score " + str(metrics.accuracy_score(Y_test, y_pred_test)))
    print("dev_score " + str(metrics.accuracy_score(Y_dev, y_pred_dev)))
    count += 1

    from sklearn.metrics import classification_report, confusion_matrix

    print(classification_report(X_train_tf, y_pred_train, Y_test))
    confusion_matrix(X_train_tf, y_pred_train)






else:
    sys.exit()