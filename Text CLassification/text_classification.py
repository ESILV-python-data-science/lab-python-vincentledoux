#--articles-list ./LeMonde2003.csv

import argparse
import logging
import time
import sys
import pickle

import clf as clf
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

    #read the doc
    df = pd.read_csv(args.articles_list, header=0, sep='\t')
    df.dropna(axis=0, how='all')
    print(df.size)
    categories = ['ENT','INT','ART','SOC','FRA','SPO','LIV','TEL','UNE']
    df_isin = df["category"].isin(categories)
    df=df[df_isin]
    print(df.size)
    sns.countplot(df['category'])
    plt.show()





else:
    sys.exit()