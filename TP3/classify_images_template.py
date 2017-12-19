# -*- coding: utf-8 -*-
"""
Classify digit images

C. Kermorvant - 2017
"""


import argparse
import logging
import time
import sys
import pickle

import clf as clf
from tqdm import tqdm
import pandas as pd
from PIL import Image, ImageFilter
from sklearn.cluster import KMeans
from sklearn import svm, metrics, neighbors
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sys import stdin

import numpy as np
# Setup logging
logger = logging.getLogger('classify_images.py')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)




def KNN(X_train, Y_train, X_test, Y_test, X_valid, Y_valid):

    neigh = KNeighborsClassifier(n_neighbors=1)
    x = neigh.fit(X_train, Y_train)
    neigh.predict(X_test)
    neigh.predict(Y_test)
    x = neigh.fit(X_test, Y_test)
    neigh.predict(X_valid)
    neigh.predict(Y_valid)
    y =metrics.accuracy_score(X_valid, Y_valid)



    return y

def Linear(X_train, Y_train, X_test, Y_test, X_valid, Y_valid):
    regr = linear_model.LinearRegression()
    regr.fit(X_train, Y_train)
    np.mean((regr.predict(X_test) - Y_test) ** 2)
    x = regr.score(X_test, Y_test)
    regr.git(X_train, Y_train)
    np.mean((regr.predict(X_valid) - Y_valid) **2)
    x = regr.score(X_valid, Y_valid)
    return x

def Linear2(X_train, Y_train, X_test, Y_test):
    regr = linear_model.LinearRegression()
    regr.fit(X_train, Y_train)
    np.mean((regr.predict(X_test) - Y_test) ** 2)
    x = regr.score(X_test, Y_test)
    return x




def extract_features_subresolution(img,img_feature_size = (8, 8)):

    gray_img = img.convert('L')

    # reduce the image to a given size
    reduced_img = gray_img.resize(
        img_feature_size, Image.BOX).filter(ImageFilter.SHARPEN)

    # return the values of the reduced image as features
    return [255 - i for i in reduced_img.getdata()]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract features, train a classifier on images and test the classifier')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--images-list',help='file containing the image path and image class, one per line, comma separated')
    input_group.add_argument('--load-features', help='read features and class from pickle file')
    parser.add_argument('--save-features', help='save features in pickle format')
    parser.add_argument('--limit-samples',type=int, help='limit the number of samples to consider for training')
    classifier_group = parser.add_mutually_exclusive_group(required=True)
    classifier_group.add_argument('--nearest-neighbors',type=int)
    classifier_group.add_argument('--features-only', action='store_true', help='only extract features, do not train classifiers')
    args = parser.parse_args()

    image_pickle = []
    if args.load_features:

        with(open(args.load_features, "rb")) as myImage:
            image_pickle.append(myImage)

        all_df = pd.DataFrame(image_pickle)
        args.save_features()
    else:


        # Load the image list from CSV file using pd.read_csv
        # see the doc for the option since there is no header ;
        # specify the column names :  filename , class
        file_list = []
        #BASE_DIR = 'C:/Users/Vincent/Documents'
        column_names = ['filename', 'class']

        df = pd.read_csv(args.images_list, names=column_names, header=None)
        #file = df.filename
        for i in df:
            file_list.extend(i)

        logger.info('Loaded {} images in {}'.format(all_df.shape,args.images_list))


        # Extract the feature vector on all the pages found
        # Modify the extract_features from TP_Clustering to extract 8x8 subresolution values
        # white must be 0 and black 255
        data = []
        for i_path in tqdm(file_list):
            page_image = Image.open(i_path)
            data.append(extract_features_subresolution(page_image))

        # check that we have data
        if not data:
            logger.error("Could not extract any feature vector or class")
            sys.exit(1)



        # convert to np.array
        X = np.array(data)


    all_df = []
    # save features
    if args.save_features:
        # convert X to dataframe with pd.DataFrame and save to pickle with to_pickle
        #logger.info('Saved {} features and class to {}'.format(df_features.shape,args.save_features))
        all_df = pd.DataFrame(file_list)


    if args.features_only:
        logger.info('No classifier to train, exit')
        sys.exit()
    train, test = train_test_split(all_df, test_size=0.2)
    neigh = KNeighborsClassifier(n_neighbors=1)
    x = neigh.fit(train, test)
    metrics.accuracy_score(x)



    # Train classifier
    logger.info("Training Classifier")
    X_train, Y_train, X_temp, Y_temp = train_test_split(all_df['Class'], all_df['Filename'], test_size=0.4)
    X_test, Y_test, X_valid, Y_Valid = train_test_split(X_temp, Y_temp, test_size=0.5)
    # Use train_test_split to create train/test split
    logger.info("Train set size is {}".format(X_train.shape))
    logger.info("Test set size is {}".format(X_test.shape))


    if args.nearest_neighbors:
        # create KNN classifier with args.nearest_neighbors as a parameter
        logger.info('Use kNN classifier with k= {}'.format(args.nearest_neighbors))
    else:
        logger.error('No classifier specified')
        sys.exit()

    # Do Training@
    t0 = time.time()
    logger.info("Training  done in %0.3fs" % (time.time() - t0))

    # Do testing
    logger.info("Testing Classifier")
    t0 = time.time()
    predicted = clf.predict(X_test)

    # Print score produced by metrics.classification_report and metrics.accuracy_score
    logger.info("Testing  done in %0.3fs" % (time.time() - t0))
    X_train, Y_train, X_temp, Y_temp = train_test_split(all_df['Class'], all_df['Filename'], test_size=0.4)
    X_test, Y_test, X_valid, Y_Valid = train_test_split(X_temp, Y_temp, test_size=0.5)

    print("which algorithm do you wish ? ")
    print("1. KNN")
    print("2. Linear Regression")
    print("3.Linear with no validation")
    x = stdin.read(1)
    userinput = stdin.readline()
    if userinput == "1":
        KNN(X_train, Y_train, X_test, Y_test, X_valid, Y_Valid)
    if userinput == "2":
        Linear(X_train, Y_train, X_test, Y_test, X_valid, Y_Valid)
    if userinput == "3":
        Linear2(X_train, Y_train, X_test, Y_test)
    else:
        print('you choose the wrong numner')






    
