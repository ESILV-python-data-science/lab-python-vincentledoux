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
from sklearn.linear_model import LogisticRegression
from sys import stdin

import numpy as np
import matplotlib.pyplot as plt
# Setup logging
logger = logging.getLogger('classify_images.py')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)



def KNNQ4(X_train, Y_train, X_test, Y_test):

    # Train classifier
    logger.info("Training Classifier")
    neigh = KNeighborsClassifier(n_neighbors=args.nearest_neighbors)
    # Use train_test_split to create train/test split
    logger.info("Train set size is {}".format(X_train.shape))
    logger.info("Test set size is {}".format(X_test.shape))

    if args.nearest_neighbors:
        # create KNN classifier with args.nearest_neighbors as a parameter
        logger.info('Use kNN classifier with k= {}'.format(args.nearest_neighbors))
        neigh = KNeighborsClassifier(n_neighbors=args.nearest_neighbors)
    else:
        logger.error('No classifier specified')
        sys.exit()

    # Do Training@
    t0 = time.time()
    x = neigh.fit(X_test, Y_test)
    logger.info("Training  done in %0.3fs" % (time.time() - t0))

    # Do testing
    logger.info("Testing Classifier")
    t0 = time.time()
    predicted = neigh.predict(X_test)
    return x



def Linear(X_train, Y_train, X_test, Y_test):
    regr = LogisticRegression()
    regr.fit(X_train, Y_train)

    #np.mean((regr.predict(X_test) - Y_test) ** 2)
    x = regr.predict(X_test)
    test = metrics.accuracy_score(Y_test, x)
    return test




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
    parser.add_argument('--limit-samples', type=int, help='limit the number of samples to consider for training')
    parser.add_argument('--learning-curve', type=int)
    classifier_group = parser.add_mutually_exclusive_group(required=True)
    classifier_group.add_argument('--nearest-neighbors',type=int)
    classifier_group.add_argument('--logistic-regression', type=int)
    classifier_group.add_argument('--features-only', action='store_true', help='only extract features, do not train classifiers')
    args = parser.parse_args()

    image_pickle = []
    if args.load_features:

        with(open(args.load_features, "rb")) as myImage:
            image_pickle.append(myImage)

        df = pd.read_pickle(image_pickle)
        df.columns = ['filename', 'class']


    else:


        # Load the image list from CSV file using pd.read_csv
        # see the doc for the option since there is no header ;
        # specify the column names :  filename , class
        file_list = []
        column_names = ['filename', 'class']

        file_list = pd.read_csv(args.images_list, names=column_names, header=None)
        logger.info('Loaded {} images in {}'.format(file_list.shape, args.images_list))


        # Extract the feature vector on all the pages found
        # Modify the extract_features from TP_Clustering to extract 8x8 subresolution values
        # white must be 0 and black 255

        data = []
        for i_path in tqdm(file_list['filename']):
            page_image = Image.open(i_path)
            data.append(extract_features_subresolution(page_image))

        # check that we have data
        if not data:
            logger.error("Could not extract any feature vector or class")
            sys.exit(1)



        # convert to np.array
        X = np.array(data)
        Y = np.array(file_list['class'])


    all_df = []
    # save features
    if args.save_features:
        # convert X to dataframe with pd.DataFrame and save to pickle with to_pickle

        all_df = pd.DataFrame(X)
        all_df['class'] = Y
        all_df.to_pickle("all_df.pk1")
        logger.info('Saved {} features and class to {}'.format(all_df.shape, args.save_features))



    if args.features_only:
        logger.info('No classifier to train, exit')
        sys.exit()

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    if args.nearest_neighbors:
        if args.learning_curve:
            X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.4)
            X_test, X_valid, Y_test, Y_Valid = train_test_split(X_temp, Y_temp, train_size=0.5)
            # Use train_test_split to create train/test split
            logger.info("Train set size is {}".format(X_train.shape))
            logger.info("Test set size is {}".format(X_test.shape))

            # create KNN classifier with args.nearest_neighbors as a parameter
            logger.info('Use kNN classifier with k= {}'.format(args.nearest_neighbors))
            neigh = KNeighborsClassifier(n_neighbors=args.nearest_neighbors)
            # Do Training@
            t0 = time.time()
            # x = neigh.fit(X_train, Y_train)
            # logger.info("Training  done in %0.3fs" % (time.time() - t0))

            # Do testing
            logger.info("Testing Classifier")
            t0 = time.time()
            # neigh.predict(X_test)
            # predicted = neigh.predict(X_test)

            # Print score produced by metrics.classification_report and metrics.accuracy_score
            x = KNNQ4(x_train, y_train, x_test, y_test)
            logger.info("Testing  done in %0.3fs" % (time.time() - t0))
            score = KNNQ4(X_train, Y_train, X_test, Y_test)
            scoreTV = KNNQ4(X_train, Y_train, X_valid, Y_Valid)
            scoretv = KNNQ4(X_test, Y_test, X_valid, Y_Valid)
            print(str(score))
            print(str(scoreTV))
            print(str(scoretv))
            Max = score
            if scoreTV > Max:
                Max = scoreTV
            if scoretv > Max:
                Max = scoretv
                if scoreTV > scoretv:
                    Max = scoreTV
        else :
            # Train classifier
            logger.info("Training Classifier")
            size = [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1]
            result = {}
            X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.4)
            for i in size:
                X_test, X_valid, Y_test, Y_Valid = train_test_split(X_temp, Y_temp, train_size=i)
                # Use train_test_split to create train/test split
                logger.info("Train set size is {}".format(X_train.shape))
                logger.info("Test set size is {}".format(X_test.shape))

                # create KNN classifier with args.nearest_neighbors as a parameter
                logger.info('Use kNN classifier with k= {}'.format(args.nearest_neighbors))
                neigh = KNeighborsClassifier(n_neighbors=args.nearest_neighbors)
                # Do Training@
                t0 = time.time()
                # x = neigh.fit(X_train, Y_train)
                # logger.info("Training  done in %0.3fs" % (time.time() - t0))

                # Do testing
                logger.info("Testing Classifier")
                t0 = time.time()
                # neigh.predict(X_test)
                # predicted = neigh.predict(X_test)

                # Print score produced by metrics.classification_report and metrics.accuracy_score
                x = KNNQ4(x_train, y_train, x_test, y_test)
                logger.info("Testing  done in %0.3fs" % (time.time() - t0))
                score = KNNQ4(X_train, Y_train, X_test, Y_test)
                scoreTV = KNNQ4(X_train, Y_train, X_valid, Y_Valid)
                scoretv = KNNQ4(X_test, Y_test, X_valid, Y_Valid)
                print(str(score))
                print(str(scoreTV))
                print(str(scoretv))
                Max = score
                if scoreTV > Max:
                    Max = scoreTV
                if scoretv > Max:
                    Max = scoretv
                    if scoreTV > scoretv:
                        Max = scoreTV
                plt.figure()
                plt.title("KNN" + i)
                plt.xlabel("set size")
                plt.ylabel("Accuracy")
                plt.grid()
                plt.plot(X_train, x, linewidth=2.5, label="Train")
                plt.plot(X_test, y, linewidth=2.5, label="Test")
                plt.show()

    elif args.logistic_regression:
        best = 0
        myscore = 0
        size = [0.01, 0.1, 0.2, 0.4, 0.6, 0.8]
        for i in size:
            X_train, X_var, Y_train, Y_var = train_test_split(x_train, y_train, train_size=i)
            myscore = Linear(X_train, Y_train, x_test, y_test)
            logger.info("score of " + str(i) + " is " + str(myscore))
            if args.learning_curve:
                plt.figure()
                plt.title("Logistic" + i)
                plt.xlabel("set size")
                plt.ylabel("Accuracy")
                plt.grid()
                plt.plot(X_train, myscore, linewidth=2.5, label="Train")
                plt.plot(x_test, y, linewidth=2.5, label="Test")
                plt.show()


        logger.info("best score is " + str(best))
    else:
        logger.error('No classifier specified')
        sys.exit()





