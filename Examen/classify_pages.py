
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

logger = logging.getLogger('classify_exam.py')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)




def image_loader(image):
    if ':' in image:
        image = image.replace(':', '_')
    myImage = Image.open(image)
    return myImage

def extract_features_subresolution(img,img_feature_size = (8, 8)):

    gray_img = img.convert('L')

    # reduce the image to a given size
    reduced_img = gray_img.resize(
        img_feature_size, Image.BOX).filter(ImageFilter.SHARPEN)

    # return the values of the reduced image as features
    return [255 - i for i in reduced_img.getdata()]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract features, train a classifier on text and test the classifier')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image-list',help='file containing the text path and text class, one per line, comma separated')
    parser.add_argument('--save-features', help='save features in pickle format')
    parser.add_argument('--limit-samples', type=int, help='limit the number of samples to consider for training')
    input_group.add_argument('--load-features', help='read features and class from pickle file')
    args = parser.parse_args()
    Y = []
    image_pickle = []
    if args.load_features:
        with(open(args.load_features, "rb")) as myImage:
            image_pickle.append(myImage)

        all_df = pd.read_pickle(image_pickle)
        all_df.columns = ['file', 'type', 'ref']
        if 'class' in all_df.columns:
            X = all_df.drop(['type'], axis=1)
            Y = all_df['type']

    column_names = ['file', 'type', 'reference']
    df = pd.read_csv(args.image_list, header=None, names=column_names, sep=',')
    if args.limit_samples:
        df = df[:args.limit_samples]


    categories = ['reliure', 'pageblanche', 'miniature + texte', 'calendrier', 'texte', 'texte + miniature', 'miniature']
    df_isin = df["type"].isin(categories)
    df = df[df_isin]
    print(df.size)
    sns.countplot(df['type'])
    plt.show()

    data = []


    for i_path in tqdm(df['file']):
        page_image = image_loader(i_path)
        data.append(extract_features_subresolution(page_image))
    print("done")

    # check that we have data
    if not data:
        logger.error("Could not extract any feature vector or class")
        sys.exit(1)
    X = np.array(data)
    Y = df['type']

    my_df = []
    # save features
    if args.save_features:
        # convert X to dataframe with pd.DataFrame and save to pickle with to_pickle

        my_df = pd.DataFrame(X)
        my_df['type'] = Y
        my_df.to_pickle("all_df.pk1")
        logger.info('Saved {} features and class to {}'.format(my_df.shape, args.save_features))

    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.4)
    X_test, X_dev, Y_test, Y_dev = train_test_split(X_temp, Y_temp, test_size=0.5)

    nearest = [1,3,5,6,8]
    best = 0
    pred_y = 0
    pred_x = 0
    bestN = 0
    for i in nearest:
        neigh = neighbors.KNeighborsClassifier(n_neighbors=nearest)
        neigh.fit(X_train, Y_train)

        train_predicted = neigh.predict(X_train)
        test_predicted = neigh.predict(X_test)
        dev_predicted = neigh.predict(X_dev)
        accuracy_train = metrics.accuracy_score(Y_train, train_predicted)
        accuracy_test = metrics.accuracy_score(Y_test, test_predicted)
        accuracy_dev = metrics.accuracy_score(Y_dev, dev_predicted)
        if accuracy_test>best:
            best = accuracy_test
            bestN = i
            pred_y = train_predicted
            pred_x = test_predicted
        print("at nearest neighbor of : " + str(i))
        print("accuracy train : " + str(accuracy_train))
        print("accuracy test : " + str(accuracy_test))
        print("accuracy test : " + str(accuracy_dev))

    print("my best accuracy is : ")
    print("KNN of : " + str(bestN))
    print("Accuracy of: " + str(best))
    plt.figure()
    plt.title("KNN" + best)
    plt.xlabel("set size")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.plot(X_train, pred_y, linewidth=2.5, label="Train")
    plt.plot(X_test, pred_x, linewidth=2.5, label="Test")
    plt.show()










else:
    sys.exit()