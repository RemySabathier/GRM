# Code to train ML model on the training set
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import random
import scipy.io as sio
import os
from main import Features
from itertools import combinations
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pickle

if __name__ == '__main__':

    X = np.load('X_train300.npy').reshape((-1, 8))
    y = np.load('y_train300.npy').flatten()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=42, stratify=y)

    clf = LogisticRegression()

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print('Mean Accuracy Train: {:.3f}'.format(clf.score(X_train, y_train)))
    print('Mean Accuracy Test: {:.3f}'.format(clf.score(X_test, y_test)))

    # Save the model as pickle file
    with open('model_1.pk', 'wb') as f:
        pickle.dump(clf, f)
