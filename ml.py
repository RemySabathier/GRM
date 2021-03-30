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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.dummy import DummyClassifier

if __name__ == '__main__':

    X = np.load('feature_training_set/X_train_features.npy')
    X = X.reshape((-1,X.shape[-1]))
    y = np.load('feature_training_set/y_train_features.npy').flatten()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=42, stratify=y)

    clf = LogisticRegression(max_iter=10000)
    # clf = AdaBoostClassifier(
    #    base_estimator=LogisticRegression(), n_estimators=10)
    clf.fit(X_train, y_train)

    #Dummy clf for comparison
    dummy_clf = DummyClassifier(strategy="prior")
    dummy_clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print('Mean Accuracy Train: {:.3f}'.format(clf.score(X_train, y_train)))
    print('Mean Accuracy Test: {:.3f}'.format(clf.score(X_test, y_test)))
    print('Mean Accuracy Test (DUMMY): {:.3f}'.format(dummy_clf.score(X_test, y_test)))

    # Save the model as pickle file
    with open('model_1.pk', 'wb') as f:
        pickle.dump(clf, f)
