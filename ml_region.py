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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder

if __name__ == '__main__':

    X = np.load('X_train_region.npy')
    y = np.load('y_train_region.npy')
    y_global, y_vertical = y[:,0], y[:,1]
    
    enc = OneHotEncoder()

    #First step : Global model training ###########################################

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_global, train_size=0.8, random_state=42, stratify=y_global)

    #Encode in Multilabel
    y_train = enc.fit_transform(y_train.reshape(-1,1))
    y_test = enc.transform(y_test.reshape(-1,1))

    clf = OneVsRestClassifier(AdaBoostClassifier(n_estimators=8))
    clf.fit(X_train, y_train)

    #Dummy clf for comparison
    dummy_clf = DummyClassifier(strategy="prior")
    dummy_clf.fit(X_train, y_train)
    

    y_pred = clf.predict(X_test).toarray()

    print('Mean Accuracy Train: {:.3f}'.format(clf.score(X_train, y_train)))
    print('Mean Accuracy Test: {:.3f}'.format(clf.score(X_test, y_test)))
    print('Mean Accuracy Test (DUMMY): {:.3f}'.format(dummy_clf.score(X_test, y_test)))

    # Save the model as pickle file
    with open('model_general.pk', 'wb') as f:
        pickle.dump(clf, f)


    #Second step : Vertical model training ###########################################

    #Remove 0 label from the dataset
    X = X[y_vertical != 0]
    y_vertical = y_vertical[y_vertical != 0]-1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_vertical, train_size=0.8, random_state=42, stratify=y_vertical)

    #Encode in Multilabel
    y_train = enc.fit_transform(y_train.reshape(-1,1))
    y_test = enc.transform(y_test.reshape(-1,1))

    clf = OneVsRestClassifier(AdaBoostClassifier(n_estimators=8))
    clf.fit(X_train, y_train)

    #Dummy clf for comparison
    dummy_clf = DummyClassifier(strategy="prior")
    dummy_clf.fit(X_train, y_train)

    
    y_pred = clf.predict(X_test)

    print('Mean Accuracy Train: {:.3f}'.format(clf.score(X_train, y_train)))
    print('Mean Accuracy Test: {:.3f}'.format(clf.score(X_test, y_test)))
    print('Mean Accuracy Test (DUMMY): {:.3f}'.format(dummy_clf.score(X_test, y_test)))
    
    # Save the model as pickle file
    with open('model_vertical.pk', 'wb') as f:
        pickle.dump(clf, f)
