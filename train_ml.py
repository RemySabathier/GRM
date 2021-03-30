# Code to generate data for first ML model

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import random
import os
from main import Features
from itertools import combinations
import random
from tqdm import tqdm
from utils import load_annotation


if __name__ == '__main__':

    dataset_path = 'dataset'
    annotation_path = r'dataset\allimsegs2.mat'
    size = 300

    # Step 1: Create a dataset of same-label and different-label superpixels
    annotations = load_annotation(annotation_path)
    N_img = len(annotations)  # Number of images

    training_data_X, training_data_y = [], []

    for img_path in tqdm(annotations):

        image = np.array(Image.open(os.path.join(dataset_path, img_path)))
        initial_superpixel = annotations[img_path]['seg_image']

        feature_store = Features(initial_superpixel, image,model_1_path=None)        
        d_features = feature_store.compute_features()

        # Extract the feature vector and the label for each superpixel in the image
        X = np.array([d_features[i]
                      for i in range(1, len(d_features)+1)])
        y = annotations[img_path]['labels'].flatten()
        N = X.shape[0]

        # Generate random possible combinations
        comb_it = combinations(range(N), 2)

        comb = [(np.abs(X[r1]-X[r2]), int(y[r1] == y[r2]))
                for r1, r2 in comb_it]

        comb_0 = [c[0] for c in comb if c[1] == 0]
        comb_1 = [c[0] for c in comb if c[1] == 1]
        samp_0 = min(len(comb_0), int(size/2))
        samp_1 = min(len(comb_1), int(size/2))
        random.shuffle(comb_0)
        random.shuffle(comb_1)

        training_data_X.append(comb_0[:samp_0] + comb_1[:samp_1])
        training_data_y.append([0]*samp_0+[1]*samp_1)

    # Save the dataset into .npy format
    X_train = np.array(training_data_X)
    y_train = np.array(training_data_y)
    np.save('X_train300.npy', X_train)
    np.save('y_train300.npy', y_train)
