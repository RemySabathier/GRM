# Code to generate data for other ML model on regions

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import random
import os
from main import Features, Regions
from itertools import combinations
import random
from tqdm import tqdm
import time
import pickle
from utils import load_annotation


if __name__ == '__main__':

    dataset_path = 'dataset'
    model_1_path = 'model_1.pk'
    annotation_path = r'dataset\allimsegs2.mat'
    train_repartition_path = 'dataset/train_test_repartition.pk'

    # Create a dataset of different class regions
    # Possible Classes are : Ground, Vertical, Sky or Mixed (4)
    # Possible Vertical Classes are: Nothing, Left, Center, Right, Porous, Solid or Mixed (7)

    # Step 1: Create a dataset of same-label and different-label superpixels
    annotations = load_annotation(annotation_path)
    N_img = len(annotations)  # Number of images

    #Training images
    with open(train_repartition_path, 'rb') as f:
        training_img_path = pickle.load(f)['Train']

    #Hyperparameters
    nb_hypothesis = 5
    number_regions_hypothesis = [3,4,5,7,9,11,15,20,25]

    list_hypothesis = random.sample(number_regions_hypothesis,nb_hypothesis)

    training_data_X, training_data_y = [], []

    for img_path in tqdm(training_img_path):

        image = np.array(Image.open(os.path.join(dataset_path, img_path)))
        superpixel_array = annotations[img_path]['seg_image']

        fr = Features(superpixel_array, image, model_1_path)

        # Compute a set of region segmentation with *k_sel* regions
        start = time.time()
        seg_list = [fr.segmentation(k_sel=k) for k in list_hypothesis]
        st1 = time.time()

        # Next: Create a list of regions classes per proposed segmentation
        reg_list = [Regions(seg, image) for seg in seg_list]
        st2 = time.time()

        for regions in reg_list:

            # Compute the features of all regions (for this hypothesis)
            X_features = regions.compute_features()

            # For each region, return list of superpixels inside
            d_sp_in_r = regions.compute_region_sp_list(superpixel_array)

            # We look in the annotations to find the category of each superpixels
            for r in d_sp_in_r:

                labels_id = [i-1 for i in d_sp_in_r[r]]

                vertical_labels = annotations[img_path]['horz_labels'][labels_id]
                global_labels = annotations[img_path]['vert_labels'][labels_id]
                npixels_labels = annotations[img_path]['npixels'][labels_id]

                count_global = [np.sum(npixels_labels[global_labels==i]) for i in range(1,4)]               
                #If there is a main class (>80%) then tag the region with this category, otherwise tag MIX
                if max(count_global)/sum(count_global) >= 0.8:
                    r_global = np.argmax(count_global)
                else:
                    r_global = 3 #The MIXED label


                count_vertical = [np.sum(npixels_labels[vertical_labels==i]) for i in range(0,6)]          
                #If there is a main class (>80%) then tag the region with this category, otherwise tag MIX
                if max(count_vertical)/sum(count_vertical) >= 0.8:
                    r_vertical = np.argmax(count_vertical)
                else:
                    r_vertical = 6 #The MIXED label
            
                #Add the region to the training set
                training_data_X.append(X_features[r])
                training_data_y.append([r_global,r_vertical])

        stop= time.time()
        print('')
        print('Segmentation: {:.3f}s / Regions: {:.3f}s / Data Generation: {:.3f}s'.format(st1-start,st2-st1,stop-st2))

    # Save the dataset into .npy format
    X_train = np.array(training_data_X)
    y_train = np.array(training_data_y)
    np.save('X_train_region.npy', X_train)
    np.save('y_train_region.npy', y_train)
