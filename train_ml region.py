# Code to generate data for other ML model on regions

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import random
import scipy.io as sio
import os
from main import Features, Regions
from itertools import combinations
import random
from tqdm import tqdm


def load_annotation(path):
    '''Load the Matlab annotation file and return a python dict'''
    data = sio.loadmat(path)
    X = data['imsegs'][0]
    N = X.shape[0]
    d = {}
    for n in range(N):
        param = {}
        file_name = X[n][0][0]
        param['seg_image'] = X[n][2]
        param['npixels'] = X[n][4]
        param['vlabels'] = X[n][6]
        param['hlabels'] = X[n][7]
        param['labels'] = X[n][8]
        param['vert_labels'] = X[n][9].flatten()
        param['horz_labels'] = X[n][10].flatten()
        param['label_names'] = X[n][11]
        param['vert_names'] = X[n][12]
        param['horz_names'] = X[n][13]
        d[file_name] = param
    return d


if __name__ == '__main__':

    dataset_path = 'dataset'
    model_1_path = 'model_1.pk'
    annotation_path = r'dataset\allimsegs2.mat'

    # Create a dataset of different class regions
    # Possible Classes are : Ground, Vertical, Sky or Mixed (4)
    # Possible Vertical Classes are: Left, Center, Right, Porous, Solid or Mixed (6)

    # Step 1: Create a dataset of same-label and different-label superpixels
    annotations = load_annotation(annotation_path)
    N_img = len(annotations)  # Number of images

    #Hyperparameters
    nb_hypothesis = 5
    number_regions_hypothesis = [3,4,5,7,9,11,15,20,25]

    list_hypothesis = random.sample(number_regions_hypothesis,nb_hypothesis)

    training_data_X, training_data_y = [], []

    for img_path in tqdm(annotations):

        image = np.array(Image.open(os.path.join(dataset_path, img_path)))
        superpixel_array = annotations[img_path]['seg_image']

        fr = Features(superpixel_array, image, model_1_path)

        # Compute a set of region segmentation with *k_sel* regions
        seg_list = [fr.segmentation(k_sel=k) for k in list_hypothesis]

        # Next: Create a list of regions classes per proposed segmentation
        reg_list = [Regions(seg, image) for seg in seg_list]

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

                unique_global, count_global = np.unique(global_labels,return_counts=True)                
                #If there is a main class (>80%) then tag the region with this category, otherwise tag MIX
                if max(count_global)/sum(count_global) >= 0.8:
                    r_global = unique_global[np.argmax(count_global)]
                else:
                    r_global = 3 #The MIXED label

                unique_vertical, count_vertical = np.unique(vertical_labels,return_counts=True)                
                #If there is a main class (>80%) then tag the region with this category, otherwise tag MIX
                if max(count_vertical)/sum(count_vertical) >= 0.8:
                    r_vertical = unique_vertical[np.argmax(count_vertical)]
                else:
                    r_vertical = 5 #The MIXED label
            
                #Add the region to the training set
                training_data_X.append(X_features[r])
                training_data_y.append([r_global,r_vertical])

    # Save the dataset into .npy format
    X_train = np.array(training_data_X)
    y_train = np.array(training_data_y)
    np.save('X_train_region.npy', X_train)
    np.save('y_train_region.npy', y_train)
