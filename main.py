import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import felzenszwalb
from skimage.segmentation import mark_boundaries
from skimage import measure
from PIL import Image
import random
import scipy.io as sio


def compute_features(region_array):
    '''Compute the feature vector per region/superpixel'''
    unique_regions = list(np.unique(region_array))
    d = {r: np.random.randn(10) for r in unique_regions}
    return d


def similarity(reg_a, reg_b, feature_dict):
    '''Compute the similarity of two regions based on the feature dictionary'''
    return np.sum(feature_dict[reg_a]*feature_dict[reg_b])


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
        param['vert_labels'] = X[n][9]
        param['horz_labels'] = X[n][10]
        param['label_names'] = X[n][11]
        param['vert_names'] = X[n][12]
        param['horz_names'] = X[n][13]

        d[file_name] = param

    return d


def segmentation(image, k_sel=32, n_iterations=50):
    '''Main function to compute the segmentation'''

    # Array storing the region of pixels
    region_map = np.zeros((image.shape[0], image.shape[1]))

    # First step : Generate superpixels
    segments_sp = felzenszwalb(image, scale=100, sigma=0.1, min_size=1000)

    # Boolean to monitor the termination
    change = True

    while change:

        change = False

        # Compute the feature dict of each region
        d_feat = compute_features(segments_sp)

        # Randomly order them and assign the first k to different regions
        unique = np.unique(segments_sp)
        np.random.shuffle(unique)
        random_reg, remaining_sp = unique[:k_sel], unique[k_sel:]

        # Dict storing the new regions
        d_new_reg = {r: r for r in random_reg}
        for r_sp in remaining_sp:
            d_new_reg[r_sp] = sorted([(similarity(r_sp, x, d_feat), x)
                                      for x in random_reg], key=lambda tup: tup[0])[-1][1]

        for k in d_new_reg:
            if d_new_reg[k] != k:
                change = True
                segments_sp = np.where(
                    segments_sp == k, d_new_reg[k], segments_sp)

    return segments_sp
