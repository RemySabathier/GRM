import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import felzenszwalb
from skimage.segmentation import mark_boundaries
from skimage import measure
from skimage.color import rgb2hsv
from PIL import Image
import random
import scipy.io as sio
import pickle


class Features():

    def __init__(self, region_array, image, model_1_path):

        self.region_array = region_array
        self.image = image
        self.hsv_image = rgb2hsv(self.image)
        self.d_features = {}  # dict of features
        self.region_prop = {}

        # Load the first model for similarity computation
        with open(model_1_path, 'rb') as f:
            self.model_1 = pickle.load(f)

    def compute_features(self):
        '''Compute the feature vector for each region'''

        # Update region_prop
        self.update_region_prop()
        # Reset the region feature vector
        self.d_features = {}

        for region in self.region_prop:

            label = region.label
            coords = region.coords
            pix_intensity = self.image[coords[:, 0], coords[:, 1], :]
            pix_intensity_hsv = self.hsv_image[coords[:, 0], coords[:, 1], :]

            # RGB mean values
            mean_intensity_r = np.mean(pix_intensity[:, 0])/255.
            mean_intensity_g = np.mean(pix_intensity[:, 1])/255.
            mean_intensity_b = np.mean(pix_intensity[:, 2])/255.

            # HSV mean values
            mean_intensity_h = np.mean(pix_intensity_hsv[:, 0])
            mean_intensity_s = np.mean(pix_intensity_hsv[:, 1])
            mean_intensity_v = np.mean(pix_intensity_hsv[:, 2])

            # Coords mean values
            mean_coords_x = np.mean(coords[:, 0])/self.image.shape[0]
            mean_coords_y = np.mean(coords[:, 1])/self.image.shape[1]

            # DOOG Filters mean abs response of 12 filters

            self.d_features[label] = np.array([
                mean_coords_x, mean_coords_y, mean_intensity_r, mean_intensity_g, mean_intensity_b, mean_intensity_h, mean_intensity_s, mean_intensity_v])

    def update_region_prop(self):
        self.region_prop = measure.regionprops(self.region_array)

    def similarity(self, reg_a, reg_b):
        '''Compute the similarity of two regions based on the feature dictionary'''
        # It should be a learned pairwise affinity function
        # It should be a ML model that gives a probability/similarity
        return self.model_1.predict_proba(
            np.abs(self.d_features[reg_a]-self.d_features[reg_b]).reshape((1, -1)))[0][1]


def segmentation(image, k_sel=32):
    '''Main function to compute the segmentation'''

    # Array storing the region of pixels
    image = np.array(image)
    region_map = np.zeros((image.shape[0], image.shape[1]))

    # First step : Generate superpixels
    segments_sp = 1 + felzenszwalb(image, scale=100, sigma=0.1, min_size=1000)

    # Instanciate Feature Object storing feature vectors of each superpixel
    feat_store = Features(segments_sp, image, 'model_1.pk')

    # Boolean to monitor the termination
    change = True

    while change:

        change = False

        # Compute the feature dict of each region
        feat_store.compute_features()

        # Randomly order them and assign the first k to different regions
        unique = np.unique(segments_sp)
        np.random.shuffle(unique)
        random_reg, remaining_sp = unique[:k_sel], unique[k_sel:]

        # Dict storing the new regions
        d_new_reg = {r: r for r in random_reg}
        for r_sp in remaining_sp:
            d_new_reg[r_sp] = sorted([(feat_store.similarity(r_sp, x), x)
                                      for x in random_reg], key=lambda tup: tup[0])[-1][1]

        for k in d_new_reg:
            if d_new_reg[k] != k:
                change = True
                segments_sp = np.where(
                    segments_sp == k, d_new_reg[k], segments_sp)

    return segments_sp


if __name__ == '__main__':

    # Open the image
    path_test = 'dataset/structure31.jpg'
    image = Image.open(path_test)

    # Compute the segmentation
    X = segmentation(image, k_sel=4)
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.imshow(X)
    plt.show()
