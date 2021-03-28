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

    def __init__(self, superpixel_array, image, model_1_path):

        self.superpixel_array = superpixel_array
        self.image = image
        self.hsv_image = rgb2hsv(self.image)

        # Load the first model for similarity computation
        with open(model_1_path, 'rb') as f:
            self.model_1 = pickle.load(f)

    def compute_features(self, region_props):
        '''Compute the feature vector for each region'''

        # Reset the region feature vector
        d_features = {}

        for region in region_props:

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

            d_features[label] = np.array([
                mean_coords_x, mean_coords_y, mean_intensity_r, mean_intensity_g, mean_intensity_b, mean_intensity_h, mean_intensity_s, mean_intensity_v])

        return d_features

    def compute_region_prop(self, array):
        '''Compute the region properties of a labeled array'''
        return measure.regionprops(array)

    def similarity(self, reg_a, reg_b, d_features):
        '''Compute the similarity of two regions based on the feature dictionary'''
        return self.model_1.predict_proba(
            np.abs(d_features[reg_a]-d_features[reg_b]).reshape((1, -1)))[0][1]

    def segmentation(self, k_sel=32):
        '''Main function to compute a region segmentation'''
        # Boolean to monitor the termination
        change = True

        # The region array : start with the initial superpixels segmentation
        region_array = np.copy(self.superpixel_array)

        while change:
            change = False

            # Compute the regions props
            region_props = self.compute_region_prop(region_array)

            # Compute the feature dict of each region
            d_features = self.compute_features(region_props)

            # Randomly order them and assign the first k to different regions
            unique = np.unique(region_array)
            np.random.shuffle(unique)
            selected_reg, remaining_sp = unique[:k_sel], unique[k_sel:]

            # Dict storing the new regions
            d_new_reg = {r: r for r in selected_reg}
            for r_sp in remaining_sp:
                d_new_reg[r_sp] = sorted([(self.similarity(r_sp, x, d_features), x)
                                          for x in selected_reg], key=lambda tup: tup[0])[-1][1]

            # Update the region array with the merges
            for k in d_new_reg:
                if d_new_reg[k] != k:
                    change = True
                    region_array = np.where(
                        region_array == k, d_new_reg[k], region_array)

        return region_array


if __name__ == '__main__':

    # Open the image
    path_test = 'dataset/structure31.jpg'
    model_1_path = 'model_1.pk'
    image = np.array(Image.open(path_test))

    # Compute a set of superpixels
    superpixels = 1 + felzenszwalb(image, scale=100, sigma=0.1, min_size=1000)

    # Initialize the class to compute regions
    fr = Features(superpixels, image, model_1_path)

    # Compute a region segmentation with *k_sel* regions
    X = fr.segmentation(k_sel=4)
    X2 = fr.segmentation(k_sel=4)

    # Visualization
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.subplot(1, 3, 2)
    plt.imshow(X)
    plt.subplot(1, 3, 3)
    plt.imshow(X2)
    plt.show()
