import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import felzenszwalb
from skimage.segmentation import mark_boundaries
from skimage.color import label2rgb
from skimage import measure
from skimage.color import rgb2hsv
from PIL import Image
import random
import scipy.io as sio
import pickle
from scipy.stats import entropy


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
            d_new_reg[r_sp] = sorted([(np.mean([self.similarity(r_sp, u, d_features) for u in d_new_reg if d_new_reg[u] == x]), x)
                                        for x in selected_reg], key=lambda tup: tup[0])[-1][1]

        # Update the region array with the merges
        for k in d_new_reg:
            if d_new_reg[k] != k:
                region_array = np.where(
                    region_array == k, d_new_reg[k], region_array)

        return region_array


class Regions():
    '''Region class to compute region-level features'''

    def __init__(self, segmentation_image, image):

        self.segmentation_image = segmentation_image
        self.image = image
        self.hsv_image = rgb2hsv(self.image)

        # compute the regions_props
        self.region_props = self.compute_region_prop(self.segmentation_image)

    def compute_region_prop(self, array):
        '''Compute the region properties of a labeled array'''
        return measure.regionprops(array)

    def compute_region_sp_list(self, superpixel_array):
        '''For each region, return label of superpixels inside'''

        d_sp_in_r = {}

        for region in self.region_props:

            coords = region.coords
            label = region.label
            sp_labels = np.unique(
                superpixel_array[coords[:, 0], coords[:, 1]])
            d_sp_in_r[label] = sp_labels

        return d_sp_in_r

    def compute_features(self):
        '''Compute the feature vector for each region'''


        # Reset the region feature vector
        d_features = {}

        for region in self.region_props:

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

            # Location percentile/number
            perc_10_coords_x = np.percentile(coords[:, 0], 10)
            perc_10_coords_y = np.percentile(coords[:, 1], 10)
            perc_90_coords_x = np.percentile(coords[:, 0], 90)
            perc_90_coords_y = np.percentile(coords[:, 1], 90)

            # Number of superpixels
            nb_superpixels = region.area
            perc_convex = region.convex_area/region.area

            # DOOG Filters mean abs response of 12 filters

            d_features[label] = np.array([
                mean_coords_x, mean_coords_y, mean_intensity_r, mean_intensity_g, mean_intensity_b, mean_intensity_h, mean_intensity_s, mean_intensity_v,
                perc_10_coords_x, perc_10_coords_y, perc_90_coords_x, perc_90_coords_y, nb_superpixels, perc_convex
                ])

        return d_features


if __name__ == '__main__':

    # Open the image
    path_test = 'dataset/structure31.jpg'
    model_1_path = 'model_1.pk'
    image = np.array(Image.open(path_test))

    # Compute a set of superpixels
    superpixels = 1 + felzenszwalb(image, scale=3., sigma=0.9, min_size=1000)

    # Initialize the class to compute regions
    fr = Features(superpixels, image, model_1_path)

    # Compute a set of region segmentation with *k_sel* regions
    seg_list = [fr.segmentation(k_sel=7) for _ in range(2)]

    # Next: Create a class to compute features on region
    reg = Regions(seg_list[0], image)
    F = reg.compute_features()

    # Estimate the likelihood of a class to be homogenous

    # Estimate the class of a region1

    # Visualization
    plt.subplot(1, 3, 1)
    plt.imshow(label2rgb(superpixels, image))
    plt.subplot(1, 3, 2)
    plt.imshow(label2rgb(seg_list[0], image))
    plt.subplot(1, 3, 3)
    plt.imshow(label2rgb(seg_list[1], image))
    plt.show()
