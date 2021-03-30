import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import felzenszwalb
from skimage.segmentation import mark_boundaries
from skimage.color import label2rgb
from skimage import measure
from skimage.color import rgb2hsv, rgb2gray
from PIL import Image
import random
import scipy.io as sio
import pickle
from scipy.stats import entropy
import matplotlib.colors as mcolors
from matplotlib import cm
from utils import get_doog_filter_list, compute_filter_response
from utils import compute_line_features
import time
import cProfile
class Features():

    def __init__(self, superpixel_array, image, model_1_path):

        self.superpixel_array = superpixel_array
        self.image = image
        self.hsv_image = rgb2hsv(self.image)

        # Load the first model for similarity computation
        if model_1_path:
            with open(model_1_path, 'rb') as f:
                self.model_1 = pickle.load(f)
        else:
            self.model_1 = None 
        
        #Load the DooG filter list
        self.doog_filters = get_doog_filter_list()
        self.doog_response = compute_filter_response(self.doog_filters,self.image)

    def compute_features(self, region_props=None):
        '''Compute the feature vector for each region'''

        if not region_props:
            region_props = self.compute_region_prop(self.superpixel_array)

        # Reset the region feature vector
        d_features = {}

        for region in region_props:

            label = region.label
            coords = region.coords
            pix_intensity = self.image[coords[:, 0], coords[:, 1], :]
            pix_intensity_hsv = self.hsv_image[coords[:, 0], coords[:, 1], :]

            d_features[label] = []

            # Coords mean values
            mean_coords_x = np.mean(coords[:, 0])/self.image.shape[0]
            mean_coords_y = np.mean(coords[:, 1])/self.image.shape[1]
            d_features[label].extend([mean_coords_x,mean_coords_y])

            # RGB mean values   
            mean_intensity_r = np.mean(pix_intensity[:, 0])/255.
            mean_intensity_g = np.mean(pix_intensity[:, 1])/255.
            mean_intensity_b = np.mean(pix_intensity[:, 2])/255.
            d_features[label].extend([mean_intensity_r,mean_intensity_g,mean_intensity_b])

            # HSV mean values
            mean_intensity_h = np.mean(pix_intensity_hsv[:, 0])
            mean_intensity_s = np.mean(pix_intensity_hsv[:, 1])
            mean_intensity_v = np.mean(pix_intensity_hsv[:, 2])
            d_features[label].extend([mean_intensity_h,mean_intensity_s,mean_intensity_v])

            # DOOG Filters mean abs response of 12 filters
            doog_response = self.doog_response[:,coords[:,0],coords[:,1]]
            doog_mean_response = np.mean(doog_response,axis=1)
            dog_global_mean = np.mean(doog_mean_response)
            dog_argmax = np.argmax(doog_mean_response)
            dog_stats = np.max(doog_mean_response) - np.median(doog_response)   
            d_features[label].extend(list(doog_mean_response)+[dog_global_mean,dog_argmax,dog_stats])
            
            d_features[label] = np.array(d_features[label])

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
        unique = list(np.unique(region_array))
        nb_unique = len(unique)
        np.random.shuffle(unique)
        selected_reg, remaining_sp = unique[:k_sel], unique[k_sel:]

        # Dict storing the new regions
        d_new_reg2 = {r: r for r in selected_reg}

        X_features = np.array([[np.abs(d_features[reg_a]-d_features[reg_b]) for reg_b in unique] for reg_a in unique]).reshape((-1,23))
        #Predict all the features
        y_probas = self.model_1.predict_proba(X_features)[:,1].reshape((len(d_features),len(d_features)))

        for r_sp in remaining_sp:
            d_new_reg2[r_sp] = sorted([(np.mean([y_probas[unique.index(r_sp),unique.index(u)] for u in d_new_reg2 if d_new_reg2[u] == x]), x)
                                        for x in selected_reg], key=lambda tup: tup[0])[-1][1]

        # Update the region array with the merges
        for k in d_new_reg2:
            if d_new_reg2[k] != k:
                region_array = np.where(
                    region_array == k, d_new_reg2[k], region_array)

        return region_array


class Regions():
    '''Region class to compute region-level features'''

    def __init__(self, segmentation_image, image):

        self.segmentation_image = segmentation_image
        self.image = image
        self.hsv_image = rgb2hsv(self.image)

        # compute the regions_props
        self.region_props = self.compute_region_prop(self.segmentation_image)

        #Load the DooG filter list
        self.doog_filters = get_doog_filter_list()
        self.doog_response = compute_filter_response(self.doog_filters,self.image)

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


            d_features[label] = []

            # Coords mean values
            mean_coords_x = np.mean(coords[:, 0])/self.image.shape[0]
            mean_coords_y = np.mean(coords[:, 1])/self.image.shape[1]
            d_features[label].extend([mean_coords_x,mean_coords_y])

            # RGB mean values   
            mean_intensity_r = np.mean(pix_intensity[:, 0])/255.
            mean_intensity_g = np.mean(pix_intensity[:, 1])/255.
            mean_intensity_b = np.mean(pix_intensity[:, 2])/255.
            d_features[label].extend([mean_intensity_r,mean_intensity_g,mean_intensity_b])

            # HSV mean values
            mean_intensity_h = np.mean(pix_intensity_hsv[:, 0])
            mean_intensity_s = np.mean(pix_intensity_hsv[:, 1])
            mean_intensity_v = np.mean(pix_intensity_hsv[:, 2])
            d_features[label].extend([mean_intensity_h,mean_intensity_s,mean_intensity_v])

            # Hue/Saturation Histogram and entropy
            hue_hist = np.histogram(pix_intensity_hsv[:, 0],bins=5, range=(0.,1.))[0]
            hue_hist = hue_hist/sum(hue_hist)
            saturation_hist = np.histogram(pix_intensity_hsv[:, 1],bins=3, range=(0.,1.))[0]
            saturation_hist = saturation_hist/sum(saturation_hist)
            hue_entropy = entropy(hue_hist)
            saturation_entropy = entropy(saturation_hist)
            d_features[label].extend(list(hue_hist)+list(saturation_hist)+[hue_entropy,saturation_entropy])


            # DOOG Filters mean abs response of 12 filters
            doog_response = self.doog_response[:,coords[:,0],coords[:,1]]
            doog_mean_response = np.mean(doog_response,axis=1)
            dog_global_mean = np.mean(doog_mean_response)
            dog_argmax = np.argmax(doog_mean_response)
            dog_stats = np.max(doog_mean_response) - np.median(doog_response)   
            d_features[label].extend(list(doog_mean_response)+[dog_global_mean,dog_argmax,dog_stats])
            
            # Location percentile/number
            perc_10_coords_x = np.percentile(coords[:, 0], 10)
            perc_10_coords_y = np.percentile(coords[:, 1], 10)
            perc_90_coords_x = np.percentile(coords[:, 0], 90)
            perc_90_coords_y = np.percentile(coords[:, 1], 90)
            d_features[label].extend([perc_10_coords_x,perc_10_coords_y,perc_90_coords_x,perc_90_coords_y])

            # Number of superpixels
            nb_superpixels = region.area
            perc_convex = region.convex_area/region.area
            d_features[label].extend([nb_superpixels,perc_convex])

            # Convex Hull
            nb_convex_contours = len(measure.find_contours(region.convex_image, 0.8))
            d_features[label].extend([nb_convex_contours])

            #Line features (Geometry)
            geom_features = compute_line_features(region.bbox,self.image)
            d_features[label].extend(geom_features)

            d_features[label] = np.array(d_features[label])

        return d_features


## Visualization Function ##
def global_prediction(image,superpixels,prediction_dict):
    '''Compute the vizual output of the global prediction'''

    output_label = np.zeros_like(superpixels)

    for sp in np.unique(superpixels):        
        sp_prediction = np.argmax(prediction_dict[sp])
        output_label = np.where(superpixels==sp,sp_prediction,output_label)
    
    return mix_image_global(image,output_label), output_label

def mix_image_global(image,global_prediction_array, alpha=0.4):
    seg_array = global_prediction_array/2
    colors = [(0, "green"), (1/2, "red"), (2/2, 'blue')]
    mycmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
    return alpha*mycmap(seg_array)[:,:,:3] + (1-alpha)*image/255.


def vertical_prediction(image,superpixels,prediction_dict,vertical_dict):
    '''Compute the vizual output of the global prediction'''

    output_label = np.zeros_like(superpixels)-1.

    for sp in np.unique(superpixels):
        
        sp_prediction_general = np.argmax(prediction_dict[sp])   
        if sp_prediction_general == 1:
            sp_prediction_vertical = np.argmax(vertical_dict[sp])
            output_label = np.where(superpixels==sp,sp_prediction_vertical,output_label)

    return mix_image_vertical(image,output_label), output_label

def mix_image_vertical(image,vertical_prediction_array, alpha=0.7):
    seg_array = (vertical_prediction_array+1)/5
    #Right, Center, Left, Porous, Solid
    colors = [(0., "black"), (1/5, "blue"), (2/5, "orange"), (3/5,"red"), (4/5,'green'), (1.,'purple')]
    mycmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
    return alpha*mycmap(seg_array)[:,:,:3] + (1-alpha)*image/255.



def compute_global_subvertical_segmentation(
    path_test,
    model_1_path='model_1.pk',
    model_global_path='model_general.pk',
    model_vertical_path='model_vertical.pk',
    nb_hypothesis = 5,
    number_regions_hypothesis = [3,4,5,7,9,11,15,20,25],
    min_superpixel_size = 1000,

):
    '''
    Main function computing the global and subvertical segmentation
    Params:
    path_test : path of the image to segment
    model_1_path: path of the superpixel_affinity saved model
    model_global_path: path of the global segmentation saved model
    model_vertical_path: path of the sub-vertical segmentation saved model
    nb_hypothesis: number of different segmentations to perform to get predictions
    number_region_hypothesis: list of number of desired regions at the end of last algorithm
    min_superpixel_size: minimum size of superpixels for the initial segmentation
    '''
    
    image = np.array(Image.open(path_test))

    list_hypothesis = random.sample(number_regions_hypothesis,nb_hypothesis)

    # Load the global model 
    with open(model_global_path, 'rb') as f:
        model_global = pickle.load(f)
    
    # Load the vertical model 
    with open(model_vertical_path, 'rb') as f:
        model_vertical = pickle.load(f)

    # Compute a set of superpixels
    superpixels = 1 + felzenszwalb(image, scale=3., sigma=0.9, min_size=min_superpixel_size)


    #For each superpixel, store a score per global region
    superpixel_score_general = {k:[] for k in np.unique(superpixels)}
    superpixel_score_vertical = {k:[] for k in np.unique(superpixels)}


    # Initialize the class to compute regions
    fr = Features(superpixels, image, model_1_path)

    # Compute a set of region segmentation with *k_sel* regions
    seg_list = [fr.segmentation(k_sel=k) for k in list_hypothesis]

    # Next: Create a class to compute features on region
    reg_list = [Regions(seg, image) for seg in seg_list]

    for regions in reg_list:

        # For each region, return list of superpixels inside
        d_sp_in_r = regions.compute_region_sp_list(superpixels)

        # Compute feature vector for each region
        feat_dict = regions.compute_features()

        # Estimate the likelihood of a class to be homogenous
        predict_proba_classes_general = {k:model_global.predict_proba(feat_dict[k].reshape(1,-1))[0] for k in feat_dict} 
        predict_proba_classes_vertical = {k:model_vertical.predict_proba(feat_dict[k].reshape(1,-1))[0] for k in feat_dict} 


        #For each superpixel, look at in which region it is and add the proba_arrays to superpixel_score
        for region in d_sp_in_r:
            superpixels_inside = d_sp_in_r[region]

            proba_vector_general = predict_proba_classes_general[region]
            proba_vector_general[:3] = proba_vector_general[:3]/sum(proba_vector_general[:3])
            proba_vector_general[-1] = 1-proba_vector_general[-1]

            proba_vector_vertical = predict_proba_classes_vertical[region]
            proba_vector_vertical[:5] = proba_vector_vertical[:5]/sum(proba_vector_vertical[:5])
            proba_vector_vertical[-1] = 1-proba_vector_vertical[-1]

            for superpixel in superpixels_inside:
                superpixel_score_general[superpixel].append(proba_vector_general)
                superpixel_score_vertical[superpixel].append(proba_vector_vertical)

    
    #Superpixel prediction
    superpixels_pred_general = {}
    superpixels_pred_vertical = {}

    #For each superpixel, compute the final score per region
    for superpixel in superpixel_score_general:
        
        homogeneity_general = [hypoth[-1] for hypoth in superpixel_score_general[superpixel]]
        homogeneity_general = homogeneity_general/sum(homogeneity_general) #Normalize

        #Compute the prediction per global class
        prediction_general = np.array([h*label_prediction[:3] for h,label_prediction in zip(homogeneity_general,superpixel_score_general[superpixel])]).reshape((-1,3))
        prediction_general = np.sum(prediction_general, axis=0)
        superpixels_pred_general[superpixel] = prediction_general

        homogeneity_vertical = [hypoth[-1] for hypoth in superpixel_score_vertical[superpixel]]
        homogeneity_vertical = homogeneity_vertical/sum(homogeneity_vertical) #Normalize

        #Compute the prediction per global class
        prediction_vertical = np.array([h*label_prediction[:5] for h,label_prediction in zip(homogeneity_vertical,superpixel_score_vertical[superpixel])]).reshape((-1,5))
        prediction_vertical = np.sum(prediction_vertical, axis=0)
        superpixels_pred_vertical[superpixel] = prediction_vertical


    # Vizualization of the global segmentation
    global_output, g_label_map = global_prediction(image,superpixels, superpixels_pred_general)
    # Visualization of the vertical segmentation
    vertical_output, v_label_map = vertical_prediction(image,superpixels, superpixels_pred_general, superpixels_pred_vertical)

    result = {
        'global_output':global_output,
        'g_label_map':g_label_map,
        'vertical_output':vertical_output,
        'v_label_map':v_label_map,
        'initial_superpixels':superpixels,
        'segmentation_hyp':seg_list,
        }
    return result


if __name__ == '__main__':

    # Open the image
    path_test = 'dataset_test\city10.jpg' 
    model_1_path = 'model_1.pk'
    model_global_path = 'model_general.pk'
    model_vertical_path = 'model_vertical.pk'
    image = np.array(Image.open(path_test))

    #Hyperparameters
    nb_hypothesis = 5
    number_regions_hypothesis = [3,4,5,7,9,11,15,20,25]
    min_superpixel_size = 1000

    cp = cProfile.Profile()
    cp.enable()

    result = compute_global_subvertical_segmentation(
        path_test,
        model_1_path,
        model_global_path,
        model_vertical_path,
        nb_hypothesis,
        number_regions_hypothesis,
        min_superpixel_size
    )

    cp.disable()
    filename = 'profile.prof'  # You can change this if needed
    cp.dump_stats(filename)

    # Visualization
    plt.subplot(1, 5, 1)
    plt.title('Superpixel Segmentation')
    plt.imshow(label2rgb(result['initial_superpixels'], image))
    plt.subplot(1, 5, 2)
    plt.title('Region Hypothesis 1')
    plt.imshow(label2rgb(result['segmentation_hyp'][0], image))
    plt.subplot(1, 5, 3)
    plt.title('Region Hypothesis 2')
    plt.imshow(label2rgb(result['segmentation_hyp'][1], image))
    plt.subplot(1, 5, 4)
    plt.title('Global Segmentation')
    plt.imshow(result['global_output'])
    plt.subplot(1, 5, 5)
    plt.title('Vertical Segmentation')
    plt.imshow(result['vertical_output'])
    plt.show()