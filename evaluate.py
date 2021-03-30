import numpy as np
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from skimage.color import rgb2hsv, rgb2gray
from PIL import Image
import random
import scipy.io as sio
import pickle
import os
import matplotlib.colors as mcolors
from matplotlib import cm
from tqdm import tqdm
from utils import load_annotation
from main import compute_global_subvertical_segmentation
from main import mix_image_global, mix_image_vertical

from sklearn.metrics import classification_report, confusion_matrix

def ground_truth_from_annotation(annotations,img_path):

    img_params = annotations[img_path]
    segmentation = annotations[img_path]['seg_image']
    global_predictions = annotations[img_path]['vert_labels']
    vertical_predictions = annotations[img_path]['horz_labels']

    output_global = np.zeros_like(segmentation)
    output_vertical = np.zeros_like(segmentation)-1

    for sp in np.unique(segmentation):
        prediction_global = global_predictions[sp-1]-1
        prediction_vertical = vertical_predictions[sp-1]-1
        output_global = np.where(segmentation==sp,prediction_global,output_global)
        output_vertical = np.where(segmentation==sp,prediction_vertical,output_vertical)
    
    return output_global,output_vertical

def accuracy(prediction,ground_truth,n_classes):
    acc = [-1]*n_classes
    gt_labels = np.unique(ground_truth)
    for i in gt_labels:
        acc[i] = np.sum(prediction[ground_truth==i]==i)/np.sum(ground_truth==i)
    return acc

def conf_matrix(prediction,ground_truth,classes):
    
    n_classes = len(classes)

    confm = np.zeros((n_classes,n_classes))
    for id_i, i in enumerate(classes):
        for id_j, j in enumerate(classes):
            confm[id_i,id_j] = np.sum(prediction[ground_truth==j]==i)
        s = np.sum(confm[id_i])
        if s>0:
            confm[id_i]=confm[id_i]/s
    
    return confm


if __name__ == '__main__':

    train_repartition_path = 'dataset/train_test_repartition.pk'

    #Load the annotations
    annotation_path = r'dataset\allimsegs2.mat'
    dataset_path = 'dataset'
    annotations = load_annotation(annotation_path)

    #Test images
    with open(train_repartition_path, 'rb') as f:
        test_img_path = pickle.load(f)['Test']
    
    #Parameters
    model_1_path = 'pretrained_models/model_superpixel_similarity.pk'
    model_global_path = 'pretrained_models/model_general_segmentation.pk'
    model_vertical_path = 'pretrained_models/model_vertical_segmentation.pk'

    #Hyperparameters
    nb_hypothesis = 5
    number_regions_hypothesis = [3,4,5,7,9,11,15,20,25]
    min_superpixel_size = 1000

    metric_dict = {}

    for img_path in tqdm(test_img_path):
        
        path_test = os.path.join(dataset_path,img_path)
        image = np.array(Image.open(path_test))

        #Compute the segmentation
        result = compute_global_subvertical_segmentation(
            path_test,
            model_1_path,
            model_global_path,
            model_vertical_path,
            nb_hypothesis,
            number_regions_hypothesis,
            min_superpixel_size
        )
        
        prediction_g = result['g_label_map']
        prediction_v = result['v_label_map']

        #Build the ground truth from annotation vector
        gt_general, gt_vertical = ground_truth_from_annotation(annotations,img_path)

        #Compute the metrics

        cr_g = classification_report(gt_general.flatten(),prediction_g.flatten(),labels=[0,1,2],output_dict=True, zero_division=0)
        cfm_g = confusion_matrix(gt_general.flatten(),prediction_g.flatten(),labels=[0,1,2])

        cr_v = classification_report(gt_vertical.flatten(),prediction_v.flatten(),labels=[-1,0,1,2,3,4],output_dict=True, zero_division=0)
        cfm_v = confusion_matrix(gt_vertical.flatten(),prediction_v.flatten(),labels=[-1,0,1,2,3,4])
        
        metric_dict[img_path] = {
            'Metrics Global': cr_g,
            'Confusion Matrix Global': cfm_g,
            'Metrics Vertical': cr_v,
            'Confusion Matrix Vertical': cfm_v,
        }
    
    # Save the metric dictionnary
    with open('metric_report.pk', 'wb') as f:
        pickle.dump(metric_dict,f)