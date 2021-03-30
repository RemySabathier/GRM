# Difference of Absolute Gaussians filters implementation
# Code largely inspired : https://github.com/teeters/GeometricContext

from skimage.filters import gaussian
import numpy as np
from skimage.transform import rotate
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage.color import rgb2gray
import math
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line
from skimage.transform import hough_line, hough_line_peaks
import scipy.io as sio
from numba import jit
import time


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
        param['npixels'] = X[n][4].flatten()
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


def Doog_filters(sigma, r, theta, gSize):
    '''DooG filter implementation'''
    
    sigmaX = r*sigma
    sigmaY = sigma
    y0 = sigma
    a, b, c = -1, 2, 1

    #Create 3 gaussians
    gaussian1 = create_gaussian(0,y0,sigmaX,sigmaY,gSize)
    gaussian2 = create_gaussian(0,0,sigmaX,sigmaY,4*gSize+1)
    gaussian3 = create_gaussian(0,-y0,sigmaX,sigmaY,gSize)

    gaussian2 = rotate(gaussian2,theta)

    i1 = np.int((4*gSize+1)/2 - gSize/2 + 1)
    i2 = i1 + gSize

    gaussian2 = gaussian2[i1:i2,i1:i2]
    gaussian2 = gaussian2/np.sum(gaussian2)

    output = a*gaussian1 + b*gaussian2 + c*gaussian3
    return output/np.sum(output)

@jit(nopython=True)
def create_gaussian(x0,y0,sigmaX,sigmaY,gsize):

    radius = int((gsize-1)/2)
    G = np.zeros((gsize,gsize))

    for x in range(-radius,radius):
        for y in range(-radius,radius):
            G[x+radius+1,y+radius+1] = 1/(2*np.pi*sigmaX*sigmaY)*np.exp(-1*((x-x0)**2)/(2*sigmaX**2) - (y-y0)**2/(2*sigmaY**2))
    
    G = G/np.sum(G)
    return G

def get_doog_filter_list():
    '''return a list of DooG filters'''
    
    angle_step = 15
    filter_size = 5
    sigma = 1.5
    r = 0.25
    num_angles= int(180/angle_step)

    filter_list = []
    for angle in range(0,180,angle_step):
        filter_list.append(Doog_filters(sigma,r,angle,filter_size))
    
    return filter_list


def compute_filter_response(filter_list,img):
    grey_img = rgb2gray(img)
    return np.array([np.abs(convolve2d(grey_img,filter,mode='same') - grey_img) for filter in filter_list])

    
@jit(nopython=True)
def dot(vA, vB):
    return vA[0]*vB[0]+vA[1]*vB[1]

@jit(nopython=True)
def ang(lineA, lineB):
    # Get nicer vector form
    vA = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
    vB = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]
    # Get dot prod
    dot_prod = dot(vA, vB)
    # Get magnitudes
    magA = dot(vA, vA)**0.5
    magB = dot(vB, vB)**0.5
    # Get angle in radians and then convert to degrees
    angle = math.acos(dot_prod/(magB*magA))
    # Basically doing angle <- angle mod 360
    ang_deg = math.degrees(angle)%360
    if ang_deg-180>=0:
        # As in if statement
        return 360 - ang_deg
    else: 
        return ang_deg

@jit(nopython=True)
def det(a, b):
    return a[0] * b[1] - a[1] * b[0]

@jit(nopython=True)
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    div = det(xdiff, ydiff)
    if div == 0:
        #Lines do not intersect
        return [0.,0.]
       
    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return [x, y]

def compute_line_features(bbox,image):
    
    image_region = rgb2gray(image[bbox[0]:bbox[2],bbox[1]:bbox[3]])
    x_length, y_length = image_region.shape
    
    edges = canny(image_region, 2)
    lines = probabilistic_hough_line(edges, threshold=15, line_length=50, line_gap=10)

    intersections_lines = np.array([[line_intersection(lineA,lineB) for lineB in lines if lineB!=lineA] for lineA in lines]).reshape((-1,2))
    angle_lines = np.array([[ang(lineA,lineB) for lineB in lines if lineB!=lineA] for lineA in lines]).reshape((-1,1))

    return features_from_lines(intersections_lines,angle_lines,x_length,y_length,len(lines))


@jit(nopython=True)
def features_from_lines(intersections_lines,angle_lines,x_length,y_length,n_lines):

    nearly_par = np.sum(angle_lines<=5)
    right_inter, up_inter = np.sum(intersections_lines>0, axis=0)
    far_inter =  np.sum(intersections_lines[:,0]**2 + intersections_lines[:,1]**2 >x_length**2 + y_length**2)
    very_far_inter =  np.sum(intersections_lines[:,0]**2 + intersections_lines[:,1]**2 >3*(x_length**2 + y_length**2))

    total = n_lines*(n_lines-1)
    #Geometric Features
    G1 = n_lines
    G2 = nearly_par/(total+0.000001)
    G3 = total
    G4 = right_inter/(total+0.0001)
    G5 = up_inter/(total+0.00001)
    G6 = far_inter/(total+0.00001)
    G7 = very_far_inter/(total+0.00001)
    return [G1,G2,G3,G4,G5,G6,G7]