# Difference of Absolute Gaussians filters implementation
# Code largely inspired : https://github.com/teeters/GeometricContext

from skimage.filters import gaussian
import numpy as np
from skimage.transform import rotate
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage.color import rgb2gray

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

    
