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

    
import math
def dot(vA, vB):
    return vA[0]*vB[0]+vA[1]*vB[1]
def ang(lineA, lineB):
    # Get nicer vector form
    vA = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
    vB = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]
    # Get dot prod
    dot_prod = dot(vA, vB)
    # Get magnitudes
    magA = dot(vA, vA)**0.5
    magB = dot(vB, vB)**0.5
    # Get cosine value
    cos_ = dot_prod/magA/magB
    # Get angle in radians and then convert to degrees
    angle = math.acos(dot_prod/magB/magA)
    # Basically doing angle <- angle mod 360
    ang_deg = math.degrees(angle)%360

    if ang_deg-180>=0:
        # As in if statement
        return 360 - ang_deg
    else: 

        return ang_deg

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return (x, y)