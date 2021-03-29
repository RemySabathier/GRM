# GRM
Geometric Context from single Image (GRM Assignment)


## Steps

1. Over-segmentation method by Felzenszwalbet al (Efficient graph-basedimage segmentation,IJCV)

2. More complex grometric features:
* Standard segmentation algorithm : features for this algo are local : little chanve of obatianing large regions

* Evaluate all possible segmentations : sample a small number of segamentation (sampling sets of superpixels)

3. Multiple segmentation hypotheses : find wxhich part of the hypotheses are correct

## Features

"Automatic photopop-up" Hoiem et al.

olor16C1. RGB values: mean3C2. HSV values: C1 in HSV space3C3. Hue: histogram (5 bins) and entropy6C4. Saturation: histogram (3 bins) and entropy4Texture15T1. DOOG filters: mean abs response of 12 filters12T2. DOOG stats: mean of variables in T11T3. DOOG stats: argmax of variables in T11T4. DOOG stats: (max - median) of variables in T11Location and Shape12L1. Location: normalized x and y, mean2L2. Location: norm. x and y,10thand90thpctl4L3. Location: norm. y wrt horizon,10th,90thpctl2L4. Shape: number of superpixels in region1L5. Shape: number of sides of convex hull1L6. Shape:num pixels/area(convex hull)1L7. Shape: whether the region is contiguous∈{0,1}13D Geometry35G1. Long Lines: total number in region1G2. Long Lines: % of nearly parallel pairs of lines1G3. Line Intsctn: hist. over 12 orientations, entropy13G4. Line Intsctn: % right of center1G5. Line Intsctn: % above center1G6. Line Intsctn: % far from center at 8 orientations8G7. Line Intsctn: % very far from center at 8 orient.8G8. Texture gradient: x and y “edginess” (T2) center


## Learning

300 images, each image is over segmented (150 000 superpixels)
50 images to train
250 to cross validate
dataset : http://www.cs.cmu.edu/afs/cs.cmu.edu/Web/People/dhoiem/projects/context/index.html

### Generating segmentation

per image : multiple segmenjtation
varying the number of regions and the initializations

Greedy algo :
1. Randomly order superpixels
2. assign the first $n_r$ superpixels to different regions
3. iteratively assign each remaining superpixel based on a learned pairwise affinity function
4. repeat step 3 several times

Number of regions : $n_r \in \left{ 3, 4, 5, 7, 9, 11, 15, 20, 25 \right}$

Sample pairs of same label and different label superpixels from our training set
likelyhood that two superpixels have the same label
logistic regression form of Adaboost
Each likelihood function in the weak learner is obtained using kernel density estimation



### Geometric Labeling

For each region (set of homogeneous and contiguous superpixels)
Compute features 


## Results

