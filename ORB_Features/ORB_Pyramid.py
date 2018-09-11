from skimage import data
import numpy as np
from skimage import transform as tf
from skimage.feature import (match_descriptors, corner_harris,
                             corner_peaks, ORB, plot_matches)
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from skimage.transform import pyramid_gaussian

from skimage import color

import cv2

def scale_space (image, downscale_val, sigma_val):

 #rows, cols, dim = image.shape
 rows, cols = image.shape

 # pyramid contains the images convolved with Gaussian kernels
 pyramid = tuple(pyramid_gaussian(image, downscale=downscale_val, sigma=sigma_val))

 #no_pyramid_images=len(pyramid)
 #print(no_pyramid_images)
  
 #for i in range (1, no_pyramid_images):  
 # fig, ax = plt.subplots()
 # ax.imshow(pyramid[i])  # pyramid[0] is the original image
 # plt.show()
 return pyramid

# ---------------------------------------------
# Main
# ---------------------------------------------

img1 = rgb2gray(data.astronaut())

#img1=rgb2gray(cv2.imread('Elaine.tiff'))


downscale_val=1.2
sigma_val=2.5

img_pyramid = scale_space (img1, downscale_val, sigma_val)

#no_pyramid_images=len(img_pyramid)
#print(no_pyramid_images)
img1=img_pyramid[0]  # Original image
#print(img1.dtype)

descriptor_extractor = ORB(n_keypoints=200)
descriptor_extractor.detect_and_extract(img1)
keypoints1 = descriptor_extractor.keypoints
descriptors1 = descriptor_extractor.descriptors

fig, ax = plt.subplots(nrows=7, ncols=1)
plt.gray()

for i in range (1, 8):
 
 img2=img_pyramid[i]
 descriptor_extractor.detect_and_extract(img2)
 keypoints2 = descriptor_extractor.keypoints
 descriptors2 = descriptor_extractor.descriptors

 matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)

 print("\n No. of matching points with pyramid image {} is {}".format(i,len(matches12)))
 #print(matches12)
 
 plot_matches(ax[i-1], img1, img2, keypoints1, keypoints2, matches12)
 ax[i-1].axis('off')
 

plt.show()

 
