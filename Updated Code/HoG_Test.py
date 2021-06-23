import numpy as np
import matplotlib.pyplot as plt
import cv2


# HOG features
# ---------------------------------------------

from skimage.feature import hog
from skimage import data, color, exposure

# -----------------------------------------------------------------------------
def HoG_Func ( input_image, cell_size ):

 image = color.rgb2gray(input_image)
 
 fd, hog_image = hog(image, orientations=8, pixels_per_cell=(cell_size, cell_size), cells_per_block=(1, 1), visualize=True)

 fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

 ax1.axis('on')
 ax1.imshow(image, cmap=plt.cm.gray)
 ax1.set_title('Input image')
 ax1.set_adjustable('box')

 # Rescale histogram for better display
 hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

 ax2.axis('on')
 ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
 ax2.set_title('Histogram of Oriented Gradients')
 ax2.set_adjustable('box')
 plt.show()

# -----------------------------------------------------------------------------


# ------------------------------------
# Main Function
# ------------------------------------
my_image=cv2.imread('Lena.tiff')
#my_image=cv2.imread('Elaine.tiff')

#print(my_image.shape)
#print(my_image.dtype)
HoG_Func(my_image, 32)


