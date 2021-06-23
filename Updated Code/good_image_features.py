'''
Shi-Tomasi Corner Detector & Good Features to Track
N strongest corners in the image by Shi-Tomasi method (or Harris Corner Detection, if you specify it)
'''

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('Lena.tiff') # read the image as gray-scale
print(img.shape)

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
print(gray.shape)

# How many points you want to detect?
NoOfPoints=25;

corners = cv2.goodFeaturesToTrack(gray,NoOfPoints,0.01,10)
corners = np.int0(corners)

for i in corners:
    x, y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)


img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()
