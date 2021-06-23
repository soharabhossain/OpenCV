'''

Counting the Number of Objects in an Image

Counting number of objects from an image is a challenging task.

If we have an image with non-overlapping clear object contours then we can go as follows:

i) Convert the image in grayscale

ii) Reduce noise by applying smoothing (blurring)

iii) Apply edge detection -e.g. Canny's algorithm

iv) Finding contours on the edge map found from the previous steps

v) Count the number of closed contours found

'''

# import libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load image
img = cv2.imread("Bounding.png")


# Convert the original to grey scale and blur it to reduce noise
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)


# Display the gray image
plt.imshow(gray, cmap='gray')
plt.title('Gray Image')
plt.show()

# Display the blurred image
plt.imshow(blurred, cmap='gray')
plt.title('Blurred Image')
plt.show()


# Canny edge detection
canny = cv2.Canny(blurred, 10, 150)

# Show Canny's edge map
plt.imshow(canny, cmap='gray')
plt.title('Canny Edge Map')
plt.show()

# Approach 1

# find the contours
#(_, contours, _) = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
( contours, _) = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours: -1 will draw all contours
img1 = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
plt.imshow(img1)
plt.title('Contour Plot')
plt.show()


# Print how many coins we found
print("\n Found %i objects in the image." % len(contours))
