
#import cv2

img1 = cv2.imread('/home/soharab/images/Lena.tiff')
img2 = cv2.imread('/home/soharab/images/Soharab.png')

cv2.imshow('Lena', img1)
cv2.imshow('Soharab', img2)

cv2.waitKey()
cv2.destroyAllWindows()
