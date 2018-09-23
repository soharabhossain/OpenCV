
# Realtime Edge Detection with OpenCV

# import libraries   
import cv2  
  
# import numpy with standard alias 
import numpy as np 
  
  
# Capture frames from a camera 
cap = cv2.VideoCapture(0) 
  
  
# Loop over all the frames captured by the camera
# Continue doing it until the user presses the 'Escape' key
while(1): 
  
    # Reads frames from a camera 
    ret, frame = cap.read() 
  
    # Converting RGB frame to GRAY frame 
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) 
      

    # -------------------------------------------------------  
    # Compute Edge maps
    # -------------------------------------------------------  
    
    # Find edges in the input image and create the edge-map 
    edges = cv2.Canny(gray,100,200) 

    # Calcution of Sobelx 
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5) 
      
    # Calculation of Sobely 
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5) 
      
    # Calculation of Laplacian 
    laplacian = cv2.Laplacian(gray, cv2.CV_64F) 


    # ----------------------------------------------
    # Display the original frame and the edge maps 
    # -------------------------------------------------------  

    # Show the original frame
    cv2.imshow('Original',frame) 

    # Display edge maps 
    cv2.imshow('Canny', edges) 
    cv2.imshow('Sobel-x', sobelx) 
    cv2.imshow('Sobel-y', sobely) 
    cv2.imshow('Laplacian', laplacian) 

    # ----------------------------------------------
  
    # Wait for Esc key to stop 
    k = cv2.waitKey(5) & 0xFF
    if k == 27: 
        break
 
  
# Close the video stream 
cap.release() 
  
# De-allocate any associated memory usage 
cv2.destroyAllWindows()  





