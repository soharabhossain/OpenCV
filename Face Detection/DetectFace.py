
# import the necessary packages
import cv2


#-----------------------------------------------------------------------------------------------
class DetectFace:
	def __init__(self, faceCascadePath):
		# load the face detector
		self.faceCascade = cv2.CascadeClassifier(faceCascadePath)

	def detect(self, image, scaleFactor = 1.1, minNeighbors = 5, minSize = (30, 30)):
		# detect faces in the image
		rects = self.faceCascade.detectMultiScale(image,
			scaleFactor = scaleFactor, minNeighbors = minNeighbors,
			minSize = minSize, flags = cv2.CASCADE_SCALE_IMAGE)

		# return the rectangles representing bounding
		# boxes around the faces
		return rects

#----------------------------------------------------------------------------------------------


# Select the input file 
#input_file ="C:\Python27\PyML\My Special Collection\CV_FaceDetection\images\sundar.png"
input_file ="C:\Python27\PyML\My Special Collection\CV_FaceDetection\images\group_pic_2.png"

# Face detector haarcascade-file
face ="C:\Python27\PyML\My Special Collection\CV_FaceDetection\cascades\haarcascade_frontalface_default.xml"

# load the image and convert it to grayscale
image = cv2.imread(input_file)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find faces in the image
fd = DetectFace(face)
faceRects = fd.detect(gray, scaleFactor = 1.2, minNeighbors = 5, minSize = (30, 30))

print("\n Found {} no. of faces".format(len(faceRects)))

# Loop over the faces and draw a rectangle around each
for (x, y, w, h) in faceRects:
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Mark the facial regions on the input-image
cv2.imshow("Faces", image)

cv2.waitKey(0)

