import cv2
from CV import PreProcessor, FaceDetector
import matplotlib.image as mpimg

# import numpy as np
from matplotlib import pyplot as plt

# rgb image picked up in bgr format
imgrgb = cv2.imread('Images/manFace.jpeg')

preProcessor = PreProcessor()
# convert to BGR then gray scale
greyScaleImg = preProcessor.applygreyscale(imgrgb)
# Smooth image
bFilter = preProcessor.applybilateralfilter(greyScaleImg)

# Detect Faces
faceDetector = FaceDetector()
faces = faceDetector.detectfaces(bFilter)

print("run ran")

# cv2.imwrite("bilateralFilter.png", bFilter)

# display image
for face in faces:
    plt.subplot(122), plt.imshow(face), plt.title('Faces')
    plt.xticks([]), plt.yticks([])
    plt.show()
