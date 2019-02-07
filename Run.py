import cv2
from CV import PreProcessor, FaceDetector

import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Images/abba.png')
preProcessor = PreProcessor()

greyScaleImg = preProcessor.applygreyscale(img)
bFilter = preProcessor.applybilateralfilter(greyScaleImg)

faceDetector = FaceDetector()
faces = faceDetector.detectfaces(bFilter)


print("run ran")

#cv2.imwrite("bilateralFilter.png", bFilter)

#display image
for face in faces:
    plt.subplot(122), plt.imshow(face), plt.title('Faces')
    plt.xticks([]), plt.yticks([])
    plt.show()
