import cv2
from CV import PreProcessor, FaceDetector

import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Images/manFace.jpeg')
preProcessor = PreProcessor()

greyScaleImg = preProcessor.applygreyscale(img)
bFilter = preProcessor.applybilateralfilter(greyScaleImg)

faceDetector = FaceDetector()
faces = faceDetector.detectfaces(bFilter)
faceDetector.drawresult(faces, bFilter)

print("run ran")

#cv2.imwrite("bilateralFilter.png", bFilter)

#display image

plt.subplot(122), plt.imshow(bFilter), plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()
