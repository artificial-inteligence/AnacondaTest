import cv2
from PreProcessing import PreProcessor

import PreProcessing
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Images/face-2.jpg')
preProcessor = PreProcessor()

greyScaleImg = preProcessor.applygreyscale(img)
bFilter = preProcessor.applybilateralfilter(greyScaleImg)
print("run ran")

#cv2.imwrite("bilateralFilter.png", bFilter)

#display image

plt.subplot(122), plt.imshow(bFilter), plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()
