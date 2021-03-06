# openCV
import cv2

import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray


class PreProcessor:

    def __init__(self):
        pass

    def applybilateralfilter(self, img):
        # apply bilateral filter on image
        blur = cv2.bilateralFilter(img, 9, 75, 75)
        return blur

    def applygreyscale(self, img):
        greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return greyscale
#
# # display image
# plt.subplot(121), plt.imshow(img), plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(blur), plt.title('Blurred')
# plt.xticks([]), plt.yticks([])
# plt.show()
