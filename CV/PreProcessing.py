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

    def applygreyscale(self, imgrgb):
        imgbgr = cv2.cvtColor(imgrgb, cv2.COLOR_RGB2BGR)

        grey_scale = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2GRAY)

        return grey_scale
#
# # display image
# plt.subplot(121), plt.imshow(img), plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(blur), plt.title('Blurred')
# plt.xticks([]), plt.yticks([])
# plt.show()
