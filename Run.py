import cv2

import BilateralFilter
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('baboon.png')



#cv2.imwrite("bilateralFilter.png", bFilter)

#display image
plt.subplot(122), plt.imshow(bFilter), plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()
