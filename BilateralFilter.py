# openCV
import cv2
import numpy as np
from matplotlib import pyplot as plt

# apply bilateral filter
img = cv2.imread('baboon.png')
blur = cv2.bilateralFilter(img, 9, 75, 75)
cv2.imwrite("bilateralFilter.png", blur)

# display image
plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(blur), plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()
