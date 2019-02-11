import cv2
import numpy as np
from matplotlib import pyplot as plt


filename = 'Images/manFace.jpeg'
img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst, None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.005*dst.max()]=[255,0,0]

plt.subplot(122), plt.imshow(img), plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()

