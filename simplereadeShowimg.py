import cv2
from matplotlib import pyplot as plt
import numpy as np

img = cv2.imread('Images/band.jpeg')
print type(img)

plt.subplot(122), plt.imshow(img), plt.title('normal read')
plt.xticks([]), plt.yticks([])
plt.show()
imgbgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

plt.subplot(122), plt.imshow(imgbgr), plt.title('converted to BRG color')
plt.xticks([]), plt.yticks([])
plt.show()

imggrey = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2GRAY)
plt.subplot(122), plt.imshow(imggrey), plt.title('BRG to grey')
plt.xticks([]), plt.yticks([])
plt.show()


imgrgbgray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.subplot(122), plt.imshow(imgrgbgray), plt.title('RGB to grey')
plt.xticks([]), plt.yticks([])
plt.show()