import cv2
import numpy as np
from matplotlib import pyplot as plt


class HarrisonCornerDetector:
    def __init__(self):
        pass

    def detect_cornders(self, img, detail=0.05, rgbcolour=None):
        if rgbcolour is None:
            rgbcolour = [255, 0, 0]

        imgGray32 = np.float32(img)
        dst = cv2.cornerHarris(imgGray32, 2, 3, 0.04)

        # result is dilated for marking the corners, not important
        dst = cv2.dilate(dst, None)

        # Threshold for an optimal value, it may vary depending on the image.
        img[dst > detail * dst.max()] = rgbcolour
        plt.subplot(122), plt.imshow(img), plt.title('Faces')
        plt.xticks([]), plt.yticks([])
        plt.show()
        return img
