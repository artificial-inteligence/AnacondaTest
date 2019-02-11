import cv2
from CV import PreProcessor, FaceDetector, HarrisonCornerDetector
import matplotlib.image as mpimg

# import numpy as np
from matplotlib import pyplot as plt

# rgb image picked up in bgr format
imgrgb = cv2.imread('Images/manFace.jpeg')
#rgb image colour is off as it reads bgr
plt.subplot(122), plt.imshow(imgrgb), plt.title('original image')
plt.xticks([]), plt.yticks([])
plt.show()

preProcessor = PreProcessor()
# convert to BGR
imgbgr = preProcessor.convert_to_bgr(imgrgb)

plt.subplot(122), plt.imshow(imgbgr), plt.title('after convert to bgr image')
plt.xticks([]), plt.yticks([])
plt.show()


# gray scale returns blue/green...
greyScaleImg = preProcessor.applygreyscale(imgrgb)
plt.subplot(122), plt.imshow(greyScaleImg), plt.title('after Applying Grey Scale')
plt.xticks([]), plt.yticks([])
plt.show()

# Smooth image
bFilter = preProcessor.applybilateralfilter(imgbgr)

# Detect Faces
faceDetector = FaceDetector()
faces = faceDetector.detectfaces(bFilter)

print("run ran")

detector = HarrisonCornerDetector()

# display image
for face in faces:
    results = detector.detect_cornders(face)
    plt.subplot(122), plt.imshow(results), plt.title('Faces')
    plt.xticks([]), plt.yticks([])
    plt.show()
