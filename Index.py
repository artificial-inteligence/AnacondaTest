import cv2
from matplotlib import pyplot as plt

from skimage import io, color
img = io.imread('baboon.png')
dimensions = color.guess_spatial_dimensions(img)
print(dimensions)

# get  shape
import skimage.io as io
from skimage.color import rgb2gray
img = io.imread('baboon.png')
print (img.shape)

# create grey scale
import skimage.io as io
from skimage.color import rgb2gray
img = io.imread('baboon.png')
img_grayscale = rgb2gray(img)
io.imsave('baboon-gs.png', img_grayscale)

# view image
show_grayscale = io.imshow(img_grayscale)
io.show()





#opencv
# apply bilateral filter
img = cv2.imread('baboon-gs.png')
blur = cv2.bilateralFilter(img, 9, 75, 75)
cv2.imwrite("bilateralFilter.png", blur)

# display image
plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(blur), plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()