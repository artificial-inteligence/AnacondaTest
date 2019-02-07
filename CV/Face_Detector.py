import cv2
from matplotlib import pyplot as plt


class FaceDetector:

    def __init__(self):
        self.cascPath = "CV/haarcascade_frontalface.xml"
        self.faceCascade = cv2.CascadeClassifier(self.cascPath)
        pass

    def detectfaces(self, greyScaleImg):
        faces = self.faceCascade.detectMultiScale(
            greyScaleImg,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces

    def drawresult(self, facesimage, originalimage):
        print "Found {0} faces!".format(len(facesimage))

        for (x, y, w, h) in facesimage:
            cv2.rectangle(originalimage, (x, y), (x + w, y + h), (0, 255, 0), 2)

            region = originalimage[y:(y + h), x:(x + w)]
            # display image
            plt.subplot(122), plt.imshow(region), plt.title('Blurred')
            plt.xticks([]), plt.yticks([])
            plt.show()

