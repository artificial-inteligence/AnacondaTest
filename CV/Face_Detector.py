import cv2
from matplotlib import pyplot as plt


class FaceDetector:

    def __init__(self):
        self.cascPath = "CV/haarcascade_frontalface.xml"
        self.faceCascade = cv2.CascadeClassifier(self.cascPath)
        pass

    def detectfaces(self, Img):
        # detect faces
        faces = self.faceCascade.detectMultiScale(
            Img,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        # extract faces
        print("Found {0} faces!".format(len(faces)))
        extractedfaces = []
        for (x, y, w, h) in faces:
            # cv2.rectangle(greyScaleImg, (x, y), (x + w, y + h), (0, 255, 0), 2)
            region = Img[y:(y + h), x:(x + w)]
            extractedfaces.append(region)
        return extractedfaces
