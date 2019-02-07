import cv2


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
        for (x, y, w, h) in facesimage:
            cv2.rectangle(originalimage, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return originalimage
