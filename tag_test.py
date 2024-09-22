import cv2
import numpy as np
from robotpy_apriltag import AprilTagDetector as apriltag

imagepath = 'test2.png'
img = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
detector = apriltag()
detector.addFamily("tagStandard41h12")

detections = detector.detect(img)

# Put red dots on the tags and display the image
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

for detection in detections:
    center = detection.getCenter()
    cv2.circle(img, (int(center.x), int(center.y)), 5, (0, 0, 255), -1)
cv2.imshow("apriltag", img)
cv2.waitKey(0)