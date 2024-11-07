import cv2
import numpy as np
from robotpy_apriltag import AprilTagDetector as apriltag

video_path = 'videos/new_fiducials.mjpeg'
# Open the video file
cap = cv2.VideoCapture(video_path)
    
# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")

_, img = cap.read()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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