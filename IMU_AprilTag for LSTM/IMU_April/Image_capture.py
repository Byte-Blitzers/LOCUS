import cv2
import numpy as np
from pupil_apriltags import Detector

# === Initialize camera ===
cap = cv2.VideoCapture("http://10.251.2.214:8080/video")  # Replace with your IP cam URL if needed
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# === AprilTag detector ===
at_detector = Detector(
    families='tag36h11',
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)

print("Press ESC to exit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = at_detector.detect(gray)

    for tag in tags:
        corners = tag.corners
        pts = [tuple(map(int, pt)) for pt in corners]
        cv2.polylines(frame, [np.array(pts)], True, (0, 255, 0), 2)
        cv2.putText(frame, str(tag.tag_id), pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("AprilTag Detection", frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
