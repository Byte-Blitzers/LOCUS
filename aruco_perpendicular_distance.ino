"""
INSTRUCTIONS FOR ACCURATE DISTANCE ESTIMATION USING ARUCO MARKERS:

1. This script estimates the distance between the camera and ArUco markers using computer vision.
2. It uses pre-calibrated camera parameters for accurate pose estimation.
3. The physical size of the marker must be known and correctly set in MARKER_LENGTH.
4. The camera must face the marker **directly and perpendicularly** (i.e., in line of sight).
5. Any significant tilt or angle between the camera and the marker will reduce accuracy.
6. Use a flat, well-lit surface and avoid lens distortion for best results.
7. Press 'q' to exit the video stream.
"""

import cv2
import cv2.aruco as aruco
import numpy as np

# ==== Load camera calibration ====
with np.load("camera_calibration.npz") as data:
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]

# ==== Set marker size in meters ====
MARKER_LENGTH = 0.05  # 5 cm

# ==== ArUco marker dictionary ====
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
()

# ==== Start video capture ====
cap = cv2.VideoCapture(2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_LENGTH, camera_matrix, dist_coeffs)

        for i in range(len(ids)):
            aruco.drawDetectedMarkers(frame, corners)
            
            #  Use drawFrameAxes instead of aruco.drawAxis
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.03)

            distance_m = tvecs[i][0][2]
            distance_cm = distance_m * 100* 2

            x, y = int(corners[i][0][0][0]), int(corners[i][0][0][1])
            text = f"ID:{ids[i][0]} Dist:{distance_cm:.1f}cm"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Aruco Distance Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()