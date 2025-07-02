import cv2
import numpy as np

# === Parameters ===
ip_url = 'http://10.251.2.214:8080/video'  # Replace with your phone's IP stream
calib_file = 'calibration_result.npz'

# === Load Camera Calibration ===
with np.load(calib_file) as data:
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']

# === Open the IP Camera ===
cap = cv2.VideoCapture(ip_url)
if not cap.isOpened():
    print("Failed to connect to the camera stream.")
    exit()

# Set resolution to 1080p (1920x1080)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

print("Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Undistort the frame
    undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)

    # Show both frames
    cv2.imshow("Original", frame)
    cv2.imshow("Undistorted", undistorted)

    if cv2.waitKey(1) == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
