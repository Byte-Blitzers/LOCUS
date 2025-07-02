import cv2
import numpy as np
from pupil_apriltags import Detector

# === Calibration data ===
cameraMatrix = np.array([[1490.1842, 0.0, 983.576162],
                         [0.0, 1488.64741, 529.259638],
                         [0.0, 0.0, 1.0]])
distCoeffs = np.array([0.0499695131, -0.360262769, -0.00039303398, 0.00151007513, 0.986424759])

# Tag size in meters (5.5 cm)
tag_size = 0.055

# === Initialize webcam or IP camera ===
cap = cv2.VideoCapture("http://10.251.2.214:8080/video")  # Replace with your stream URL if needed

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# === Initialize detector ===
detector = Detector(
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

    tags = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=(cameraMatrix[0, 0], cameraMatrix[1, 1], cameraMatrix[0, 2], cameraMatrix[1, 2]),
        tag_size=tag_size
    )

    for tag in tags:
        # Draw tag border
        corners = np.int32(tag.corners)
        cv2.polylines(frame, [corners], isClosed=True, color=(0, 255, 0), thickness=2)

        # Tag ID
        cv2.putText(frame, f"ID: {tag.tag_id}", tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Pose
        t = tag.pose_t.flatten()
        x, y, z = t
        text = f"X: {x:.2f}m Y: {y:.2f}m Z: {z:.2f}m"
        cv2.putText(frame, text, (corners[0][0], corners[0][1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("AprilTag Pose Detection", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
