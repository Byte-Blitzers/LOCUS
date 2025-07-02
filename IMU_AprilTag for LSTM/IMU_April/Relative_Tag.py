import cv2
import numpy as np
from pupil_apriltags import Detector

# --- Camera calibration (from your earlier result) ---
camera_matrix = np.array([
    [1490.18420, 0.0,        983.576162],
    [0.0,        1488.64741, 529.259638],
    [0.0,        0.0,        1.0]
])
dist_coeffs = np.array([0.0499695131, -0.360262769, -0.00039303398, 0.00151007513, 0.986424759])

# AprilTag and Tag size (5.5 cm => 0.055 m)
tag_size = 0.055
# Set your tag IDs accordingly
ROVER_ID = 0
ANCHOR_ID = 1

# Initialize the detector
detector = Detector(families="tag36h11",
                    nthreads=1,
                    quad_decimate=1.0,
                    quad_sigma=0.0,
                    refine_edges=1,
                    decode_sharpening=0.25,
                    debug=0)

# Initialize video capture (replace 0 with your IP camera stream URL if needed)
cap = cv2.VideoCapture("http://10.251.2.214:8080/video")
if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect tags with pose estimation
    tags = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=(camera_matrix[0,0], camera_matrix[1,1], camera_matrix[0,2], camera_matrix[1,2]),
        tag_size=tag_size
    )

    poses = {}
    for tag in tags:
        tag_id = tag.tag_id
        poses[tag_id] = tag.pose_t.flatten()

        # Draw tag edges and ID
        corners = np.int32(tag.corners)
        cv2.polylines(frame, [corners], True, (0, 255, 0), 2)
        cv2.putText(frame, f"ID {tag_id}", tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # Compute relative position if both tags detected
    if ROVER_ID in poses and ANCHOR_ID in poses:
        rover = poses[ROVER_ID]
        anchor = poses[ANCHOR_ID]
        rel = rover - anchor  # vector from anchor to rover (in camera frame)
        cv2.putText(frame, f"Rel Pos (m): x={rel[0]:.2f}, y={rel[1]:.2f}, z={rel[2]:.2f}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    cv2.imshow("Rover-vs-Anchor AprilTags", frame)
    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
