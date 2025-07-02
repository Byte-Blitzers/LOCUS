import cv2
import numpy as np
import csv
import time
import threading
import socket
from pupil_apriltags import Detector

# === IMU Receiver Thread ===
latest_imu = {
    "timestamp": 0,
    "accel": ["null", "null", "null"],
    "gyro": ["null", "null", "null"]
}

def imu_listener():
    global latest_imu
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(("10.251.2.40", 8000))  # Replace with your ESP8266 IP
            print("[INFO] Connected to ESP8266 IMU")
            while True:
                line = s.recv(128).decode().strip()
                parts = line.split(",")
                if len(parts) == 7:
                    latest_imu = {
                        "timestamp": int(parts[0]),
                        "accel": [float(parts[1]), float(parts[2]), float(parts[3])],
                        "gyro":  [float(parts[4]), float(parts[5]), float(parts[6])]
                    }
        except Exception as e:
            print(f"[WARN] IMU connection error: {e}")
            time.sleep(1)

# Start IMU thread
threading.Thread(target=imu_listener, daemon=True).start()

# === Camera Calibration ===
camera_matrix = np.array([
    [1490.18420, 0.0,        983.576162],
    [0.0,        1488.64741, 529.259638],
    [0.0,        0.0,        1.0]
])
dist_coeffs = np.array([0.0499695131, -0.360262769, -0.00039303398, 0.00151007513, 0.986424759])

# === AprilTag Detector ===
tag_size = 0.055  # meters
ROVER_ID = 0
ANCHOR_ID = 1

detector = Detector(families="tag36h11",
                    nthreads=1,
                    quad_decimate=1.0,
                    quad_sigma=0.0,
                    refine_edges=1,
                    decode_sharpening=0.25,
                    debug=0)

# === Video Stream ===
cap = cv2.VideoCapture("http://10.251.2.214:8080/video")
if not cap.isOpened():
    print("Cannot open camera stream")
    exit()

# === CSV Setup ===
csv_file = open("april_imu_log.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "timestamp_ms",
    "tag_id",
    "pose_x", "pose_y", "pose_z",
    "rel_x", "rel_y", "rel_z",
    "imu_time",
    "accel_x", "accel_y", "accel_z",
    "gyro_x", "gyro_y", "gyro_z"
])

print("Logging started. Press ESC to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame capture failed")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=(camera_matrix[0, 0], camera_matrix[1, 1],
                       camera_matrix[0, 2], camera_matrix[1, 2]),
        tag_size=tag_size
    )

    poses = {}
    for tag in tags:
        tag_id = tag.tag_id
        pose = tag.pose_t.flatten()
        poses[tag_id] = pose

        corners = np.int32(tag.corners)
        cv2.polylines(frame, [corners], True, (0, 255, 0), 2)
        cv2.putText(frame, f"ID {tag_id}", tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        now = int(time.time() * 1000)
        imu = latest_imu

        csv_writer.writerow([
            now,
            tag_id,
            *pose,
            "null", "null", "null",
            imu["timestamp"],
            *imu["accel"],
            *imu["gyro"]
        ])

    if ROVER_ID in poses and ANCHOR_ID in poses:
        rover = poses[ROVER_ID]
        anchor = poses[ANCHOR_ID]
        rel = rover - anchor
        cv2.putText(frame, f"Rel Pos (m): x={rel[0]:.2f}, y={rel[1]:.2f}, z={rel[2]:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        now = int(time.time() * 1000)
        imu = latest_imu

        csv_writer.writerow([
            now,
            "REL",
            "null", "null", "null",
            *rel,
            imu["timestamp"],
            *imu["accel"],
            *imu["gyro"]
        ])

    cv2.imshow("AprilTag + IMU", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()
print("Saved as april_imu_log.csv")
