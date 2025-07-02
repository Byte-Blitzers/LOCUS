import cv2
import numpy as np
import glob
import os

# === Calibration Settings ===
CHECKERBOARD = (10, 7)  # Inner corners (columns - 1, rows - 1)
square_size = 15  # mm

# === Prepare object points ===
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []  # 3D real-world points
imgpoints = []  # 2D image points

# === Load Images ===
image_folder = 'captured_images'  # Ensure this folder exists
images = glob.glob(os.path.join(image_folder, '*.jpg'))

if not images:
    print("No calibration images found.")
    exit()

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Could not read image: {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # === Find corners with robust flags ===
    ret, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if ret:
        corners_sub = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        objpoints.append(objp)
        imgpoints.append(corners_sub)

        cv2.drawChessboardCorners(img, CHECKERBOARD, corners_sub, ret)
        cv2.imshow('Corners Detected', img)
        cv2.waitKey(100)
    else:
        print(f"Warning: Corners not detected in {fname}")

cv2.destroyAllWindows()

# === Camera Calibration ===
if len(objpoints) < 5:
    print("Not enough valid images with corners detected. Calibration aborted.")
    exit()

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    None,
    None
)

# === Save Results ===
np.savez('calibration_result.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

# === Output Results ===
print("\n=== Calibration Results ===")
print("Camera Matrix:\n", camera_matrix)
print("\nDistortion Coefficients:\n", dist_coeffs.ravel())
print("\nRe-projection Error:", ret)
print("Calibration saved to calibration_result.npz")
