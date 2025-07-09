# ============================================================
# Kalman Filter Workflow (Short Summary of Steps)
# ============================================================
# 1) Define state vector [X] and control input [U] from Sensor 1
# 2) Predict next state using non-linear motion model
# 3) Compute Jacobian matrix [F] to linearize prediction
# 4) Define process noise covariance matrix [Q] (from Sensor 1 datasheet)
# 5) Fine-tune Q for responsiveness vs smoothness
# 6) Predict error covariance matrix [P]
# 7) Get measurement vector [Z] from Sensor 2 (same format as X)
# 8) Compute predicted measurement vector [Ẑ] from state prediction
# 9) Define measurement Jacobian [H]
# 10) Compute innovation (residual): Y = Z - Ẑ
# 11) Define measurement noise covariance matrix [R] (from Sensor 2 datasheet)
# 12) Compute innovation covariance [S]
# 13) Calculate Kalman Gain [K]
# 14) Update state estimate: X = X + K * Y → Final filtered values (x, y, yaw)
# 15) Update error covariance: P = (I - K * H) * P
# ============================================================

import numpy as np
import time

class KalmanFilter:
    def __init__(self):
        # === 1) State Vector Initialization ===
        self.x = np.zeros((3, 1))  # [x, y, yaw]
        self.v = 0.2               # Constant velocity (control input U)

        # === 6) Initial Error Covariance Matrix [P] ===
        self.P = np.eye(3) * 1.0

        # === 4) Process Noise Covariance Matrix [Q] ===
        self.Q = np.diag([1e-9, 1e-9, 1e-9])  # From Sensor 1 (IMU) datasheet

        # === 11) Measurement Noise Covariance Matrix [R] ===
        self.R = np.array([
            [0.01, 0, 0],
            [0, 0.01, 0],
            [0, 0, 2.45e-7]
        ])  # From Sensor 2 (e.g., UWB/GPS) datasheet

        # === 9) Measurement Jacobian [H] ===
        self.H = np.eye(3)

        # === 7) Placeholder for measurement vector [Z] ===
        self.z = None

        # Time interval
        self.dt = 0.1

    # === 3) Compute Jacobian Matrix [F] ===
    def get_jacobian(self, theta):
        dt = self.dt
        v = self.v
        Fk = np.array([
            [1, 0, -v * np.sin(theta) * dt],
            [0, 1,  v * np.cos(theta) * dt],
            [0, 0, 1]
        ])
        return Fk

    # === 7) Set Measurement Vector [Z] ===
    def set_measurement(self, x, y, yaw):
        self.z = np.array([[x], [y], [yaw]])

    def step(self):
        theta = self.x[2, 0]
        dt = self.dt
        v = self.v

        # === 2) Predict Next State (Non-linear model) ===
        self.x[0, 0] += v * np.cos(theta) * dt
        self.x[1, 0] += v * np.sin(theta) * dt

        # === 3) Compute Jacobian ===
        Fk = self.get_jacobian(theta)

        # === 6) Predict Error Covariance Matrix ===
        self.P = Fk @ self.P @ Fk.T + self.Q

        if self.z is not None:
            # === 8) Predicted Measurement Vector [Ẑ] ===
            z_hat = self.H @ self.x

            # === 10) Innovation (Residual) ===
            Y = self.z - z_hat

            # === 12) Innovation Covariance [S] ===
            S = self.H @ self.P @ self.H.T + self.R

            # === 13) Kalman Gain [K] ===
            K = self.P @ self.H.T @ np.linalg.inv(S)

            # === 14) Update State Estimate ===
            self.x = self.x + K @ Y
            # 🔹 Final filtered output values (x, y, yaw) after correction using sensor data

            # === 15) Update Error Covariance ===
            I = np.eye(3)
            self.P = (I - K @ self.H) @ self.P

            self.z = None

        return self.x


# === Example Simulation ===
if __name__ == '__main__':
    kf = KalmanFilter()

    # Simulated measurements from Sensor 2: (x, y, yaw)
    uwb_measurements = [
        (1.0, 0.5, 0.0),
        (1.2, 0.6, 0.05),
        (1.4, 0.7, 0.1),
        (1.6, 0.8, 0.15)
    ]

    for i in range(20):
        if i < len(uwb_measurements):
            kf.set_measurement(*uwb_measurements[i])

        state = kf.step()
        print(f"Step {i+1}: x = {state[0,0]:.3f}, y = {state[1,0]:.3f}, yaw = {state[2,0]:.3f}")
        time.sleep(kf.dt)