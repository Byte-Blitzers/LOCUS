import sys
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from joblib import load  # for loading scaler
import matplotlib.pyplot as plt

# === Add Data_Loader path ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data_Loader')))

# === Load trained model and scaler ===
model = load_model('best_model.keras')  # or best_model.h5
scaler = load('scaler.save')

# === Test File Path ===
test_file = 'trolley/merged_data/data1/merged7.csv'

# === Prepare Data ===
def load_test_sequence(file_path, sequence_length=30, stride=5):
    df = pd.read_csv(file_path, header=None)
    data = df.values.astype(np.float32)

    acc_data = data[:, :3]         # AX, AY, AZ
    gt_positions = data[:, 3:5]    # GT X, Y

    acc_scaled = scaler.transform(acc_data)

    X = []
    y_true = []

    for i in range(0, len(acc_scaled) - sequence_length, stride):
        X.append(acc_scaled[i:i+sequence_length])
        y_true.append(gt_positions[i + sequence_length - 1])

    return np.array(X), np.array(y_true)

print("[INFO] Loading test sequence...")
X_test, y_true = load_test_sequence(test_file)
print(f"[INFO] X_test shape: {X_test.shape}")
print(f"[INFO] y_true shape: {y_true.shape}")

# === Prediction ===
print("[INFO] Predicting...")
y_pred = model.predict(X_test)

# === 2D Plot ===
plt.figure(figsize=(8, 6))
plt.plot(y_true[:, 0], y_true[:, 1], label='Ground Truth', color='blue')
plt.plot(y_pred[:, 0], y_pred[:, 1], label='Predicted', color='orange')
plt.title("2D Trajectory: Ground Truth vs Predicted")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.legend()
plt.grid(True)
plt.axis('equal')  # Keeps aspect ratio square
plt.tight_layout()
plt.show()
