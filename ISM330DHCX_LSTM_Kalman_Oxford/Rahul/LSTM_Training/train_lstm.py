import sys
import os

# === Add Data_Loader to path ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data_Loader')))
from data_loader import load_merged_data

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# === Load the data ===
print("[INFO] Loading data...")
X, y = load_merged_data('trolley/merged_data', sequence_length=30, stride=5, normalize=True)
print(f"[INFO] Loaded data: X shape = {X.shape}, y shape = {y.shape}")

# === Split into training and validation ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"[INFO] Training set: {X_train.shape}, Validation set: {X_val.shape}")

# === Build the LSTM model ===
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.3),
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(3)  # Output: x, y, z
])

model.compile(optimizer=Adam(learning_rate=1e-4), loss='huber', metrics=['mae'])
model.summary()

# === Callbacks ===
checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True, monitor="val_loss", mode="min")
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)

# === Train the model ===
print("[INFO] Training model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=[checkpoint, early_stop, reduce_lr],
    verbose=1
)

# === Evaluate the model ===
loss, mae = model.evaluate(X_val, y_val)
print(f"[INFO] Validation MAE: {mae:.4f}")

# === Plot training history ===
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title("MAE")
plt.legend()

plt.tight_layout()
plt.show()

# === Predict and plot ===
y_pred = model.predict(X_val[:100])

# Plot X coordinate
plt.figure(figsize=(10, 5))
plt.plot(y_val[:100, 0], label='True X')
plt.plot(y_pred[:, 0], label='Predicted X')
plt.title("X Coordinate: Prediction vs Ground Truth")
plt.xlabel("Sample")
plt.ylabel("X position")
plt.legend()
plt.grid()
plt.show()

# 2D trajectory plot
plt.figure(figsize=(8, 6))
plt.plot(y_val[:100, 0], y_val[:100, 1], label="True Trajectory", marker='o')
plt.plot(y_pred[:, 0], y_pred[:, 1], label="Predicted Trajectory", marker='x')
plt.title("2D Position Prediction (X vs Y)")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid()
plt.axis("equal")
plt.show()
