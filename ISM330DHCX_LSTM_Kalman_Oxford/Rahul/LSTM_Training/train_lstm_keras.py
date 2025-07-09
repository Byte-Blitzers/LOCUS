import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# === Import data loader ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data_Loader')))
from data_loader import load_merged_data

# === Load data ===
print("[INFO] Loading data...")
X, y = load_merged_data('trolley/merged_data', sequence_length=30, stride=5, normalize=True)
print(f"[INFO] Loaded: X = {X.shape}, y = {y.shape}")

# === Train-test split ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"[INFO] Train shape = {X_train.shape}, Val shape = {X_val.shape}")

# === LSTM Model ===
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(2)  # Output: X, Y
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# === Callbacks ===
checkpoint = ModelCheckpoint("best_model.keras", monitor="val_loss", save_best_only=True, mode="min")
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1)

# === Train ===
print("[INFO] Training model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=[checkpoint, early_stop, reduce_lr],
    verbose=1
)

# === Final Evaluation ===
loss, mae = model.evaluate(X_val, y_val)
print(f"[INFO] Final Validation MAE: {mae:.4f}")

# === Plot Loss & MAE ===
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title("MAE Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.legend()

plt.tight_layout()
plt.show()

# === Prediction Sanity Check ===
y_pred = model.predict(X_val[:100])

plt.figure(figsize=(10, 5))
plt.plot(y_val[:100, 0], label='True X')
plt.plot(y_pred[:, 0], label='Pred X')
plt.title("X Coordinate Prediction")
plt.xlabel("Sample")
plt.ylabel("X position")
plt.legend()
plt.grid(True)
plt.show()
