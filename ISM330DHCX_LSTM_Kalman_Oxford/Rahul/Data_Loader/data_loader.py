import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

def load_merged_data(base_path, sequence_length=30, stride=5, normalize=True):
    X = []
    y = []

    all_csv_paths = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith('.csv'):
                all_csv_paths.append(os.path.join(root, file))

    all_csv_paths.sort()

    scaler = MinMaxScaler()

    # Collect all data for fitting scaler
    raw_inputs = []

    for file_path in all_csv_paths:
        df = pd.read_csv(file_path, header=None)
        if df.shape[1] < 5:
            print(f"[WARNING] Skipping {file_path}, expected at least 5 columns.")
            continue
        raw_inputs.append(df.iloc[:, :3].values.astype(np.float32))  # AX, AY, AZ

    raw_inputs_concat = np.vstack(raw_inputs)

    # Fit scaler if needed
    if normalize:
        scaler.fit(raw_inputs_concat)
        joblib.dump(scaler, 'scaler.save')
        print("[INFO] Scaler fitted and saved as 'scaler.save'.")

    # Process files again for sequence creation
    for file_path in all_csv_paths:
        df = pd.read_csv(file_path, header=None)
        data = df.values.astype(np.float32)

        acc_data = data[:, :3]    # AX, AY, AZ
        pos_data = data[:, 3:5]   # GT_X, GT_Y

        if normalize:
            acc_data = scaler.transform(acc_data)

        for i in range(0, len(acc_data) - sequence_length, stride):
            X.append(acc_data[i:i+sequence_length])
            y.append(pos_data[i + sequence_length - 1])

    return np.array(X), np.array(y)
