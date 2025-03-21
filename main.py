import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense, Dropout
import requests
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

plt.style.use("fivethirtyeight")

def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(sequence):
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Load dataset (modify path if needed)
while True:
    try:
        dataset = pd.read_csv("dataset.csv")  # Ensure this file exists
        break  # Exit loop if loading is successful
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        time.sleep(10)  # Retry after 10 seconds

# Preprocessing
dataset.columns = dataset.columns.str.strip()
df = dataset[(dataset["Gender"] == "f") & (dataset["Age"] == 19)]
hr = df["HR"].values.reshape(-1, 1)

# Process heart rate data
lk = len(hr)
newl = [hr[0]]
gol = 59

for i in range(lk):
    if i == 483:
        break
    if gol == 0:
        newl.append(hr[i])
        gol = 59
    gol -= 1

newl = np.array(newl).reshape(-1, 1)

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_hr = scaler.fit_transform(newl)

# Prepare LSTM dataset
n_steps_in, n_steps_out = 3, 3
X, y = split_sequence(scaled_hr, n_steps_in, n_steps_out)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

# Build model
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(150, return_sequences=False))
model.add(Dropout(0.6))
model.add(Dense(50))
model.add(Dense(1))

optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss="mean_squared_error")

# Train model
model.fit(X_train, y_train, batch_size=1, epochs=100)

# API URLs
bpm_url = "https://www.tejasswami.shop/bpm/1737205139712"
alert_url = "https://www.tejasswami.shop/alerts/alert"

def send_request():
    """Fetch BPM data from the API."""
    try:
        response = requests.get(bpm_url, timeout=10)
        response.raise_for_status()
        bpm_data = response.json()
        return bpm_data.get("heartBPM", 0)
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {e}")
        return None  # Return None if request fails

# Keep running indefinitely
g = [80, 81, 80]

while True:
    try:
        BPM1 = send_request()
        if BPM1 is None:
            logging.warning("Skipping iteration due to API failure.")
            time.sleep(60)
            continue

        g.append(BPM1)
        del g[0]

        # Reshape & scale
        g = np.array(g).reshape(-1, 1)
        g = scaler.transform(g).reshape((g.shape[0], g.shape[1], 1))

        # Make predictions
        predictions = model.predict(g)
        predictions = scaler.inverse_transform(predictions)

        Validate = predictions[-1][0]  # Get last prediction value
        logging.info(f"Predicted BPM: {Validate}")

        # Send alert if BPM is too high or too low
        if Validate >= 160 or Validate <= 50:
            try:
                response = requests.post(alert_url, json={"fPrint": "1737205139712"})
                response.raise_for_status()
                logging.info(f"Alert sent! Response: {response.json()}")
            except requests.exceptions.RequestException as e:
                logging.error(f"Failed to send alert: {e}")

        time.sleep(60)  # Run every minute

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        time.sleep(10)  # Retry after 10 seconds
