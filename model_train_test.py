import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("historical_stock_data.csv")  # Ensure this file exists
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

# Use only "Close" prices
data = df["Close"].values.reshape(-1, 1)

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Define parameters
TIME_STEP = 60  # Days used for training
EPOCHS = 50
BATCH_SIZE = 32
TRAIN_RATIO = 0.7  # 70% training, 30% testing

# Prepare training data
X, y = [], []
for i in range(TIME_STEP, len(data_scaled)):
    X.append(data_scaled[i - TIME_STEP:i, 0])
    y.append(data_scaled[i, 0])

X, y = np.array(X), np.array(y)
train_size = int(len(X) * TRAIN_RATIO)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape input for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(TIME_STEP, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)  # Output layer (stock price prediction)
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")

# Train model
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))

# Save model
model.save("stock_price_model.h5")

# Save test data for future predictions
np.save("recent_stock_data.npy", X_test[-1])  # Last window for predictions
np.save("actual_stock_prices.npy", scaler.inverse_transform(y_test.reshape(-1, 1)))  # Actual prices
np.save("stock_dates.npy", df.index[-len(y_test):].astype(str))  # Corresponding dates

# Plot training loss
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Test Loss")
plt.legend()
plt.show()

print("✅ Model trained and saved successfully!")
