import numpy as np
import tensorflow as tf
import pandas as pd

# Load trained model
model = tf.keras.models.load_model("stock_price_model.h5")

# Load recent stock data for predictions
past_data = np.load("recent_stock_data.npy")  # Ensure this file exists

# Define parameters
TIME_STEP = 60  # Number of past days used for prediction
future_days = 90  # Predict for 3 months (daily)

def predict_future_prices(model, past_data, future_days=90):
    """
    Predicts future stock prices based on trained model.

    Parameters:
    - model: Trained LSTM model
    - past_data: Recent stock prices used as input
    - future_days: Number of future days to predict

    Returns:
    - future_predictions: List of predicted stock prices
    """
    future_predictions = []
    input_data = past_data[-TIME_STEP:].reshape(1, TIME_STEP, 1)  # Initial input

    for _ in range(future_days):
        predicted_price = model.predict(input_data)  # Predict next day’s price
        future_predictions.append(predicted_price[0, 0])  # Store prediction

        # Append predicted price for the next prediction step
        input_data = np.append(input_data[:, 1:, :], [[predicted_price]], axis=1)

    return future_predictions

# Generate predictions
predicted_prices = predict_future_prices(model, past_data, future_days)

# Load actual stock data for comparison
actual_data = np.load("actual_stock_prices.npy")  # Ensure this file exists
dates = np.load("stock_dates.npy")  # Load dates

# Save predictions for visualization
np.save("predicted_prices.npy", predicted_prices)

print("Predictions saved! Run visualization.py to see the results.")
