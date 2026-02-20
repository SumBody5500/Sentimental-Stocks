import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import timedelta

def plot_predictions(actual_prices, predicted_prices, actual_dates):
    """
    Plots actual vs. predicted stock prices.

    Parameters:
    - actual_prices: List of real stock prices (used for testing).
    - predicted_prices: List of predicted stock prices.
    - actual_dates: List of dates corresponding to actual stock prices.
    """

    # Convert dates into a Pandas series for better handling
    actual_dates = pd.to_datetime(actual_dates)

    # Generate future dates for predicted prices
    future_dates = [actual_dates.iloc[-1] + timedelta(days=i) for i in range(1, len(predicted_prices) + 1)]

    # Plot actual prices
    plt.figure(figsize=(12, 6))
    plt.plot(actual_dates, actual_prices, label="Actual Prices", color="blue", linestyle="-", marker="o")

    # Plot predicted prices
    plt.plot(future_dates, predicted_prices, label="Predicted Prices", color="red", linestyle="dashed", marker="x")

    # Formatting the graph
    plt.xlabel("Date")
    plt.ylabel("Stock Price (USD)")
    plt.title("Stock Price Prediction vs Actual")
    plt.legend()
    plt.grid(True)
    
    # Rotate x-axis labels for better visibility
    plt.xticks(rotation=45)
    
    # Show the graph
    plt.show()

# Example Usage (Load Data and Call Function)
if __name__ == "__main__":
    # Load actual stock data (Modify this to match your dataset)
    actual_data = np.load("actual_stock_prices.npy")  # Load actual prices
    predicted_data = np.load("predicted_prices.npy")  # Load predicted prices
    dates = np.load("stock_dates.npy")  # Load corresponding dates

    # Plot the updated graph
    plot_predictions(actual_data, predicted_data, dates)
