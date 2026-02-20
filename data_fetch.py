import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def fetch_stock_data(stock_symbol="AAPL", start_date="2020-01-01", end_date="2024-01-01"):
    data = yf.download(stock_symbol, start=start_date, end=end_date, interval="1d")
    return data[['Close']]

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

if __name__ == "__main__":
    data = fetch_stock_data()
    print(data.head())  # Check if data is fetched properly
