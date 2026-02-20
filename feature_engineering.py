import pandas as pd
import numpy as np

def add_technical_indicators(df):
    """
    Add moving averages, RSI, and Fourier Transform features to the stock data.
    """
    df["SMA_20"] = df["Close"].rolling(window=20).mean()  # 20-day Simple Moving Average
    df["SMA_50"] = df["Close"].rolling(window=50).mean()  # 50-day Simple Moving Average
    df["RSI"] = compute_rsi(df["Close"], window=14)       # Relative Strength Index (RSI)

    # Fourier Transform for frequency domain analysis
    fft = np.fft.fft(df["Close"].values)
    df["Fourier_Real"] = np.real(fft)[:len(df)]           # Add real part of FFT
    df["Fourier_Imag"] = np.imag(fft)[:len(df)]           # Add imaginary part of FFT

    df.dropna(inplace=True)                               # Drop rows with NaN values
    return df

def compute_rsi(series, window):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
