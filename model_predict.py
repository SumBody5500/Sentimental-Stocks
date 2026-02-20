import numpy as np
import tensorflow as tf
import joblib
from data_fetch import fetch_stock_data, preprocess_data

def predict_stock_prices():
    model = tf.keras.models.load_model("lstm_stock_model.h5")
    scaler = joblib.load("scaler.pkl")
    
    # Fetch latest stock data
    data = fetch_stock_data()
    scaled_data, _ = preprocess_data(data)

    # Prepare last 60 days of data for prediction
    last_60_days = scaled_data[-60:]
    last_60_days = np.reshape(last_60_days, (1, 60, 1))

    # Predict next day's price
    prediction = model.predict(last_60_days)
    predicted_price = scaler.inverse_transform(prediction)[0, 0]
    
    return predicted_price

if __name__ == "__main__":
    print(f"Predicted Next Closing Price: ${predict_stock_prices():.2f}")
