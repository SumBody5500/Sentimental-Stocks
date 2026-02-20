from flask import Flask, render_template, request
from data_fetcher import DataFetcher
from feature_engineering import add_technical_indicators
from model import build_neural_network
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model  # Import Model
import logging # Import Logging for debugging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        symbol = request.form.get("symbol").strip()

        if not symbol:
            return render_template("index.html", error="Please enter a valid stock symbol.")

        fetcher = DataFetcher()
        try:
            # Fetch stock data
            stock_data = fetcher.fetch_stock_data(symbol=symbol, start_date="2014-01-01", end_date="2025-03-01")

            # Check if stock_data is None or empty
            if stock_data is None or stock_data.empty:
                return render_template("index.html", error="No stock data available for this symbol.")

            # Add technical indicators
            stock_data = add_technical_indicators(stock_data.copy())  # Create a copy to avoid modifying the original DataFrame

            # Handle missing values by filling with the mean
            stock_data.fillna(stock_data.mean(), inplace=True)

            # News Sentiment Analysis
            news_sentiment = fetcher.fetch_news_sentiment(symbol)
            stock_data['News_Sentiment'] = [news_sentiment] * len(stock_data)  # Ensure sentiment is 1-D

            # Prepare data for training
            X = stock_data[["SMA_20", "SMA_50", "RSI", "Fourier_Real", "Fourier_Imag", 'News_Sentiment']].values
            y = stock_data["Close"].values

            # Check for NaN or inf values in X and y
            if np.isnan(X).any() or np.isinf(X).any():
                logging.error("NaN or inf values found in input features (X)")
                return render_template("index.html", error="Data contains invalid numeric values (NaN or inf). Please check the data and feature engineering.")

            if np.isnan(y).any() or np.isinf(y).any():
                logging.error("NaN or inf values found in target variable (y)")
                return render_template("index.html", error="Data contains invalid numeric values (NaN or inf) in the target variable. Please check the data and feature engineering.")


            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Scale the data
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Build and train the neural network model
            input_dim = X_train.shape[1]
            model = build_neural_network(input_dim)
            model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)  # Set verbose=0 to suppress output

            # Make predictions
            y_pred = model.predict(X_test).flatten()
    
            if np.isnan(y_pred).any() or np.isinf(y_pred).any():
                logging.error("NaN or inf values found in prediction (y_pred)")
                return render_template("index.html", error="Model produced invalid predictions (NaN or inf). Please check the data, model architecture, and training process.")

            # Create a DataFrame for actual vs predicted values
            predictions_df = pd.DataFrame({
                'Date': stock_data.iloc[len(X_train):]["Date"].values,
                'Actual': y_test,
                'Predicted': y_pred.flatten()   # Ensure predictions are flattened
            })

            # Render the predictions in HTML format
            return render_template("index.html",
                                   stock_data=stock_data.to_html(classes='table table-striped', index=False),
                                   predictions=predictions_df.to_html(classes='table table-striped', index=False))

        except Exception as e:
            logging.exception("An error occurred:")
            return render_template("index.html", error=f"An unexpected error occurred: {str(e)}. Check the logs for details.")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
