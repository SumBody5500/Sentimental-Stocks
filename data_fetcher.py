import yfinance as yf
import pandas as pd
import requests

class DataFetcher:
    def fetch_stock_data(self, symbol, start_date, end_date):
        """
        Fetch historical stock price data using yfinance.
        """
        try:
            # Download historical stock data with auto_adjust=False
            df = yf.download(symbol, start=start_date, end=end_date, auto_adjust=False)
            
            # Ensure the required columns are available
            df.reset_index(inplace=True)  # Reset index to make 'Date' a column
            
            # Check if Adj Close is in columns
            if 'Adj Close' not in df.columns:
                df['Adj Close'] = df['Close']  # If Adj Close is missing, use Close price
            
            df = df[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
            
            return df
        except Exception as e:
            raise Exception(f"Error fetching stock data: {str(e)}")

    def fetch_news_sentiment(self, query):
        """
        Fetch news sentiment data using News API.
        """
        url = f"https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "apiKey": "6145075a74eb444185a489b761f33e03",  # Replace with your actual News API key
            "language": "en",
            "sortBy": "relevancy"
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            articles = response.json().get("articles", [])
            
            sentiments = []
            for article in articles:
                # Dummy sentiment score (replace with real sentiment analysis API if needed)
                sentiments.append(len(article["title"]) % 2)  # Example: even length -> positive sentiment
            
            return sum(sentiments) / len(sentiments) if sentiments else 0
        except requests.exceptions.RequestException as e:
            print(f"Error fetching news: {e}")
            return 0  # Return neutral sentiment in case of error
