from model_predict import predict_stock_prices
from news_analysis import get_financial_news, analyze_sentiment
import numpy as np

def generate_trade_signal():
    predicted_price = predict_stock_prices()
    headlines = get_financial_news()
    sentiment_scores = analyze_sentiment(headlines)

    avg_sentiment = np.mean(sentiment_scores)
    
    # Generate Trading Signal
    if avg_sentiment > 0.2:
        decision = "BUY"
    elif avg_sentiment < -0.2:
        decision = "SELL"
    else:
        decision = "HOLD"

    print(f"Predicted Price: ${predicted_price:.2f}")
    print(f"Average Sentiment Score: {avg_sentiment:.2f}")
    print(f"Recommended Action: {decision}")

if __name__ == "__main__":
    generate_trade_signal()
