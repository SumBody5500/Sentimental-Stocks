import requests
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from config import NEWS_API_KEY

nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

def get_financial_news():
    url = f"https://newsapi.org/v2/everything?q=stock%20market&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    news_data = response.json()
    
    if news_data["status"] == "ok":
        return [article["title"] for article in news_data["articles"][:10]]  # Top 10 headlines
    return []

def analyze_sentiment(headlines):
    scores = [sia.polarity_scores(headline)["compound"] for headline in headlines]
    return scores

if __name__ == "__main__":
    headlines = get_financial_news()
    scores = analyze_sentiment(headlines)
    
    for i in range(len(headlines)):
        print(f"News: {headlines[i]}\nSentiment Score: {scores[i]}\n")
