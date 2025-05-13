
import streamlit as st
import pandas as pd
import yfinance as yf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import tweepy
import datetime
import matplotlib.pyplot as plt

# --- Twitter API credentials (replace with your own keys or use secrets in deployment) ---
bearer_token = "AAAAAAAAAAAAAAAAAAAAAJKy1AEAAAAAx1Idx1wuT0kNUDkO6l5CoLnt5hI%3DyusZa0SLrGZci9cxz0GBn5c9HZ0i83GdOhYj3nJxkiuxNGNfW3"

# --- FinBERT Model ---
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

sentiment_pipeline = load_model()

# --- Streamlit UI ---
st.title("📈 Sentiment-Based Trading Strategy")
stock_symbol = st.text_input("Enter Stock Symbol (e.g., TSLA):", value="TSLA")
search_keywords = st.text_input("Enter keywords for sentiment (e.g., Tesla):", value="Tesla")

if st.button("Analyze"):
    # --- Step 1: Fetch Tweets ---
    st.write("🔍 Fetching tweets...")
    client = tweepy.Client(bearer_token=bearer_token)
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)

    tweets = client.search_recent_tweets(
        query=f"{search_keywords} -is:retweet lang:en",
        max_results=50,
        tweet_fields=['created_at', 'text']
    )
    tweet_texts = [tweet.text for tweet in tweets.data] if tweets.data else []

    # --- Step 2: Sentiment Analysis ---
    st.write("🧠 Analyzing sentiment...")
    sentiments = sentiment_pipeline(tweet_texts)
    sentiment_df = pd.DataFrame(sentiments)
    sentiment_counts = sentiment_df['label'].value_counts()
    pos = sentiment_counts.get('positive', 0)
    neg = sentiment_counts.get('negative', 0)
    total = len(tweet_texts)
    sentiment_score = (pos - neg) / total if total > 0 else 0
    st.metric("Sentiment Score", round(sentiment_score, 2))

    # --- Step 3: Stock Data ---
    st.write("📉 Downloading stock data...")
    df = yf.download(stock_symbol, period="6mo")
    df['Return'] = df['Adj Close'].pct_change()
    df['Signal'] = 0
    df.loc[df.index[-1], 'Signal'] = 1 if sentiment_score > 0.1 else -1 if sentiment_score < -0.1 else 0
    df['Strategy_Return'] = df['Signal'].shift(1) * df['Return']

    # --- Step 4: Plotting ---
    st.write("📊 Plotting strategy performance...")
    df['Cumulative_Market'] = (1 + df['Return']).cumprod()
    df['Cumulative_Strategy'] = (1 + df['Strategy_Return']).cumprod()

    st.line_chart(df[['Cumulative_Market', 'Cumulative_Strategy']])

    st.write("✅ Strategy completed. You can now interpret the plots or adjust keywords/stocks.")
