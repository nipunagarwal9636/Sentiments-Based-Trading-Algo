
import streamlit as st
import pandas as pd
import yfinance as yf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import praw
import datetime

# --- Reddit API credentials (use secrets in deployment) ---
reddit = praw.Reddit(
    client_id=st.secrets["REDDIT_CLIENT_ID"],
    client_secret=st.secrets["REDDIT_CLIENT_SECRET"],
    user_agent=st.secrets["REDDIT_USER_AGENT"]
)

# --- FinBERT Model ---
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

sentiment_pipeline = load_model()

# --- Streamlit UI ---
st.title("📈 Reddit-Based Sentiment Trading Strategy")
stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL):", value="AAPL")
subreddit_name = st.text_input("Enter Subreddit (e.g., stocks):", value="stocks")
keyword = st.text_input("Enter Keyword to Filter Posts (e.g., Apple):", value="Apple")

if st.button("Analyze"):
    # --- Step 1: Fetch Reddit Posts ---
    st.write("🧵 Fetching Reddit posts...")
    subreddit = reddit.subreddit(subreddit_name)
    posts = subreddit.search(keyword, sort='new', time_filter='day', limit=50)
    post_texts = [post.title + " " + (post.selftext or "") for post in posts]

    # --- Step 2: Sentiment Analysis ---
    st.write("🧠 Analyzing sentiment...")
    sentiments = sentiment_pipeline(post_texts)
    sentiment_df = pd.DataFrame(sentiments)
    sentiment_counts = sentiment_df['label'].value_counts()
    pos = sentiment_counts.get('positive', 0)
    neg = sentiment_counts.get('negative', 0)
    total = len(post_texts)
    sentiment_score = (pos - neg) / total if total > 0 else 0
    st.metric("Sentiment Score", round(sentiment_score, 2))

    # --- Step 3: Stock Data ---
    st.write("📉 Downloading stock data...")
    df = yf.download(stock_symbol, period="6mo")
    df['Return'] = df['Adj Close'].pct_change()
    df['Signal'] = 0
    df.loc[df.index[-1], 'Signal'] = 1 if sentiment_score > 0.1 else -1 if sentiment_score < -0.1 else 0
    df['Strategy_Return'] = df['Signal'].shift(1) * df['Return']

    # --- Step 4: Performance Chart ---
    st.write("📊 Strategy Performance")
    df['Cumulative_Market'] = (1 + df['Return']).cumprod()
    df['Cumulative_Strategy'] = (1 + df['Strategy_Return']).cumprod()
    st.line_chart(df[['Cumulative_Market', 'Cumulative_Strategy']])

    st.success("✅ Strategy complete! Adjust keywords or symbols to explore further.")
