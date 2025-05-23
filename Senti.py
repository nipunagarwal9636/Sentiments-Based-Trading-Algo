import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import yfinance as yf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import praw
import datetime
import torch
import torch.nn.functional as F

# --- Reddit API credentials (use secrets in deployment) ---
reddit = praw.Reddit(
    client_id="yMjAkr-zxWXFhT87ugfc9w",
    client_secret="50lLxGXaCk1h4Nk1rRxupmB02BAP5w",
    user_agent="python:sentiment.reddit.app:v1.0 (by u/Nipun27)"
)

# --- FinBERT Model ---
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone", torch_dtype=torch.float32).to("cpu")


def get_sentiment(texts):
    results = []
    for text in texts:
        inputs = tokenizer(text[:512], return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            scores = F.softmax(outputs.logits, dim=1)[0]
            label = torch.argmax(scores).item()
            label_name = ["neutral", "positive", "negative"][label]
            results.append({"label": label_name, "score": scores[label].item()})
    return results



# --- Streamlit UI ---
st.title("📈 Reddit-Based Sentiment Trading Strategy")
st.subheader("Developed by Nipun Agarwal and Garv Joshi")
st.markdown("---")
stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL):", value="AAPL")
subreddit_name = st.text_input("Enter Subreddit (e.g., stocks):", value="stocks")
keyword = st.text_input("Enter Keyword to Filter Posts (e.g., Apple):", value="Apple")

if st.button("Analyze"):
    # --- Step 1: Fetch Reddit Posts ---
    st.write("🧵 Fetching Reddit posts...")
    subreddit = reddit.subreddit(subreddit_name)
    posts = subreddit.search(keyword, sort='new', time_filter='day', limit=50)
    post_texts = [
        (post.title + " " + (post.selftext or ""))[:512]  # truncate to avoid model input size error
        for post in posts ]


    # --- Step 2: Sentiment Analysis ---
    st.write("🧠 Analyzing sentiment...")
    sentiments = get_sentiment(post_texts)
    sentiment_df = pd.DataFrame(sentiments)
    sentiment_counts = sentiment_df['label'].value_counts()
    pos = sentiment_counts['positive'] if 'positive' in sentiment_counts else 0
    neg = sentiment_counts['negative'] if 'negative' in sentiment_counts else 0
    total = len(post_texts)
    sentiment_score = (pos - neg) / total if total > 0 else 0
    st.metric("Sentiment Score", round(sentiment_score, 2))

    # --- Step 3: Stock Data ---
    st.write("📉 Downloading stock data...")
    df = yf.download(stock_symbol, period="3mo")
    if df.empty:
        st.error("❌ No stock data found. Please check the symbol.")
        st.stop()
    if 'Adj Close' not in df.columns:
        if 'Close' in df.columns:
            df['Adj Close'] = df['Close']
        else:
            st.error("⚠️ Could not find stock price data for this symbol.")
            st.stop()
    df['Return'] = df['Adj Close'].pct_change()
    df['Signal'] = 0
    df.loc[df.index[-1], 'Signal'] = 1 if sentiment_score > 0.1 else -1 if sentiment_score < -0.1 else 0
    df['Strategy_Return'] = df['Signal'].shift(1) * df['Return']

    # --- Step 4: Performance Chart ---
    st.write("📊 Strategy Performance")
    df['Cumulative_Market'] = (1 + df['Return']).cumprod()
    df['Cumulative_Strategy'] = (1 + df['Strategy_Return']).cumprod()
    #df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
    #st.line_chart(df[['Cumulative_Market', 'Cumulative_Strategy']])
    # Step 1: Make sure 'Adj Close' exists
    if 'Adj Close' not in df.columns:
        if 'Close' in df.columns:
            df['Adj Close'] = df['Close']
        else:
            st.error("Missing 'Adj Close' and 'Close' columns in stock data.")
            st.stop()

    # Step 2: Calculate returns and strategy
    df['Return'] = df['Adj Close'].pct_change()
    df['Signal'] = 0
    df.loc[df.index[-1], 'Signal'] = ( 1 if sentiment_score > 0.1 else -1 if sentiment_score < -0.1 else 0)
    df['Strategy_Return'] = df['Signal'].shift(1) * df['Return']

    # Step 3: Calculate cumulative returns
    df['Cumulative_Market'] = (1 + df['Return'].fillna(0)).cumprod()
    df['Cumulative_Strategy'] = (1 + df['Strategy_Return'].fillna(0)).cumprod()

    # Step 4: Debug – print column names to verify
    st.write("✅ Columns in DataFrame:", df.columns.tolist())

    # Step 5: Check and plot
    if 'Cumulative_Market' in df.columns and 'Cumulative_Strategy' in df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df['Cumulative_Market'], label='Market Return', color='blue')
        plt.plot(df.index, df['Cumulative_Strategy'], label='Strategy Return', color='green')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.title('Market vs Strategy Cumulative Returns')
        plt.legend()
        plt.grid(True)

    # Show plot in Streamlit
        st.pyplot(plt)
    else:
        st.warning("🚫 Required columns not found in DataFrame.")



    st.success("✅ Strategy complete! Adjust keywords or symbols to explore further.")
