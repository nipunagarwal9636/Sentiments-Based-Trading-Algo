#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
import os

from transformers import pipeline, BertTokenizer, BertForSequenceClassification
import tweepy


# In[2]:


BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAJKy1AEAAAAAx1Idx1wuT0kNUDkO6l5CoLnt5hI%3DyusZa0SLrGZci9cxz0GBn5c9HZ0i83GdOhYj3nJxkiuxNGNfW3'
#BEARER_TOKEN = '1676310337547796480-gSKAiE88bImZjQ3VgHppHX1IxAnwpI'
TICKER = 'GOOG'
SENTIMENT_FILE = 'sentiment_log.csv'
model_name = "yiyanghkust/finbert-tone"


# In[3]:


st.set_page_config(page_title="Sentiment Trading Dashboard", layout="centered")
client = tweepy.Client(bearer_token=BEARER_TOKEN)
finbert = pipeline("sentiment-analysis",
                   model=BertForSequenceClassification.from_pretrained(model_name),
                   tokenizer=BertTokenizer.from_pretrained(model_name))


# In[4]:


def fetch_tweets(keyword="Google", max_tweets=50):
    query = f"{keyword} -is:retweet lang:en"
    tweets = client.search_recent_tweets(query=query, max_results=max_tweets)
    return [tweet.text for tweet in tweets.data] if tweets.data else []

def get_sentiment_score(tweets):
    sentiments = finbert(tweets)
    score = 0
    for s in sentiments:
        if s['label'] == 'positive':
            score += 1
        elif s['label'] == 'negative':
            score -= 1
    return score / len(sentiments) if sentiments else 0

def log_sentiment(score):
    today = dt.date.today().strftime("%Y-%m-%d")
    new_row = pd.DataFrame([{"Date": today, "Sentiment": score}])
    if os.path.exists(SENTIMENT_FILE):
        df = pd.read_csv(SENTIMENT_FILE)
        if today not in df['Date'].values:
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(SENTIMENT_FILE, index=False)
    else:
        new_row.to_csv(SENTIMENT_FILE, index=False)

def load_sentiment_data():
    if os.path.exists(SENTIMENT_FILE):
        return pd.read_csv(SENTIMENT_FILE, parse_dates=['Date']).set_index("Date")
    else:
        return pd.DataFrame(columns=["Date", "Sentiment"]).set_index("Date")

def merge_stock_sentiment(sentiment_df):
    start = sentiment_df.index.min().strftime('%Y-%m-%d')
    end = dt.date.today().strftime('%Y-%m-%d')
    price = yf.download(TICKER, start=start, end=end)
    price.index = price.index.normalize()
    return price.join(sentiment_df, how="inner")

def generate_signals(df, threshold=0.1):
    df = df.copy()
    df['Signal'] = 0
    df.loc[df['Sentiment'] > threshold, 'Signal'] = 1
    df.loc[df['Sentiment'] < -threshold, 'Signal'] = -1
    df['Daily Return'] = df['Adj Close'].pct_change()
    df['Strategy Return'] = df['Signal'].shift(1) * df['Daily Return']
    return df

def plot_returns(df):
    cumulative = (1 + df['Strategy Return'].fillna(0)).cumprod()
    benchmark = (1 + df['Daily Return'].fillna(0)).cumprod()
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(cumulative, label="Strategy")
    ax.plot(benchmark, label="Buy & Hold")
    ax.set_title("Backtest: Sentiment Strategy vs Benchmark")
    ax.legend()
    st.pyplot(fig)


# In[5]:


st.title("📈 Sentiment-Based Trading Strategy Dashboard")
st.write(f"Tracking {TICKER} using Twitter sentiment (FinBERT model)")

if st.button("🔁 Fetch & Log Today's Sentiment"):
    tweets = fetch_tweets("Google")
    if tweets:
        score = get_sentiment_score(tweets)
        log_sentiment(score)
        st.success(f"Today's Sentiment Score: {score:.2f}")
    else:
        st.warning("No tweets fetched.")

sentiment_data = load_sentiment_data()
st.line_chart(sentiment_data['Sentiment'])

if len(sentiment_data) > 3:
    st.subheader("📊 Strategy Backtest")
    merged = merge_stock_sentiment(sentiment_data)
    result = generate_signals(merged)
    plot_returns(result)
else:
    st.info("Not enough sentiment history to backtest.")


# In[6]:





# In[ ]:




