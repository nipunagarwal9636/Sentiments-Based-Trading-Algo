# log_sentiment.py
from your_existing_functions import fetch_tweets, get_sentiment_score, log_sentiment

tweets = fetch_tweets("Google")
if tweets:
    score = get_sentiment_score(tweets)
    log_sentiment(score)
    print(f"Logged score: {score}")
else:
    print("No tweets found.")
