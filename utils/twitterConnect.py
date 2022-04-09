import tweepy
import os
from dotenv import load_dotenv

load_dotenv()
TWITTER_API_KEY= os.getenv("TWITTER_API_KEY")
TWITTER_SECRET_API_KEY= os.getenv("TWITTER_SECRET_API_KEY")
TWITTER_ACCESS_TOKEN= os.getenv("TWITTER_ACCESS_TOKEN")
TWITTER_SECRET_ACCESS_TOKEN= os.getenv("TWITTER_BEARER_TOKEN")

auth = tweepy.OAuth1UserHandler(
   TWITTER_API_KEY, TWITTER_SECRET_API_KEY, TWITTER_ACCESS_TOKEN, TWITTER_SECRET_ACCESS_TOKEN
)

api = tweepy.API(auth)

public_tweets = api.home_timeline()
for tweet in public_tweets:
    print(tweet.text)

