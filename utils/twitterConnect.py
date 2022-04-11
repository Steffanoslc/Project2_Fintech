import os
import re
import tweepy
import pandas as pd
import hvplot.pandas
from textblob import TextBlob
from dotenv import load_dotenv

load_dotenv()
TWITTER_API_KEY= os.getenv("TWITTER_API_KEY")
TWITTER_SECRET_API_KEY= os.getenv("TWITTER_SECRET_API_KEY")
TWITTER_BEARER_TOKEN= os.getenv("TWITTER_BEARER_TOKEN")
TWITTER_ACCESS_TOKEN= os.getenv("TWITTER_ACCESS_TOKEN")
TWITTER_SECRET_ACCESS_TOKEN= os.getenv("TWITTER_BEARER_TOKEN")

auth = tweepy.OAuth1UserHandler(
   TWITTER_API_KEY,TWITTER_SECRET_API_KEY,
   TWITTER_ACCESS_TOKEN,TWITTER_SECRET_ACCESS_TOKEN
)
api = tweepy.API(auth)

search_term="#bitcoin -filter:retweets"
tweets= tweepy.Cursor(api.search_tweets,q=search_term,lang="en",count=100,tweet_mode="extended").items(300)
all_tweets= [tweet.full_text for tweet in tweets]
tweets_df= pd.DataFrame(all_tweets, columns=["Tweets"])

def cleanTwt(twt):
    twt= re.sub("#[A-Za-z0-9]+","",twt)
    twt= re.sub("\\n","",twt)
    twt= re.sub("https?:\/\/\S+","",twt)
    return twt

tweets_df["Tweets"]= tweets_df["Tweets"].apply(cleanTwt)

def getSubjectivity(twt):
    return TextBlob(twt).sentiment.subjectivity
def getPolarity(twt):
    return TextBlob(twt).sentiment.polarity

tweets_df["Subjectivity"]= tweets_df["Tweets"].apply(getSubjectivity)
tweets_df["Polarity"]= tweets_df["Tweets"].apply(getPolarity)

def getSentiment(score):
    if score < 0:
        return "Negative"
    elif score == 0:
        return "Neutral"
    else:
        return "Positive"

tweets_df["Sentiment"] = tweets_df["Polarity"].apply(getSentiment)