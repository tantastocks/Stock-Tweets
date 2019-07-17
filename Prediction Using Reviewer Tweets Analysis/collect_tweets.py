import tweepy
from datetime import datetime, timedelta
import pandas as pd 

# twitter api credentials
consumer_key = "pisboa4NgbIxIsdp9BHLhfTEH"
consumer_secret = "ZDEYrHpwi0C4tJwtblhIHeESx9N3Wu5JDguFa41AFkf4GCxeDu"
access_token = "1122579665405841418-ETP260DUd5HF0G03PRpHj0ykOyakqd"
access_token_secret = "3ZnhgKgI0lfqHORsyP9AH9UGAzkWXF1BJvA9YZ56WI0Yb"

# instantiate the api
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True,wait_on_rate_limit_notify=True)

database = ["facebook.facebook_tweets", "apple.apple_tweets", "amazon.amazon_tweets","nokia.nokia_tweets", "intel.intel_tweets", "microsoft.microsoft_tweets", "NVIDIA.NVIDIA_tweets", "AMD.AMD_tweets", "twitter.twitter_tweets", "google.google_tweets"]
csv_names = ['facebook_tweets.csv', 'apple_tweets.csv', 'amazon_tweets.csv', 'nokia_tweets.csv', 'intel_tweets.csv', 'microsoft_tweets.csv', 'NVIDIA_tweets.csv', 'AMD_tweets.csv', 'twitter_tweets.csv', 'google_tweets.csv']
hashtags = ["#FB", "#AAPL", "#AMZN", "#NOKIA", "#INTC", "#MSFT", "#NVDA", "#AMD","#TWTR",  "#GOOGL"]

for i in range (0,10):
    dataframe = pd.DataFrame(columns=['index', 'date', 'retweet_count', 'likes', 'text', 'user_name'])
    dataframe.set_index('index',inplace=True)
    
    for tweet in tweepy.Cursor(api.search,q=hashtags[i],count=100,
                               lang="en",
                               since=datetime.today().strftime('%Y-%m-%d')).items():
         print (tweet.created_at,"\t",tweet.retweet_count,"\t",tweet.text)
         dataframe = dataframe.append({'date': tweet.created_at, 'retweet_count': tweet.retweet_count, 'likes': tweet.favorite_count, 'text': tweet.text.encode('utf-8'), 'user_name': tweet.user.screen_name}, ignore_index=True)
     
    #dataframe.to_csv(csv_names[i])
    full_table_id = database[i]
    project_id = 'tanta-stocks'
    dataframe.to_gbq(full_table_id, project_id=project_id, if_exists='replace')
