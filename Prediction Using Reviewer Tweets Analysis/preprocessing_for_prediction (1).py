import pandas as pd 
import re

# Packages for data preparation
from nltk.corpus import stopwords
from google.cloud import bigquery as bq

client = bq.Client()


facebook_query= client.query("""
    SELECT *  FROM `tanta-stocks.facebook.facebook_tweets`
    """)
apple_query = client.query("""
    SELECT *  FROM `tanta-stocks.apple.apple_tweets`
    """)
amazon_query = client.query("""
    SELECT *  FROM `tanta-stocks.amazon.amazon_tweets`
    """)
nokia_query = client.query("""
    SELECT *  FROM `tanta-stocks.nokia.nokia_tweets`
    """)
intel_query = client.query("""
    SELECT *  FROM `tanta-stocks.intel.intel_tweets`
    """)
microsoft_query = client.query("""
    SELECT *  FROM `tanta-stocks.microsoft.microsoft_tweets`
    """)
NVIDIA_query = client.query("""
    SELECT *  FROM `tanta-stocks.NVIDIA.NVIDIA_tweets`
    """)
AMD_query = client.query("""
    SELECT *  FROM `tanta-stocks.AMD.AMD_tweets`
    """)
twitter_query = client.query("""
    SELECT *  FROM `tanta-stocks.twitter.twitter_tweets`
    """)
google_query = client.query("""
    SELECT *  FROM `tanta-stocks.google.google_tweets`
    """)

def remove_stopwords(input_text):
        stopwords_list = stopwords.words('english')
        # Some words which might indicate a certain sentiment are kept via a whitelist
        whitelist = ["n't", "not", "no"]
        words = input_text.split() 
        clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1] 
        return " ".join(clean_words) 
    
def remove_mentions(input_text):
        return re.sub(r'@\w+', '', input_text)
    
#csv_names = ['facebook_tweets.csv', 'apple_tweets.csv', 'amazon_tweets.csv', 'nokia_tweets.csv', 'intel_tweets.csv', 'microsoft_tweets.csv', 'NVIDIA_tweets.csv', 'AMD_tweets.csv', 'twitter_tweets.csv', 'google_tweets.csv']
database = ["facebook.facebook_tweets_processed", "apple.apple_tweets_processed", "amazon.amazon_tweets_processed","nokia.nokia_tweets_processed", "intel.intel_tweets_processed", "microsoft.microsoft_tweets_processed", "NVIDIA.NVIDIA_tweets_processed", "AMD.AMD_tweets_processed", "twitter.twitter_tweets_processed", "google.google_tweets_processed"]
#csv_names2 = ['facebook_tweets_processed.csv', 'apple_tweets_processed.csv', 'amazon_tweets_processed.csv', 'nokia_tweets_processed.csv', 'intel_tweets_processed.csv', 'microsoft_tweets_processed.csv', 'NVIDIA_tweets_processed.csv', 'AMD_tweets_processed.csv','twitter_tweets_processed.csv',  'google_tweets_processed.csv']
dataset = [facebook_query, apple_query, amazon_query, nokia_query, intel_query, microsoft_query, NVIDIA_query, AMD_query, twitter_query, google_query]

for i in range (0,10):
    #df = pd.read_csv(csv_names[i],encoding='latin-1')
    df = dataset[i].result().to_dataframe()
    #df = df.drop(["Unnamed: 0"], axis=1)       
    df.text = df.text.apply(remove_stopwords).apply(remove_mentions)
    #df.to_csv(csv_names2[i])
    full_table_id = database[i]
    project_id = 'tanta-stocks'
    df.to_gbq(full_table_id, project_id=project_id, if_exists='replace')
