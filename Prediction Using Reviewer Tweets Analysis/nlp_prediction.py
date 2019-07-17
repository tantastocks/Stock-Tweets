import pandas as pd 
import numpy as np

# Packages for data preparation
from keras.preprocessing.text import Tokenizer

# Packages for modeling
from keras.models import load_model
NB_WORDS = 10000  # Parameter indicating the number of words we'll put in the dictionary

model = load_model('model.h5')
model.summary()

def one_hot_seq(seqs, nb_features = NB_WORDS):
    ohs = np.zeros((len(seqs), nb_features))
    for i, s in enumerate(seqs):
        ohs[i, s] = 1.
    return ohs
from google.cloud import bigquery as bq

client = bq.Client()


facebook_query= client.query("""
    SELECT *  FROM `tanta-stocks.facebook.facebook_tweets_processed`
    """)
apple_query = client.query("""
    SELECT *  FROM `tanta-stocks.apple.apple_tweets_processed`
    """)
amazon_query = client.query("""
    SELECT *  FROM `tanta-stocks.amazon.amazon_tweets_processed`
    """)
nokia_query = client.query("""
    SELECT *  FROM `tanta-stocks.nokia.nokia_tweets_processed`
    """)
intel_query = client.query("""
    SELECT *  FROM `tanta-stocks.intel.intel_tweets_processed`
    """)
microsoft_query = client.query("""
    SELECT *  FROM `tanta-stocks.microsoft.microsoft_tweets_processed`
    """)
NVIDIA_query = client.query("""
    SELECT *  FROM `tanta-stocks.NVIDIA.NVIDIA_tweets_processed`
    """)
AMD_query = client.query("""
    SELECT *  FROM `tanta-stocks.AMD.AMD_tweets_processed`
    """)
twitter_query = client.query("""
    SELECT *  FROM `tanta-stocks.twitter.twitter_tweets_processed`
    """)
google_query = client.query("""
    SELECT *  FROM `tanta-stocks.google.google_tweets_processed`
    """)

database = ["facebook.facebook_day_score", "apple.apple_day_score", "amazon.amazon_day_score","nokia.nokia_day_score", "intel.intel_day_score", "microsoft.microsoft_day_score", "NVIDIA.NVIDIA_day_score", "AMD.AMD_day_score", "twitter.twitter_day_score", "google.google_day_score"]
#csv_names = ['facebook_day_score.csv', 'apple_day_score.csv', 'amazon_day_score.csv', 'nokia_day_score.csv', 'intel_day_score.csv', 'microsoft_day_score.csv', 'NVIDIA_day_score.csv', 'AMD_day_score.csv', 'twitter_day_score.csv', 'google_day_score.csv']
database2 = ["facebook.facebook_tweets_score", "apple.apple_tweets_score", "amazon.amazon_tweets_score","nokia.nokia_tweets_score", "intel.intel_tweets_score", "microsoft.microsoft_tweets_score", "NVIDIA.NVIDIA_tweets_score", "AMD.AMD_tweets_score", "twitter.twitter_tweets_score", "google.google_tweets_score"]
#csv_names2 = ['facebook_tweets_processed.csv', 'apple_tweets_processed.csv', 'amazon_tweets_processed.csv', 'nokia_tweets_processed.csv', 'intel_tweets_processed.csv', 'microsoft_tweets_processed.csv', 'NVIDIA_tweets_processed.csv', 'AMD_tweets_processed.csv','twitter_tweets_processed.csv',  'google_tweets_processed.csv']
#csv_names3 = ['facebook_tweets_score.csv', 'apple_tweets_score.csv', 'amazon_tweets_score.csv', 'nokia_tweets_score.csv', 'intel_tweets_score.csv', 'microsoft_tweets_score.csv', 'NVIDIA_tweets_score.csv', 'AMD_tweets_score.csv','twitter_tweets_score.csv',  'google_tweets_score.csv']
dataset = [facebook_query, apple_query, amazon_query, nokia_query, intel_query, microsoft_query, NVIDIA_query, AMD_query, twitter_query, google_query]
database3 = ["facebook.facebook_last_day_score", "apple.apple_last_day_score", "amazon.amazon_last_day_score","nokia.nokia_last_day_score", "intel.intel_last_day_score", "microsoft.microsoft_last_day_score", "NVIDIA.NVIDIA_last_day_score", "AMD.AMD_last_day_score", "twitter.twitter_last_day_score", "google.google_last_day_score"]

for j in range (0,10):
    #df1 = pd.read_csv(csv_names2[j],encoding='latin-1')
    df1 = dataset[j].result().to_dataframe()
    data1=df1.text
    
    tk = Tokenizer(num_words=NB_WORDS,
                   filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                   lower=True,
                   split=" ")
    
    tk.fit_on_texts(data1)
    data_seq1 = tk.texts_to_sequences(data1)
    
    data_oh1 = one_hot_seq(data_seq1)
    
    results=model.predict(data_oh1)
    
    df1['neg_score']=results[:,0]
    df1['pos_score']=results[:,1]
    class_type = []
    df1['class_type']=0
        
    for i in range(len(results)):
        if (df1.iloc[i,4]>df1.iloc[i,3]):
            class_type.insert(i, 1)
        else:
            class_type.insert(i, 0)
    
    df1['class_type'] = pd.Series(class_type, index=df1.index)
    class_=0
    if results[:,0].sum()/len(results) >results[:,0].sum()/len(results):
        class_=0
    else:
        class_=1
    day_score_facebook=pd.DataFrame(columns=['date','pos','neg','class_type'])
    day_score_facebook['date'] = pd.to_datetime(day_score_facebook['date']).dt.strftime('%d-%m-%Y 00:00:00')
    day_score_facebook=day_score_facebook.append({'date':df1.iloc[0,0],'pos':results[:,0].sum()/len(results),'neg':results[:,1].sum()/len(results),'class_type':class_}, ignore_index=True)
    #day_score_facebook.to_csv(csv_names[j], index=False)
    full_table_id = database[j]
    full_table_id2 = database3[j]
    project_id = 'tanta-stocks'
    day_score_facebook['date'] = day_score_facebook['date'].dt.date
    day_score_facebook.to_gbq(full_table_id, project_id=project_id, if_exists='append', table_schema = [{'name': 'date', 'type': 'STRING'}, {'name': 'pos', 'type': 'FLOAT'}, {'name': 'neg', 'type': 'FLOAT'}, {'name': 'class_type', 'type': 'INTEGER'}])
    day_score_facebook['date'] = pd.to_datetime(day_score_facebook['date']).dt.strftime('%d-%m-%Y')
    #day_score_facebook['date'] = day_score_facebook['date'].dt.date
    day_score_facebook.to_gbq(full_table_id2, project_id=project_id, if_exists='replace')
#df1.to_csv(csv_names3[j],index=False)
    full_table_id = database2[j]
    project_id = 'tanta-stocks'
    df1.to_gbq(full_table_id, project_id=project_id, if_exists='replace')
  





