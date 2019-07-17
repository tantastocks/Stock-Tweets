import pandas as pd 
import numpy as np

# Packages for data preparation
from keras.preprocessing.text import Tokenizer

# Packages for modeling
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

from google.cloud import bigquery as bq

client = bq.Client()

model = load_model('nlpmodel86.13.h5')
model.summary()

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

database = ["facebook.facebook_day_score_final", "apple.apple_day_score_final", "amazon.amazon_day_score_final","nokia.nokia_day_score_final", "intel.intel_day_score_final", "microsoft.microsoft_day_score_final", "NVIDIA.NVIDIA_day_score_final", "AMD.AMD_day_score_final", "twitter.twitter_day_score_final", "google.google_day_score_final"]
#csv_names = ['facebook_day_score_final.csv', 'apple_day_score_final.csv', 'amazon_day_score_final.csv', 'nokia_day_score_final.csv', 'intel_day_score_final.csv', 'microsoft_day_score_final.csv', 'NVIDIA_day_score_final.csv', 'AMD_day_score_final.csv', 'twitter_day_score_final.csv', 'google_day_score_final.csv']
database2 = ["facebook.facebook_tweets_score_final", "apple.apple_tweets_score_final", "amazon.amazon_tweets_score_final","nokia.nokia_tweets_score_final", "intel.intel_tweets_score_final", "microsoft.microsoft_tweets_score_final", "NVIDIA.NVIDIA_tweets_score_final", "AMD.AMD_tweets_score_final", "twitter.twitter_tweets_score_final", "google.google_tweets_score_final"]
#csv_names2 = ['facebook_tweets_processed.csv', 'apple_tweets_processed.csv', 'amazon_tweets_processed.csv', 'nokia_tweets_processed.csv', 'intel_tweets_processed.csv', 'microsoft_tweets_processed.csv', 'NVIDIA_tweets_processed.csv', 'AMD_tweets_processed.csv','twitter_tweets_processed.csv',  'google_tweets_processed.csv']
#csv_names3 = ['facebook_tweets_score_final.csv', 'apple_tweets_score_final.csv', 'amazon_tweets_score_final.csv', 'nokia_tweets_score_final.csv', 'intel_tweets_score_final.csv', 'microsoft_tweets_score_final.csv', 'NVIDIA_tweets_score_final.csv', 'AMD_tweets_score_final.csv','twitter_tweets_score_final.csv',  'google_tweets_score_final.csv']
dataset = [facebook_query, apple_query, amazon_query, nokia_query, intel_query, microsoft_query, NVIDIA_query, AMD_query, twitter_query, google_query]

for j in range (0,10):

    #df1 = pd.read_csv(csv_names2[j],encoding='latin-1',index_col=None)
    df1 = dataset[j].result().to_dataframe()
    data1=df1.text
    tk = Tokenizer(num_words=10000,
                   filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                   lower=True,
                   split=" ")
    
    tk.fit_on_texts(data1)
    data_seq1 = tk.texts_to_sequences(data1)
    data_seq1 = pad_sequences(data_seq1, maxlen=300)
    results=model.predict(data_seq1)
    results_pos=[]
    for i in range (len(results)):
        temp = 1-results[i]
        results_pos.append(temp)
        
    
    data_score=pd.DataFrame(columns=['date','pos','neg'])
    data_score['date'] = pd.to_datetime(df1['date']).dt.strftime('%Y-%m-%d 00:00:00')
    data_score = data_score.assign(neg=results)
    data_score['pos'] = pd.Series(results_pos, index=data_score.index)
    #data_score['neg'] = pd.Series(results, index=data_score.index)
    data_score['class_type'] = np.where(data_score['pos']>=data_score['neg'], 1, 0)
    #data_score.to_csv(csv_names3[j],index=False)
    full_table_id = database2[j]
    project_id = 'tanta-stocks'
    data_score.to_gbq(full_table_id, project_id=project_id, if_exists='replace')
    
    day_score_facebook=pd.DataFrame(columns=['date','pos','neg'])
    day_score_facebook=day_score_facebook.append({'date':data_score.iloc[0,0],'pos':data_score['pos'].sum()/len(data_score),'neg':data_score['neg'].sum()/len(data_score)}, ignore_index=True)
    day_score_facebook['class_type'] = np.where(day_score_facebook['pos']>=day_score_facebook['neg'], 1, 0)
    #day_score_facebook.to_csv(csv_names[j], index=False)
    full_table_id = database[j]
    project_id = 'tanta-stocks'
    day_score_facebook.to_gbq(full_table_id, project_id=project_id, if_exists='replace')



