import pandas as pd
import numpy as np
import re
from string import punctuation
from sklearn.externals import joblib
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

#generate series based on date
day_out=[]
def gen_fun(data_frame):
    
    out= []
    try:

        for i in range(len(data_frame['date'])):

            if data_frame.iloc[i,0] == data_frame.iloc[i+1,0]:
                out.append(data_frame.iloc[i,0])

            else : 
                out.append(data_frame.iloc[i,0])

                break

    except IndexError:
        pass
        
    return out

#generate sub local data frame and new global data frame

def dec_fun(series):
    
    dff= pd.DataFrame({'date2':series})
    
    dff= df_tweet_preds[df_tweet_preds['date'].isin(dff['date2'])]
    
    #df_tweet_preds= df_tweet_preds[~df_tweet_preds['date'].isin(dff['date'])]
     
    return dff
   

#split the data frame on 3 regions "positive"

def split_fn_pos(data_frame):
    
    out_low = data_frame[(data_frame.retweets <= 10)]
    out_low_predict = out_low['positive']

    out_med = data_frame[(data_frame.retweets < 20) & (data_frame.retweets > 10)] 
    out_med_predict = out_med['positive']

    out_high = data_frame[(data_frame.retweets >= 20)]
    out_high_predict = out_high['positive']
    
    out_all_pos= pd.DataFrame({'high':out_high_predict , 'medium':out_med_predict , 'low' : out_low_predict})
    
    return out_all_pos


#split the data frame on 3 regions "negative"

def split_fn_neg(data_frame):
    
    out_low = data_frame[(data_frame.retweets <= 15)]
    out_low_predict = out_low['negative']

    out_med = data_frame[(data_frame.retweets < 30) & (data_frame.retweets > 15)] 
    out_med_predict = out_med['negative']

    out_high = data_frame[(data_frame.retweets >= 30)]
    out_high_predict = out_high['negative']
    
    out_all_neg= pd.DataFrame({'high':out_high_predict , 'medium':out_med_predict , 'low' : out_low_predict})
    
    return out_all_neg


#score function

def score_fn(data_frame):
    score = 0.0
    score = data_frame.mean()
    
    return score

#     print("User Positive Score is :" , score , "%")

#     pos_score = pd.DataFrame(score, columns = ['Positive Score']) 

#     pos_score.to_csv(path_or_buf='pos_score.csv',index=False)

#     return pos_score


#final score out

def outt_fn (h,m,l):

    final_score= []
    score= []
    percentage_low= 0.0
    percentage_med= 0.0
    percentage_high= 0.0

    try:

        percentage_low = score_fn(l)
        percentage_med = score_fn(m)
        percentage_high = score_fn(h)

    except ZeroDivisionError:
        pass

    try:

        score.append(percentage_low)

    except IndexError:
        pass

    try:
        
        score.append(percentage_med*1.05)

    except IndexError:
        pass


    try:

        score.append(percentage_high*1.15)
        
    except IndexError:
        pass
    
    score_df= pd.DataFrame(score)
    
    final_score.append(score_df.mean())
        
    pos_score = final_score[0] 

    return pos_score

#helper function to clean tweets
def processTweet(tweet):
    # Remove HTML special entities (e.g. &amp;)
    tweet = re.sub(r'\&\w*;', '', tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','',tweet)
    # Remove tickers
    tweet = re.sub(r'\$\w*', '', tweet)
    # To lowercase
    tweet = tweet.lower()
    # Remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*\/\w*', '', tweet)
    # Remove hashtags
    tweet = re.sub(r'#\w*', '', tweet)
    # Remove Punctuation and split 's, 't, 've with a space for filter
    tweet = re.sub(r'[' + punctuation.replace('@', '') + ']+', ' ', tweet)
    # Remove words with 2 or fewer letters
    tweet = re.sub(r'\b\w{1,2}\b', '', tweet)
    # Remove whitespace (including new line characters)
    tweet = re.sub(r'\s\s+', ' ', tweet)
    tweet = re.sub(r'أ ب ت ث ج ح خ د ذ ر ز س ش ص ض ط ظ ع غ ف ق ك ل م ن هـ و ي', ' ', tweet)
    # Remove single space remaining at the front of the tweet.
    tweet = tweet.lstrip(' ') 
    # Remove characters beyond Basic Multilingual Plane (BMP) of Unicode:
    tweet = ''.join(c for c in tweet if c <= '\uFFFF') 
    return tweet
# ______________________________________________________________
dataset = [facebook_query, apple_query, amazon_query, nokia_query, intel_query, microsoft_query, NVIDIA_query, AMD_query, twitter_query, google_query]
database = ["facebook.facebook_day_score_sentiment", "apple.apple_day_score_sentiment", "amazon.amazon_day_score_sentiment","nokia.nokia_day_score_sentiment", "intel.intel_day_score_sentiment", "microsoft.microsoft_day_score_sentiment", "NVIDIA.NVIDIA_day_score_sentiment", "AMD.AMD_day_score_sentiment", "twitter.twitter_day_score_sentiment", "google.google_day_score_sentiment"]

for j in range (0,10):  
    test_data = dataset[j].result().to_dataframe()
    #test_data = test_data.drop(["Unnamed: 0"], axis=1)
    test_data.replace('', np.nan , inplace=True)
    test_data.columns = ['date','retweets', 'likes', 'message', 'user_name']
    
    test_data = test_data.dropna()
    
    #clean your train data
    
    test_set = test_data.copy()
    test_set['message'] = test_data['message'].apply(processTweet)
    
    # load from file and predict using the best configs found in the CV step
    model_NB = joblib.load("twitter_sentiment.pkl" )
    
    #run predictions on new twitter dataset
    
    tweet_preds = model_NB.predict(test_data['message'])
    
    # append predictions to dataframe
    df_tweet_preds = test_data.copy()
    df_tweet_preds['predictions'] = tweet_preds
    
    # get proba out way 2
    
    tweet_preds = model_NB.predict_proba(test_data['message'])
    
    # append predictions to dataframe
    df_tweet_preds_proba = df_tweet_preds.copy()
    
    df_tweet_preds_proba['negative'] = tweet_preds[:,:1]
    df_tweet_preds_proba['positive'] = tweet_preds[:,1:]
    
    df_tweet_preds = df_tweet_preds_proba.copy()
    
    df_tweet_preds['date'] = pd.to_datetime(df_tweet_preds.date).dt.strftime('%Y-%m-%d')
    
    df_tweet_preds['retweets'] = pd.to_numeric(df_tweet_preds.retweets , errors='coerce')
    
    day_out= []
    out_pos= []
    out_neg= []
    final= []
    
    while len(df_tweet_preds) != 0 :
    
        out1= gen_fun(df_tweet_preds)
        out2= dec_fun(out1)
        out3= split_fn_pos(out2)
        out3_2= split_fn_neg(out2)
            
        high_1 = out3['high']
        med_1 = out3['medium']
        low_1 = out3['low']
    
        high_pos = high_1[~pd.isnull(high_1)]
        med_pos = med_1[~pd.isnull(med_1)]
        low_pos = low_1[~pd.isnull(low_1)]
            
        high_2 = out3_2['high']
        med_2 = out3_2['medium']
        low_2 = out3_2['low']
    
        high_neg = high_2[~pd.isnull(high_2)]
        med_neg = med_2[~pd.isnull(med_2)]
        low_neg = low_2[~pd.isnull(low_2)]
    
        out_pos.append(outt_fn(high_pos,med_pos,low_pos)[0])
            
        out_neg.append(outt_fn(high_neg,med_neg,low_neg)[0])
            
        df_tweet_preds= df_tweet_preds[~df_tweet_preds['date'].isin(out2['date'])]
            
        day_out.append(out1[0])
            
        
    final= pd.DataFrame({'date':day_out, 'positive':out_pos , 'negative' : out_neg})
        
        
    full_table_id = database[j]
    project_id = 'tanta-stocks'
    final.to_gbq(full_table_id, project_id=project_id, if_exists='replace')
    
