import pandas as pd
import numpy as np
import pickle
import datetime
from google.cloud import bigquery as bq

def to_sequences(seq_size, data,close):
    x = []
    y = []

    for i in range(len(data)-seq_size-1):
        window = data[i:(i+seq_size)]
        after_window = close[i+seq_size]
        window = [[x] for x in window]
        x.append(window)
        y.append(after_window)
        
    return np.array(x),np.array(y)

client = bq.Client()

facebook_query= client.query("""
    SELECT *  FROM `tanta-stocks.facebook.facebook_day_score` ORDER BY Date DESC LIMIT 9
    """)
apple_query = client.query("""
    SELECT *  FROM `tanta-stocks.apple.apple_day_score` ORDER BY Date DESC LIMIT 9
    """)
amazon_query = client.query("""
    SELECT *  FROM `tanta-stocks.amazon.amazon_day_score` ORDER BY Date DESC LIMIT 9
    """)
nokia_query = client.query("""
    SELECT *  FROM `tanta-stocks.nokia.nokia_day_score` ORDER BY Date DESC LIMIT 9
    """)
intel_query = client.query("""
    SELECT *  FROM `tanta-stocks.intel.intel_day_score` ORDER BY Date DESC LIMIT 9
    """)
microsoft_query = client.query("""
    SELECT *  FROM `tanta-stocks.microsoft.microsoft_day_score` ORDER BY Date DESC LIMIT 9
    """)
NVIDIA_query = client.query("""
    SELECT *  FROM `tanta-stocks.NVIDIA.NVIDIA_day_score` ORDER BY Date DESC LIMIT 9
    """)
AMD_query = client.query("""
    SELECT *  FROM `tanta-stocks.AMD.AMD_day_score` ORDER BY Date DESC LIMIT 9
    """)
twitter_query = client.query("""
    SELECT *  FROM `tanta-stocks.twitter.twitter_day_score` ORDER BY Date DESC LIMIT 9
    """)
google_query = client.query("""
    SELECT *  FROM `tanta-stocks.google.google_day_score` ORDER BY Date DESC LIMIT 9
    """)

facebook_query2 = client.query("""
    SELECT *  FROM `tanta-stocks.facebook.facebook_data` ORDER BY Date DESC LIMIT 9
    """)
apple_query2 = client.query("""
    SELECT *  FROM `tanta-stocks.apple.apple_data` ORDER BY Date DESC LIMIT 9
    """)
amazon_query2 = client.query("""
    SELECT *  FROM `tanta-stocks.amazon.amazon_data` ORDER BY Date DESC LIMIT 9
    """)
nokia_query2 = client.query("""
    SELECT *  FROM `tanta-stocks.nokia.nokia_data` ORDER BY Date DESC LIMIT 9
    """)
intel_query2 = client.query("""
    SELECT *  FROM `tanta-stocks.intel.intel_data` ORDER BY Date DESC LIMIT 9
    """)
microsoft_query2 = client.query("""
    SELECT *  FROM `tanta-stocks.microsoft.microsoft_data` ORDER BY Date DESC LIMIT 9
    """)
NVIDIA_query2 = client.query("""
    SELECT *  FROM `tanta-stocks.NVIDIA.NVIDIA_data` ORDER BY Date DESC LIMIT 9
    """)
AMD_query2 = client.query("""
    SELECT *  FROM `tanta-stocks.AMD.AMD_data` ORDER BY Date DESC LIMIT 9
    """)
twitter_query2 = client.query("""
    SELECT *  FROM `tanta-stocks.twitter.twitter_data` ORDER BY Date DESC LIMIT 9
    """)
google_query2 = client.query("""
    SELECT *  FROM `tanta-stocks.google.google_data` ORDER BY Date DESC LIMIT 9
    """)

database = ["facebook.facebook_CNN", "apple.apple_CNN", "amazon.amazon_CNN","nokia.nokia_CNN", "intel.intel_CNN", "microsoft.microsoft_CNN", "NVIDIA.NVIDIA_CNN", "AMD.AMD_CNN", "twitter.twitter_CNN", "google.google_CNN"]
dataset = [facebook_query, apple_query, amazon_query, nokia_query, intel_query, microsoft_query, NVIDIA_query, AMD_query, twitter_query, google_query]
dataset2 = [facebook_query2, apple_query2, amazon_query2, nokia_query2, intel_query2, microsoft_query2, NVIDIA_query2, AMD_query2, twitter_query2, google_query2]

for j in range(0,10):
    #LSTM_df = pd.read_csv("FB (1).csv")
    #day_score = pd.read_csv("day_score.csv")
    LSTM_df = dataset2[j].result().to_dataframe()
    day_score = dataset[j].result().to_dataframe()
    pos = []
    neg = []
    for i in range (0,9):
        pos.insert(i, day_score.at[i,'pos'])
        neg.insert(i, day_score.at[i,'neg'])
    
    LSTM_df['pos'] = pd.Series(pos, index=LSTM_df.index)
    LSTM_df['neg'] = pd.Series(neg, index=LSTM_df.index)
    LSTM_dff = LSTM_df.copy()
    
    LSTM_df = LSTM_df.drop(['AdjClose'], axis=1)
    LSTM_df = LSTM_df.drop(['Date'], axis=1)
    close_df = LSTM_df['Close']
    filename = 'CNN_scaler.sav'
    scaler = pickle.load(open(filename, 'rb'))
    LSTM_df = scaler.transform(LSTM_df)
    
    percent_30 = len(LSTM_df) - 9
    train = LSTM_df[0:percent_30]
    test = LSTM_df[percent_30:len(LSTM_df)]
    
    close_train = close_df[0:percent_30].values
    close_test = close_df[percent_30:len(close_df)].values
    
    
    SEQUENCE_SIZE = 7
    
    x_test,y_test = to_sequences(SEQUENCE_SIZE,test,close_test)
    
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],x_test.shape[3]))
    
    filename = 'CNN_model.sav'
    cnn = pickle.load(open(filename, 'rb'))
    cnn.load_weights('best_weights_cnn.hdf5') # load weights from best model
    x_test = x_test.reshape((-1,1,7,7))
    pred = cnn.predict(x_test)
    pred=pred.reshape((-1,1))
    
    Current_Date = datetime.datetime.today()
    NextDay_Date = datetime.datetime.today() + datetime.timedelta(days=1)
    columns = ['Date','Prediction']
    realtime_row1 = pd.DataFrame(columns=columns)
    realtime_row1['Date'] = pd.Series(NextDay_Date)
    realtime_row1['Prediction'] = pd.Series(pred.flatten())
    full_table_id = database[j]
    project_id = 'tanta-stocks'
    realtime_row1.to_gbq(full_table_id, project_id=project_id, if_exists='replace')
