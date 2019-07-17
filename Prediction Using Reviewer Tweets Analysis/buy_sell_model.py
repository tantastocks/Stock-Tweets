import pandas as pd 
import pickle
from google.cloud import bigquery as bq

client = bq.Client()

facebook_query= client.query("""
    SELECT *  FROM `tanta-stocks.facebook.facebook_last_day_score` 
    """)
apple_query = client.query("""
    SELECT *  FROM `tanta-stocks.apple.apple_last_day_score` 
    """)
amazon_query = client.query("""
    SELECT *  FROM `tanta-stocks.amazon.amazon_last_day_score` 
    """)
nokia_query = client.query("""
    SELECT *  FROM `tanta-stocks.nokia.nokia_last_day_score` 
    """)
intel_query = client.query("""
    SELECT *  FROM `tanta-stocks.intel.intel_last_day_score`
    """)
microsoft_query = client.query("""
    SELECT *  FROM `tanta-stocks.microsoft.microsoft_last_day_score` 
    """)
NVIDIA_query = client.query("""
    SELECT *  FROM `tanta-stocks.NVIDIA.NVIDIA_last_day_score`
    """)
AMD_query = client.query("""
    SELECT *  FROM `tanta-stocks.AMD.AMD_last_day_score` 
    """)
twitter_query = client.query("""
    SELECT *  FROM `tanta-stocks.twitter.twitter_last_day_score` 
    """)
google_query = client.query("""
    SELECT *  FROM `tanta-stocks.google.google_last_day_score` 
    """)

facebook_query2 = client.query("""
    SELECT *  FROM `tanta-stocks.facebook.facebook_data` ORDER BY Date DESC LIMIT 3
    """)
apple_query2 = client.query("""
    SELECT *  FROM `tanta-stocks.apple.apple_data` ORDER BY Date DESC LIMIT 3
    """)
amazon_query2 = client.query("""
    SELECT *  FROM `tanta-stocks.amazon.amazon_data` ORDER BY Date DESC LIMIT 3
    """)
nokia_query2 = client.query("""
    SELECT *  FROM `tanta-stocks.nokia.nokia_data` ORDER BY Date DESC LIMIT 3
    """)
intel_query2 = client.query("""
    SELECT *  FROM `tanta-stocks.intel.intel_data` ORDER BY Date DESC LIMIT 3
    """)
microsoft_query2 = client.query("""
    SELECT *  FROM `tanta-stocks.microsoft.microsoft_data` ORDER BY Date DESC LIMIT 3
    """)
NVIDIA_query2 = client.query("""
    SELECT *  FROM `tanta-stocks.NVIDIA.NVIDIA_data` ORDER BY Date DESC LIMIT 3
    """)
AMD_query2 = client.query("""
    SELECT *  FROM `tanta-stocks.AMD.AMD_data` ORDER BY Date DESC LIMIT 3
    """)
twitter_query2 = client.query("""
    SELECT *  FROM `tanta-stocks.twitter.twitter_data` ORDER BY Date DESC LIMIT 3
    """)
google_query2 = client.query("""
    SELECT *  FROM `tanta-stocks.google.google_data` ORDER BY Date DESC LIMIT 3
    """)

database = ["facebook.facebook_Decision", "apple.apple_Decision", "amazon.amazon_Decision","nokia.nokia_Decision", "intel.intel_Decision", "microsoft.microsoft_Decision", "NVIDIA.NVIDIA_Decision", "AMD.AMD_Decision", "twitter.twitter_Decision", "google.google_Decision"]
dataset = [facebook_query, apple_query, amazon_query, nokia_query, intel_query, microsoft_query, NVIDIA_query, AMD_query, twitter_query, google_query]
dataset2 = [facebook_query2, apple_query2, amazon_query2, nokia_query2, intel_query2, microsoft_query2, NVIDIA_query2, AMD_query2, twitter_query2, google_query2]

for j in range (0,10):
    df1=dataset2[j].result().to_dataframe()
    df1['Date'] = pd.to_datetime(df1['Date']).dt.date
    df1 = df1.sort_values('Date', ascending=True)
    del df1["High"]
    del df1["Low"]
    del df1["Open"]
    del df1["AdjClose"]
    del df1["Volume"]
    
    df2=dataset[j].result().to_dataframe()
    df2['date'] = pd.to_datetime(df2['date']).dt.date    
    df1['Date'] = pd.to_datetime(df1['Date'])
    df2['date']  = pd.to_datetime(df2['date'])
    realtime_row = pd.concat([df1.set_index('Date'), 
                     df2.set_index('date')], 
                     axis=1, 
                     keys=('a', 'b'))
    realtime_row.columns = realtime_row.columns.map('_'.join)
    
    #creat new column of the same size 
    change=[0]*len(realtime_row.a_Close)
    momentum=[0]*len(realtime_row.a_Close)
    decision=[0]*len(realtime_row.a_Close)
    #calculate momentum and change
    for i in range(1,len(realtime_row.a_Close)):
        if realtime_row.a_Close[i]>realtime_row.a_Close[i-1]:
            momentum[i]='1'
            change[i]=(realtime_row.a_Close[i]-realtime_row.a_Close[i-1])/realtime_row.a_Close[i-1]
        else:
            momentum[i]='0'
            change[i]=(realtime_row.a_Close[i-1]-realtime_row.a_Close[i])/realtime_row.a_Close[i-1]
    realtime_row['change']=change
    realtime_row['momentum']=momentum
    realtime_row
    
    realtime_row1=realtime_row[2:]
    
    
    realtime_row1.rename(columns={"a_Close":"close","b_pos":"positive","b_neg":"negative","b_class_type":"sentiment"},inplace=True)
    print(realtime_row1)
    print("0")
    realtime_row1.momentum=realtime_row1.momentum.astype(int)
    print("1.55555555555")
    realtime_row1.sentiment=realtime_row1.sentiment.astype(int)
    print("11111111111111")
    realtime =realtime_row1.iloc[0:,0:].values
    
    # Feature Scaling
    filename = 'Scaler_sellbuy.sav'
    sc = pickle.load(open(filename, 'rb'))
    realtime= sc.transform(realtime)
    
    filename = 'DecisionTree_Model.sav'
    classifier = pickle.load(open(filename, 'rb'))
    
    ypred = classifier.predict(realtime)
    
    realtime_row1['Decision'] = pd.Series(ypred, index=realtime_row1.index)
    
    full_table_id = database[j]
    project_id = 'tanta-stocks'
    realtime_row1.to_gbq(full_table_id, project_id=project_id, if_exists='replace')
