import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from fbprophet import Prophet
plt.style.use('fivethirtyeight')
from datetime import datetime
plt.rcParams['figure.figsize']=(18,6)
plt.style.use('ggplot')
from fbprophet.diagnostics import cross_validation, performance_metrics
from fbprophet.plot import plot_cross_validation_metric
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from google.cloud import bigquery as bq

client = bq.Client()


facebook_query= client.query("""
    SELECT *  FROM `tanta-stocks.facebook.facebook_data`
    """)
apple_query = client.query("""
    SELECT *  FROM `tanta-stocks.apple.apple_data`
    """)
amazon_query = client.query("""
    SELECT *  FROM `tanta-stocks.amazon.amazon_data`
    """)
nokia_query = client.query("""
    SELECT *  FROM `tanta-stocks.nokia.nokia_data`
    """)
intel_query = client.query("""
    SELECT *  FROM `tanta-stocks.intel.intel_data`
    """)
microsoft_query = client.query("""
    SELECT *  FROM `tanta-stocks.microsoft.microsoft_data`
    """)
NVIDIA_query = client.query("""
    SELECT *  FROM `tanta-stocks.NVIDIA.NVIDIA_data`
    """)
AMD_query = client.query("""
    SELECT *  FROM `tanta-stocks.AMD.AMD_data`
    """)
twitter_query = client.query("""
    SELECT *  FROM `tanta-stocks.twitter.twitter_data`
    """)
google_query = client.query("""
    SELECT *  FROM `tanta-stocks.google.google_data`
    """)
database = [facebook_query, apple_query, amazon_query, nokia_query, intel_query, microsoft_query, NVIDIA_query, AMD_query, twitter_query, google_query]
prediction_database = ["facebook.facebook_daily_prediction", "apple.apple_daily_prediction", "amazon.amazon_daily_prediction", "nokia.nokia_daily_prediction", "intel.intel_daily_prediction", "microsoft.microsoft_daily_prediction", "NVIDIA.NVIDIA_daily_prediction", "AMD.AMD_daily_prediction", "twitter.twitter_prediction", "google.google_daily_prediction"]

for j in range(0,10):
    
    df2 = database[j].result().to_dataframe()
    df2.tail()
    
    df2['Date'] = df2['Date'].dt.strftime('%Y-%m-%d')
    
    del df2['High']
    del df2['Low']
    del df2['Open']
    del df2['Close']
    del df2['Volume']
    
    print(df2.shape[0] , 'Rows And' , df2.shape[1] , 'Cols')
    print('Min date' , df2.Date.min() , 'Max date' , df2.Date.max())
    df2.head()
    
    df2.info()
    
    df2.plot(x = 'Date', y = 'AdjClose')
    plt.xlabel("Date",fontsize=12,fontweight='bold',color='gray')
    plt.ylabel('Close',fontsize=12,fontweight='bold',color='gray')
    plt.title("Apple data",fontsize=18)
    plt.show()
    
    train = df2[0:df2.shape[0]*2//3]
    test = df2[-500:]
    
    train = train.rename(index=str, columns={"Date": "ds", "AdjClose": "y"})
    test = test.rename(index=str, columns={"Date": "ds", "AdjClose": "y"})
    df2 = df2.rename( columns={"Date": "ds", "AdjClose": "y"})
    
    model = Prophet()
    #model.add_seasonality('self_define_cycle',period=30,fourier_order=8,mode='additive')
    model.fit(test)
    future = model.make_future_dataframe(periods=30) #forecasting for 1 year from now.
    forecast = model.predict(future)
    forecast['day_week'] = forecast.ds.dt.weekday_name
    forecast = forecast[forecast.day_week != 'Sunday']
    forecast = forecast[forecast.day_week != 'Saturday']
    figure=model.plot(forecast)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    len(forecast)
    
    fig2 = model.plot_components(forecast)
    
    df_cv = cross_validation(model, horizon='30 days')
    df_p = performance_metrics(df_cv)
    
    
    df_p.tail()
    
    fig3 = plot_cross_validation_metric(df_cv, metric='mape')
    
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(20)
    
    prediction = forecast[['ds', 'yhat']].copy()
    
    prediction.columns = ['Date', 'daily_prediction']
    
    full_table_id = prediction_database[j]
    project_id = 'tanta-stocks'
    
    prediction.to_gbq(full_table_id, project_id=project_id, if_exists='replace')


