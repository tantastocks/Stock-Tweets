import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
from google.cloud import bigquery as bq

matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

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
prediction_database = ["facebook.facebook_monthly_prediction", "apple.apple_monthly_prediction", "amazon.amazon_monthly_prediction", "nokia.nokia_monthly_prediction", "intel.intel_monthly_prediction", "microsoft.microsoft_monthly_prediction", "NVIDIA.NVIDIA_monthly_prediction", "AMD.AMD_monthly_prediction", "twitter.twitter_monthly_prediction", "google.google_monthly_prediction"]

for j in range(0,10):
    
    # Reading the data
    
    df = database[j].result().to_dataframe()
    stock = df
    
    stock['Date'].min()
    stock['Date'].max()
    
    stock.Date = pd.to_datetime(stock.Date, format='%Y%m%d', errors='ignore')
    
    cols = ['High', 'Low', 'Open', 'Volume', 'AdjClose']
    stock.drop(cols, axis=1, inplace=True)
    stock = stock.sort_values('Date')
    
    stock.isnull().sum()
    
    stock = stock.groupby('Date')['Close'].sum().reset_index()
    
    stock = stock.set_index('Date')
    stock.index
    
    #y = stock['Close'].resample('M').mean()
    stock.index = pd.to_datetime(stock.index)
    
    monthly_mean = stock.Close.resample('M').mean()
    
    
    
    from pylab import rcParams
    rcParams['figure.figsize'] = 18, 8
    
    decomposition = sm.tsa.seasonal_decompose(monthly_mean, model='additive')
    fig = decomposition.plot()
    plt.show()
    
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    
    print('Examples of parameter combinations for Seasonal ARIMA...')
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
    
    l_param = []
    l_param_seasonal=[]
    l_results_aic=[]
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(monthly_mean,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
    
                results = mod.fit()
    
                print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
                
                l_param.append(param)
                l_param_seasonal.append(param_seasonal)
                l_results_aic.append(results.aic)
            except:
                continue
            
    minimum=l_results_aic[0]
    for i in l_results_aic[1:]:
        if i < minimum: 
            minimum = i
    i=l_results_aic.index(minimum)
    
    mod = sm.tsa.statespace.SARIMAX(monthly_mean,
                                    order=l_param[i],
                                    seasonal_order=l_param_seasonal[i],
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    
    results = mod.fit()
    
    print(results.summary().tables[1])
    
    results.plot_diagnostics(figsize=(16, 8))
    plt.show()
    
    pred_uc = results.get_forecast(steps=100)
    pred_ci = pred_uc.conf_int()
    print(pred_uc.predicted_mean)
    
    
    prediction = pd.DataFrame(pred_uc.predicted_mean)
    prediction.reset_index(level=0, inplace=True)
    prediction.columns = ['Date', 'monthly_prediction']
    
    full_table_id = prediction_database[j]
    project_id = 'tanta-stocks'
    
    prediction.to_gbq(full_table_id, project_id=project_id, if_exists='replace')

