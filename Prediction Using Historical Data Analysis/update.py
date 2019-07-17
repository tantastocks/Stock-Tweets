import pandas_datareader.data as web
import datetime
import pandas

codes = ["FB", "AAPL", "AMZN", "NOK", "INTC", "MSFT", "NVDA", "AMD", "TWTR", "GOOG"]
database = ["facebook.facebook_data", "apple.apple_data", "amazon.amazon_data","nokia.nokia_data", "intel.intel_data", "microsoft.microsoft_data", "NVIDIA.NVIDIA_data", "AMD.AMD_data", "twitter.twitter_data", "google.google_data"]
start = datetime.datetime(2000, 1, 1)
Current_Date = datetime.date.today()

for i in range (0,10):
    dataframe = web.DataReader( codes[i], 'yahoo', str(Current_Date))
    dataframe.reset_index(level=0, inplace=True)
    dataframe.columns = ['Date', 'High', 'Low', 'Open', 'Close', 'Volume', 'AdjClose']
    dataframe.columns = ['Date', 'High', 'Low', 'Open', 'Close', 'Volume', 'AdjClose']
    dataframe['High'] = dataframe['High'].astype(float)
    dataframe['Low'] = dataframe['Low'].astype(float)
    dataframe['Open'] = dataframe['Open'].astype(float)
    dataframe['Close'] = dataframe['Close'].astype(float)
    dataframe['Volume'] = dataframe['Volume'].astype(int)
    dataframe['AdjClose'] = dataframe['AdjClose'].astype(float)
    full_table_id = database[i]
    project_id = 'tanta-stocks'
    dataframe.to_gbq(full_table_id, project_id=project_id, if_exists='append', table_schema = [{'name': 'Date', 'type': 'TIMESTAMP'},{'name': 'High', 'type': 'FLOAT'},{'name': 'Low', 'type': 'FLOAT'},{'name': 'Open', 'type':'FLOAT'},{'name': 'Close', 'type': 'FLOAT'},{'name': 'Volume', 'type': 'INTEGER'},{'name': 'AdjClose', 'type': 'FLOAT'}])
