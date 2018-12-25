import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import requests
from bs4 import BeautifulSoup


def RSI_indicator(Close, RSI_window):
    df = pd.DataFrame(Close)
    df.columns = ['Close']
    df['Up'] = df.Close - df.Close.shift(1)
    df['Up'][df['Up'] < 0] = 0
    df['Down'] = df.Close.shift(1) - df.Close
    df['Down'][df['Down'] < 0] = 0
    df['first_AG'] = df['Up'].rolling(RSI_window).mean()
    df['first_AL'] = df['Down'].rolling(RSI_window).mean()
    df = df.dropna()
    df['Up'].iloc[0] = df['first_AG'].iloc[0]
    df['Down'].iloc[0] = df['first_AL'].iloc[0]
    df['AG'] = df['Up'].ewm(com=RSI_window-1).mean()
    df['AL'] = df['Down'].ewm(com=RSI_window-1).mean()
    df['RS'] = df['AG']/df['AL']
    df['RSI'] = 100 - 100/(1+df['RS'])
    df = df.dropna()
    return df.RSI


def rolling_zScore(Close, zScore_window):
    df = pd.DataFrame(Close)
    df.columns = ['Close']
    df['rolling_zScore'] = df.Close.rolling(window=zScore_window+1).apply(lambda x: (x[-1]- x[:-1].mean())/x[:-1].std(), raw=True)
    df = df.dropna()
    return df['rolling_zScore']


def rsi_return(df, RSI_window, RSI_threshold):
    df = pd.DataFrame(df)
    df.columns = ['Price']
    df['RSI_14'] = RSI_indicator(df.copy(), 14)
    df['Price_lead5'] = df.iloc[:, 0].shift(-5)    
    df = df[df.RSI_14 < RSI_threshold]
    df = df.dropna()
    return_5day = (df['Price_lead5'] - df['Price'])
    return return_5day

df = pd.read_excel('Dymon Candidate Test.xlsx', 'Task 1').reset_index()
ccy_list = df.columns.values[1::2].tolist()
ccy_list.remove('usdjpy.1')
df_list = []
for ccy in ccy_list:
    ccy_loc = df.columns.get_loc(ccy)
    ccy_name = ccy[:6].upper()
    df_ccy = df.iloc[:, [ccy_loc-1, ccy_loc]]
    df_ccy.columns = ['Date', ccy_name]
    df_ccy = df_ccy.dropna()
    df_ccy = df_ccy.set_index('Date')
    df_ccy = df_ccy.dropna()
    df_ccy = df_ccy[(stats.zscore(df_ccy)<3.0) & (stats.zscore(df_ccy)>-3.0)]
    df_ccy = df_ccy[df_ccy[ccy_name]>0.0]
    df_list.append(df_ccy)
df_all = pd.concat(df_list, axis=1) 

# use USDCAD as an example
df_5ay_return = rsi_return(df_all['USDCAD'].copy(), RSI_window=14, RSI_threshold=30)
df_rolling_zscore = rolling_zScore(df_all['USDCAD'].copy(), zScore_window=100)


url = "https://www.federalreserve.gov/releases/h10/weights/default.htm"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'lxml')
table = soup.find("table", attrs={"title":"Total Trade Weights"})
df_weights = pd.read_html(str(table))[0]


ccy_list = [ii[:6].upper() for ii in ccy_list]
ccy_dict = {'USDCNY': 'China', 'EURUSD': 'Euro area', 
            'USDJPY': 'Japan', 'USDCHF': 'Switzerland',
            'USDMXN': 'Mexico', 'USDCAD': 'Canada',
            'AUDUSD': 'Australia'}

df_weights = pd.read_html(str(table))[0]
df_weights = df_weights.set_index(df_weights.columns.values[0])
df_weights = df_weights.iloc[1:, :]
df_weights.index = df_weights.index.str.replace('*', '')
df_weights = df_weights.loc[list(ccy_dict.values()), :]
df_weights.index = ccy_dict.keys()
df_weights = df_weights.div(df_weights.sum(axis=0), axis=1)


df_all = df_all.dropna()
df_agg_list = []
for year in df_all.index.year.unique():
    df_agg = (df_all[str(year)] * df_weights[str(year)]).sum(axis=1)
    df_agg_list.append(df_agg)

df_dollar_index = pd.concat(df_agg_list, axis=0)
df_dollar_index = df_dollar_index.sort_index()
df_dollar_index = df_dollar_index*100.0/df_dollar_index['1997-01-02']

df_benchmark = pd.read_excel('Dymon Candidate Test.xlsx', 'Task 2')

df_combined = pd.concat([df_benchmark, df_dollar_index], axis=1).dropna()
df_combined.columns = [df_benchmark.columns[0], 'dollar index']
df_combined.plot()
df_combined.to_csv('dollar_index.csv')

plt.show()




