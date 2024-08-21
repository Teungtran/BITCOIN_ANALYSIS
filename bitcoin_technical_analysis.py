import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.io as pio
import talib as tb

pio.renderers.default= 'browser'
ticker = 'BTC-USD'
start = '2023-01-01'
end = '2024-01-01'

# extract data from yahoo finance
df = yf.download(ticker, start, end)

#Simple Moving Average (sma of Close price)
df['SMA'] = tb.SMA(df["Close"], timeperiod = 30)

# Relative strength index (RSI)
df['RSI'] = tb.RSI(df['Close'], timeperiod = 15)

# 2 standard deviation to the center SMA line ( check volatility)
df['Upper_BB'], df['Middle_BB'], df['Lower_BB'] = tb.BBANDS(df['Close'], timeperiod = 20, nbdevup = 2, nbdevdn=2, matype=0)

#visualization
fig = make_subplots(cols = 1, rows = 2, shared_xaxes= True, vertical_spacing=0.2, row_heights=[0.7, 0.3],
                    subplot_titles=[f'{ticker} PRICE AND INDICATORS', 'RSI', ]) 

# create lines for the indicators
candles = go.Candlestick(
    x = df.index,
    open = df.Open,
    high = df.High,
    close = df.Close,
    low = df.Low,
    name = 'Price'
)

sma_line = go.Scatter(
    x = df.index,
    y = df.SMA,
    line = {'color': 'blue', 'width' : 2},
    name = 'SMA'
)

rsi_line = go.Scatter(
    x = df.index,
    y = df.RSI,
    line = {'color': 'orange', 'width' : 2},
    name = 'RSI'
)

upper_BB = go.Scatter(
    x = df.index,
    y = df['Upper_BB'],
    line = {'color': 'red', 'width' : 2},
    name = 'Upper'
)

lower_BB = go.Scatter(
    x = df.index,
    y = df['Lower_BB'],
    line = {'color': 'cyan', 'width' : 2},
    name = 'Lower'
)

middle_BB = go.Scatter(
    x = df.index,
    y = df['Middle_BB'],
    line = {'color': 'green', 'width' : 2},
    name = 'Middle'
)

# add trace for the chart
fig.add_trace(candles, row =1,col= 1)
fig.add_trace(upper_BB, row =1,col= 1)
fig.add_trace(lower_BB, row =1,col= 1)
fig.add_trace(middle_BB, row =1,col= 1)
fig.add_trace(sma_line, row =1,col= 1)
fig.add_trace(rsi_line, row =2,col= 1)

# update layout ( title, name...)
fig.update_layout(
    title = f'{ticker} Technical Analysis',
    yaxis_title = 'Price',
    xaxis_title = 'Date',
    xaxis_rangeslider_visible = False,
    height = 900,
    template = 'plotly_dark'
)
fig.update_yaxes( range = [0,100], row =2, col =1 )

fig.show()