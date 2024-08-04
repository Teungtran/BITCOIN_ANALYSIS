
import pandas as pd 
import matplotlib.pyplot as plt 
import mplfinance as mpf
import datetime as dt
import seaborn as sns
start = dt.datetime(2024,7,1)
end = dt.datetime.now()
df = pd.read_csv("BTC-USD.csv",parse_dates = ["Date"], index_col = "Date")
# draw overall chart
fig = df.plot(legend = True, kind ='line', xlabel = "Years", ylabel = "Price").figure
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)   
plt.figure(figsize = (10,5))    
plt.show()   

# draw general candle plots
timeframe = df.loc[start :end ,:]
fig = mpf.plot(timeframe, type = 'candle', volume =True, style = 'yahoo')
plt.show()

# draw line chart using seaborn
fig, ax = plt.subplots(figsize=(32, 10), dpi=100)
ax.set_title("Bitcoin CLOSE Price from 01/07-31/07")
ax.set_xlabel("Date")
ax.set_ylabel("CLOSE Price")
sns.lineplot(data=timeframe["Close"], palette= ['red'], linewidth=1, legend=True )
plt.show()
