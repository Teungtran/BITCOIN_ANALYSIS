import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import datetime as dt 
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
# clean data
crypto = 'BTC'
curr ='USD'
start = dt.datetime(2023,1,7)
end = dt.datetime.now()
df = pd.read_csv("BTC-USD.csv",parse_dates = ["Date"], index_col = "Date")
timeframe = df.loc[start :end ,:]

# prepare data

# give range (Scale the data)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(timeframe['Close'].values.reshape(-1,1))
prediction_days = 60
future = 30
# give detail of the chart (split data into x_train, y_train) 
x_train = []
y_train =[]

for x in range(prediction_days, len(scaled_data)-future):
    x_train.append(scaled_data[x-prediction_days:x,0])
    y_train.append(scaled_data[x+future,0])
    
x_train = np.array(x_train)
y_train = np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))

# create model LSTM from RNN(RECURRENT NEURAL NETWORK)
model = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1],1)))
model.add(Dropout(0,2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0,2))
model.add(LSTM(units = 50))
model.add(Dropout(0,2))
model.add(Dense(units = 1))
#compiling and training 
model.compile(optimizer = 'adam', loss ='mean_squared_error')
# epochs: How many times the data is trained in network
# batch_size: The number of training samples in the network at once 
model.fit(x_train, y_train, epochs  = 25, batch_size = 30)
# testing
test_start = dt.datetime(2023,1,6)
test_end = dt.datetime.now()
test_timeframe = df.loc[test_start:test_end,:]  
actual_price = test_timeframe['Close'].values
total = pd.concat((timeframe['Close'], test_timeframe['Close']), axis = 0)

model_input = total[len(total)-len(test_timeframe)-prediction_days:].values
model_input = model_input.reshape(-1,1)
model_input = scaler.fit_transform(model_input)

x_test =[]
for x in range (prediction_days, len(model_input)):
    x_test.append(model_input[x-prediction_days:x, 0])
    
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

predicted_price = model.predict(x_test)
predicted_price = scaler.inverse_transform(predicted_price)

# visualize (line chart)
plt.plot(actual_price, color ='blue',label= 'Actual Price')
plt.plot(predicted_price, color = 'r', label ='Predicted Price')
plt.title('Bitcoin Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc = 'upper left')
plt.show()

# PREDICT A DAY AHEAD
real_data = [model_input[len(model_input)+1-prediction_days:len(model_input)+1,0]]
real_data = np.array(real_data)
real_data = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
prediction = prediction[0]

print("prediction price for the next day is", prediction)


