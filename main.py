import streamlit as st 
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data
from datetime import datetime
from keras.models import load_model

# Set start and end dates for data retrieval
start_date = '2012-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')

# Streamlit app title
st.title("Stock Trend Prediction")

# Input for stock symbol
stock_symbol = st.text_input("Enter the Stock Symbol", "AAPL")

# Validate stock symbol and fetch data
try:
    df =  yf.download(stock_symbol, start=start_date, end=end_date)
except:
    st.error("Error: Invalid stock symbol or data retrieval issue. Please enter a valid stock symbol.")
    st.stop()
    

df = pd.DataFrame(df)
# Display data description
st.subheader(f"Data for {stock_symbol} from {start_date} to {end_date}")
st.write(df.describe())

# Plot closing price of the stock
st.subheader("Closing Price Plot")
fig, ax = plt.subplots(figsize=(15, 8))
plt.plot(df['Close'])
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.title(f"Closing Price Plot for {stock_symbol}")
st.pyplot(fig)

st.subheader("Closing Price vs  100MA")
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(15,8))
plt.plot(ma100)
st.pyplot(fig)

st.subheader("Closing Price vs  100MA & 200MA")
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(15,8))
plt.plot(ma100,'r')
plt.plot(ma200, 'g')
plt.plot(df.Close,'b')
st.pyplot(fig)

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_training_scale = scaler.fit_transform(data_training)

X_train = []
y_train = []

for i in range(100, data_training.shape[0]):
    X_train.append(data_training_scale[i-100: i])
    y_train.append(data_training_scale[i,0])

X_train, y_train = np.array(X_train), np.array(y_train)    
    
model = load_model("Keras_Model.h5")

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

X_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    X_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])
    
X_test, y_test = np.array(X_test), np.array(y_test)

y_pred = model.predict(X_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_pred = y_pred * scale_factor
y_test = y_test * scale_factor

st.subheader("Original vs Prediction")
fig2 = plt.figure(figsize=(15,8))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_pred, 'r', label = 'Predicted Price')
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig2)


last_100_days = df['Close'][-100:].values.reshape(1, 100, 1) 

# Recreate the scaler and fit it to the training data
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_scale = scaler.fit_transform(data_training)

# Scale the last 100 days data using the new scaler
last_100_days_scaled = scaler.transform(last_100_days.reshape(-1, 1)).reshape(1, 100, 1)  # Scale and reshape

# Make a prediction for the next day's price
next_day_pred_scaled = model.predict(last_100_days_scaled)

# Inverse scale the prediction
next_day_pred = next_day_pred_scaled[0, 0] * scale_factor

# Get today's closing price
today_closing_price = df['Close'][-1]

# Determine the trend based on the comparison
if next_day_pred > today_closing_price:
    trend = "Up"
elif next_day_pred < today_closing_price:
    trend = "Down"
else:
    trend = "Stable"

st.subheader("Predicted Trend for Tomorrow:")
st.write(f"The predicted trend for tomorrow is {trend}.")

