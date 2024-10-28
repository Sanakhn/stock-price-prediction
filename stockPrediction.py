import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf

st.title("Stock Price Prediction")

stock = st.text_input("Enter the Stock ID", "AAPL")

from datetime import datetime
end = datetime.now()
start = datetime(end.year-20, end.month, end.day)

apple_data = yf.download(stock, start, end)

model = load_model("Stock_price_model.keras")
st.subheader("Stock Data")
st.write(apple_data)

splitting_len = int(len(apple_data)*0.7)
X_test = pd.DataFrame(apple_data.Close[splitting_len:])

def plot(figsize, values, full_data, extra_data = 0, extra_dataset = None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values,'Orange')
    plt.plot(full_data.Close, 'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig

st.subheader('Original Close Price and MA for 250 days')
apple_data['MA_for_250_days'] = apple_data.Close.rolling(250).mean()
st.pyplot(plot((15,6), apple_data['MA_for_250_days'], apple_data,0))

st.subheader('Original Close Price and MA for 200 days')
apple_data['MA_for_200_days'] = apple_data.Close.rolling(200).mean()
st.pyplot(plot((15,6), apple_data['MA_for_200_days'], apple_data,0))

st.subheader('Original Close Price and MA for 100 days')
apple_data['MA_for_100_days'] = apple_data.Close.rolling(100).mean()
st.pyplot(plot((15,6), apple_data['MA_for_100_days'], apple_data,0))

st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
st.pyplot(plot((15,6), apple_data['MA_for_100_days'], apple_data,1,apple_data['MA_for_250_days']))

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(X_test[['Close']])

X = []
y = []

for i in range(100,len(scaled_data)):
    X.append(scaled_data[i-100:i])
    y.append(scaled_data[i])

X, y = np.array(X), np.array(y)

prediction = model.predict(X)

prediction = scaler.inverse_transform(prediction)
y_test = scaler.inverse_transform(y)

plottingPredictions = pd.DataFrame(
 {
  'originalData': y_test.reshape(-1),
    'predictions': prediction.reshape(-1)
 } ,
    index = apple_data.index[splitting_len+100:]
)
st.subheader("Original values vs Predicted values")
st.write(plottingPredictions)

st.subheader('Original Close Price vs Predicted Close price')
fig = plt.figure(figsize=(15,6))
plt.plot(pd.concat([apple_data.Close[:splitting_len+100],plottingPredictions], axis=0))
plt.legend(["Data- not used", "Original Test data", "Predicted Test data"])
st.pyplot(fig)