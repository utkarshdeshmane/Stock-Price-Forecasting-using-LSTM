import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Multi-day Stock Price Forecasting using LSTM + Matplotlib")


@st.cache_data
def fetch_historical_data(stock_symbol, period):
    stock_symbol = stock_symbol.strip() + ".NS"
    stock_data = yf.download(stock_symbol, period=period, interval="1d", progress=False)
    if stock_data.empty or len(stock_data) < 60:
        raise ValueError(f"Insufficient data found for stock symbol: {stock_symbol}")
    return stock_data


def prepare_lstm_data(data, time_step=60):
    X, y = [], []
    for i in range(len(data)-time_step-1):
        X.append(data[i:(i+time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)


st.sidebar.header("Input Parameters")
stock_symbol = st.sidebar.text_input("Enter stock symbol:", "TCS").upper().strip()
period = st.sidebar.selectbox("Select data period:", ["6mo", "1y", "2y"], index=1)
epochs = st.sidebar.slider("Training epochs:", 5, 50, 10, 5)
forecast_days = st.sidebar.slider("Forecast days:", 1, 10, 3, 1)

try:

    stock_data = fetch_historical_data(stock_symbol, period)
    df = stock_data[['Close']]

    st.markdown("### Latest Data Snapshot")
    st.dataframe(df.tail())

    st.markdown("### Historical Closing Prices")
    plt.figure(figsize=(10,4))
    plt.plot(df.index, df['Close'], label='Close Price', color='blue')
    plt.title(f"{stock_symbol} Closing Prices ({period})")
    plt.xlabel("Date")
    plt.ylabel("Price (INR)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()


    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(df.values)

 
    time_step = 60
    X, y_data = prepare_lstm_data(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

 
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_step,1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

 
    st.markdown("### Training LSTM Model")
    with st.spinner("Training... (please wait)"):
        history = model.fit(X, y_data, batch_size=32, epochs=epochs, verbose=0)

    
    st.markdown(f"### Forecasting Next {forecast_days} Days")
    predictions = []
    input_seq = scaled_data[-time_step:].reshape(1, time_step, 1)

    for _ in range(forecast_days):
        next_pred = model.predict(input_seq)
        predictions.append(next_pred[0,0])
        
       
        next_pred_reshaped = next_pred.reshape(1,1,1)
        

        input_seq = np.concatenate((input_seq[:,1:,:], next_pred_reshaped), axis=1)


    predictions_inv = scaler.inverse_transform(np.array(predictions).reshape(-1,1))

 
    last_date = df.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='B')

    forecast_df = pd.DataFrame(predictions_inv, index=forecast_dates, columns=['Close'])
    combined_df = pd.concat([df, forecast_df])

    plt.figure(figsize=(10,4))
    plt.plot(df.index, df['Close'], label='Historical', color='green')
    plt.plot(forecast_df.index, forecast_df['Close'], label='Forecast', color='red')
    plt.title(f"{stock_symbol} Closing Prices with {forecast_days}-Day Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price (INR)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

    st.markdown("### Forecasted Prices")
    st.dataframe(forecast_df)

    st.info("Note: LSTM predictions are based purely on historical prices without external market factors. Use cautiously for decision-making.")

except Exception as e:
    st.error(f"Error: {e}")
