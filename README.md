# 📈 Multi-day Stock Price Forecasting using LSTM + Streamlit

This project is a **Streamlit-based interactive web application** that predicts the **next N days' closing prices** of a stock using an **LSTM (Long Short-Term Memory) neural network**.

---

## 🚀 **Features**

✅ Fetches historical stock data from **Yahoo Finance**  
✅ Preprocesses and scales data for LSTM training  
✅ Builds a **deep LSTM model** with Dropout layers to reduce overfitting  
✅ Supports **multi-day sequence forecasting** using recursive prediction  
✅ Visualizes:
- Historical closing prices
- Forecasted future prices

✅ Displays forecast results in a **clean data table**  
✅ Simple, responsive **Streamlit UI**

---

## 🛠️ **Installation**

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/lstm-stock-forecast.git
cd lstm-stock-forecast

python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows

streamlit
yfinance
numpy
pandas
scikit-learn
tensorflow
matplotlib

streamlit run app.py
