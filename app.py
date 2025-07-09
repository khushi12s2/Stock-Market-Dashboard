import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
import statsmodels.api as sm
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from fpdf import FPDF
import sqlite3

st.set_page_config(page_title="Stock Dashboard", layout="wide")

# SQLite DB for caching
conn = sqlite3.connect("stock_data.db")
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS stock_cache (
    ticker TEXT,
    date TEXT,
    close REAL,
    PRIMARY KEY (ticker, date)
)
""")

# THEME TOGGLE
st.sidebar.header("Settings")
theme_option = st.sidebar.selectbox("🎨 Theme Mode", ["Light", "Dark"])
if theme_option == "Dark":
    st.markdown("""
        <style>
        body, .stApp { background-color: #0e1117; color: white; }
        .css-1v0mbdj, .css-1d391kg, .css-1cpxqw2, .css-1v3fvcr { color: white !important; }
        .css-18e3th9 { background-color: #0e1117; }
        </style>
    """, unsafe_allow_html=True)

refresh = st.sidebar.button("🔁 Refresh Data")
st.title("📈 Stock Dashboard & Predictor")

# CHART TYPE SWITCH
chart_type = st.sidebar.radio("📊 Chart Type", ["Line", "Candlestick", "Area"])

# EXPORT TO PDF
save_pdf = st.sidebar.button("📋 Save Report as PDF")

# SESSION CACHE
@st.cache_resource(show_spinner=False)
def fetch_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end)

ticker_groups = {
    "Tech": ["AAPL", "GOOGL", "MSFT"],
    "Automotive": ["TSLA"],
    "E-Commerce": ["AMZN"]
}
group_choice = st.sidebar.selectbox("Select Sector", list(ticker_groups.keys()), index=0)
tickers = st.sidebar.multiselect("Select Tickers", ticker_groups[group_choice], default=[ticker_groups[group_choice][0]])

start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", datetime.now())

st.sidebar.markdown("---")
show_rsi = st.sidebar.checkbox("Show RSI", True)
show_macd = st.sidebar.checkbox("Show MACD/Signal", True)
show_obv = st.sidebar.checkbox("Show OBV", True)

arima_toggle = st.sidebar.checkbox("🔁 Include ARIMA Forecast")
rsiperiod = st.sidebar.slider("RSI Period", 5, 30, 14)

st.sidebar.markdown("---")
if st.sidebar.button("⬇️ Download Data CSV"):
    st.session_state.download_trigger = True

show_model_comparison = st.sidebar.checkbox("📊 Compare All Models", True)
export_format = st.sidebar.multiselect("📤 Export Formats", ["CSV", "Excel", "JSON"], default=["CSV"])

all_comparison = pd.DataFrame()

for ticker in tickers:
    st.header(f"📊 {ticker} Stock Data")
    data = fetch_data(ticker, start_date, end_date) if not refresh else yf.download(ticker, start=start_date, end=end_date)

    if not data.empty:
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip() for col in data.columns.values]

        valid_cols = list(data.columns)
        plot_column = next((col for col in valid_cols if 'Close' in col and ticker in col), None)
        if not plot_column:
            st.error(f"❌ Neither 'Adj Close' nor 'Close' found in columns: {valid_cols}")
            st.stop()

        volume_col = next((col for col in valid_cols if 'Volume' in col and ticker in col), None)
        if not volume_col:
            st.error(f"❌ 'Volume' column not found in columns: {valid_cols}")
            st.stop()

        data['MA20'] = data[plot_column].rolling(window=20).mean()
        data['MA50'] = data[plot_column].rolling(window=50).mean()
        rolling_std = data[plot_column].rolling(window=20).std().squeeze()
        data['BB_upper'] = data['MA20'] + 2 * rolling_std
        data['BB_lower'] = data['MA20'] - 2 * rolling_std

        def compute_rsi(series, period=14):
            delta = series.diff()
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            avg_gain = pd.Series(gain.flatten(), index=series.index).rolling(window=period).mean()
            avg_loss = pd.Series(loss.flatten(), index=series.index).rolling(window=period).mean()
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))

        data['RSI'] = compute_rsi(data[plot_column], period=rsiperiod)
        data['EMA12'] = data[plot_column].ewm(span=12, adjust=False).mean()
        data['EMA26'] = data[plot_column].ewm(span=26, adjust=False).mean()
        data['MACD'] = data['EMA12'] - data['EMA26']
        data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

        obv = [0]
        for i in range(1, len(data)):
            curr = float(data[plot_column].iloc[i])
            prev = float(data[plot_column].iloc[i - 1])
            if curr > prev:
                obv.append(obv[-1] + data[volume_col].iloc[i])
            elif curr < prev:
                obv.append(obv[-1] - data[volume_col].iloc[i])
            else:
                obv.append(obv[-1])
        data['OBV'] = obv

        data['Buy_Signal'] = np.where((data['MACD'] > data['Signal']) & (data['RSI'] < 30), data[plot_column], np.nan)
        data['Sell_Signal'] = np.where((data['MACD'] < data['Signal']) & (data['RSI'] > 70), data[plot_column], np.nan)

        if chart_type == "Line":
            st.line_chart(data[[plot_column, 'MA20', 'MA50']])
        elif chart_type == "Area":
            fig_area = px.area(data, x=data.index, y=plot_column, title="Area Chart")
            st.plotly_chart(fig_area)
        elif chart_type == "Candlestick":
            fig_candle = go.Figure(data=[go.Candlestick(x=data.index,
                                                        open=data[f'Open_{ticker}'],
                                                        high=data[f'High_{ticker}'],
                                                        low=data[f'Low_{ticker}'],
                                                        close=data[plot_column])])
            fig_candle.add_trace(go.Scatter(x=data.index, y=data['Buy_Signal'], mode='markers', marker=dict(symbol='triangle-up', color='green'), name='Buy'))
            fig_candle.add_trace(go.Scatter(x=data.index, y=data['Sell_Signal'], mode='markers', marker=dict(symbol='triangle-down', color='red'), name='Sell'))
            fig_candle.update_layout(title='Candlestick Chart')
            st.plotly_chart(fig_candle)

        if show_rsi:
            st.line_chart(data[['RSI']])
        if show_macd:
            st.line_chart(data[['MACD', 'Signal']])
        if show_obv:
            st.line_chart(data[['OBV']])

        if show_model_comparison:
            st.subheader("📉 Model Comparison")

            lr_model = LinearRegression()
            X = np.arange(len(data)).reshape(-1, 1)
            y = data[plot_column].values
            lr_model.fit(X, y)
            preds_lr = lr_model.predict(X)
            data['LR_Prediction'] = preds_lr
            st.line_chart(data[['LR_Prediction']])
            all_comparison[ticker+'_LR'] = pd.Series(preds_lr.flatten(), index=data.index)

            scaled_data = MinMaxScaler().fit_transform(y.reshape(-1, 1))
            X_lstm, y_lstm = [], []
            for i in range(60, len(scaled_data)):
                X_lstm.append(scaled_data[i-60:i, 0])
                y_lstm.append(scaled_data[i, 0])

            if len(X_lstm) > 0:
                X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
                X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))

                lstm_model = Sequential()
                lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_lstm.shape[1], 1)))
                lstm_model.add(Dropout(0.2))
                lstm_model.add(LSTM(units=50))
                lstm_model.add(Dropout(0.2))
                lstm_model.add(Dense(1))
                lstm_model.compile(optimizer='adam', loss='mean_squared_error')
                lstm_model.fit(X_lstm, y_lstm, epochs=3, batch_size=32, verbose=0)

                predicted_lstm = lstm_model.predict(X_lstm)
                predicted_lstm = MinMaxScaler().fit(y.reshape(-1, 1)).inverse_transform(predicted_lstm)
                lstm_series = pd.Series(predicted_lstm.flatten(), index=data.index[-len(predicted_lstm):])
                st.line_chart(pd.DataFrame({"LSTM Prediction": lstm_series}))
                all_comparison[ticker+'_LSTM'] = lstm_series
            else:
                st.warning("📉 Not enough data to train LSTM (requires > 60 rows).")

            if arima_toggle:
                try:
                    arima_model = sm.tsa.ARIMA(data[plot_column], order=(5, 1, 0))
                    arima_result = arima_model.fit()
                    forecast = arima_result.forecast(steps=10)
                    st.line_chart(pd.DataFrame({"ARIMA Forecast": forecast}, index=pd.date_range(start=data.index[-1], periods=10, freq='D')))
                except Exception as e:
                    st.warning(f"ARIMA model error: {e}")

        if export_format:
            if "Excel" in export_format:
                excel_bytes = BytesIO()
                data.to_excel(excel_bytes, index=True)
                st.download_button("📥 Export Excel", data=excel_bytes.getvalue(), file_name=f"{ticker}_data.xlsx")
            if "JSON" in export_format:
                st.download_button("📥 Export JSON", data=data.to_json().encode('utf-8'), file_name=f"{ticker}_data.json")
            if "CSV" in export_format:
                st.download_button("📥 Export CSV", data=data.to_csv().encode('utf-8'), file_name=f"{ticker}_data.csv")

        if save_pdf:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt=f"Stock Summary Report: {ticker}", ln=True, align='C')
            for i, row in data.tail(10).iterrows():
                pdf.cell(200, 8, txt=f"{i.date()} | Close: {row[plot_column]:.2f} | RSI: {row['RSI']:.2f}", ln=True)
            pdf_bytes = BytesIO()
            pdf.output(pdf_bytes)
            st.download_button("📋 Download PDF Report", data=pdf_bytes.getvalue(), file_name=f"{ticker}_report.pdf")

if not all_comparison.empty:
    st.subheader("📊 Multi-Ticker Model Comparison")
    st.line_chart(all_comparison.dropna())
if 'download_trigger' in st.session_state and st.session_state.download_trigger:
    for ticker in tickers:
        data = fetch_data(ticker, start_date, end_date)
        if not data.empty:
            data.to_csv(f"{ticker}_data.csv")
            st.success(f"✅ {ticker} data downloaded as CSV.")
    st.session_state.download_trigger = False   