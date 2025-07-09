import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from datetime import datetime, timedelta

# === PAGE CONFIG ===
st.set_page_config(page_title="📈 Stock & Portfolio Dashboard", layout="wide")
st.title("📊 Stock Dashboard & Portfolio Tracker")

# === SIDEBAR ===
st.sidebar.header("Stock Selection")
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", datetime.now())

st.sidebar.header("Portfolio Tracker")
portfolio_symbols = st.sidebar.text_input("Stock Symbols (comma-separated)", "AAPL,GOOGL,TSLA")
portfolio_quantities = st.sidebar.text_input("Quantities (comma-separated)", "10,5,2")

# === FETCH STOCK DATA ===
@st.cache_data(show_spinner=False)
def fetch_stock_data(ticker, start, end):
    try:
        return yf.download(ticker, start=start, end=end)
    except Exception as e:
        st.error(f"❌ Error fetching stock data for {ticker}: {e}")
        return pd.DataFrame()

data = fetch_stock_data(ticker, start_date, end_date)

# === PLOTTING CHARTS ===
if not data.empty:
    st.subheader(f"📈 {ticker} Price Charts")

    plot_column = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
    st.write(f"Using `{plot_column}` for charts")

    # ✅ FIXED: ensure 1D Series with .squeeze()
    fig1 = px.line(x=data.index, y=data[plot_column].squeeze(), title=f"{ticker} Price Trend")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(x=data.index, y=data[plot_column].squeeze(), title=f"{ticker} Price Scatter")
    st.plotly_chart(fig2, use_container_width=True)

    # === TABS ===
    pricing_data, fundamental_data, news = st.tabs(["💹 Pricing Data", "📊 Fundamental Data", "📰 Top 10 News"])

    with pricing_data:
        st.header("Price Movements")
        data2 = data.copy()
        data2['% Change'] = data2[plot_column].pct_change()
        data2.dropna(inplace=True)
        st.dataframe(data2)

        annual_return = data2['% Change'].mean() * 252 * 100
        st.metric("Annual Return (%)", f"{annual_return:.2f}")

        stdev = data2['% Change'].std() * np.sqrt(252) * 100
        st.metric("Volatility (Std Dev %)", f"{stdev:.2f}")

        if stdev != 0:
            risk_adj_return = annual_return / stdev
            st.metric("Risk-Adjusted Return", f"{risk_adj_return:.2f}")
        else:
            st.warning("⚠️ Standard deviation is 0. Risk-adjusted return cannot be calculated.")

    with fundamental_data:
        try:
            from alpha_vantage.fundamentaldata import FundamentalData # type: ignore
            key = "JGHJPATK0FJ8HUU2"  # Replace with your Alpha Vantage API key
            st.header(f"Fundamental Data for {ticker}")
            fd = FundamentalData(key, output_format="pandas")

            st.subheader("Balance Sheet")
            bs_raw = fd.get_balance_sheet_annual(ticker)[0]
            bs = bs_raw.T[2:]
            bs.columns = bs_raw.T.iloc[0]
            st.dataframe(bs)

            st.subheader("Income Statement")
            is_raw = fd.get_income_statement_annual(ticker)[0]
            is1 = is_raw.T[2:]
            is1.columns = is_raw.T.iloc[0]
            st.dataframe(is1)

            st.subheader("Cash Flow Statement")
            cf_raw = fd.get_cash_flow_annual(ticker)[0]
            cf = cf_raw.T[2:]
            cf.columns = cf_raw.T.iloc[0]
            st.dataframe(cf)
        except Exception as e:
            st.error(f"❌ Error fetching fundamental data: {e}")

    with news:
        try:
            from stocknews import StockNews # type: ignore
            st.header(f"Top News for {ticker}")
            sn = StockNews(ticker, save_news=False)
            df_news = sn.read_rss()
            for i in range(min(10, len(df_news))):
                st.subheader(f"📰 News {i + 1}")
                st.write(df_news['published'][i])
                st.write(df_news['title'][i])
                st.write(df_news['summary'][i])
                st.write(f"📌 Title Sentiment: {df_news['sentiment_title'][i]}")
                st.write(f"🧠 News Sentiment: {df_news['sentiment_summary'][i]}")
        except Exception as e:
            st.error(f"❌ Error fetching news: {e}")
else:
    st.warning("⚠️ Please enter a valid ticker or date range to fetch stock data.")

# === PORTFOLIO TRACKER ===
st.header("📦 Portfolio Summary")

@st.cache_data(show_spinner=False)
def fetch_portfolio_prices(symbols_list):
    try:
        df = yf.download(tickers=" ".join(symbols_list), period="1d")['Close']
        return df.iloc[0] if isinstance(df, pd.DataFrame) else df
    except Exception as e:
        st.error(f"❌ Error fetching portfolio prices: {e}")
        return {}

def get_portfolio_summary(symbols, quantities):
    symbols_list = [s.strip().upper() for s in symbols.split(',')]
    quantities_list = list(map(float, quantities.split(',')))
    prices = fetch_portfolio_prices(symbols_list)

    summary = []
    total_value = 0
    for i, symbol in enumerate(symbols_list):
        price = prices[symbol] if symbol in prices else 0
        value = price * quantities_list[i]
        total_value += value
        summary.append({
            'Symbol': symbol,
            'Price': round(price, 2),
            'Quantity': quantities_list[i],
            'Total Value': round(value, 2)
        })
    return summary, total_value

if portfolio_symbols and portfolio_quantities:
    try:
        summary, total = get_portfolio_summary(portfolio_symbols, portfolio_quantities)
        st.dataframe(pd.DataFrame(summary))
        st.success(f"✅ Total Portfolio Value: ${total:,.2f}")
    except Exception as e:
        st.error(f"❌ Portfolio error: {e}")
else:
    st.warning("⚠️ Please enter valid stock symbols and quantities.")
