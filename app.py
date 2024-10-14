import streamlit as st # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import yfinance as yf  # type: ignore
import plotly.express as px # type: ignore

st.title('Stock Dashboard')
ticker = st.sidebar.text_input('Ticker')
start_date = st.sidebar.date_input('Start_date')
end_date = st.sidebar.date_input('End_Date')

data = yf.download(ticker,start=start_date,end=end_date)
data

fig1 = px.line(data, x=data.index,y=data['Adj Close'],title = ticker)
st.plotly_chart(fig1)
fig2 = px.scatter(data, x=data.index,y=data['Adj Close'],title = ticker)
st.plotly_chart(fig2)
pricing_data,fundamental_data,news=st.tabs(["Pricing Data","Fundamental Data","Top 10 News"]) 
with pricing_data:
    st.header('Price Movements')
    data2 = data
    data2['% Change'] = data['Adj Close']/data['Adj Close'].shift(1)-1
    data2.dropna(inplace = True)
    st.write(data2)
    annual_return = data2['% Change'].mean()*252*100
    st.write("Annual Return is:",annual_return,"%")
    stdev = np.std(data2['% Change'])*np.sqrt(252)
    st.write("Standard Deviation is:",stdev*100,"%")
    st.write("Risk Adj.Return is:",annual_return/(stdev*100))

from alpha_vantage.fundamentaldata import FundamentalData # type: ignore
with fundamental_data:
    key = "IKR4UZYWX3XP0YGF"
    fd = FundamentalData(key,output_format="pandas")
    st.subheader('Balance Sheet')
    balance_sheet = fd.get_balance_sheet_annual(ticker)[0]
    bs = balance_sheet.T[2:]
    bs.columns = list(balance_sheet.T.iloc[0])
    st.write(bs)
    st.subheader('Income Statement')
    income_statement = fd.get_income_statement_annual(ticker)[0]
    is1 = income_statement.T[2:]
    is1.columns = list(income_statement.T.iloc[0])
    st.write(is1)
    st.subheader("Cash Flow Statement")
    cash_flow = fd.get_cash_flow_annual(ticker)[0]
    cf = cash_flow.T[2:]
    cf.columns = list(cash_flow.T.iloc[0])
    st.write(cf)


from stocknews import StockNews # type: ignore
with news:
    st.header(f'News of {ticker}')
    sn = StockNews(ticker,save_news = False)
    df_news = sn.read_rss()
    for i in range (10):
        st.subheader(f'News {i+1}')
        st.write(df_news['published'][i])
        st.write(df_news['title'][i])
        st.write(df_news['summary'][i])
        title_sentiment = df_news['sentiment_title'][i]
        st.write(f'Title Sentiment {title_sentiment}')
        news_sentiment = df_news['sentiment_summary'][i]
        st.write(f'News Sentiment {news_sentiment}')

# Existing Code for Stock Dashboard ...

# Sidebar Inputs for Portfolio Tracker
st.sidebar.header('Portfolio Tracker')
portfolio_symbols = st.sidebar.text_input('Enter stock symbols (comma-separated):', 'AAPL,GOOGL,TSLA')
portfolio_quantities = st.sidebar.text_input('Enter quantities (comma-separated):', '10,5,2')

# Function to fetch portfolio data
def fetch_portfolio_data(symbols, quantities):
    portfolio_data = {}
    symbols_list = symbols.split(',')
    quantities_list = list(map(float, quantities.split(',')))

    for i, symbol in enumerate(symbols_list):
        stock_data = yf.Ticker(symbol.strip()).history(period="1d")['Close'][0]
        portfolio_data[symbol] = {
            'price': stock_data,
            'quantity': quantities_list[i],
            'total_value': stock_data * quantities_list[i]
        }
    return portfolio_data


# Existing stock fetching and charting code...

# Portfolio Tracker Section
st.subheader('Portfolio Summary')

if portfolio_symbols and portfolio_quantities:
    portfolio_data = fetch_portfolio_data(portfolio_symbols, portfolio_quantities)

    total_portfolio_value = 0
    for stock, data in portfolio_data.items():
        st.write(f"**{stock}:** Price: ${data['price']:.2f}, Quantity: {data['quantity']}, Total Value: ${data['total_value']:.2f}")
        total_portfolio_value += data['total_value']

    st.write(f"### Total Portfolio Value: ${total_portfolio_value:.2f}")
else:
    st.write("Please enter valid stock symbols and quantities.")
