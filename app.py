import streamlit as st
import pandas as pd 
import numpy as np 
import yfinance as yf 
import plotly.express as px

st.title('Stock Dashboard')
ticker = st.sidebar.text_input('Ticker')
start_date = st.sidebar.date_input('Start_date')
end_date = st.sidebar.date_input('End_Date')

data = yf.download(ticker,start=start_date,end=end_date)
data

fig = px.line(data, x=data.index,y=data['Adj Close'],title = ticker)
st.plotly_chart(fig)

pricing_data,fundamental_data,news=st.tabs(["Pricing Data","Fundamental Data","Top !0 News"]) 

with pricing_data:
    st.header('Price Movements')
    data2 = data
    data2['% Change'] = data['Adj Close']/data['Adj Close'].shift(1)-1
    data2.dropnone(inplace = True)
    st.write(data2)
    annual_return = data2['% Change'].mean()*252*100
    st.write("Annual Return is:",annual_return,"%")

with fundamental_data:
    st.write("Fundamental")

with news:
    st.write("News")