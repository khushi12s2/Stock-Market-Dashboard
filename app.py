import streamlit as st
import pandas as pd 
import numpy as np 
import yfinance as yf 
import plotly.express as px

st.title('Stock Dashboard')
ticker = st.sidebar.text_input('Ticker')
start_date = st.sidebar.date_input('Start_date')
end_date = st.sidebar.date_input('End_Date')

data = yf.download(ticker,start =start_date,end = end_date)
data