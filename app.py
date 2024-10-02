import streamlit as st
import pandas as pd 
import numpy as np 
import yfinance as yf 
import plotly.express as px

st.title('Stock Dashboard')
Ticker = st.sidebar.text_input('Ticker')
Start_date = st.sidebar.date_input('Start_date')
End_Date = st.sidebar.date_input('End_Date')