import streamlit as st
import pandas as pd 
import numpy as np 
import yfinance as yf 
import plotly.express as px

st.title('Stock Dashboard')
st.sidebar.text_input('Ticker')