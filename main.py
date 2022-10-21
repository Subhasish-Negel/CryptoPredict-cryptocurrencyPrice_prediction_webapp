import streamlit as st
import pandas_datareader as pdr
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly as pplt
import plotly.graph_objs as go

st.set_page_config(page_title='CryptoPredict', page_icon=':chart_with_upwards_trend:', layout='wide')


st.title('CryptoPredict: Crypto Currency Price History & Prediction App:')

stocks = ['BTC-USD', 'ETH-USD', 'YFI-USD', 'WBTC-USD', 'PAXG-USD', 'SOL-USD']
selected_stocks = st.selectbox("Select Your Cryptocurrency", stocks)
st.subheader('Select The Date Range for Training Dataset')

START = st.date_input('Start', value=pd.to_datetime("2017-01-01"))
TODAY = st.date_input('End(Today)', value=pd.to_datetime("today"))

st.subheader('Select Days of Prediction(1 - 180)')
n_years = st.slider("", 1, 180)
period = n_years * 1


@st.cache
def load_data(ticker):
    data = pdr.get_data_yahoo(ticker, start=START, end=TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text("Loading Data...")
data = load_data(selected_stocks)
data_load_state.text('Loading Data...Done!')




def plot_raw_date():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Adj Close Price'))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)

st.subheader('Raw Data(Last 7 Days):')


left_col, right_col = st.columns(2)
with left_col:
    plot_raw_date()
with right_col:
    st.markdown('##')
    st.markdown('##')
    st.markdown('##')
    st.write(data.tail(7))

# Forecasting

df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={'Date': 'ds', 'Close': 'y'})

model = Prophet()
model.fit(df_train)

future = model.make_future_dataframe(periods=period)
forecast = model.predict(future)


st.subheader('Raw Predicted Data')
st.table(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(8))


# Plotting predicted data


col1, col2, col3 = st.columns(3)

with col2:
    st.subheader('Predicted Graph:')
    fig1 = pplt(model, forecast)
    st.plotly_chart(fig1)

# Seasonality Check

st.subheader('Price Seasonality Check:')
fig2 = model.plot_components(forecast)
st.write(fig2)
