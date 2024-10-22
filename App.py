import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import datetime as dt
from itertools import islice
from plotly.subplots import make_subplots
import yfinance as yf


st.set_page_config(page_title="Stocks Dashboard", page_icon="📈", layout="wide")
st.html("styles.html")
pio.templates.default = "plotly_white"


start = dt.datetime(2020, 1, 1)
end = dt.datetime.now()

# List of ticker symbols
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NFLX"]


# Streamlit layout
st.title("Stock Data Cards")

@st.cache_data
def getdata(tickers,start,end):
    stock_hist={}
    stock_info={}
    for ticker in tickers:
        stock_hist[ticker]=yf.download(ticker, start=start,end=end)
        stock_info[ticker]=yf.Ticker(ticker).info
    return stock_hist,stock_info


# Base URL for the logos
base_logo_url = "https://assets.parqet.com/logos/symbol/{}"

def getliveprice(tickers):
    live_df={}
    for ticker in tickers:
        live_df[ticker]=yf.download(ticker, interval="1m", period="1d").iloc[-1]

    return live_df

stock_hist,stock_info = getdata(tickers,start,end)
for t in stock_hist:
    st.dataframe(stock_hist[t].iloc[-1])
    #st.write(stock_hist[t].iloc[-1])
    info_card=['longName','previousClose','volume','averageVolume','sharesOutstanding','dayHigh','dayLow','marketCap','forwardPE','trailingEps','website']
    for i in info_card:

        st.write(i,stock_info[t][i])

def batched(iterable, n_cols):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n_cols < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n_cols)):
        yield batch

def plot_sparkline(data,line_color="red"):
    fillcolor="pink" if line_color=="red" else "green"
    fig_spark = go.Figure(
        data=go.Scatter(name="",
            y=data.iloc[:, 0].tolist(),
            x=data.index.tolist(),
            mode="lines",
            fill="tozeroy",
            line_color=line_color,
            fillcolor=fillcolor,
            fillgradient=dict(
                type="horizontal",
                colorscale=[(0.0, "white"), (0.5, "light"+fillcolor), (1.0, fillcolor)],
            ),
        ),
    )
    fig_spark.update_traces(hovertemplate="Price: $ %{y:.2f} %{x}")
    fig_spark.update_xaxes(visible=False, fixedrange=True)
    fig_spark.update_yaxes(visible=False, fixedrange=True)
    fig_spark.update_layout(
        showlegend=False,
        plot_bgcolor="white",
        height=50,
        margin=dict(t=10, l=0, b=0, r=0, pad=0),
    )
    return fig_spark

def display_watchlist_card(ticker, symbol_name, last_price, change,change_pct, open):
    with st.container(border=True):
        st.html(f'<span class="watchlist_card"></span>')

        tl, tr = st.columns([2, 1])
        bl, br = st.columns([1, 1])

        with tl:
            st.html(f'<span class="watchlist_symbol_name"></span>')
            st.markdown(f"{symbol_name}")

        with tr:
            st.html(f'<span class="watchlist_ticker"></span>')
            st.markdown(f"{ticker}")
            negative_gradient = float(change_pct) < 0
            st.markdown(
                f":{'red' if negative_gradient else 'green'}[{'▼' if negative_gradient else '▲'} $ {change: .2f} ({change_pct:.3f} %)]"
            )

        with bl:
            with st.container():
                st.html(f'<span class="watchlist_price_label"></span>')
                st.markdown(f"Current Value")

            with st.container():
                st.html(f'<span class="watchlist_price_value"></span>')
                st.markdown(f"$ {last_price:.2f}")

        with br:
            fig_spark = plot_sparkline(open,'red' if negative_gradient else 'green')
            st.html(f'<span class="watchlist_br"></span>')
            st.plotly_chart(
                fig_spark, config=dict(displayModeBar=False), use_container_width=True
            )



live_data=getliveprice(tickers)
def watch_cards(stock_hist,stock_info,live_data):
    for ticker in stock_hist:
        display_watchlist_card( ticker, stock_info[ticker]['longName'], 
                               live_data[ticker]['Close'].iloc[-1], 
                               float((live_data[ticker]['Close'].iloc[-1]-stock_hist[ticker]['Close'].iloc[-2])),
                               float((live_data[ticker]['Close'].iloc[-1]-stock_hist[ticker]['Close'].iloc[-2])/stock_hist[ticker]['Close'].iloc[-2]), stock_hist[ticker]['Open'])

watch_cards(stock_hist,stock_info,live_data)
