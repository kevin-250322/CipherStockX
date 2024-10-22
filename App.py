import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import datetime as dt
from plotly.subplots import make_subplots
import yfinance as yf


st.set_page_config(page_title="Stocks Dashboard", page_icon="📈", layout="wide")
st.html("styles.html")
pio.templates.default = "plotly_white"


start = dt.datetime(2020, 1, 1)
end = dt.datetime.now()

# List of ticker symbols
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA","NFLX"]


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
        des,mt=st.columns([100,1])
        eg,rg,gg=st.columns([1,1,1])
        em,w52,mt=st.columns([1,1,1])


        with tl:
            st.image(base_logo_url.format(ticker), width=75)
            st.html(f'<span class="watchlist_symbol_name" title={stock_info[ticker]['website']}></span>')
            st.markdown(f"{symbol_name} [🌐](%s)"%stock_info[ticker]['website'])
            st.write("check out this [link](%s)" % stock_info[ticker]['website'])
            

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
        
        with des:
            with st.container():
                st.write(stock_info[ticker]['longBusinessSummary'][:500]+"...")

        with br:
            fig_spark = plot_sparkline(open,'red' if negative_gradient else 'green')
            st.html(f'<span class="watchlist_br"></span>')
            st.plotly_chart(
                fig_spark, config=dict(displayModeBar=False), use_container_width=True
            )
        
        with eg:  
            with st.container():
                st.html(f'<span class="watchlist_price_label"></span>')
                st.markdown(f"Earnings Growth")

            with st.container():
                st.html(f'<span class="watchlist_price_value"></span>')
                st.markdown(f"{stock_info[ticker]['earningsGrowth']*100:.2f} %")
        
        with rg:
            with st.container():
                st.html(f'<span class="watchlist_price_label"></span>')
                st.markdown(f"Revenue Growth")

            with st.container():
                st.html(f'<span class="watchlist_price_value"></span>')
                st.markdown(f"{stock_info[ticker]['revenueGrowth']*100:.2f} %")
        
        with gg:
            with st.container():
                st.html(f'<span class="watchlist_price_label"></span>')
                st.markdown(f"Gross Margins")

            with st.container():
                st.html(f'<span class="watchlist_price_value"></span>')
                st.markdown(f"{stock_info[ticker]['grossMargins']*100:.2f} %")
        
        with em:
            with st.container():
                st.html(f'<span class="watchlist_price_label"></span>')
                st.markdown(f"EBITDA Margins")

            with st.container():
                st.html(f'<span class="watchlist_price_value"></span>')
                st.markdown(f"{stock_info[ticker]['ebitdaMargins']*100:.2f} %")
        
        with w52:
            with st.container():
                st.html(f'<span class="watchlist_price_label"></span>')
                st.markdown(f"General Market Trend")

            with st.container():
                st.html(f'<span class="watchlist_price_value"></span>')
                st.markdown(f"{stock_info[ticker]['recommendationKey']} ")
        
        







live_data=getliveprice(tickers)

def watch_cards(stock_hist,stock_info,live_data):
    
    for i in range(0, len(stock_hist), 2):
        col1, col2 = st.columns([1,1])  # Create 2 columns

    # Place the item in the first column
        with col1:
            display_watchlist_card( tickers[i], stock_info[tickers[i]]['longName'], 
                               live_data[tickers[i]]['Close'].iloc[-1], 
                               float((live_data[tickers[i]]['Close'].iloc[-1]-stock_hist[tickers[i]]['Close'].iloc[-2])),
                               float((live_data[tickers[i]]['Close'].iloc[-1]-stock_hist[tickers[i]]['Close'].iloc[-2])/stock_hist[tickers[i]]['Close'].iloc[-2]), stock_hist[tickers[i]]['Open'])


    # Place the next item in the second column, if it exists
        if i + 1 < len(stock_hist):
            with col2:
             display_watchlist_card( tickers[i+1], stock_info[tickers[i+1]]['longName'], 
                               live_data[tickers[i+1]]['Close'].iloc[-1], 
                               float((live_data[tickers[i+1]]['Close'].iloc[-1]-stock_hist[tickers[i+1]]['Close'].iloc[-2])),
                               float((live_data[tickers[i+1]]['Close'].iloc[-1]-stock_hist[tickers[i+1]]['Close'].iloc[-2])/stock_hist[tickers[i+1]]['Close'].iloc[-2]), stock_hist[tickers[i+1]]['Open'])

    # for ticker in stock_hist:
    

    #     display_watchlist_card( ticker, stock_info[ticker]['longName'], 
    #                            live_data[ticker]['Close'].iloc[-1], 
    #                            float((live_data[ticker]['Close'].iloc[-1]-stock_hist[ticker]['Close'].iloc[-2])),
    #                            float((live_data[ticker]['Close'].iloc[-1]-stock_hist[ticker]['Close'].iloc[-2])/stock_hist[ticker]['Close'].iloc[-2]), stock_hist[ticker]['Open'])

watch_cards(stock_hist,stock_info,live_data)
