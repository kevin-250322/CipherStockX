import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import datetime as dt
from plotly.subplots import make_subplots
import yfinance as yf
import humanize as hm
import os

import google.generativeai as genai
import json
import requests
from bs4 import BeautifulSoup
from sec_cik_mapper import StockMapper
import os
import json
import subprocess


from Llm import llm
from Plotting import plotting
from Vectorbase import vectorbase
llms=llm()
plottings=plotting()
st.set_page_config(page_title="Stocks Dashboard", page_icon="📈", layout="wide")
st.html("styles.html")
pio.templates.default = "plotly_white"
@st.cache_data
def vector():
    return vectorbase()
vectorbases=vector()

def display_watchlist_card(ticker, symbol_name, last_price, change,change_pct, open):
    with st.container(border=True):
        st.html(f'<span class="watchlist_card"></span>')

        tl, tr = st.columns([2, 1])
        bl, br = st.columns([1, 1])
        des,mt=st.columns([100,1])
        eg,rg,gg=st.columns([1,1,1])
        em,w52,mt=st.columns([1,1,1])


        with tl:
            #st.image(base_logo_url.format(ticker), width=75)
            st.image(base_logo_url.format(ticker) if requests.get(base_logo_url.format(ticker)).status_code == 200 else default_logo_url, width=75)

            st.html(f'<span class="watchlist_symbol_name" title={stock_info[ticker]['website']}></span>')
            st.markdown(f"{symbol_name} [🌐](%s)"%stock_info[ticker]['website'])
            

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
                st.write(stock_info[ticker]['longBusinessSummary'][:350]+"...")

        with br:
            fig_spark = plottings.plot_sparkline(open,'red' if negative_gradient else 'green')
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

@st.fragment
def display_symbol_history(stock_hist):
    with st.container():
        selected_ticker, selected_period = filter_symbol_widget()
        mapping_period = {"Week": 7, "Month": 31, "Trimester": 90, "Year": 365}
        history_df=stock_hist[selected_ticker].tail(mapping_period[selected_period])

    
        left_chart, right_indicator = st.columns([1.5, 1])

        f_candle = plottings.plot_candlestick(history_df)
    
        with left_chart:
        
            st.html('<span class="column_plotly"></span>')
            st.plotly_chart(f_candle, use_container_width=True)

        with right_indicator:
            st.html('<span class="column_indicator"></span>')
            st.subheader("Period Metrics")
            l, r = st.columns(2)

            with l:
                st.html('<span class="low_indicator"></span>')
                st.metric("Lowest Volume Day", f'{history_df["Volume"].min().iloc[0]:,}')
                st.metric("Lowest Close Price", f'$ {history_df["Close"].min().iloc[0]:,.2f}')
            with r:
                st.html('<span class="high_indicator"></span>')
                st.metric("Highest Volume Day", f'{history_df["Volume"].max().iloc[0]:,}')
                st.metric("Highest Close Price", f'$ {history_df["Close"].max().iloc[0]:,.2f}')

            with st.container():
                st.html('<span class="bottom_indicator"></span>')
           
                st.metric("Average Daily Volume", f'{history_df["Volume"].mean().iloc[0]:,.2f}')
                st.metric("Current Market Cap"," $ "+hm.intword(stock_info[selected_ticker]["marketCap"]))
    return selected_ticker

@st.cache_data
def getdata(tickers,start,end):
    stock_hist={}
    stock_info={}
    for ticker in tickers:
        stock_hist[ticker]=yf.download(ticker, start=start,end=end)
        stock_info[ticker]=yf.Ticker(ticker).info
    return stock_hist,stock_info

@st.cache_data
def getliveprice(tickers):
    live_df={}
    for ticker in tickers:
        live_df[ticker]=yf.download(ticker, interval="1m", period="1d").iloc[-1]
    return live_df

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

def filter_symbol_widget():
    with st.container():
        left_widget, right_widget, _ = st.columns([1, 1, 3])

    selected_ticker = left_widget.selectbox(
        "📰 Currently Showing", list(tickers)
    )
    selected_period = right_widget.selectbox(
        "⌚ Period", ("Week", "Month", "Trimester", "Year"), 2
    )
    

    return selected_ticker, selected_period

def analyze_sec_filing(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    facts = data['facts']

    # Performance Score
    performance_score = calculate_performance_score(facts)

    # Risk Factor Score
    risk_score = calculate_risk_score(facts)

    # Growth Potential Score
    growth_score = calculate_growth_score(facts)

    # Market Position Score
    market_score = calculate_market_score(facts)

    return {
        "Market Position Score": market_score,
        "Growth Potential Score": growth_score,
        "Performance Score": performance_score,
        "Risk Factor Score": risk_score
    }

def calculate_performance_score(facts):
    # Use EntityCommonStockSharesOutstanding as a proxy for performance
    shares_outstanding = facts['dei']['EntityCommonStockSharesOutstanding']['units']['shares']
    latest_shares = shares_outstanding[-1]['val']
    previous_shares = shares_outstanding[-2]['val']

    performance_change = (latest_shares - previous_shares) / previous_shares
    performance_score = min(max(performance_change * 100 + 50, 0), 100)
    return round(performance_score, 2)

def calculate_risk_score(facts):
    # Use AccountsPayableCurrent as a proxy for risk
    if 'AccountsPayableCurrent' in facts.get('us-gaap', {}):
        accounts_payable = facts['us-gaap']['AccountsPayableCurrent']['units']['USD']
        latest_payable = accounts_payable[-1]['val']
        previous_payable = accounts_payable[-2]['val']

        risk_change = (latest_payable - previous_payable) / previous_payable
        risk_score = 100 - min(max(risk_change * 100 + 50, 0), 100)
    else:
        risk_score = 50  # Default score if data is not available

    return round(risk_score, 2)

def calculate_growth_score(facts):
    # Use EntityPublicFloat as a proxy for growth potential
    public_float = facts['dei']['EntityPublicFloat']['units']['USD']
    latest_float = public_float[-1]['val']
    previous_float = public_float[-2]['val']

    growth_rate = (latest_float - previous_float) / previous_float
    growth_score = min(max(growth_rate * 50 + 50, 0), 100)
    return round(growth_score, 2)

def calculate_market_score(facts):
    # Example: Market Value relative to industry benchmark
    try:
        public_float = facts['dei']['EntityPublicFloat']['units']['USD'][-1]['val']
        industry_benchmark = 1000000000000  # Replace with actual industry data
        market_position_score = (public_float / industry_benchmark) * 100
        return min(max(market_position_score, 0), 100)  # Limit score between 0 and 100
    except KeyError:
        return 0
    
def download_and_analyze(tickers):
    sec_score={}
    stock_mapper = StockMapper()

    # Loop through each ticker in the list
    for ticker in tickers:
        try:
            # Get the CIK number
            cik = stock_mapper.ticker_to_cik.get(ticker)
            if cik is None:
                print(f"CIK not found for {ticker}. Skipping...")
                continue

            # Format the CIK as a string with leading zeros to match the SEC API format
            cik_str = f"CIK{cik.zfill(10)}"  # Ensure CIK is 10 digits long

            # Build the SEC API URL
            sec_url = f"https://data.sec.gov/api/xbrl/companyfacts/{cik_str}.json"

            # Define the local file path for saving the JSON
            file_path = f"{cik_str}.json"  # Save in the current working directory

            # Use requests to download the JSON file from the SEC API
            headers = {
                'User-Agent': 'for research AdminContact@example.com'
            }
            response = requests.get(sec_url, headers=headers)

            # Check if the request was successful
            if response.status_code == 200:
                with open(file_path, 'w') as json_file:
                    json.dump(response.json(), json_file)
                # Analyze the downloaded file
                results = analyze_sec_filing(file_path)
                results["path"]=file_path
                results['url']=sec_url
                sec_score[ticker]=results
                print(f"Results for {ticker}: {results}")
            else:
                print(f"Failed to download data for {ticker}: {response.status_code}")

        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
        
    return sec_score    





start = dt.datetime(2020, 1, 1)
end = dt.datetime.now()
st.title("Automated Stock Analysis")
keyword=st.text_input(value="Automotive",label="Enter a description of the kinds of stocks you are looking for:")

# Base URL for the logos
base_logo_url = "https://assets.parqet.com/logos/symbol/{}"

default_logo_url=""

#if st.button("find stocks"):
if st.button("finds stocks"):
    serch_query=llms.eloberateprompt(keyword)
    ticks=vectorbases.find_similar_stocks(serch_query)
    tickers=ticks['Symbol'].tolist()
    #st.write(tickers)
    

#if True:
# # List of ticker symbols
    #tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA","NFLX"]

# tickers=find_similar_stocks(keyword)['Symbol'].tolist()
# print(tickers)

# # Streamlit layout
    st.title("Stock Data Cards")

    

    stock_hist,stock_info = getdata(tickers,start,end)
    live_data=getliveprice(tickers)
    
    watch_cards(stock_hist,stock_info,live_data)

# 

    st.title("Stock Price Comparison")
    st.write()

    st.plotly_chart(plottings.plotallscatter(stock_hist))

    st.write()
    st.title("Stock Comparison Summary")
    rt=llms.basiccomapre(json.dumps(stock_info))




    for item in rt['stock_comparison']:
        st.write("- "+item['summary'])
    st.write(rt['overall_summary'])


    st.divider()

    st.title("Candelstick Dashboard")
    ticker_picked=display_symbol_history(stock_hist)


    st.divider()
    st.title("Sentiment Analysis")

    sentiment_analysis=llms.sentiment_data(tickers)
    sentiment_analysis_dict=llms.sentiment_analysis_tick(sentiment_analysis)

    bardata={}
    for ticker, info in sentiment_analysis_dict.items():
        bardata[ticker] = info["sentiment_score"]
    
    df = pd.DataFrame(list(bardata.items()), columns=['Company', 'Sentiment Score'])

    st.plotly_chart(plottings.sentibar(df))

    st.write("Sentiment Analysis Summary")
    for tick in sentiment_analysis_dict:
        st.write("- "+sentiment_analysis_dict[tick]['analysis'])

    st.divider()
    st.title('Sec Filling Analysis')



    # Run the download and analysis for all tickers
    sec_status=download_and_analyze(tickers)
    st.write()
    for ticker in tickers:
        left,right=st.columns([1,1])
        with left:
            st.write(ticker)

        if ticker not in sec_status:
            with left:
                st.write("failed parsing sec filling imcomplete recent filling")
            
        else:
            with right: 
                st.markdown(f" [View Sec Filling ](%s)"%sec_status[ticker]['url'])
            with left:
                name,no=st.columns([9,1])
                with name:
                    st.progress( int(sec_status[ticker]['Market Position Score']), text= 'Market Position Score')
                with no:
                    st.write(str(int(sec_status[ticker]['Market Position Score'])))
                with name:
                    st.progress( int(sec_status[ticker]['Growth Potential Score']), text='Growth Potential Score')
                with no:
                    st.write(str(int(sec_status[ticker]['Growth Potential Score'])))

            with right:
                name,no=st.columns([9,1])
                with name:
                    st.progress(int( sec_status[ticker]['Performance Score']), 'Performance Score')
                
                with no:
                    st.write(str(int(sec_status[ticker]['Performance Score'])))
                with name:
                    st.progress( int(sec_status[ticker]['Risk Factor Score']), 'Risk Factor Score')
                with no:
                    st.write(str(int(sec_status[ticker]['Risk Factor Score'])))
        st.divider()
        
