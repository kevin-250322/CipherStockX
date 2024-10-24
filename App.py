import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import datetime as dt
from plotly.subplots import make_subplots
import yfinance as yf
import humanize as hm
import os
from typing import List
from pydantic import BaseModel
import google.generativeai as genai
import json


class StockInfo(BaseModel):
    ticker: str
    earnings_growth: str
    revenue_growth: str
    ebitda_margin: str
    gross_margin: str
    fifty_two_week_change: str
    summary: str

class StockComparisonSummary(BaseModel):
    stock_comparison: List[StockInfo]
    overall_summary: str

st.set_page_config(page_title="Stocks Dashboard", page_icon="📈", layout="wide")
st.html("styles.html")
pio.templates.default = "plotly_white"


genai.configure(api_key=os.getenv("API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash-8b")

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

@st.cache_data
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

def plot_candlestick(history_df):
    f_candle = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.1,horizontal_spacing=0.1
    )
  
    f_candle.add_trace(
        go.Candlestick(
            x=history_df.index.tolist(),
            open=history_df["Open"].iloc[:, 0].tolist(),
            high=history_df["High"].iloc[:, 0].tolist(),
            low=history_df["Low"].iloc[:, 0].tolist(),
            close=history_df["Close"].iloc[:, 0].tolist(),
            name="Dollars",
        ),
        row=1,
        col=1,
    )
    f_candle.add_trace(
        go.Bar(x=history_df.index.tolist(), y=history_df["Volume"].iloc[:, 0].tolist(), name="Volume Traded"),
        row=2,
        col=1,
    )
    f_candle.update_layout(
        title=" Stock Price Trends",
        showlegend=True,
        xaxis_rangeslider_visible=False,
        # xaxis=dict(title="date"),
        yaxis1=dict(title="OHLC"),
        yaxis2=dict(title="Volume"),
        hovermode="x",
       
    )
    f_candle.update_layout(
        title_font_family="Open Sans",
        title_font_color="#174C4F",
        title_font_size=32,
        font_size=16,
        margin=dict(l=80, r=80, t=100, b=80, pad=0),
        height=500,
        
    )
    f_candle.update_xaxes(title_text="Date", row=2, col=1)
    f_candle.update_traces(selector=dict(name="Dollars"), showlegend=True)
    return f_candle

def plotallscatter():
    normalized_data = pd.DataFrame()
    for ticker, prices in stock_hist.items():
    # Calculate the percentage change
        pct_chnage_data=pd.DataFrame(prices['Adj Close'])
        normalized_data[ticker] = (pct_chnage_data.pct_change().apply(lambda x: (1+x).cumprod())-1)*100

# Plotting with Plotly
    fig = make_subplots(rows=1, cols=1)

# Add traces for each stock
    for ticker in normalized_data.columns:
        fig.add_trace(
        go.Scatter(x=normalized_data.index, y=normalized_data[ticker], mode='lines', name=ticker)
    )

# Customize the layout
    fig.update_layout(
    title="Normalized Stock Price History (% Change)",
    xaxis_title="Date",
    yaxis_title="Normalized Price Change (%)",
    legend_title="Stock Ticker",
    template="ggplot2"
    )

# Show the plot
    st.plotly_chart(fig)

st.title("Stock Price Comparison")
st.write()

plotallscatter()


st.write()
st.title("Stock Comparison Summary")

stock_info_json = json.dumps(stock_info)  # Convert your stock_info dictionary to a JSON string

prompt = f"""
Please provide a detailed stock comparison summary in JSON format. Compare each stock based on these metrics:
1. Earnings growth
2. Revenue growth
3. EBITDA margins
4. Gross margins
5. 52-week change

For each stock, create an individual summary in JSON, followed by a brief paragraph explaining the comparison results in a human-readable format. Ensure the output is well-structured in JSON.

Input JSON for stocks:
{stock_info_json}

Output format:
{{
    "stock_comparison": [
        {{
            "ticker": "AMT",
            "earnings_growth": "88.20%",
            "revenue_growth": "4.60%",
            "ebitda_margin": "71.45%",
            "gross_margin": "52.71%",
            "52_week_change": "29.41%",
            "summary": "Stock AMT has the highest earnings growth (88.20%) and gross margins (71.45%), along with a strong
            52-week change (29.41%), but its revenue growth (4.60%) is lower compared to EQIX and DLR."
        }},
        ...
    ],
    "overall_summary": "In Summary: This comparison highlights a variety of performance metrics among the stocks. AMT stands out for its significant earnings growth and high margins, while EQIX offers a balance between growth and decent margins. DLR provides strong margins albeit with declining growth, and HPE struggles with both low growth and margins. NTTYY exhibits the lowest performance across multiple metrics. Investors may prioritize different features based on their risk appetite and investment objectives."
}}

later on i shld be able to use like this json like this to print 
'
Stock Comparison Summary
• Stock AMT has the highest earnings growth (88.20%) and gross margins (71.45%), along with a strong
52-week change (29.41%), but its revenue growth (4.60%) is lower compared to EQIX and DLR.
• Stock EQIX has impressive earnings growth (43.00%) and consistent revenue growth (6.90%), with solid EBITDA margins (37.71%), but its gross margins are lower than AMT and DLR.
• Stock DLR has the highest gross margins (52.71%) and EBITDA margins (43.63%), but it faces A significant declines in both earnings growth (-41.10%) and revenue growth (-4.10%), despite a reasonably high 52-week change (15.24%).
• Stock HPE displays declining earnings growth (-25.20%) and modest revenue growth (3.30%), with the lowest gross margins (34.97%) and EBITDA margins (17.28%), alongside a moderate 52-week change (7.38%).
• Stock NTTYY shows declining earnings growth (-26.00%) and relatively lower gross margins (29.14%)
compared to its peers, with a negative 52-week change (-6.15%).
In Summary: This comparison highlights a variety of performance metrics among the stocks. AMT stands out for its significant earnings growth and high margins, while EQIX offers a balance between growth and decent margins. DLR provides strong margins albeit with declining growth, and HPE struggles with both low growth and margins. NTTYY exhibits the lowest performance across multiple metrics. Investors may prioritize different features based on their risk appetite and investment objectives.'
MAKE SURE TO USE % METRICS IN SUMMARY OF EACH TICKER and ONLY OUTPUT JSON AND NO PREAMBEL OR POST COMMNETS
"""

# Generating structured output using Gemini's JSON mode
response = model.generate_content(prompt)
def find_braces_positions(s: str):
    first_brace = s.find('{')  # Find the position of the first '{'
    last_brace = s.rfind('}')  # Find the position of the last '}'
    return first_brace, last_brace
Start,End=find_braces_positions(response.text)

rt=json.loads(response.text[Start:End+1])

for item in rt['stock_comparison']:
    st.write("- "+item['summary'])
st.write(rt['overall_summary'])


st.divider()

st.title("Candelstick Dashboard")


@st.fragment
def display_symbol_history(stock_hist):
    with st.container():
        selected_ticker, selected_period = filter_symbol_widget()
        mapping_period = {"Week": 7, "Month": 31, "Trimester": 90, "Year": 365}
        history_df=stock_hist[selected_ticker].tail(mapping_period[selected_period])

    
        left_chart, right_indicator = st.columns([1.5, 1])

        f_candle = plot_candlestick(history_df)
    
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
                    



ticker_picked=display_symbol_history(stock_hist)




    

