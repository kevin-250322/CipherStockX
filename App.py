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
import requests
from bs4 import BeautifulSoup
from sec_cik_mapper import StockMapper
import os
import json
import subprocess
import faiss
import numpy as np

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

# Load the FAISS index
index = faiss.read_index("Assets/faiss_index.bin",label="enter sector")
genai.configure(api_key=os.getenv("API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash-8b")

start = dt.datetime(2020, 1, 1)
end = dt.datetime.now()

keyword=st.text_input(value="Automobile")

data = pd.read_csv("/content/nasdaq_with_summaries.csv")
data['Summary'] = data['Summary'].fillna(" ")
data['Industry'] = data['Industry'].fillna(" ")
data['Sector'] = data['Sector'].fillna(" ")

# Select relevant columns for embedding
data['Description'] = data['Summary'] + " " + data['Industry'] + " " + data['Sector']


# Function to perform search based on the FAISS index
def find_similar_stocks(query, top_k=6):
    # Generate an embedding for the query text
    query_embedding = genai.embed_content(model="models/text-embedding-004", content=[query])['embedding'][0]
    query_vector = np.array(query_embedding).astype("float32").reshape(1, -1)

    # Search the FAISS index with the query embedding
    distances, indices = index.search(query_vector, top_k)

    # Retrieve and display the top matching results, sorted by Market Cap
    results = data.iloc[indices[0]].assign(distance=distances[0]).sort_values(by='Market Cap', ascending=False)  # Change to ascending=True for ascending sort

    return results[['Symbol', 'Name', 'Industry', 'Sector', 'Summary', 'distance', 'Market Cap']]


# List of ticker symbols
#tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA","NFLX"]

tickers=find_similar_stocks(keyword)['Symbol'].tolist()
print(tickers)
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


st.divider()
st.title("Sentiment Analysis")

def UrlTextScrape(url):

  try:
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    page_text = soup.get_text(separator=' ', strip=True)
    return page_text

  except requests.RequestException as e:
    return

sentiments = {}


def sentiment_data():

    for tick in tickers:
        stock = yf.Ticker(tick)
        sentiments[tick] = []
        for i in range(0,len(stock.news)):
            if '/m/' not in stock.news[i]['link']:
                text = UrlTextScrape(stock.news[i]['link'])
                sentiments[tick].append(
            {
                'date': stock.news[i]['providerPublishTime'],
                'title': stock.news[i]['title'],
                'text': text,
                'liink': stock.news[i]['link'],
                'relatedTickers': stock.news[i]['relatedTickers']

            }
        )
sentiment_data()
sentiment_analysis = json.dumps(sentiments)

prompt2=f"""
perform sentiment analysis for each ticker and give analysis(write in the tone of an analysit and mentioning number and 
comprasion data if possible) and sentiment score(0-100) output in json format here is the input json {sentiment_analysis}

Example output format:
{{
  "AMT": {{
    "analysis": str,
    "sentiment_score": 95
  }},
  ........
}}
"""
response2 = model.generate_content(prompt2)
Start1,End1=find_braces_positions(response2.text)

sentiment_analysis_dict=json.loads(response2.text[Start1:End1+1])

bardata={}
for ticker, info in sentiment_analysis_dict.items():
    bardata[ticker] = info["sentiment_score"]

df = pd.DataFrame(list(bardata.items()), columns=['Company', 'Sentiment Score'])

# Create a bar chart using Plotly
fig = go.Figure(data=[
    go.Bar(x=df['Company'], y=df['Sentiment Score'], marker_color='skyblue')
])

# Update layout
fig.update_layout(
    title='Average Sentiment Score by stock',
    xaxis_title='Company Ticker',
    yaxis_title='Average Sentiment Score',
    template='plotly_white'
)
st.plotly_chart(fig)
st.write("Sentiment Analysis Summary")
for tick in sentiment_analysis_dict:
    st.write("- "+sentiment_analysis_dict[tick]['analysis'])

st.divider()

st.title('Sec Filling Analysis')

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
        
