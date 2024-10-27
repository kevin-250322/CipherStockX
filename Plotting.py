import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import datetime as dt
from plotly.subplots import make_subplots
import yfinance as yf
import humanize as hm
import os



class plotting:

    def plot_sparkline(self,data,line_color="red"):
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


    def plot_candlestick(self,history_df):
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

    def plotallscatter(self,stock_hist):
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
        return fig
    
    def sentibar(self,df):
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
        return fig





