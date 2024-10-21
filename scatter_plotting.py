import datetime as dt
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd

def get_pattern(df):
    if len(df) < 2:
        return 'Neutral'  # Not enough data to determine pattern

    # Check the price change from two days ago
    price_today = float(df['Close'].iloc[-1].iloc[0])
    price_yesterday = float(df['Close'].iloc[-2].iloc[0])

    price_today = price_today
    price_yesterday = price_yesterday
    
    price_change_percentage = ((price_today - price_yesterday) / price_yesterday) * 100
    print(price_change_percentage)
    # Define the pattern based on price change
    if price_change_percentage > 1:  # 1% increase
        return 'Bullish'
    elif price_change_percentage < -1:  # 1% decrease
        return 'Bearish'
    else:
        return 'Neutral'

def plot_stock(stock_symbol, start_date, end_date):
    print(end_date)
    # Download stock data
    df = yf.download(stock_symbol, start=start_date, end=end_date)
    print(df.head())
    # Calculate percentage change
    df['Pct_Change'] = df['Close'].pct_change() * 100

    # Create trace for stock price
    trace = go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name=stock_symbol
    )

    # Get latest price and percentage change
    latest_price = df['Close'].iloc[-1]
    latest_pct_change = df['Pct_Change'].iloc[-1]



    # Get pattern (in a real scenario, you'd get this from your data source)
    pattern = get_pattern(df)

    return trace, latest_price, latest_pct_change, pattern

def clacy(i,len):
  return 1-(.175*(i-1))


def plot_multiple_stocks(stock_list, start_date, end_date):
    # Create subplot figure with increased vertical spacing
    fig = make_subplots(rows=len(stock_list), cols=1,
                        shared_xaxes=True,)
                        #vertical_spacing=0.05)  # Increased vertical spacing

    # Add each stock to the subplot
    for i, stock in enumerate(stock_list, start=1):
        trace, latest_price, latest_pct_change, pattern = plot_stock(stock, start_date, end_date)
        fig.add_trace(trace, row=i, col=1)

        # Add ticker name annotation at the top left
        fig.add_annotation(
            text=f"{stock}",
            xref="paper", yref="paper",
            x=0.01, y=clacy(i,len(stock_list)),  # Adjusted for spacing
            xanchor="left", yanchor="bottom",
            showarrow=False,
            font=dict(size=14, color="white")  # Increased font size
        )

        # Add annotations for price and percentage change in the middle
        fig.add_annotation(
            text=f"${latest_price.iloc[0]:.2f} ({latest_pct_change:+.2f}%)",
            xref="paper", yref="paper",
            x=0.5, y=clacy(i,len(stock_list)),  # Centered horizontally
            xanchor="center", yanchor="bottom",
            showarrow=False,
            font=dict(size=14, color="white")  # Increased font size
        )

        # Add pattern annotation at the middle top right
        color = 'green' if pattern == 'Bullish' else 'red' if pattern == 'Bearish' else 'gray'
        fig.add_annotation(
            text=pattern,
            xref="paper", yref="paper",
            x=0.95, y=clacy(i,len(stock_list)),
            xanchor="right", yanchor="bottom",
            showarrow=False,
            font=dict(size=14, color=color)  # Increased font size
        )

    # Update layout
    fig.update_layout(
       height=300 * len(stock_list),
        width=1000,  # Increased width to accommodate annotations
        plot_bgcolor='black',  # Set background to black
        paper_bgcolor='black',  # Set paper background to black
        font_color='white',  # Set default font color to white
        showlegend=False
    )

    # Remove grid lines
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    # Show plot
    return fig
