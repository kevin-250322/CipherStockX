import datetime as dt
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def get_pattern(df):
    if len(df) < 2:
        return 'Neutral'  # Not enough data to determine pattern

    # Check the price change from two days ago
    price_today = df['Close'].iloc[-1]
    price_yesterday = df['Close'].iloc[-2]

    price_change_percentage = ((price_today - price_yesterday) / price_yesterday) * 100

    # Define the pattern based on price change
    if price_change_percentage > 1:  # 1% increase
        return 'Bullish'
    elif price_change_percentage < -1:  # 1% decrease
        return 'Bearish'
    else:
        return 'Neutral'

def plot_stock(stock_symbol, start_date, end_date):
    # Download stock data
    df = yf.download(stock_symbol, start=start_date, end=end_date)

    # Create trace for stock price
    trace = go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name=stock_symbol
    )

    # Get latest price and percentage change
    latest_price = df['Close'].iloc[-1]
    latest_pct_change = ((latest_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
    pattern = get_pattern(df)

    return trace, latest_price, latest_pct_change, pattern

def plot_multiple_stocks(stock_list, start_date, end_date):
    # Create subplot figure with increased vertical spacing
    fig = make_subplots(rows=len(stock_list), cols=1, shared_xaxes=True)

    # Add each stock to the subplot
    for i, stock in enumerate(stock_list, start=1):
        trace, latest_price, latest_pct_change, pattern = plot_stock(stock, start_date, end_date)
        fig.add_trace(trace, row=i, col=1)

        # Add annotations (omitted for brevity)
        # ... [Use your existing annotation code here]

    # Update layout
    fig.update_layout(
        height=300 * len(stock_list),
        width=1000,
        plot_bgcolor='black',
        paper_bgcolor='black',
        font_color='white',
        showlegend=False
    )

    # Remove grid lines
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    return fig
