import streamlit as st
import yfinance as yf


# List of ticker symbols
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NFLX"]

# Streamlit layout
st.title("Stock Data Cards")


# Base URL for the logos
base_logo_url = "https://assets.parqet.com/logos/symbol/{}"



# Create data cards with boxes
cols = st.columns(3)
for i, tick in enumerate(tickers):
    with cols[i % 3]:  # Cycle through the columns
        with st.container():  # Create a container for the card
            stockdata = yf.Ticker(tick)
            st.markdown("---")  # Horizontal line for separation
            st.image(base_logo_url.format(tick), width=50)
            st.markdown(f"**{tick}**: {stockdata.info['longName']}")
            st.write(f"**Description**: {stockdata.info['longBusinessSummary']}")
            st.write(f"**Location**: {stockdata.info['country']}")
            st.write(f"**Current Value**: {stockdata.info['previousClose']}")
            st.markdown(f"[Visit Website]({stockdata.info['website']})")
            st.markdown("---")  # Horizontal line for separation

