import streamlit as st

# Sample stock data
stocks = [
    {
        "icon": "https://example.com/icon1.png",
        "ticker": "AAPL",
        "name": "Apple Inc.",
        "description": "Technology company",
        "location": "Cupertino, CA",
        "current_value": "$145.09",
        "website": "https://www.apple.com"
    },
    {
        "icon": "https://example.com/icon2.png",
        "ticker": "MSFT",
        "name": "Microsoft Corp.",
        "description": "Software company",
        "location": "Redmond, WA",
        "current_value": "$299.35",
        "website": "https://www.microsoft.com"
    },
    {
        "icon": "https://example.com/icon3.png",
        "ticker": "GOOGL",
        "name": "Alphabet Inc.",
        "description": "Search engine company",
        "location": "Mountain View, CA",
        "current_value": "$2,748.45",
        "website": "https://www.google.com"
    },
    {
        "icon": "https://example.com/icon4.png",
        "ticker": "AMZN",
        "name": "Amazon.com Inc.",
        "description": "E-commerce and cloud computing",
        "location": "Seattle, WA",
        "current_value": "$3,325.00",
        "website": "https://www.amazon.com"
    },
    {
        "icon": "https://example.com/icon5.png",
        "ticker": "FB",
        "name": "Meta Platforms Inc.",
        "description": "Social media company",
        "location": "Menlo Park, CA",
        "current_value": "$353.05",
        "website": "https://www.facebook.com"
    },
    {
        "icon": "https://example.com/icon6.png",
        "ticker": "TSLA",
        "name": "Tesla Inc.",
        "description": "Electric vehicle manufacturer",
        "location": "Palo Alto, CA",
        "current_value": "$1,025.00",
        "website": "https://www.tesla.com"
    }
]

# Streamlit layout
st.title("Stock Data Cards")

# Create data cards with boxes
cols = st.columns(3)
for i, stock in enumerate(stocks):
    with cols[i % 3]:  # Cycle through the columns
        # Create a box for each card
        with st.container():
            st.markdown(
                f"""
                <div style="border: 1px solid #ccc; border-radius: 8px; padding: 16px; margin: 10px; background-color: #f9f9f9;">
                    <img src="{stock['icon']}" width="50" alt="{stock['name']} logo">
                    <h5>{stock['ticker']}: {stock['name']}</h5>
                    <p><strong>Description:</strong> {stock['description']}</p>
                    <p><strong>Location:</strong> {stock['location']}</p>
                    <p><strong>Current Value:</strong> {stock['current_value']}</p>
                    <p><a href="{stock['website']}">Visit Website</a></p>
                </div>
                """,
                unsafe_allow_html=True
            )


