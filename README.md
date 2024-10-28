# CipherStockX

**CipherStockX** is an innovative stock analysis platform designed to decode market trends and insights. Leveraging advanced AI and machine learning models, it empowers users with real-time data, sentiment analysis, and performance scoring to enhance trading decisions. The interactive and sleek web interface, powered by Streamlit, provides a comprehensive view of market metrics, stock performance, and SEC filing analyses.

## Demo

![Demo GIF](Assets/Demo.gif)

## Key Features

### Stock Analysis and Visualization
- **Real-time Stock Data**: Real-time prices and performance trends with visual insights.
- **Sentiment Analysis**: Analysis of major stocks based on news and social media trends.
- **Performance Scoring**: Insights from SEC filings to evaluate growth potential, risk, and market positioning.

### Interactive Data Dashboard
- **Comprehensive Visualizations**: View stock data with candlestick charts, sparklines, and indicators for revenue growth, gross margins, EBITDA margins, and market trends.
- **Watchlist Functionality**: Track selected stocks with performance indicators and trendlines.

### Sentiment and SEC Filing Analysis
- **Sentiment Scores**: Integration with Google Generative AI for sentiment analysis.
- **Performance and Risk Metrics**: Scores based on metrics like public float and accounts payable, derived from SEC filings.

## Usage

- **Watchlist**: Displays summaries of selected stocks with visuals on price changes, earnings growth, revenue growth, gross margins, and other critical indicators.
- **Stock History and Analysis**: Filter historical data by periods (Week, Month, Trimester, Year) and view comprehensive charts.
- **SEC Filing Analysis**: Analyze JSON-based SEC filings for performance, growth potential, market position, and risk factors.

## Code Overview

- **`app.py`**: The main Streamlit app that manages data retrieval from Yahoo Finance, visualizations, and SEC filing analysis.
- **`llm.py`**: Manages interactions with language models for sentiment analysis.
- **`plotting.py`**: Contains utility functions for generating various stock visualizations.
- **`vectorbase.py`**: Provides vector-based search and filtering for relevant stock data.

## Dependencies

- **Python Libraries**: `pandas`, `plotly`, `streamlit`, `yfinance`, `humanize`, `google-generativeai`, `BeautifulSoup`, `sec_cik_mapper`
- **External API**: Google Generative AI for enhanced sentiment analysis.

## Future Enhancements

- **Expanded SEC Metrics**: Additional SEC filing metrics for detailed risk assessment.
- **Diverse Data Sources**: Incorporate more sources for comprehensive sentiment analysis.
- **Sector-Specific Analysis**: Enable vector-based sector-specific searches.

## License

This project is licensed under the MIT License.
