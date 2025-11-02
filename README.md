# Stock Intelligence Dashboard

A comprehensive web-based stock analysis dashboard that provides real-time financial data, market sentiment, and industry comparisons.

## Features

✅ **Real-time Stock Data** - Current prices, market cap, P/E ratios, and more from Yahoo Finance
✅ **Financial Metrics** - Revenue, gross profit, operating income, net income, EBITDA
✅ **Growth Analysis** - Revenue growth tracking and year-over-year comparisons
✅ **Industry Comparison** - Compare P/E ratios and other metrics with industry averages
✅ **Price History** - Interactive charts showing historical price movements
✅ **Significant Events** - Automatic detection of major price movements (>5%)
✅ **Latest News** - Real-time news from Yahoo Finance
✅ **Market Sentiment** - Reddit discussions from investing subreddits (optional)
✅ **Analyst Ratings** - Analyst recommendations and price targets
✅ **Beautiful UI** - Modern, responsive interface with gradient design

## Installation

1. Make sure you have Python 3.12+ installed

2. Install dependencies:
```bash
pip install flask yfinance pandas plotly requests praw beautifulsoup4 python-dotenv
```

Or if using the virtual environment:
```bash
source .venv/bin/activate  # On macOS/Linux
pip install flask yfinance pandas plotly requests praw beautifulsoup4 python-dotenv
```

## Usage

1. Start the Flask server:
```bash
python main.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Enter a stock ticker (e.g., AAPL, TSLA, MSFT, GOOGL) and click "Analyze"

## Optional: Reddit Integration

To enable Reddit sentiment analysis:

1. Create a Reddit app at https://www.reddit.com/prefs/apps
2. Choose "script" as the app type
3. Copy `.env.example` to `.env` and add your credentials:
```bash
cp .env.example .env
```
4. Edit `.env` with your Reddit API credentials

## Data Sources

- **Yahoo Finance** - Primary source for stock prices, financials, and news
- **Reddit** - Community discussions and sentiment (optional)
- All data is fetched in real-time via the `yfinance` library

## Metrics Explained

- **Revenue Growth** - Year-over-year change in total revenue
- **Gross Profit** - Revenue minus cost of goods sold
- **Operating Income** - Profit from business operations
- **Net Income** - Total profit after all expenses
- **EBITDA** - Earnings before interest, taxes, depreciation, and amortization
- **P/E Ratio** - Price-to-earnings ratio (stock price / earnings per share)
- **Industry Comparison** - How the stock's P/E compares to industry average

## Tech Stack

- **Backend**: Flask (Python)
- **Data**: yfinance, pandas
- **Visualization**: Plotly.js
- **Frontend**: HTML, CSS, JavaScript
- **APIs**: Reddit API (PRAW)

## Screenshots

The dashboard displays:
- Company overview with key metrics
- Financial statements and profitability margins
- Industry comparisons and P/E analysis
- Interactive price charts
- Significant price movements timeline
- Latest news articles
- Reddit community discussions

## Notes

- Some stocks may have limited financial data depending on reporting
- Reddit integration requires API credentials (see setup above)
- Data is fetched in real-time, so analysis may take a few seconds
- Historical data typically goes back 1-5 years depending on the stock

## License

MIT License - feel free to use and modify!
