import streamlit as st
import yfinance as yf
import praw
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import requests
from PIL import Image
import io
import time
import math
import random

# App configuration
st.set_page_config(
    page_title="StockSense Pro - Market Analysis Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    :root {
        --primary-color: #1E88E5;
        --primary-light: rgba(30, 136, 229, 0.1);
        --primary-medium: rgba(30, 136, 229, 0.3);
        --accent-color: #4CAF50;
        --accent-light: rgba(76, 175, 80, 0.1);
        --accent-medium: rgba(76, 175, 80, 0.3);
        --negative-color: #F44336;
        --text-color: #E0E0E0;
        --text-light: #BBBBBB;
        --background-dark: #121212;
        --background-medium: #1E1E1E;
        --background-light: #2A2A2A;
        --card-bg: #1E1E1E;
        --border-color: #333333;
    }
    
    /* Main background */
    .main .block-container {
        background-color: var(--background-dark);
        padding: 2rem;
        border-radius: 10px;
    }
    
    /* Global text color */
    body {
        color: var(--text-color);
        background-color: var(--background-dark);
    }
    
    /* Headers */
    .main-header {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: var(--primary-color) !important;
        margin-bottom: 1rem !important;
        text-shadow: 0 0 10px rgba(30, 136, 229, 0.3);
    }
    .sub-header {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        color: var(--text-color) !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
    }
    
    /* Cards */
    .card {
        background: linear-gradient(145deg, var(--card-bg), #252525);
        border-radius: 10px;
        padding: 20px;
        border-left: 4px solid var(--primary-color);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }
    .accent-card {
        background: linear-gradient(145deg, var(--card-bg), #252525);
        border-radius: 10px;
        padding: 20px;
        border-left: 4px solid var(--accent-color);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }
    
    /* Metrics */
    .metric-value {
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    .metric-label {
        font-size: 1rem !important;
        color: var(--text-light) !important;
    }
    .trend-up {
        color: var(--accent-color) !important;
    }
    .trend-down {
        color: var(--negative-color) !important;
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background-color: var(--primary-color);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, var(--background-medium), var(--background-light));
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(90deg, var(--primary-color), #1565C0);
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #1565C0, var(--primary-color));
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Accent button */
    .accent-button>button {
        background: linear-gradient(90deg, var(--accent-color), #388E3C);
        color: white;
    }
    .accent-button>button:hover {
        background: linear-gradient(90deg, #388E3C, var(--accent-color));
    }
    
    /* Tables */
    .dataframe {
        border: 1px solid var(--border-color);
        border-radius: 5px;
        background-color: var(--background-medium);
    }
    .dataframe th {
        background-color: var(--background-light);
        color: var(--text-color);
        font-weight: 600;
    }
    .dataframe td {
        background-color: var(--background-medium);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: var(--background-medium);
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--background-light);
        border-bottom: 2px solid var(--primary-color);
    }
    
    /* Inputs */
    .stTextInput>div>div>input {
        background-color: var(--background-light);
        color: var(--text-color);
        border: 1px solid var(--border-color);
    }
    .stSelectbox>div>div>div {
        background-color: var(--background-light);
        color: var(--text-color);
    }
    
    /* Plotly charts */
    .js-plotly-plot .plotly {
        background-color: var(--background-medium) !important;
    }
    .js-plotly-plot .bg {
        fill: var(--background-medium) !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for keeping track of watchlist
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []

# Download NLTK Data
@st.cache_resource
def download_nltk():
    nltk.download("vader_lexicon")

download_nltk()

# Reddit API setup
def setup_reddit():
    """Setup Reddit API connection"""
    try:
        reddit = praw.Reddit(
            client_id="EO5pAkkNYea4g6mYu4LImw",
        client_secret="3gUKK1NyUEFht4CHGtpGEZM-HU88vg",
        user_agent="StockSentimentApp",
        )
        return reddit
    except Exception as e:
        st.error(f"Error setting up Reddit API: {e}")
        return None

reddit = setup_reddit()

# NewsAPI setup
def setup_newsapi():
    """Setup NewsAPI connection"""
    # Replace with your actual NewsAPI key
    return "7e93999fa88c408eadcde081d8a4de2c"

newsapi_key = setup_newsapi()

# Predefined stock options by sector
STOCKS_BY_SECTOR = {
    "Technology": ["AAPL", "MSFT", "GOOG", "NVDA", "AMD", "INTC", "IBM", "CSCO", "ORCL", "TSM"],
    "E-Commerce": ["AMZN", "SHOP", "ETSY", "BABA", "EBAY", "JD", "MELI", "CPNG", "WMT", "TGT"],
    "Social Media": ["META", "SNAP", "PINS", "TWTR", "SPOT", "MTCH", "BMBL", "RBLX"],
    "Automotive": ["TSLA", "F", "GM", "TM", "LCID", "RIVN", "NIO", "LI", "XPEV"],
    "Finance": ["JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "PYPL", "SQ", "AXP"],
    "Healthcare": ["JNJ", "PFE", "MRK", "ABBV", "ABT", "UNH", "CVS", "ISRG", "GILD"],
    "Consumer": ["KO", "PEP", "MCD", "SBUX", "NKE", "LULU", "PG", "CL", "DIS", "NFLX"]
}

# Flatten for search functionality
ALL_STOCKS = [stock for sector_stocks in STOCKS_BY_SECTOR.values() for stock in sector_stocks]

# Add a function to fetch real news from NewsAPI
def fetch_news_articles(symbol, days=7):
    """Fetches news articles about the stock from NewsAPI"""
    articles = []
    
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Format dates for API
        from_date = start_date.strftime('%Y-%m-%d')
        to_date = end_date.strftime('%Y-%m-%d')
        
        # Make API request
        url = f"https://newsapi.org/v2/everything?q={symbol}&from={from_date}&to={to_date}&language=en&sortBy=publishedAt&apiKey={newsapi_key}"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            
            if data["status"] == "ok":
                for article in data["articles"]:
                    # Use VADER sentiment analyzer to determine sentiment
                    analyzer = SentimentIntensityAnalyzer()
                    sentiment_score = analyzer.polarity_scores(article["title"] + " " + (article["description"] or ""))["compound"]
                    
                    # Determine sentiment category
                    if sentiment_score >= 0.05:
                        sentiment = "Positive"
                    elif sentiment_score <= -0.05:
                        sentiment = "Negative"
                    else:
                        sentiment = "Neutral"
                    
                    # Parse date
                    published_date = datetime.strptime(article["publishedAt"], "%Y-%m-%dT%H:%M:%SZ")
                    
                    articles.append({
                        "headline": article["title"],
                        "source": article["source"]["name"],
                        "sentiment": sentiment,
                        "sentiment_score": sentiment_score,
                        "date": published_date,
                        "url": article["url"],
                        "description": article["description"]
                    })
        else:
            st.warning(f"Could not fetch news data: API returned status code {response.status_code}")
            
    except Exception as e:
        st.warning(f"Error fetching news data: {e}")
        
    # If no articles were found or there was an error, create minimal fallback data
    if not articles:
        # Create minimal fallback data with a clear indication it's fallback data
        fallback_dates = [(end_date - timedelta(days=i)) for i in range(5)]
        
        fallback_articles = [
            {
                "headline": f"[FALLBACK] {symbol} Market Analysis",
                "source": "StockSense Pro",
                "sentiment": "Neutral",
                "sentiment_score": 0.0,
                "date": fallback_dates[0],
                "url": "#",
                "description": "This is fallback data. The NewsAPI request failed or returned no results."
            },
            {
                "headline": f"[FALLBACK] {symbol} Recent Performance",
                "source": "StockSense Pro",
                "sentiment": "Neutral",
                "sentiment_score": 0.0,
                "date": fallback_dates[1],
                "url": "#",
                "description": "This is fallback data. The NewsAPI request failed or returned no results."
            }
        ]
        
        articles = fallback_articles
        st.info("Using fallback news data. To see real news, please ensure your NewsAPI key is valid and the stock symbol is correct.")
    
    # Sort by date (most recent first)
    articles.sort(key=lambda x: x["date"], reverse=True)
    
    # Create time series data for sentiment trend
    dates = [(end_date - timedelta(days=i)).date() for i in range(min(14, days), -1, -1)]
    
    # Group articles by date and calculate average sentiment for each date
    sentiment_by_date = {}
    for article in articles:
        article_date = article["date"].date()
        if article_date in sentiment_by_date:
            sentiment_by_date[article_date].append(article["sentiment_score"])
        else:
            sentiment_by_date[article_date] = [article["sentiment_score"]]
    
    # Calculate average sentiment for each date
    sentiment_trend = []
    for date in dates:
        if date in sentiment_by_date and sentiment_by_date[date]:
            avg_sentiment = sum(sentiment_by_date[date]) / len(sentiment_by_date[date])
        else:
            # If no data for this date, use interpolation or previous value
            if sentiment_trend:
                avg_sentiment = sentiment_trend[-1]["Sentiment"]
            else:
                avg_sentiment = 0
        sentiment_trend.append({"Date": date, "Sentiment": avg_sentiment})
    
    # Calculate overall sentiment
    sentiment_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for article in articles:
        sentiment_counts[article["sentiment"]] += 1
    
    # Calculate average sentiment score (-1 to 1)
    total_articles = len(articles)
    avg_sentiment = sum(article["sentiment_score"] for article in articles) / total_articles if total_articles > 0 else 0
    
    return {
        "articles": articles,
        "sentiment_counts": sentiment_counts,
        "sentiment_score": avg_sentiment,
        "sentiment_trend": pd.DataFrame(sentiment_trend)
    }

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 15px; background: linear-gradient(135deg, #1A237E, #0D47A1); border-radius: 10px; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);">
        <h1 style="color: white; font-size: 1.8rem; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">StockSense Pro</h1>
        <p style="color: #BBBBBB;">Advanced Market Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation
    st.markdown("<p style='color: var(--primary-color); font-weight: 600; margin-bottom: 5px;'>NAVIGATION</p>", unsafe_allow_html=True)
    page = st.radio("", ["Dashboard", "Technical Analysis", "Sentiment Analysis", "Comparison Tool", "Watchlist"], label_visibility="collapsed")
    
    st.markdown("---")
    
    # Quick Stock Search
    st.markdown("<p style='color: var(--primary-color); font-weight: 600; margin-bottom: 5px;'>QUICK SEARCH</p>", unsafe_allow_html=True)
    quick_search = st.text_input("", placeholder="Enter stock symbol...", label_visibility="collapsed")
    if quick_search:
        filtered_stocks = [s for s in ALL_STOCKS if quick_search.upper() in s]
        if filtered_stocks:
            selected_quick_stock = st.selectbox("Select Stock:", filtered_stocks)
            if st.button("Go to Stock"):
                st.session_state.stock_symbol = selected_quick_stock
                if page != "Dashboard":
                    page = "Dashboard"
                    st.rerun()
        else:
            st.warning("No matching stocks found")
    
    st.markdown("---")
    st.caption("StockSense Pro v1.0")
    st.caption(" 2025 - Final Year Project")

# Main content
if page == "Dashboard":
    st.markdown("<h1 class='main-header'> 📊 Stock Market Dashboard</h1>", unsafe_allow_html=True)
    
    # Stock selection
    col1, col2 = st.columns([3, 1])
    with col1:
        sector = st.selectbox("Select Sector:", list(STOCKS_BY_SECTOR.keys()))
        stock_options = STOCKS_BY_SECTOR[sector]
        
    with col2:
        if 'stock_symbol' not in st.session_state:
            st.session_state.stock_symbol = stock_options[0]
            
        stock_symbol = st.selectbox("Select Stock:", stock_options, index=stock_options.index(st.session_state.stock_symbol) if st.session_state.stock_symbol in stock_options else 0)
        st.session_state.stock_symbol = stock_symbol
        
        # Add to watchlist button
        if stock_symbol not in st.session_state.watchlist:
            if st.button("📌 Add to Watchlist"):
                st.session_state.watchlist.append(stock_symbol)
                st.success(f"Added {stock_symbol} to watchlist!")
        else:
            if st.button("Remove from Watchlist"):
                st.session_state.watchlist.remove(stock_symbol)
                st.success(f"Removed {stock_symbol} from watchlist!")
    
    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        period = st.selectbox("Select Time Period:", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=2)
    with col2:
        interval = st.selectbox("Select Interval:", ["1d", "1wk", "1mo"], index=0)
    
    # Loading spinner
    with st.spinner(f"Fetching data for {stock_symbol}..."):
        # Fetch stock data
        @st.cache_data(ttl=3600)
        def get_stock_data(symbol, period, interval):
            stock = yf.Ticker(symbol)
            hist = stock.history(period=period, interval=interval)
            info = stock.info
            return hist, info
        
        stock_data, stock_info = get_stock_data(stock_symbol, period, interval)
        
        if stock_data.empty:
            st.error("No stock data found! Please try another stock.")
        else:
            # Company info section
            st.markdown("<h2 class='sub-header'>Company Overview</h2>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                # Show company logo
                if 'logo_url' in stock_info and stock_info['logo_url']:
                    try:
                        logo_response = requests.get(stock_info['logo_url'])
                        logo_img = Image.open(io.BytesIO(logo_response.content))
                        st.image(logo_img, width=100)
                    except:
                        st.info(f"{stock_symbol} Logo")
                else:
                    st.info(f"{stock_symbol} Logo")
            
            with col2:
                st.subheader(stock_info.get('longName', stock_symbol))
                st.write(stock_info.get('industry', 'N/A'), "-", stock_info.get('sector', 'N/A'))
                st.write(stock_info.get('longBusinessSummary', '')[:200] + "..." if stock_info.get('longBusinessSummary') else "No company description available.")
                
            with col3:
                st.metric(
                    "Current Price", 
                    f"${stock_data['Close'].iloc[-1]:.2f}", 
                    f"{((stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-2]) / stock_data['Close'].iloc[-2] * 100):.2f}%"
                )
            
            # Key metrics
            st.markdown("<h2 class='sub-header'>Key Metrics</h2>", unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.metric("Market Cap", f"${stock_info.get('marketCap', 0) / 1e9:.2f}B")
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.metric("P/E Ratio", f"{stock_info.get('trailingPE', 0):.2f}")
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col3:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.metric("52 Week High", f"${stock_info.get('fiftyTwoWeekHigh', 0):.2f}")
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col4:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.metric("52 Week Low", f"${stock_info.get('fiftyTwoWeekLow', 0):.2f}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Price chart
            st.markdown("<h2 class='sub-header'>Price Chart</h2>", unsafe_allow_html=True)
            
            # Create figure with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add price line
            fig.add_trace(
                go.Scatter(
                    x=stock_data.index,
                    y=stock_data['Close'],
                    name="Price",
                    line=dict(color='#1E88E5', width=2)
                ),
                secondary_y=False,
            )
            
            # Add volume bars
            fig.add_trace(
                go.Bar(
                    x=stock_data.index,
                    y=stock_data['Volume'],
                    name="Volume",
                    marker=dict(color='rgba(30, 136, 229, 0.3)')
                ),
                secondary_y=True,
            )
            
            # Set titles
            fig.update_layout(
                title_text=f"{stock_symbol} Price and Volume",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=500,
                hovermode="x unified",
                template="plotly_dark",
                margin=dict(l=0, r=0, t=30, b=0),
            )
            
            # Set y-axes titles
            fig.update_yaxes(title_text="Price ($)", secondary_y=False)
            fig.update_yaxes(title_text="Volume", secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Stock statistics
            st.markdown("<h2 class='sub-header'>Statistics</h2>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("Trading Information")
                stats_df1 = pd.DataFrame({
                    'Metric': ['Average Volume', 'Relative Volume', 'Bid-Ask Spread', 'Beta', 'Short Ratio'],
                    'Value': [
                        f"{stock_info.get('averageVolume', 0):,}",
                        f"{stock_info.get('averageVolume10days', 0) / stock_info.get('averageVolume', 1):.2f}",
                        f"{(stock_info.get('ask', 0) - stock_info.get('bid', 0)):.2f}",
                        f"{stock_info.get('beta', 0):.2f}",
                        f"{stock_info.get('shortRatio', 0):.2f}"
                    ]
                })
                st.table(stats_df1)
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("Valuation Metrics")
                stats_df2 = pd.DataFrame({
                    'Metric': ['P/E', 'Forward P/E', 'PEG Ratio', 'Price/Sales', 'Price/Book'],
                    'Value': [
                        f"{stock_info.get('trailingPE', 0):.2f}",
                        f"{stock_info.get('forwardPE', 0):.2f}",
                        f"{stock_info.get('pegRatio', 0):.2f}",
                        f"{stock_info.get('priceToSalesTrailing12Months', 0):.2f}",
                        f"{stock_info.get('priceToBook', 0):.2f}"
                    ]
                })
                st.table(stats_df2)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Recent news (simplified)
            st.markdown("<h2 class='sub-header'>Recent News</h2>", unsafe_allow_html=True)
            
            # Fetch real news data from the API
            news_data = fetch_news_articles(stock_symbol)
            
            # Display the articles
            for article in news_data["articles"][:5]:
                sentiment_color = "#4CAF50" if article["sentiment_score"] > 0.05 else "#F44336" if article["sentiment_score"] < -0.05 else "#FFC107"
                st.markdown(f"<div class='card'>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='color: {sentiment_color};'>{article['headline']}</h3>", unsafe_allow_html=True)
                date_str = article["date"].strftime("%b %d, %Y")
                st.write(f"Source: {article['source']} | Date: {date_str}")
                st.write(f"Sentiment Score: {article['sentiment_score']:.2f}")
                with st.expander("Show Summary"):
                    st.write(article["description"] or "No summary available.")
                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

elif page == "Technical Analysis":
    st.markdown("<h1 class='main-header'> 📊 Technical Analysis</h1>", unsafe_allow_html=True)
    
    if 'stock_symbol' not in st.session_state:
        st.session_state.stock_symbol = "AAPL"
    
    stock_symbol = st.text_input("Enter Stock Symbol:", st.session_state.stock_symbol)
    st.session_state.stock_symbol = stock_symbol
    
    timeframe = st.selectbox("Select Timeframe:", ["1mo", "3mo", "6mo", "1y"], index=2)
    
    with st.spinner("Generating technical analysis..."):
        # Fetch stock data
        @st.cache_data(ttl=3600)
        def get_technical_data(symbol, period):
            stock = yf.Ticker(symbol)
            hist = stock.history(period=period, interval="1d")
            return hist
        
        data = get_technical_data(stock_symbol, timeframe)
        
        if data.empty:
            st.error("No data found for this stock symbol!")
        else:
            # Calculate technical indicators
            # Moving averages
            data['MA20'] = data['Close'].rolling(window=20).mean()
            data['MA50'] = data['Close'].rolling(window=50).mean()
            data['MA200'] = data['Close'].rolling(window=200).mean()
            
            # RSI (Relative Strength Index)
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD (Moving Average Convergence Divergence)
            data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
            data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = data['EMA12'] - data['EMA26']
            data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
            data['MACD_Hist'] = data['MACD'] - data['Signal']
            
            # Create plots
            st.markdown("<h2 class='sub-header'>Price and Moving Averages</h2>", unsafe_allow_html=True)
            
            fig1 = go.Figure()
            
            # Add price line
            fig1.add_trace(go.Scatter(
                x=data.index, y=data['Close'], name='Price',
                line=dict(color='#1E88E5', width=2)
            ))
            
            # Add moving averages
            fig1.add_trace(go.Scatter(
                x=data.index, y=data['MA20'], name='20-day MA',
                line=dict(color='#FFC107', width=1.5)
            ))
            
            fig1.add_trace(go.Scatter(
                x=data.index, y=data['MA50'], name='50-day MA',
                line=dict(color='#4CAF50', width=1.5)
            ))
            
            fig1.add_trace(go.Scatter(
                x=data.index, y=data['MA200'], name='200-day MA',
                line=dict(color='#F44336', width=1.5)
            ))
            
            fig1.update_layout(
                title_text=f"{stock_symbol} Price and Moving Averages",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=500,
                hovermode="x unified",
                template="plotly_dark",
                margin=dict(l=0, r=0, t=30, b=0),
            )
            
            st.plotly_chart(fig1, use_container_width=True)
            
            # Technical interpretation
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("Moving Average Analysis")
                
                # Determine if price is above/below MA
                price = data['Close'].iloc[-1]
                ma20 = data['MA20'].iloc[-1]
                ma50 = data['MA50'].iloc[-1]
                ma200 = data['MA200'].iloc[-1]
                
                st.write(f"Price vs MA20: {'Above' if price > ma20 else 'Below'}")
                st.write(f"Price vs MA50: {'Above' if price > ma50 else 'Below'}")
                st.write(f"Price vs MA200: {'Above' if price > ma200 else 'Below'}")
                
                # Cross detection
                if data['MA20'].iloc[-2] < data['MA50'].iloc[-2] and data['MA20'].iloc[-1] > data['MA50'].iloc[-1]:
                    st.write("Golden Cross (MA20 crossed above MA50) - Bullish")
                elif data['MA20'].iloc[-2] > data['MA50'].iloc[-2] and data['MA20'].iloc[-1] < data['MA50'].iloc[-1]:
                    st.write("Death Cross (MA20 crossed below MA50) - Bearish")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("Support/Resistance Levels")
                
                # Simplified S/R levels
                price_range = data['High'].max() - data['Low'].min()
                increments = price_range / 5
                
                levels = []
                for i in range(1, 6):
                    levels.append(data['Low'].min() + increments * i)
                
                for i, level in enumerate(sorted(levels)):
                    st.write(f"Level {i+1}: ${level:.2f}")
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # RSI plot
            st.markdown("<h2 class='sub-header'>Relative Strength Index (RSI)</h2>", unsafe_allow_html=True)
            
            fig2 = go.Figure()
            
            fig2.add_trace(go.Scatter(
                x=data.index, y=data['RSI'], name='RSI',
                line=dict(color='#673AB7', width=2)
            ))
            
            # Add overbought/oversold lines
            fig2.add_shape(
                type="line", line=dict(dash='dash'),
                x0=data.index[0], y0=70, x1=data.index[-1], y1=70
            )
            
            fig2.add_shape(
                type="line", line=dict(dash='dash'),
                x0=data.index[0], y0=30, x1=data.index[-1], y1=30
            )
            
            fig2.update_layout(
                title="RSI (14) - Relative Strength Index",
                xaxis_title="Date",
                yaxis_title="RSI Value",
                height=300,
                hovermode="x unified",
                template="plotly_dark",
                margin=dict(l=0, r=0, t=30, b=0),
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # RSI interpretation
            current_rsi = data['RSI'].iloc[-1]
            
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current RSI", f"{current_rsi:.1f}")
                
            with col2:
                if current_rsi > 70:
                    st.write("Status: OVERBOUGHT")
                elif current_rsi < 30:
                    st.write("Status: OVERSOLD")
                else:
                    st.write("Status: NEUTRAL")
                    
            with col3:
                if current_rsi > 70:
                    st.write("Signal: Consider Selling")
                elif current_rsi < 30:
                    st.write("Signal: Consider Buying")
                else:
                    st.write("Signal: Hold/Neutral")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # MACD plot
            st.markdown("<h2 class='sub-header'>MACD (Moving Average Convergence Divergence)</h2>", unsafe_allow_html=True)
            
            fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.02, row_heights=[0.7, 0.3])
            
            # Add price to top plot
            fig3.add_trace(
                go.Scatter(x=data.index, y=data['Close'], name='Price'),
                row=1, col=1
            )
            
            # Add MACD to bottom plot
            fig3.add_trace(
                go.Scatter(x=data.index, y=data['MACD'], name='MACD Line',
                          line=dict(color='#1E88E5', width=2)),
                row=2, col=1
            )
            
            fig3.add_trace(
                go.Scatter(x=data.index, y=data['Signal'], name='Signal Line',
                          line=dict(color='#FF5722', width=1)),
                row=2, col=1
            )
            
            # Add histogram
            colors = ['#4CAF50' if val >= 0 else '#F44336' for val in data['MACD_Hist']]
            
            fig3.add_trace(
                go.Bar(x=data.index, y=data['MACD_Hist'], name='Histogram',
                      marker_color=colors),
                row=2, col=1
            )
            
            fig3.update_layout(height=500, title_text="MACD Analysis",
                             hovermode="x unified", legend=dict(orientation="h", y=1.02),
                             template="plotly_dark",
                             margin=dict(l=0, r=0, t=30, b=0),
                             )
            
            st.plotly_chart(fig3, use_container_width=True)
            
            # MACD interpretation
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("MACD Signal Interpretation")
            
            macd = data['MACD'].iloc[-1]
            signal = data['Signal'].iloc[-1]
            hist = data['MACD_Hist'].iloc[-1]
            
            if macd > signal and hist > 0:
                st.write("MACD is above the signal line and histogram is positive - Bullish signal")
            elif macd < signal and hist < 0:
                st.write("MACD is below the signal line and histogram is negative - Bearish signal")
            elif macd > signal and macd < 0:
                st.write("MACD is above the signal line but still negative - Potential bullish reversal")
            elif macd < signal and macd > 0:
                st.write("MACD is below the signal line but still positive - Potential bearish reversal")
                
            # Detect crossovers
            if (data['MACD'].iloc[-2] < data['Signal'].iloc[-2] and 
                data['MACD'].iloc[-1] > data['Signal'].iloc[-1]):
                st.write("Recent bullish crossover (MACD crossed above signal line)")
            elif (data['MACD'].iloc[-2] > data['Signal'].iloc[-2] and 
                  data['MACD'].iloc[-1] < data['Signal'].iloc[-1]):
                st.write("Recent bearish crossover (MACD crossed below signal line)")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Overall technical summary
            st.markdown("<h2 class='sub-header'>Technical Summary</h2>", unsafe_allow_html=True)
            
            # Simplified scoring system
            score = 0
            signals = []
            
            # Moving average signals
            if price > ma20:
                score += 1
                signals.append("Price > MA20 (Bullish)")
            else:
                score -= 1
                signals.append("Price < MA20 (Bearish)")
                
            if price > ma50:
                score += 1
                signals.append("Price > MA50 (Bullish)")
            else:
                score -= 1
                signals.append("Price < MA50 (Bearish)")
                
            if ma20 > ma50:
                score += 1
                signals.append("MA20 > MA50 (Bullish)")
            else:
                score -= 1
                signals.append("MA20 < MA50 (Bearish)")
                
            # RSI signals
            if current_rsi > 70:
                score -= 1
                signals.append("RSI > 70 (Overbought - Bearish)")
            elif current_rsi < 30:
                score += 1
                signals.append("RSI < 30 (Oversold - Bullish)")
                
            # MACD signals
            if macd > signal:
                score += 1
                signals.append("MACD > Signal (Bullish)")
            else:
                score -= 1
                signals.append("MACD < Signal (Bearish)")
                
            if hist > 0:
                score += 1
                signals.append("MACD Histogram Positive (Bullish)")
            else:
                score -= 1
                signals.append("MACD Histogram Negative (Bearish)")
            
            # Display summary
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Generate recommendation based on score
                if score >= 3:
                    recommendation = "Strong Buy"
                    color = "#4CAF50"
                elif score > 0:
                    recommendation = "Buy"
                    color = "#8BC34A"
                elif score == 0:
                    recommendation = "Neutral"
                    color = "#FFC107"
                elif score > -3:
                    recommendation = "Sell"
                    color = "#FF5722"
                else:
                    recommendation = "Strong Sell"
                    color = "#F44336"
                
                st.markdown(f"<div style='background-color: {color}; padding: 20px; border-radius: 10px; text-align: center;'>", unsafe_allow_html=True)
                st.markdown(f"<h1 style='color: white;'>{recommendation}</h1>", unsafe_allow_html=True)
                st.markdown(f"<p style='color: white;'>Score: {score}</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col2:
                st.subheader("Technical Signals")
                for signal in signals:
                    st.write(f"• {signal}")

elif page == "Sentiment Analysis":
    st.markdown("<h1 class='main-header'> 📊 Market Sentiment Analysis</h1>", unsafe_allow_html=True)
    
    if 'stock_symbol' not in st.session_state:
        st.session_state.stock_symbol = "AAPL"
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        stock_symbol = st.text_input("Enter Stock Symbol for Analysis:", st.session_state.stock_symbol)
        st.session_state.stock_symbol = stock_symbol
        
    with col2:
        platform = st.selectbox("Data Source:", ["Reddit", "Twitter*", "News*"], index=0)
        if platform != "Reddit":
            st.info("* Simulation only in this project")
    
    if st.button("Analyze Sentiment"):
        with st.spinner(f"Analyzing {stock_symbol} sentiment from {platform}..."):
            # Initialize sentiment analyzer
            sia = SentimentIntensityAnalyzer()
            
            # Function to get Reddit posts
            def fetch_reddit_posts(symbol, limit=25):
                """Fetches recent Reddit discussions about the stock"""
                posts = []
                try:
                    for submission in reddit.subreddit("wallstreetbets+stocks+investing").search(symbol, limit=limit):
                        posts.append({
                            "title": submission.title,
                            "text": submission.selftext,
                            "score": submission.score,
                            "created_utc": datetime.fromtimestamp(submission.created_utc),
                            "url": submission.url
                        })
                except Exception as e:
                    st.error(f"Error fetching Reddit data: {e}")
                return posts
            
            if platform == "Reddit":
                # Get Reddit posts
                posts = fetch_reddit_posts(stock_symbol)
                
                if not posts:
                    st.warning(f"No Reddit discussions found for {stock_symbol}!")
                else:
                    # Perform sentiment analysis
                    sentiments = {"Positive": 0, "Neutral": 0, "Negative": 0}
                    sentiment_scores = []
                    post_texts = []
                    
                    for post in posts:
                        # Combine title and text for analysis
                        text = post["title"] + " " + post["text"]
                        sentiment_score = sia.polarity_scores(text)["compound"]
                        
                        if sentiment_score > 0.05:
                            sentiments["Positive"] += 1
                            category = "Positive"
                        elif sentiment_score < -0.05:
                            sentiments["Negative"] += 1
                            category = "Negative"
                        else:
                            sentiments["Neutral"] += 1
                            category = "Neutral"
                        
                        sentiment_scores.append(sentiment_score)
                        post_texts.append({
                            "text": post["title"],
                            "score": sentiment_score,
                            "category": category,
                            "date": post["created_utc"],
                            "upvotes": post["score"]
                        })
                    
                    # Display sentiment distribution
                    st.markdown("<h2 class='sub-header'>Reddit Sentiment Distribution</h2>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Create pie chart
                        sentiment_df = pd.DataFrame(sentiments.items(), columns=["Sentiment", "Count"])
                        fig = px.pie(
                            sentiment_df, 
                            values="Count", 
                            names="Sentiment", 
                            title=f"Sentiment Distribution for {stock_symbol}",
                            color="Sentiment",
                            color_discrete_map={
                                "Positive": "#4CAF50",
                                "Neutral": "#FFC107",
                                "Negative": "#F44336"
                            },
                            template="plotly_dark"
                        )
                        fig.update_layout(
                            margin=dict(l=0, r=0, t=30, b=0),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Calculate overall sentiment score
                        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
                        
                        # Display gauge chart for overall sentiment
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=avg_sentiment,
                            domain={"x": [0, 1], "y": [0, 1]},
                            title={"text": "Overall Sentiment Score"},
                            gauge={
                                "axis": {"range": [-1, 1]},
                                "bar": {"color": "#1E88E5"},
                                "steps": [
                                    {"range": [-1, -0.05], "color": "rgba(244, 67, 54, 0.3)"},
                                    {"range": [-0.05, 0.05], "color": "rgba(255, 193, 7, 0.3)"},
                                    {"range": [0.05, 1], "color": "rgba(76, 175, 80, 0.3)"}
                                ],
                                "threshold": {
                                    "line": {"color": "black", "width": 4},
                                    "thickness": 0.75,
                                    "value": avg_sentiment
                                }
                            }
                        ))
                        
                        fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0),)
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Display sentiment over time
                    st.markdown("<h2 class='sub-header'>Sentiment Trend</h2>", unsafe_allow_html=True)
                    
                    # Sort posts by date
                    sorted_posts = sorted(post_texts, key=lambda x: x["date"])
                    
                    # Create dataframe for time series
                    trend_df = pd.DataFrame(sorted_posts)
                    
                    # Create line chart
                    fig = px.line(
                        trend_df, 
                        x="date", 
                        y="score",
                        title=f"Sentiment Trend for {stock_symbol}",
                        labels={"date": "Date", "score": "Sentiment Score"},
                        template="plotly_dark"
                    )
                    fig.update_layout(
                        margin=dict(l=0, r=0, t=30, b=0),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add horizontal lines for reference
                    fig.add_shape(
                        type="line", line=dict(dash="dash"),
                        x0=trend_df["date"].min(), y0=0.05, 
                        x1=trend_df["date"].max(), y1=0.05,
                        line_color="#4CAF50"
                    )
                    
                    fig.add_shape(
                        type="line", line=dict(dash="dash"),
                        x0=trend_df["date"].min(), y0=-0.05, 
                        x1=trend_df["date"].max(), y1=-0.05,
                        line_color="#F44336"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display top posts
                    st.markdown("<h2 class='sub-header'>Top Reddit Posts</h2>", unsafe_allow_html=True)
                    
                    # Sort by upvotes
                    top_posts = sorted(posts, key=lambda x: x["score"], reverse=True)[:10]
                    
                    for i, post in enumerate(top_posts):
                        sentiment = sia.polarity_scores(post["title"] + " " + post["text"])["compound"]
                        sentiment_color = "#4CAF50" if sentiment > 0.05 else "#F44336" if sentiment < -0.05 else "#FFC107"
                        
                        st.markdown(f"<div class='card'>", unsafe_allow_html=True)
                        st.markdown(f"<h3 style='color: {sentiment_color};'>{post['title']}</h3>", unsafe_allow_html=True)
                        st.write(f"Sentiment Score: {sentiment:.2f}")
                        st.write(f"Upvotes: {post['score']}")
                        st.write(f"Posted: {post['created_utc'].strftime('%Y-%m-%d %H:%M:%S')}")
                        if post["text"]:
                            with st.expander("Show Content"):
                                st.write(post["text"][:500] + "..." if len(post["text"]) > 500 else post["text"])
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)

            elif platform == "Twitter*":
                # Simulated Twitter sentiment analysis
                st.info("This is a simulation of Twitter sentiment analysis.")
                
                # Generate simulated data
                sentiments = {"Positive": 45, "Neutral": 30, "Negative": 25}
                
                # Display pie chart
                sentiment_df = pd.DataFrame(sentiments.items(), columns=["Sentiment", "Count"])
                fig = px.pie(
                    sentiment_df, 
                    values="Count", 
                    names="Sentiment", 
                    title=f"Simulated Twitter Sentiment for {stock_symbol}",
                    color="Sentiment",
                    color_discrete_map={
                        "Positive": "#4CAF50",
                        "Neutral": "#FFC107",
                        "Negative": "#F44336"
                    },
                    template="plotly_dark"
                )
                fig.update_layout(
                    margin=dict(l=0, r=0, t=30, b=0),
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display simulated tweets
                st.markdown("<h2 class='sub-header'>Simulated Tweets</h2>", unsafe_allow_html=True)
                
                simulated_tweets = [
                    {"text": f"Just bought more ${stock_symbol}! The company's outlook is amazing.", "sentiment": 0.8},
                    {"text": f"${stock_symbol} earnings report was solid. Holding long term.", "sentiment": 0.6},
                    {"text": f"Not sure about ${stock_symbol}, might wait for a pullback before buying.", "sentiment": 0.0},
                    {"text": f"${stock_symbol} facing increased competition. Concerned about next quarter.", "sentiment": -0.4},
                    {"text": f"Selling my ${stock_symbol} shares. Don't like the management decisions.", "sentiment": -0.7}
                ]
                
                for tweet in simulated_tweets:
                    sentiment_color = "#4CAF50" if tweet["sentiment"] > 0.05 else "#F44336" if tweet["sentiment"] < -0.05 else "#FFC107"
                    st.markdown(f"<div class='card' style='border-left: 5px solid {sentiment_color};'>", unsafe_allow_html=True)
                    st.write(tweet["text"])
                    st.write(f"Sentiment Score: {tweet['sentiment']:.2f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
            
            else:  # News
                # Simulated news sentiment analysis
                st.info("This is a simulation of News sentiment analysis.")
                
                # Generate simulated data
                dates = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(14)]
                sentiment_values = [0.3, 0.4, 0.2, -0.1, -0.3, -0.2, 0.1, 0.3, 0.5, 0.4, 0.2, 0.1, 0.0, 0.2]
                
                # Create dataframe
                news_df = pd.DataFrame({"Date": dates, "Sentiment": sentiment_values})
                
                # Plot time series
                fig = px.line(
                    news_df, 
                    x="Date", 
                    y="Sentiment",
                    title=f"Simulated News Sentiment Trend for {stock_symbol}",
                    labels={"Date": "Date", "Sentiment": "Sentiment Score"},
                    template="plotly_dark"
                )
                fig.update_layout(
                    margin=dict(l=0, r=0, t=30, b=0),
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Add horizontal lines for reference
                fig.add_shape(
                    type="line",
                    x0=news_df["Date"].min(),
                    x1=news_df["Date"].max(),
                    y0=0,
                    y1=0,
                    line=dict(color="gray", width=1, dash="dash")
                )
                
                fig.add_shape(
                    type="line",
                    x0=news_df["Date"].min(),
                    x1=news_df["Date"].max(),
                    y0=0.05,
                    y1=0.05,
                    line=dict(color="#4CAF50", width=1, dash="dash")
                )
                
                fig.add_shape(
                    type="line",
                    x0=news_df["Date"].min(),
                    x1=news_df["Date"].max(),
                    y0=-0.05,
                    y1=-0.05,
                    line=dict(color="#F44336", width=1, dash="dash")
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display simulated news articles
                st.markdown("<h2 class='sub-header'>Recent News Articles</h2>", unsafe_allow_html=True)
                
                simulated_news = [
                    {
                        "title": f"{stock_symbol} Reports Strong Quarterly Earnings, Beating Expectations",
                        "source": "Financial Times",
                        "date": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
                        "sentiment": 0.7
                    },
                    {
                        "title": f"Analysts Upgrade {stock_symbol} Rating to 'Buy' Citing Growth Potential",
                        "source": "Bloomberg",
                        "date": (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d"),
                        "sentiment": 0.6
                    },
                    {
                        "title": f"{stock_symbol} Announces New Product Line, Market Response Mixed",
                        "source": "Reuters",
                        "date": (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"),
                        "sentiment": 0.1
                    },
                    {
                        "title": f"Industry Competition Intensifies as Rival Challenges {stock_symbol}",
                        "source": "Wall Street Journal",
                        "date": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
                        "sentiment": -0.3
                    },
                    {
                        "title": f"{stock_symbol} Faces Regulatory Scrutiny Over Business Practices",
                        "source": "CNBC",
                        "date": (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d"),
                        "sentiment": -0.5
                    }
                ]
                
                for article in simulated_news:
                    sentiment_color = "#4CAF50" if article["sentiment"] > 0.05 else "#F44336" if article["sentiment"] < -0.05 else "#FFC107"
                    st.markdown(f"<div class='card'>", unsafe_allow_html=True)
                    st.markdown(f"<h3 style='color: {sentiment_color};'>{article['title']}</h3>", unsafe_allow_html=True)
                    st.write(f"Source: {article['source']} | Date: {article['date']}")
                    st.write(f"Sentiment Score: {article['sentiment']:.2f}")
                    with st.expander("Show Summary"):
                        st.write("This is a simulated news article summary. In a full implementation, this would contain the actual content of the news article.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)

elif page == "Comparison Tool":
    st.markdown("<h1 class='main-header'> 📊 Stock Comparison Tool</h1>", unsafe_allow_html=True)
    
    st.write("Compare multiple stocks to analyze their performance and metrics side by side.")
    
    # Stock selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Multi-select for stocks
        selected_stocks = st.multiselect(
            "Select Stocks to Compare:",
            ALL_STOCKS,
            default=["AAPL", "MSFT", "GOOG"] if "AAPL" in ALL_STOCKS and "MSFT" in ALL_STOCKS and "GOOG" in ALL_STOCKS else ALL_STOCKS[:3]
        )
    
    with col2:
        # Time period selection
        compare_period = st.selectbox("Time Period:", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=2)
    
    if not selected_stocks:
        st.warning("Please select at least one stock to compare.")
    elif len(selected_stocks) > 5:
        st.warning("Please select a maximum of 5 stocks for better visualization.")
    else:
        # Fetch data for all selected stocks
        with st.spinner("Fetching comparison data..."):
            @st.cache_data(ttl=3600)
            def get_comparison_data(symbols, period):
                data = {}
                info = {}
                
                for symbol in symbols:
                    try:
                        stock = yf.Ticker(symbol)
                        data[symbol] = stock.history(period=period)
                        info[symbol] = stock.info
                    except Exception as e:
                        st.error(f"Error fetching data for {symbol}: {e}")
                
                return data, info
            
            stock_data, stock_info = get_comparison_data(selected_stocks, compare_period)
            
            # Price comparison chart
            st.markdown("<h2 class='sub-header'>Price Performance Comparison</h2>", unsafe_allow_html=True)
            
            # Normalize prices to percentage change
            comparison_df = pd.DataFrame()
            
            for symbol in selected_stocks:
                if symbol in stock_data and not stock_data[symbol].empty:
                    # Calculate percentage change from first day
                    first_price = stock_data[symbol]['Close'].iloc[0]
                    comparison_df[symbol] = (stock_data[symbol]['Close'] / first_price - 1) * 100
            
            if not comparison_df.empty:
                # Create line chart
                fig = px.line(
                    comparison_df,
                    title="Percentage Price Change Comparison",
                    labels={"value": "% Change", "variable": "Stock", "index": "Date"},
                    template="plotly_dark"
                )
                fig.update_layout(
                    margin=dict(l=0, r=0, t=30, b=0),
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Key metrics comparison
                st.markdown("<h2 class='sub-header'>Key Metrics Comparison</h2>", unsafe_allow_html=True)
                
                # Create metrics dataframe
                metrics = {
                    "Current Price": [],
                    "Market Cap (B)": [],
                    "P/E Ratio": [],
                    "Forward P/E": [],
                    "PEG Ratio": [],
                    "Price/Sales": [],
                    "Price/Book": [],
                    "Beta": [],
                    "52W High": [],
                    "52W Low": []
                }
                
                for symbol in selected_stocks:
                    if symbol in stock_data and not stock_data[symbol].empty and symbol in stock_info:
                        info = stock_info[symbol]
                        price = stock_data[symbol]['Close'].iloc[-1]
                        
                        metrics["Current Price"].append(f"${price:.2f}")
                        metrics["Market Cap (B)"].append(f"${info.get('marketCap', 0) / 1e9:.2f}B")
                        metrics["P/E Ratio"].append(f"{info.get('trailingPE', 0):.2f}")
                        metrics["Forward P/E"].append(f"{info.get('forwardPE', 0):.2f}")
                        metrics["PEG Ratio"].append(f"{info.get('pegRatio', 0):.2f}")
                        metrics["Price/Sales"].append(f"{info.get('priceToSalesTrailing12Months', 0):.2f}")
                        metrics["Price/Book"].append(f"{info.get('priceToBook', 0):.2f}")
                        metrics["Beta"].append(f"{info.get('beta', 0):.2f}")
                        metrics["52W High"].append(f"${info.get('fiftyTwoWeekHigh', 0):.2f}")
                        metrics["52W Low"].append(f"${info.get('fiftyTwoWeekLow', 0):.2f}")
                
                # Create comparison table
                metrics_df = pd.DataFrame(metrics, index=selected_stocks)
                st.dataframe(metrics_df, use_container_width=True)
                
                # Volume comparison
                st.markdown("<h2 class='sub-header'>Trading Volume Comparison</h2>", unsafe_allow_html=True)
                
                # Create volume dataframe
                volume_df = pd.DataFrame()
                
                for symbol in selected_stocks:
                    if symbol in stock_data and not stock_data[symbol].empty:
                        volume_df[symbol] = stock_data[symbol]['Volume']
                
                # Create bar chart
                fig = px.bar(
                    volume_df,
                    title="Trading Volume Comparison",
                    labels={"value": "Volume", "variable": "Stock", "index": "Date"},
                    template="plotly_dark"
                )
                fig.update_layout(
                    margin=dict(l=0, r=0, t=30, b=0),
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Correlation matrix
                st.markdown("<h2 class='sub-header'>Price Correlation Matrix</h2>", unsafe_allow_html=True)
                
                # Calculate correlation between stock prices
                corr_matrix = comparison_df.corr()
                
                # Create heatmap
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    color_continuous_scale="RdBu_r",
                    title="Price Movement Correlation",
                    template="plotly_dark"
                )
                fig.update_layout(
                    margin=dict(l=0, r=0, t=30, b=0),
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk vs. Return
                st.markdown("<h2 class='sub-header'>Risk vs. Return Analysis</h2>", unsafe_allow_html=True)
                
                # Calculate daily returns
                returns_df = comparison_df.pct_change().dropna()
                
                # Calculate annualized return and volatility
                risk_return = {
                    "Stock": [],
                    "Annualized Return (%)": [],
                    "Annualized Volatility (%)": []
                }
                
                for symbol in selected_stocks:
                    if symbol in returns_df.columns:
                        # Calculate annualized return (252 trading days in a year)
                        ann_return = returns_df[symbol].mean() * 252 * 100
                        
                        # Calculate annualized volatility
                        ann_volatility = returns_df[symbol].std() * math.sqrt(252) * 100
                        
                        risk_return["Stock"].append(symbol)
                        risk_return["Annualized Return (%)"].append(ann_return)
                        risk_return["Annualized Volatility (%)"].append(ann_volatility)
                
                # Create risk-return dataframe
                risk_return_df = pd.DataFrame(risk_return)
                
                # Create scatter plot
                fig = px.scatter(
                    risk_return_df,
                    x="Annualized Volatility (%)",
                    y="Annualized Return (%)",
                    text="Stock",
                    title="Risk vs. Return Analysis",
                    size=[10] * len(risk_return_df),
                    color="Stock",
                    template="plotly_dark"
                )
                fig.update_layout(
                    margin=dict(l=0, r=0, t=30, b=0),
                )
                st.plotly_chart(fig, use_container_width=True)

elif page == "Watchlist":
    st.markdown("<h1 class='main-header'> 📋 Your Watchlist</h1>", unsafe_allow_html=True)
    
    if not st.session_state.watchlist:
        st.info("Your watchlist is empty. Add stocks from the Dashboard to track them here.")
        
        # Show some suggested stocks
        st.markdown("<h2 class='sub-header'>Suggested Stocks to Watch</h2>", unsafe_allow_html=True)
        
        suggested = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
        for stock in suggested:
            if stock in ALL_STOCKS:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{stock}** - Add this popular stock to your watchlist")
                with col2:
                    if st.button(f"Add {stock}", key=f"add_{stock}"):
                        st.session_state.watchlist.append(stock)
                        st.rerun()
    else:
        # Allow reordering
        st.write("Drag and drop to reorder your watchlist:")
        watchlist_order = st.multiselect(
            "Your Watchlist:",
            st.session_state.watchlist,
            default=st.session_state.watchlist
        )
        
        if watchlist_order:
            st.session_state.watchlist = watchlist_order
        
        # Fetch data for watchlist stocks
        with st.spinner("Fetching watchlist data..."):
            @st.cache_data(ttl=1800)
            def get_watchlist_data(symbols):
                data = {}
                
                for symbol in symbols:
                    try:
                        stock = yf.Ticker(symbol)
                        hist = stock.history(period="1mo")
                        info = stock.info
                        
                        # Calculate some metrics
                        if not hist.empty:
                            current_price = hist['Close'].iloc[-1]
                            prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                            day_change = (current_price - prev_price) / prev_price * 100
                            
                            week_price = hist['Close'].iloc[-5] if len(hist) >= 5 else hist['Close'].iloc[0]
                            week_change = (current_price - week_price) / week_price * 100
                            
                            month_price = hist['Close'].iloc[0]
                            month_change = (current_price - month_price) / month_price * 100
                            
                            data[symbol] = {
                                "price": current_price,
                                "day_change": day_change,
                                "week_change": week_change,
                                "month_change": month_change,
                                "volume": hist['Volume'].iloc[-1],
                                "name": info.get('longName', symbol),
                                "sector": info.get('sector', 'N/A'),
                                "market_cap": info.get('marketCap', 0),
                                "pe_ratio": info.get('trailingPE', 0)
                            }
                    except Exception as e:
                        st.error(f"Error fetching data for {symbol}: {e}")
                
                return data
            
            watchlist_data = get_watchlist_data(st.session_state.watchlist)
            
            # Display watchlist as cards
            st.markdown("<h2 class='sub-header'>Your Stocks</h2>", unsafe_allow_html=True)
            
            # Create columns for layout
            cols = st.columns(3)
            
            for i, symbol in enumerate(st.session_state.watchlist):
                if symbol in watchlist_data:
                    data = watchlist_data[symbol]
                    
                    with cols[i % 3]:
                        # Determine color based on day change
                        color = "#4CAF50" if data["day_change"] > 0 else "#F44336"
                        
                        st.markdown(f"""
                        <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                            <h3>{symbol} - {data["name"]}</h3>
                            <h2 style="color: {color};">${data["price"]:.2f} <span style="font-size: 0.8em;">({data["day_change"]:.2f}%)</span></h2>
                            <p>Sector: {data["sector"]}</p>
                            <p>Market Cap: ${data["market_cap"]/1e9:.2f}B</p>
                            <p>P/E Ratio: {data["pe_ratio"]:.2f}</p>
                            <hr>
                            <p>1W Change: <span style="color: {'#4CAF50' if data['week_change'] > 0 else '#F44336'};">{data["week_change"]:.2f}%</span></p>
                            <p>1M Change: <span style="color: {'#4CAF50' if data['month_change'] > 0 else '#F44336'};">{data["month_change"]:.2f}%</span></p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(f"View Details", key=f"view_{symbol}"):
                                st.session_state.stock_symbol = symbol
                                st.session_state.page = "Dashboard"
                                st.rerun()
                        with col2:
                            if st.button(f"Remove", key=f"remove_{symbol}"):
                                st.session_state.watchlist.remove(symbol)
                                st.rerun()
            
            # Performance overview
            if watchlist_data:
                st.markdown("<h2 class='sub-header'>Watchlist Performance Overview</h2>", unsafe_allow_html=True)
                
                # Create performance dataframe
                performance = {
                    "Symbol": [],
                    "1D Change (%)": [],
                    "1W Change (%)": [],
                    "1M Change (%)": []
                }
                
                for symbol in st.session_state.watchlist:
                    if symbol in watchlist_data:
                        data = watchlist_data[symbol]
                        performance["Symbol"].append(symbol)
                        performance["1D Change (%)"].append(data["day_change"])
                        performance["1W Change (%)"].append(data["week_change"])
                        performance["1M Change (%)"].append(data["month_change"])
                
                perf_df = pd.DataFrame(performance)
                
                # Create heatmap
                fig = px.imshow(
                    perf_df.set_index("Symbol").values,
                    x=["1D", "1W", "1M"],
                    y=perf_df["Symbol"],
                    color_continuous_scale=["#F44336", "#FFFFFF", "#4CAF50"],
                    color_continuous_midpoint=0,
                    text_auto=".2f",
                    title="Performance Heatmap (%)",
                    template="plotly_dark"
                )
                fig.update_layout(
                    margin=dict(l=0, r=0, t=30, b=0),
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Portfolio allocation (simulated)
                st.markdown("<h2 class='sub-header'>Portfolio Allocation (Simulated)</h2>", unsafe_allow_html=True)
                
                # Create equal-weight allocation for demonstration
                allocation = {
                    "Symbol": st.session_state.watchlist,
                    "Allocation": [100 / len(st.session_state.watchlist)] * len(st.session_state.watchlist)
                }
                
                alloc_df = pd.DataFrame(allocation)
                
                # Create pie chart
                fig = px.pie(
                    alloc_df,
                    values="Allocation",
                    names="Symbol",
                    title="Portfolio Allocation (Equal Weight)",
                    color_discrete_sequence=px.colors.sequential.Blues_r,
                    template="plotly_dark"
                )
                fig.update_layout(
                    margin=dict(l=0, r=0, t=30, b=0),
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Add a note
                st.info("This is a simulated equal-weight portfolio allocation. In a full implementation, users would be able to specify their actual holdings and allocations.")

# Footer with disclaimer
st.markdown("---")
st.markdown("""
<div style="background-color: var(--primary-light); padding: 20px; border-radius: 10px; margin-top: 30px; border-left: 4px solid var(--primary-color);">
    <h3>Disclaimer</h3>
    <p>StockSense Pro is a demonstration project for educational purposes only. The information provided should not be considered financial advice. 
    Always conduct your own research or consult with a qualified financial advisor before making investment decisions.</p>
    <p>Data is sourced from Yahoo Finance and Reddit. Market data may be delayed. Sentiment analysis is based on natural language processing and may not accurately reflect market sentiment.</p>
    <p> 2025 - Final Year Project</p>
</div>
""", unsafe_allow_html=True)

# Twitter and News Sentiment Analysis sections for the Sentiment Analysis page
if page == "Sentiment Analysis":
    # Twitter Data Section (only add if we're on the Sentiment Analysis page)
    st.markdown("<h2 class='sub-header'>Twitter Sentiment Analysis</h2>", unsafe_allow_html=True)
    
    # Simulated Twitter data
    with st.spinner("Analyzing Twitter sentiment..."):
        # Create simulated Twitter data
        @st.cache_data(ttl=3600)
        def get_simulated_twitter_data(symbol):
            # Generate a random seed based on the symbol to ensure consistent results
            seed = sum(ord(c) for c in symbol)
            random.seed(seed)
            
            # Generate simulated tweets
            tweets = []
            sentiment_choices = ["Positive", "Neutral", "Negative"]
            weights = [0.5, 0.3, 0.2]  # Default weights favoring positive sentiment
            
            # Adjust weights based on stock symbol to create variation
            if symbol in ["AAPL", "MSFT", "GOOG", "AMZN"]:
                weights = [0.6, 0.3, 0.1]  # More positive for tech giants
            elif symbol in ["GME", "AMC", "BB"]:
                weights = [0.4, 0.2, 0.4]  # More polarized for meme stocks
            
            # Generate tweets
            tweet_templates = [
                "Just bought some ${}! #investing",
                "I think ${} is going to {}",
                "My analysis shows ${} will {} in the next quarter",
                "${} earnings report looks {}",
                "The market sentiment for ${} is {}",
                "Analysts are {} about ${} future prospects",
                "${} is a {} investment right now",
                "Just read some {} news about ${}",
                "Technical indicators for ${} look {}",
                "${} chart pattern suggests {} movement ahead"
            ]
            
            sentiment_words = {
                "Positive": ["bullish", "rise", "strong", "positive", "optimistic", "great", "excellent", "promising"],
                "Neutral": ["stable", "hold", "steady", "neutral", "mixed", "unclear", "average", "moderate"],
                "Negative": ["bearish", "fall", "weak", "negative", "pessimistic", "concerning", "disappointing", "troubling"]
            }
            
            # Generate 20-40 tweets
            num_tweets = random.randint(20, 40)
            for _ in range(num_tweets):
                sentiment = random.choices(sentiment_choices, weights=weights)[0]
                template = random.choice(tweet_templates)
                
                # Replace placeholders
                tweet_text = template.format(symbol, random.choice(sentiment_words[sentiment]))
                if "{}" in tweet_text:  # If there's still a placeholder
                    tweet_text = tweet_text.format(random.choice(sentiment_words[sentiment]))
                
                # Create tweet object
                tweets.append({
                    "text": tweet_text,
                    "sentiment": sentiment,
                    "likes": random.randint(0, 100),
                    "retweets": random.randint(0, 30),
                    "timestamp": datetime.now() - timedelta(hours=random.randint(1, 72))
                })
            
            # Sort by timestamp (most recent first)
            tweets.sort(key=lambda x: x["timestamp"], reverse=True)
            
            # Calculate overall sentiment
            sentiment_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
            for tweet in tweets:
                sentiment_counts[tweet["sentiment"]] += 1
            
            # Calculate average sentiment score (-1 to 1)
            total_tweets = len(tweets)
            sentiment_score = (sentiment_counts["Positive"] - sentiment_counts["Negative"]) / total_tweets
            
            return {
                "tweets": tweets,
                "sentiment_counts": sentiment_counts,
                "sentiment_score": sentiment_score
            }
        
        twitter_data = get_simulated_twitter_data(stock_symbol)
        
        # Display sentiment distribution
        col1, col2 = st.columns(2)
        
        with col1:
            # Create pie chart
            sentiment_df = pd.DataFrame(list(twitter_data["sentiment_counts"].items()), 
                                       columns=["Sentiment", "Count"])
            
            fig = px.pie(
                sentiment_df, 
                values="Count", 
                names="Sentiment", 
                title=f"Twitter Sentiment for {stock_symbol}",
                color="Sentiment",
                color_discrete_map={
                    "Positive": "#4CAF50",
                    "Neutral": "#FFC107",
                    "Negative": "#F44336"
                },
                template="plotly_dark"
            )
            fig.update_layout(
                margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Display gauge chart for overall sentiment
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=twitter_data["sentiment_score"],
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Overall Twitter Sentiment"},
                gauge={
                    "axis": {"range": [-1, 1]},
                    "bar": {"color": "#1E88E5"},
                    "steps": [
                        {"range": [-1, -0.05], "color": "rgba(244, 67, 54, 0.3)"},
                        {"range": [-0.05, 0.05], "color": "rgba(255, 193, 7, 0.3)"},
                        {"range": [0.05, 1], "color": "rgba(76, 175, 80, 0.3)"}
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.75,
                        "value": twitter_data["sentiment_score"]
                    }
                }
            ))
            fig.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Display recent tweets
        st.markdown("<h3>Recent Tweets</h3>", unsafe_allow_html=True)
        
        for i, tweet in enumerate(twitter_data["tweets"][:5]):
            # Determine color based on sentiment
            if tweet["sentiment"] == "Positive":
                sentiment_color = "#4CAF50"
            elif tweet["sentiment"] == "Neutral":
                sentiment_color = "#FFC107"
            else:
                sentiment_color = "#F44336"
            
            # Format timestamp
            time_ago = (datetime.now() - tweet["timestamp"]).total_seconds() / 3600
            time_str = f"{int(time_ago)}h ago" if time_ago >= 1 else f"{int(time_ago * 60)}m ago"
            
            # Display tweet
            st.markdown(f"""
            <div style="border: 1px solid var(--border-color); border-radius: 10px; padding: 15px; margin-bottom: 10px; background-color: var(--background-medium);">
                <p>{tweet["text"]}</p>
                <p style="color: var(--text-light);">{time_str} • {tweet["likes"]} likes • {tweet["retweets"]} retweets</p>
            </div>
            """, unsafe_allow_html=True)
    
    # News Articles Section
    st.markdown("<h2 class='sub-header'>News Sentiment Analysis</h2>", unsafe_allow_html=True)
    
    # Fetch real news data from the API
    news_data = fetch_news_articles(stock_symbol)
    
    # Display sentiment distribution
    col1, col2 = st.columns(2)
    
    with col1:
        # Create pie chart
        sentiment_df = pd.DataFrame(list(news_data["sentiment_counts"].items()), 
                                   columns=["Sentiment", "Count"])
        
        fig = px.pie(
            sentiment_df, 
            values="Count", 
            names="Sentiment", 
            title=f"News Sentiment for {stock_symbol}",
            color="Sentiment",
            color_discrete_map={
                "Positive": "#4CAF50",
                "Neutral": "#FFC107",
                "Negative": "#F44336"
            },
            template="plotly_dark"
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Display sentiment trend
        fig = px.line(
            news_data["sentiment_trend"], 
            x="Date", 
            y="Sentiment",
            title=f"News Sentiment Trend for {stock_symbol}",
            labels={"Date": "Date", "Sentiment": "Sentiment Score"},
            template="plotly_dark"
        )
        
        # Add horizontal lines for reference
        fig.add_shape(
            type="line",
            x0=news_data["sentiment_trend"]["Date"].min(),
            x1=news_data["sentiment_trend"]["Date"].max(),
            y0=0,
            y1=0,
            line=dict(color="gray", width=1, dash="dash")
        )
        
        fig.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Display recent news articles
    st.markdown("<h3>Recent News Articles</h3>", unsafe_allow_html=True)
    
    for i, article in enumerate(news_data["articles"][:5]):
        # Determine color based on sentiment
        if article["sentiment"] == "Positive":
            sentiment_color = "#4CAF50"
        elif article["sentiment"] == "Neutral":
            sentiment_color = "#FFC107"
        else:
            sentiment_color = "#F44336"
        
        # Format date
        date_str = article["date"].strftime("%b %d, %Y")
        
        # Display article
        st.markdown(f"""
        <div style="border: 1px solid var(--border-color); border-radius: 10px; padding: 15px; margin-bottom: 10px; background-color: var(--background-medium);">
            <h4>{article["headline"]}</h4>
            <p style="color: var(--text-light);">{article["source"]} • {date_str}</p>
            <p style="color: {sentiment_color}; font-weight: bold;">Sentiment: {article["sentiment"]} ({article["sentiment_score"]:.2f})</p>
        </div>
        """, unsafe_allow_html=True)