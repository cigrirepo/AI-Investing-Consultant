 app.py  ── Senzu Financial Insights Dashboard (Enhanced Version)
# ================================================
import os
import sqlite3
import asyncio
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import time

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from textblob import TextBlob
import requests
import openai
from openai import OpenAI
import streamlit as st
from PIL import Image
import ta  # For technical indicators
import pdfkit  # For PDF exports
from io import BytesIO
import xlsxwriter

# ── session defaults ────────────────────────────────────────────────
if "loaded" not in st.session_state:
    st.session_state.loaded = False
    st.session_state.info   = None
    st.session_state.fin    = None
    st.session_state.hist   = None
    st.session_state.last_refreshed = None
    st.session_state.ticker = "AAPL"
    st.session_state.settings = {
        "time_period": "5y",
        "peer_count": 5,
        "forecast_days": 30,
        "refresh_interval_minutes": 60
    }

# ── Streamlit Page Config ──────────────────────────────────────────
st.set_page_config(page_title="Senzu Financial Insights", layout="wide", initial_sidebar_state="expanded")

# ── Top-Left Logo + Branding Bar ─────────────────────────────────────
logo_path = Path(__file__).parent / "senzu_logo.png"
# Give even more weight to the logo column
col_logo, col_brand, col_refresh = st.columns([3, 9, 3], gap="small")

with col_logo:
    if logo_path.exists():
        # Make the logo really pop
        st.image(str(logo_path), width=200)
    else:
        st.warning("⚠️ senzu_logo.png not found in repo")

with col_brand:
    st.markdown("""
    <h1 style='margin-bottom: 0px; padding-bottom: 0px;'>Senzu Financial Insights</h1>
    <p style='margin-top: 0px; padding-top: 0px; color: #888;'>Powered by AI - Deep Financial Analysis & Forecasting</p>
    """, unsafe_allow_html=True)

with col_refresh:
    if st.session_state.last_refreshed:
        st.markdown(f"**Last updated:** {st.session_state.last_refreshed.strftime('%H:%M:%S')}")
    auto_refresh = st.checkbox("Auto-refresh data", value=False, 
                              help=f"Auto-refresh every {st.session_state.settings['refresh_interval_minutes']} minutes")

# ── Keys / Secrets ─────────────────────────────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY", "")
news_api_key = os.getenv("NEWS_API_KEY", "166012e1c17248b8b0ff75d114420a72")
alpha_vantage_key = os.getenv("ALPHA_VANTAGE_KEY", "")  # For economic data

if not openai.api_key:
    st.sidebar.warning("⚠️ OpenAI API key not found. Set OPENAI_API_KEY env variable.")

# ── SQLite Watchlist ───────────────────────────────────────────────
conn = sqlite3.connect("watchlist.db", check_same_thread=False)
cur = conn.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS watchlist(ticker TEXT PRIMARY KEY, added_date TEXT)")
conn.commit()

def add_to_watchlist(ticker: str) -> None:
    """Add ticker to watchlist with current date"""
    today = datetime.now().strftime('%Y-%m-%d')
    cur.execute("INSERT OR REPLACE INTO watchlist VALUES (?, ?)", 
               (ticker.upper(), today))
    conn.commit()

def remove_from_watchlist(ticker: str) -> None:
    """Remove ticker from watchlist"""
    cur.execute("DELETE FROM watchlist WHERE ticker = ?", (ticker.upper(),))
    conn.commit()

def get_watchlist() -> List[str]:
    """Get watchlist with added dates"""
    rows = cur.execute("SELECT ticker FROM watchlist ORDER BY ticker").fetchall()
    return [r[0] for r in rows]

# ── API Rate Limiting Helper ───────────────────────────────────────
class RateLimiter:
    """Helper class to manage API rate limits with exponential backoff"""
    def __init__(self, max_calls: int, time_period: int):
        self.max_calls = max_calls
        self.time_period = time_period
        self.calls = []
        
    def wait_if_needed(self):
        """Wait if we've hit rate limits"""
        now = datetime.now()
        # Remove old calls
        self.calls = [t for t in self.calls if now - t < timedelta(seconds=self.time_period)]
        
        if len(self.calls) >= self.max_calls:
            sleep_time = (self.calls[0] + timedelta(seconds=self.time_period) - now).total_seconds()
            if sleep_time > 0:
                time.sleep(sleep_time)
                
        self.calls.append(datetime.now())

# Initialize rate limiters
yf_limiter = RateLimiter(max_calls=5, time_period=60)  # Adjust based on actual limits
news_limiter = RateLimiter(max_calls=10, time_period=60)

# ── Data Loader (cached) ───────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=60*30)  # 30-minute cache
def get_company_data(ticker: str, period: str="5y") -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Get company data with exponential backoff for API failures"""
    max_retries = 3
    backoff = 2
    
    for attempt in range(max_retries):
        try:
            yf_limiter.wait_if_needed()
            stock = yf.Ticker(ticker)
            info = stock.info
            fin = stock.quarterly_financials.T.copy() if hasattr(stock, 'quarterly_financials') else pd.DataFrame()
            hist = stock.history(period=period)
            
            # Calculate financial metrics if data exists
            if not fin.empty and "Total Revenue" in fin.columns:
                fin["YoY Rev %"] = fin["Total Revenue"].pct_change(4) * 100
                fin["QoQ Rev %"] = fin["Total Revenue"].pct_change(1) * 100
                fin["Gross Margin %"] = fin["Gross Profit"] / fin["Total Revenue"] * 100
                fin["Op Margin %"] = fin["Operating Income"] / fin["Total Revenue"] * 100
                if "Free Cash Flow" in fin.columns:
                    fin["FCF Margin %"] = fin["Free Cash Flow"] / fin["Total Revenue"] * 100
                # Add new advanced metrics
                if "Net Income" in fin.columns and "Total Assets" in fin.columns:
                    fin["ROA %"] = fin["Net Income"] / fin["Total Assets"] * 100
                if "Long Term Debt" in fin.columns and "Total Stockholder Equity" in fin.columns:
                    fin["Debt-to-Equity"] = fin["Long Term Debt"] / fin["Total Stockholder Equity"]
            
            # Add technical indicators to historical data
            if not hist.empty:
                # RSI (Relative Strength Index)
                hist['RSI'] = ta.momentum.RSIIndicator(hist['Close']).rsi()
                # MACD (Moving Average Convergence Divergence)
                macd = ta.trend.MACD(hist['Close'])
                hist['MACD'] = macd.macd()
                hist['MACD_Signal'] = macd.macd_signal()
                # Bollinger Bands
                bollinger = ta.volatility.BollingerBands(hist['Close'])
                hist['BB_Upper'] = bollinger.bollinger_hband()
                hist['BB_Lower'] = bollinger.bollinger_lband()
                # Calculate traditional SMAs
                hist["50SMA"] = hist["Close"].rolling(50).mean()
                hist["200SMA"] = hist["Close"].rolling(200).mean()
                
            return info, fin, hist
            
        except Exception as e:
            if attempt < max_retries - 1:
                st.warning(f"Retry {attempt+1}/{max_retries}: {e}")
                time.sleep(backoff ** attempt)  # Exponential backoff
            else:
                raise Exception(f"Failed to fetch data after {max_retries} attempts: {e}")

# ── Parallel Data Fetcher ─────────────────────────────────────────
def fetch_data_parallel(tickers: List[str]) -> Dict[str, dict]:
    """Fetch data for multiple tickers in parallel"""
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_ticker = {
            executor.submit(yf.Ticker, ticker): ticker for ticker in tickers
        }
        
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                stock = future.result()
                results[ticker] = {
                    "info": stock.info,
                    "hist": stock.history(period="1y")
                }
            except Exception as e:
                results[ticker] = {"error": str(e)}
    
    return results

# ── LLM Investment Thesis ──────────────────────────────────────────
@st.cache_data(ttl=24*60*60)  # Cache for 1 day
def generate_investment_thesis(ticker: str, info: dict, fin_df: pd.DataFrame = None) -> str:
    """Generate an investment thesis with more financial context"""
    if not openai.api_key:
        return "⚠️ OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
    
    # Extract additional financial metrics
    revenue_growth = "N/A"
    margins = "N/A"
    
    if fin_df is not None and not fin_df.empty and "YoY Rev %" in fin_df.columns:
        recent_growth = fin_df["YoY Rev %"].dropna().iloc[:4].tolist()
        if recent_growth:
            growth_txt = ", ".join([f"{g:.1f}%" for g in recent_growth])
            revenue_growth = f"Recent YoY Revenue Growth: {growth_txt}"
        
        if "Gross Margin %" in fin_df.columns and "Op Margin %" in fin_df.columns:
            gm = fin_df["Gross Margin %"].iloc[0] if not fin_df["Gross Margin %"].empty else "N/A"
            om = fin_df["Op Margin %"].iloc[0] if not fin_df["Op Margin %"].empty else "N/A"
            margins = f"Gross Margin: {gm:.1f}%, Operating Margin: {om:.1f}%"
    
    client = OpenAI(api_key=openai.api_key)
    prompt = f"""
You are a financial analyst. Write a concise 2-paragraph investment thesis for {ticker},
incorporating current revenue, margins, and valuation metrics.
Follow with 3 quantitative bullet points and a text matrix of recent YoY revenue growth.

Business Summary: {info.get('longBusinessSummary', 'N/A')}
Market Cap: {info.get('marketCap', 'N/A')}
Revenue (TTM): {info.get('totalRevenue', 'N/A')}
EBITDA: {info.get('ebitda', 'N/A')}
EBITDA Margin: {round((info.get('ebitda') or 0)/(info.get('totalRevenue') or 1)*100,2)}%
Trailing P/E: {info.get('trailingPE', 'N/A')}
Forward P/E: {info.get('forwardPE', 'N/A')}
EPS (TTM): {info.get('trailingEps', 'N/A')}
PEG Ratio: {info.get('pegRatio', 'N/A')}
Price/Book: {info.get('priceToBook', 'N/A')}
{revenue_growth}
{margins}
Sector: {info.get('sector', 'N/A')} | Industry: {info.get('industry', 'N/A')}

In your analysis:
1. Evaluate the company's competitive position and moat
2. Assess valuation in context of growth rate and industry peers
3. Highlight risks and potential catalysts
4. Include conservative price targets if appropriate
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4-turbo",  # Use the most recent model for better analysis
            messages=[
                {"role":"system","content":"You are a seasoned investment analyst focused on fundamental analysis. You provide balanced views with both bull and bear perspectives."},
                {"role":"user",  "content":prompt}
            ],
            temperature=0.7,
            max_tokens=800  # Allow for more detailed analysis
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Failed to generate investment thesis: {str(e)}"

# ── Plot Helpers with Enhanced Data Visualization ─────────────────────
def plot_revenue_and_growth(fin_df: pd.DataFrame) -> None:
    """Plot revenue and growth with enhanced visuals"""
    if fin_df.empty or "Total Revenue" not in fin_df.columns:
        st.warning("No revenue data available for this company")
        return
        
    df = fin_df[["Total Revenue"]].dropna().copy()
    if isinstance(df.index, pd.PeriodIndex):
        df.index = df.index.to_timestamp(how="end")
    df = df.sort_index()
    df["Quarter"] = df.index.to_period("Q").astype(str)
    df["YoY Rev %"] = df["Total Revenue"].pct_change(4) * 100

    # Highlight positive/negative growth
    colors = ["#76b852" if val >= 0 else "#ff6b6b" for val in df["YoY Rev %"].fillna(0)]

    fig = px.bar(
        df, x="Quarter", y="Total Revenue",
        labels={"Total Revenue":"Revenue ($)"},
        title="Quarterly Revenue & YoY Growth",
        template="plotly_white"  # Cleaner template
    )
    mask = df["YoY Rev %"].notna()
    fig.add_trace(go.Scatter(
        x=df["Quarter"][mask],
        y=df["YoY Rev %"][mask],
        name="YoY Rev %",
        mode="lines+markers",
        yaxis="y2",
        line=dict(color="#FFA500", width=3),
        marker=dict(size=8, color=colors, line=dict(width=2, color="#FFA500"))
    ))
    fig.update_layout(
        yaxis2=dict(overlaying="y", side="right", title="YoY %", tickformat=".1f", ticksuffix="%", showgrid=False),
        bargap=0.25, 
        xaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_stock_price_with_technicals(hist_df: pd.DataFrame, indicators: List[str] = None) -> None:
    """Plot stock price with selected technical indicators"""
    if hist_df.empty:
        st.warning("No historical price data available")
        return
        
    if indicators is None:
        indicators = ["50SMA", "200SMA"]  # Default indicators
        
    df = hist_df.copy()
    
    # Create figure with secondary y axis for volume
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="Price",
        increasing_line=dict(color='#76b852'),
        decreasing_line=dict(color='#ff6b6b')
    ))
    
    # Add selected indicators
    colors = {
        "50SMA": "#1f77b4",
        "200SMA": "#d62728",
        "RSI": "#9467bd",
        "MACD": "#2ca02c",
        "MACD_Signal": "#ff7f0e",
        "BB_Upper": "#8c564b",
        "BB_Lower": "#8c564b"
    }
    
    for indicator in indicators:
        if indicator in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[indicator],
                name=indicator,
                line=dict(color=colors.get(indicator, "#444"))
            ))
    
    # Add volume as bar chart on secondary y-axis
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name="Volume",
        yaxis="y2",
        marker=dict(color='rgba(100, 100, 100, 0.3)')
    ))
    
    # Update layout for secondary y-axis
    fig.update_layout(
        title="Price Chart with Technical Indicators",
        yaxis_title="Price",
        yaxis2=dict(
            title="Volume",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        xaxis_rangeslider_visible=False,  # Disable rangeslider for cleaner look
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ── Dynamic Peer Comparables with Enhanced Metrics ────────────────────
SP500_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
FALLBACK_PEERS = ["AAPL","MSFT","GOOGL","AMZN","META"]

@st.cache_data(ttl=24*60*60, show_spinner=False)  # Cache for 1 day
def get_sp500() -> pd.DataFrame:
    """Get S&P 500 constituents with improved error handling"""
    try:
        df = pd.read_csv(SP500_URL)
        df.columns = df.columns.str.title()
        return df
    except Exception as e:
        st.warning(f"Failed to fetch S&P 500 data: {e}")
        return pd.DataFrame()

def show_peer_comparison(focus_ticker: str, n_peers: int=5) -> None:
    """Show enhanced peer comparison with more metrics"""
    try:
        ticker_info = yf.Ticker(focus_ticker).info
        sector = ticker_info.get("sector")
        industry = ticker_info.get("industry")
    except Exception as e:
        st.error(f"Error fetching sector information: {e}")
        sector, industry = None, None
    
    # First try to match by industry (more specific), then by sector
    sp_df = get_sp500()
    if not sp_df.empty and {"Symbol", "Sector", "Industry"}.issubset(sp_df.columns):
        if industry and len(sp_df[sp_df["Industry"] == industry]) >= n_peers:
            peers = sp_df.loc[sp_df["Industry"] == industry, "Symbol"].tolist()
        elif sector and len(sp_df[sp_df["Sector"] == sector]) >= n_peers:
            peers = sp_df.loc[sp_df["Sector"] == sector, "Symbol"].tolist()
        else:
            peers = FALLBACK_PEERS
        
        peers = [t for t in peers if t != focus_ticker][:n_peers]
    else:
        peers = FALLBACK_PEERS[:n_peers]
    
    # Fetch peer data in parallel for better performance
    ticker_data = fetch_data_parallel([focus_ticker] + peers)
    
    rows = []
    for t in [focus_ticker] + peers:
        try:
            if t in ticker_data and "info" in ticker_data[t]:
                inf = ticker_data[t]["info"]
                mc, ebt = inf.get("marketCap", 0), inf.get("ebitda", 0)
                rev = inf.get("totalRevenue", 0)
                
                # Enhanced metrics
                rows.append({
                    "Ticker": t,
                    "Company": inf.get("shortName", t),
                    "Market Cap ($B)": f"{mc/1e9:,.2f}" if mc else "N/A",
                    "P/E (TTM)": f"{inf.get('trailingPE', 0):.2f}" if inf.get("trailingPE") else "N/A",
                    "P/E (FWD)": f"{inf.get('forwardPE', 0):.2f}" if inf.get("forwardPE") else "N/A",
                    "EV/EBITDA": f"{inf.get('enterpriseValue', 0)/ebt:,.2f}" if ebt and inf.get("enterpriseValue") else "N/A",
                    "P/S": f"{mc/rev:,.2f}" if mc and rev else "N/A",
                    "P/B": f"{inf.get('priceToBook', 0):.2f}" if inf.get("priceToBook") else "N/A",
                    "Div Yield (%)": f"{inf.get('dividendYield', 0)*100:.2f}" if inf.get("dividendYield") else "N/A",
                    "ROE (%)": f"{inf.get('returnOnEquity', 0)*100:.2f}" if inf.get("returnOnEquity") else "N/A"
                })
            else:
                rows.append({"Ticker": t, "Company": t, "Error": "Failed to fetch data"})
        except Exception as e:
            rows.append({"Ticker": t, "Company": t, "Error": str(e)})
    
    # Create DataFrame and highlight focus ticker
    df = pd.DataFrame(rows)
    
    # Create a list to highlight the focus ticker row
    highlight = ['background-color: #e8f4f8' if t == focus_ticker else '' for t in df['Ticker']]
    
    # Apply styling
    styled_df = df.style.apply(lambda _: ['background-color: #e8f4f8' if x == focus_ticker else '' 
                                        for x in df['Ticker']], axis=0)
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Visualization of key metrics
    if len(rows) > 1:
        # Extract metrics for visualization
        viz_data = []
        metrics = ["P/E (TTM)", "EV/EBITDA", "P/S", "P/B"]
        
        for row in rows:
            for metric in metrics:
                if metric in row and row[metric] != "N/A":
                    try:
                        value = float(row[metric].replace(",", ""))
                        viz_data.append({
                            "Ticker": row["Ticker"],
                            "Metric": metric,
                            "Value": value
                        })
                    except:
                        pass
        
        if viz_data:
            viz_df = pd.DataFrame(viz_data)
            fig = px.bar(
                viz_df, 
                x="Ticker", 
                y="Value", 
                color="Ticker",
                facet_col="Metric",
                facet_col_wrap=2,
                title="Peer Comparison - Key Valuation Metrics",
                template="plotly_white"
            )
            fig.update_layout(showlegend=False, height=500)
            st.plotly_chart(fig, use_container_width=True)

# ── Enhanced Prophet Forecast ───────────────────────────────────────
def forecast_stock_price(hist_df: pd.DataFrame, days: int = 30) -> None:
    """Create an enhanced price forecast with confidence intervals"""
    if hist_df.empty:
        st.warning("Not enough historical data for forecasting")
        return
        
    # Use Prophet for price forecasting
    df = hist_df.reset_index()[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
    
    # Create model with uncertainty intervals
    m = Prophet(
        daily_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,  # Flexibility of the trend
        uncertainty_samples=1000  # For better confidence intervals
    )
    
    # Add weekly seasonality explicitly
    m.add_seasonality(name='weekly', period=7, fourier_order=3)
    
    try:
        m.fit(df)
        future = m.make_future_dataframe(periods=days)
        fc = m.predict(future)
        
        # Create custom plot with historical data and forecast
        fig = go.Figure()
        
        # Add historical prices
        fig.add_trace(go.Scatter(
            x=df["ds"],
            y=df["y"],
            name="Historical",
            line=dict(color="navy")
        ))
        
        # Add forecast
        mask = fc["ds"] > df["ds"].iloc[-1]
        fig.add_trace(go.Scatter(
            x=fc["ds"][mask],
            y=fc["yhat"][mask],
            name="Forecast",
            line=dict(color="royalblue")
        ))
        
        # Add upper and lower bounds
        fig.add_trace(go.Scatter(
            x=fc["ds"][mask],
            y=fc["yhat_upper"][mask],
            name="Upper 95% CI",
            line=dict(width=0),
            marker=dict(color="#444"),
            showlegend=True
        ))
        
        fig.add_trace(go.Scatter(
            x=fc["ds"][mask],
            y=fc["yhat_lower"][mask],
            name="Lower 95% CI",
            line=dict(width=0),
            marker=dict(color="#444"),
            fillcolor="rgba(68, 68, 68, 0.2)",
            fill="tonexty",
            showlegend=True
        ))
        
        # Layout
        fig.update_layout(
            title=f"{days}-Day Price Forecast with 95% Confidence Intervals",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show components if desired
        if st.checkbox("Show forecast components", value=False):
            fig_comp = m.plot_components(fc)
            st.pyplot(fig_comp)
            
    except Exception as e:
        st.error(f"Forecasting error: {e}")
        st.info("Prophet requires sufficient historical data to make forecasts. Try a different stock with longer history.")

# ── Enhanced News Sentiment ─────────────────────────────────────────
@st.cache_data(ttl=60*60, show_spinner=False)  # Cache for 1 hour
def get_news_sentiment(ticker: str, limit: int = 10) -> pd.DataFrame:
    """Get news with enhanced sentiment analysis and visualization"""
    try:
        news_limiter.wait_if_needed()
        response = requests.get(
            f"https://newsapi.org/v2/everything?q={ticker}+stock&language=en&sortBy=publishedAt&apiKey={news_api_key}"
        )
        
        if response.status_code != 200:
            st.warning(f"News API error: {response.status_code} - {response.text}")
            return pd.DataFrame()
            
        arts = response.json().get("articles", [])[:limit]
        
        if not arts:
            return pd.DataFrame()
            
        rows = []
        total_polarity = 0
        
        for art in arts:
            title = art.get("title", "")
            content = art.get("description", "")
            combined = f"{title}. {content}"
            
            # Use TextBlob for sentiment
            analysis = TextBlob(combined)
            pol = analysis.sentiment.polarity
            subj = analysis.sentiment.subjectivity
            
            # Categorize sentiment
            if pol > 0.2:
                sentiment = "Positive"
            elif pol < -0.2:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
                
            # Track overall sentiment
            total_polarity += pol
                
            rows.append({
                "Date": art.get("publishedAt", "")[:10],
                "Headline": title,
                "Source": art.get("source", {}).get("name", "Unknown"),
                "URL": art.get("url", ""),
                "Polarity": round(pol, 2),
                "Subjectivity": round(subj, 2),
                "Sentiment": sentiment
            })
        
        df = pd.DataFrame(rows)
        if not df.empty:
            # Add calculated overall sentiment score
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date", ascending=False)
            
        return df
        
    except Exception as e:
        st.error(f"News sentiment error: {e}")
        return pd.DataFrame()

def visualize_sentiment(sentiment_df: pd.DataFrame) -> None:
    """Create visualizations for sentiment analysis"""
    if sentiment_df.empty:
        st.warning("No news data available for sentiment visualization")
        return
        
    # Overall sentiment score
    overall_sentiment = sentiment_df["Polarity"].mean()
    
    # Create sentiment gauge chart with better color gradient and layout
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=overall_sentiment,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Average Sentiment Score"},
        gauge={
            'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [-1, -0.5], 'color': '#ff4d4d'},
                {'range': [-0.5, -0.2], 'color': '#ffcccc'},
                {'range': [-0.2, 0.2], 'color': '#f2f2f2'},
                {'range': [0.2, 0.5], 'color': '#ccffcc'},
                {'range': [0.5, 1], 'color': '#66ff66'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': overall_sentiment
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sentiment distribution
        sentiment_counts = sentiment_df['Sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        
        # Define colors for sentiment categories
        colors = {'Positive': '#66ff66', 'Neutral': '#f2f2f2', 'Negative': '#ff4d4d'}
        
        pie_fig = px.pie(
            sentiment_counts, 
            values='Count', 
            names='Sentiment',
            color='Sentiment',
            color_discrete_map=colors,
            title="News Sentiment Distribution"
        )
        pie_fig.update_traces(textposition='inside', textinfo='percent+label')
        pie_fig.update_layout(height=250)
        st.plotly_chart(pie_fig, use_container_width=True)
    
    # Sentiment over time with enhanced visualization
    sentiment_by_date = sentiment_df.groupby(pd.Grouper(key='Date', freq='D')).agg({
        'Polarity': 'mean',
        'Headline': 'count'
    }).reset_index()
    sentiment_by_date.columns = ['Date', 'Avg_Sentiment', 'Article_Count']
    
    # Create two y-axis plot
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add sentiment line
    fig.add_trace(
        go.Scatter(
            x=sentiment_by_date['Date'],
            y=sentiment_by_date['Avg_Sentiment'],
            name="Average Sentiment",
            line=dict(color='royalblue', width=3)
        ),
        secondary_y=False,
    )
    
    # Add article count bars
    fig.add_trace(
        go.Bar(
            x=sentiment_by_date['Date'],
            y=sentiment_by_date['Article_Count'],
            name="Article Count",
            marker_color='lightgray',
            opacity=0.7
        ),
        secondary_y=True,
    )
    
    # Set titles
    fig.update_layout(
        title_text="News Sentiment Trend",
        template="plotly_white"
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Sentiment Score", secondary_y=False)
    fig.update_yaxes(title_text="Article Count", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display news table with clickable links and sentiment highlighting
    if not sentiment_df.empty:
        st.subheader("Recent News Articles")
        
        # Apply color highlighting based on sentiment
        def highlight_sentiment(val):
            if val == 'Positive':
                return 'background-color: rgba(102, 255, 102, 0.2)'
            elif val == 'Negative':
                return 'background-color: rgba(255, 77, 77, 0.2)'
            return ''
        
        # Create clickable links
        sentiment_df['Headline'] = sentiment_df.apply(
            lambda row: f"<a href='{row['URL']}' target='_blank'>{row['Headline']}</a>", 
            axis=1
        )
        
        # Select and order columns for display
        display_df = sentiment_df[['Date', 'Headline', 'Source', 'Sentiment', 'Polarity']].copy()
        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
        
        # Apply styling
        styled_df = display_df.style.applymap(highlight_sentiment, subset=['Sentiment'])
        
        # Display with HTML rendering for links
        st.write(styled_df.to_html(escape=False), unsafe_allow_html=True)

# ── Export Functionality Enhancement ─────────────────────────────────
def export_to_excel(ticker: str, info: dict, hist_df: pd.DataFrame, fin_df: pd.DataFrame = None):
    """Export data to formatted Excel file"""
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    
    # Create summary sheet
    summary_data = pd.DataFrame({
        'Metric': [
            'Company Name', 'Sector', 'Industry', 'Market Cap', 'P/E (TTM)',
            'P/E (FWD)', 'EPS (TTM)', 'PEG Ratio', '52W High', '52W Low',
            'Dividend Yield', 'Beta', 'Current Price', 'Target Price', 'Analyst Rating'
        ],
        'Value': [
            info.get('shortName', 'N/A'),
            info.get('sector', 'N/A'),
            info.get('industry', 'N/A'),
            f"${info.get('marketCap', 0)/1e9:.2f}B" if info.get('marketCap') else 'N/A',
            f"{info.get('trailingPE', 0):.2f}" if info.get('trailingPE') else 'N/A',
            f"{info.get('forwardPE', 0):.2f}" if info.get('forwardPE') else 'N/A',
            f"${info.get('trailingEps', 0):.2f}" if info.get('trailingEps') else 'N/A',
            f"{info.get('pegRatio', 0):.2f}" if info.get('pegRatio') else 'N/A',
            f"${info.get('fiftyTwoWeekHigh', 0):.2f}" if info.get('fiftyTwoWeekHigh') else 'N/A',
            f"${info.get('fiftyTwoWeekLow', 0):.2f}" if info.get('fiftyTwoWeekLow') else 'N/A',
            f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else 'N/A',
            f"{info.get('beta', 0):.2f}" if info.get('beta') else 'N/A',
            f"${info.get('currentPrice', 0):.2f}" if info.get('currentPrice') else 'N/A',
            f"${info.get('targetMeanPrice', 0):.2f}" if info.get('targetMeanPrice') else 'N/A',
            info.get('recommendationKey', 'N/A').upper()
        ]
    })
    
    summary_data.to_excel(writer, sheet_name='Summary', index=False)
    
    # Format summary sheet
    workbook = writer.book
    summary_sheet = writer.sheets['Summary']
    
    # Add formatting
    header_format = workbook.add_format({
        'bold': True,
        'text_wrap': True,
        'valign': 'top',
        'fg_color': '#D7E4BC',
        'border': 1
    })
    
    # Historical price data
    if not hist_df.empty:
        price_df = hist_df.reset_index()
        # Convert datetime to date string for Excel compatibility
        if 'Date' in price_df.columns and pd.api.types.is_datetime64_any_dtype(price_df['Date']):
            price_df['Date'] = price_df['Date'].dt.strftime('%Y-%m-%d')
        
        price_df.to_excel(writer, sheet_name='Historical Prices', index=False)
    
    # Financial data
    if fin_df is not None and not fin_df.empty:
        # Reset index to get date column
        fin_export_df = fin_df.reset_index()
        # Convert PeriodIndex to string if needed
        if 'index' in fin_export_df.columns and isinstance(fin_export_df['index'].iloc[0], pd.Period):
            fin_export_df['index'] = fin_export_df['index'].astype(str)
            fin_export_df.rename(columns={'index': 'Quarter'}, inplace=True)
        
        fin_export_df.to_excel(writer, sheet_name='Financials', index=False)
    
    # Close the writer and get the output
    writer.close()
    processed_data = output.getvalue()
    
    return processed_data

# ── PDF Report Generation ────────────────────────────────────────────
def generate_pdf_report(ticker: str, info: dict, hist_df: pd.DataFrame, fin_df: pd.DataFrame, thesis: str = None):
    """Generate PDF report with company analysis and charts"""
    try:
        # Create HTML content for PDF
        html_content = f"""
        <html>
        <head>
            <title>{ticker} - Financial Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #003366; color: white; padding: 20px; }}
                .section {{ margin: 20px 0; }}
                .data-table {{ border-collapse: collapse; width: 100%; }}
                .data-table td, .data-table th {{ border: 1px solid #ddd; padding: 8px; }}
                .data-table tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .data-table th {{ padding-top: 12px; padding-bottom: 12px; text-align: left; background-color: #4CAF50; color: white; }}
                .company-name {{ font-size: 24px; font-weight: bold; }}
                .company-details {{ font-size: 14px; }}
                .thesis {{ background-color: #f9f9f9; padding: 15px; border-left: 5px solid #003366; }}
            </style>
        </head>
        <body>
            <div class="header">
                <div class="company-name">{info.get('shortName', ticker)} ({ticker})</div>
                <div class="company-details">{info.get('sector', 'N/A')} | {info.get('industry', 'N/A')}</div>
            </div>
            
            <div class="section">
                <h2>Company Overview</h2>
                <p>{info.get('longBusinessSummary', 'No description available.')}</p>
            </div>
            
            <div class="section">
                <h2>Key Statistics</h2>
                <table class="data-table">
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Market Cap</td>
                        <td>${info.get('marketCap', 0)/1e9:.2f}B</td>
                        <td>P/E (TTM)</td>
                        <td>{info.get('trailingPE', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>P/E (FWD)</td>
                        <td>{info.get('forwardPE', 'N/A')}</td>
                        <td>EPS (TTM)</td>
                        <td>${info.get('trailingEps', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>Dividend Yield</td>
                        <td>{info.get('dividendYield', 0)*100:.2f}%</td>
                        <td>Beta</td>
                        <td>{info.get('beta', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>52W Range</td>
                        <td>${info.get('fiftyTwoWeekLow', 0):.2f} - ${info.get('fiftyTwoWeekHigh', 0):.2f}</td>
                        <td>Analyst Rating</td>
                        <td>{info.get('recommendationKey', 'N/A').upper()}</td>
                    </tr>
                </table>
            </div>
        """
        
        # Add investment thesis if available
        if thesis:
            html_content += f"""
            <div class="section">
                <h2>Investment Thesis</h2>
                <div class="thesis">
                    {thesis.replace('\n', '<br>')}
                </div>
            </div>
            """
        
        # Close HTML document
        html_content += """
        </body>
        </html>
        """
        
        # Create PDF from HTML
        pdf_content = pdfkit.from_string(html_content, False)
        return pdf_content
        
    except Exception as e:
        st.error(f"PDF generation error: {e}")
        return None

# ── Enhanced Economic Data Dashboard ────────────────────────────────
@st.cache_data(ttl=24*60*60)  # Cache for a day
def get_economic_indicators():
    """Fetch major economic indicators for context"""
    if not alpha_vantage_key:
        return None
        
    indicators = {
        "REAL_GDP": "Real GDP (Quarterly % Change)",
        "INFLATION": "CPI Annual Change (%)",
        "UNEMPLOYMENT": "Unemployment Rate (%)",
        "RETAIL_SALES": "Retail Sales MoM Change (%)",
        "FEDERAL_FUNDS_RATE": "Federal Funds Rate (%)"
    }
    
    results = {}
    
    try:
        for indicator_key, indicator_name in indicators.items():
            url = f"https://www.alphavantage.co/query?function={indicator_key}&interval=quarterly&apikey={alpha_vantage_key}"
            response = requests.get(url)
            data = response.json()
            
            if "data" in data:
                df = pd.DataFrame(data["data"])
                df["date"] = pd.to_datetime(df["date"])
                df["value"] = pd.to_numeric(df["value"])
                df = df.sort_values("date")
                results[indicator_name] = df.copy()
                
            # Wait to avoid hitting rate limits
            time.sleep(0.5)
            
        return results
    except Exception as e:
        st.error(f"Error fetching economic data: {e}")
        return None

def show_economic_dashboard():
    """Display economic indicators dashboard"""
    econ_data = get_economic_indicators()
    
    if not econ_data:
        st.warning("Economic data not available. Please set ALPHA_VANTAGE_KEY in environment variables.")
        return
        
    st.subheader("Economic Indicators Dashboard")
    
    # Create multi-indicator visualization
    fig = go.Figure()
    
    for name, df in econ_data.items():
        if not df.empty:
            fig.add_trace(go.Scatter(
                x=df["date"].iloc[-20:],  # Last 20 periods
                y=df["value"].iloc[-20:],
                name=name,
                mode="lines+markers"
            ))
    
    fig.update_layout(
        title="Economic Indicators - Recent Trends",
        xaxis_title="Date",
        yaxis_title="Value (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display current values with visual indicators
    st.subheader("Current Economic Snapshot")
    
    cols = st.columns(len(econ_data))
    
    for i, (name, df) in enumerate(econ_data.items()):
        if not df.empty:
            current = df["value"].iloc[-1]
            previous = df["value"].iloc[-2] if len(df) > 1 else current
            delta = current - previous
            with cols[i]:
                st.metric(
                    name, 
                    f"{current:.2f}%",
                    f"{delta:.2f}%",
                    delta_color="inverse" if "UNEMPLOYMENT" in name or "INFLATION" in name else "normal"
                )

# ── Main App Layout with Enhanced UI ────────────────────────────────────
def main():
    # Sidebar with enhancements
    with st.sidebar:
        st.header("Settings & Controls")
        
        ticker_input = st.text_input("Enter Stock Ticker", value=st.session_state.ticker).upper()
        
        if ticker_input != st.session_state.ticker:
            st.session_state.ticker = ticker_input
            st.session_state.loaded = False
            
        # Settings in expandable section
        with st.expander("Dashboard Settings", expanded=False):
            time_period = st.selectbox(
                "Historical Data Period",
                options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
                index=4,
                help="Time period for historical data"
            )
            
            peer_count = st.slider(
                "Number of Peer Companies",
                min_value=3,
                max_value=10,
                value=st.session_state.settings["peer_count"],
                help="Number of peer companies to compare"
            )
            
            forecast_days = st.slider(
                "Forecast Days",
                min_value=7,
                max_value=90,
                value=st.session_state.settings["forecast_days"],
                help="Number of days to forecast"
            )
            
            refresh_interval = st.slider(
                "Auto-refresh Interval (minutes)",
                min_value=15,
                max_value=120,
                value=st.session_state.settings["refresh_interval_minutes"],
                step=15,
                help="Auto-refresh interval for data"
            )
            
            # Update settings
            st.session_state.settings.update({
                "time_period": time_period,
                "peer_count": peer_count,
                "forecast_days": forecast_days,
                "refresh_interval_minutes": refresh_interval
            })
        
        # Watchlist section
        st.subheader("Watchlist")
        watchlist = get_watchlist()
        
        if st.session_state.ticker.upper() not in watchlist:
            if st.button("➕ Add to Watchlist"):
                add_to_watchlist(st.session_state.ticker)
                st.success(f"{st.session_state.ticker} added to watchlist!")
                watchlist = get_watchlist()  # Refresh list
        else:
            if st.button("➖ Remove from Watchlist"):
                remove_from_watchlist(st.session_state.ticker)
                st.success(f"{st.session_state.ticker} removed from watchlist!")
                watchlist = get_watchlist()  # Refresh list
        
        # Display watchlist as clickable buttons
        if watchlist:
            st.write("Click to switch:")
            cols = st.columns(3)
            for i, stock in enumerate(watchlist):
                with cols[i % 3]:
                    if st.button(stock, key=f"watch_{stock}"):
                        st.session_state.ticker = stock
                        st.session_state.loaded = False
                        st.rerun()
        else:
            st.info("Your watchlist is empty. Add tickers to track them.")
            
        # Attribution
        st.divider()
        st.markdown("""
        <div style='text-align: center; color: #888;'>
        <small>Data from Yahoo Finance<br>
        Powered by Prophet & OpenAI<br>
        © 2025 Senzu Financial</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    if auto_refresh and st.session_state.last_refreshed:
        elapsed = datetime.now() - st.session_state.last_refreshed
        if elapsed > timedelta(minutes=st.session_state.settings["refresh_interval_minutes"]):
            st.session_state.loaded = False
            
    # Load data if needed
    if not st.session_state.loaded:
        with st.spinner(f"Loading data for {st.session_state.ticker}..."):
            try:
                info, fin, hist = get_company_data(
                    st.session_state.ticker, 
                    period=st.session_state.settings["time_period"]
                )
                st.session_state.info = info
                st.session_state.fin = fin
                st.session_state.hist = hist
                st.session_state.loaded = True
                st.session_state.last_refreshed = datetime.now()
            except Exception as e:
                st.error(f"Error loading data: {e}")
                return
    
    # Access loaded data
    info = st.session_state.info
    fin = st.session_state.fin
    hist = st.session_state.hist
    
    # Company header with enhanced design
    if "longName" in info:
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.subheader(f"{info['longName']} ({st.session_state.ticker})")
            st.caption(f"{info.get('sector', 'N/A')} | {info.get('industry', 'N/A')}")
        
        with col2:
            current_price = info.get('currentPrice', hist['Close'].iloc[-1] if not hist.empty else 0)
            prev_close = info.get('previousClose', hist['Close'].iloc[-2] if len(hist) > 1 else current_price)
            price_change = current_price - prev_close
            price_change_pct = (price_change / prev_close) * 100 if prev_close else 0
            
            # Display metrics with conditional formatting
            st.metric(
                "Current Price", 
                f"${current_price:.2f}", 
                f"{price_change:.2f} ({price_change_pct:.2f}%)"
            )
        
        with col3:
            # Market cap with proper formatting
            mc = info.get('marketCap', 0)
            if mc >= 1e12:
                mc_str = f"${mc/1e12:.2f}T"
            elif mc >= 1e9:
                mc_str = f"${mc/1e9:.2f}B"
            elif mc >= 1e6:
                mc_str = f"${mc/1e6:.2f}M"
            else:
                mc_str = f"${mc:.2f}"
                
            st.metric("Market Cap", mc_str)
        
        # Tabs for different analysis sections
        tabs = st.tabs([
            "📈 Overview", 
            "📊 Financials", 
            "🔎 Technical Analysis",
            "🌐 Peer Comparison",
            "🔮 Price Forecast",
            "📰 News & Sentiment",
            "📑 Reports"
        ])
        
        # Overview Tab
        with tabs[0]:
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Description with "Read more" expander
                st.subheader("Business Summary")
                desc = info.get('longBusinessSummary', 'No description available.')
                if len(desc) > 500:
                    st.write(f"{desc[:500]}...")
                    with st.expander("Read more"):
                        st.write(desc)
                else:
                    st.write(desc)
                
                # Key statistics in a clean table format
                st.subheader("Key Statistics")
                metrics = [
                    {"Metric": "P/E (TTM)", "Value": f"{info.get('trailingPE', 0):.2f}" if info.get('trailingPE') else "N/A"},
                    {"Metric": "P/E (FWD)", "Value": f"{info.get('forwardPE', 0):.2f}" if info.get('forwardPE') else "N/A"},
                    {"Metric": "PEG Ratio", "Value": f"{info.get('pegRatio', 0):.2f}" if info.get('pegRatio') else "N/A"},
                    {"Metric": "Price/Book", "Value": f"{info.get('priceToBook', 0):.2f}" if info.get('priceToBook') else "N/A"},
                    {"Metric": "EV/EBITDA", "Value": f"{info.get('enterpriseToEbitda', 0):.2f}" if info.get('enterpriseToEbitda') else "N/A"},
                    {"Metric": "Profit Margin", "Value": f"{info.get('profitMargins', 0)*100:.2f}%" if info.get('profitMargins') else "N/A"},
                    {"Metric": "Dividend Yield", "Value": f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else "N/A"},
                    {"Metric": "Beta", "Value": f"{info.get('beta', 0):.2f}" if info.get('beta') else "N/A"},
                    {"Metric": "52W High", "Value": f"${info.get('fiftyTwoWeekHigh', 0):.2f}" if info.get('fiftyTwoWeekHigh') else "N/A"},
                    {"Metric": "52W Low", "Value": f"${info.get('fiftyTwoWeekLow', 0):.2f}" if info.get('fiftyTwoWeekLow') else "N/A"}
                ]
                
                metrics_df = pd.DataFrame(metrics)
                
                # Display in 2 columns for better layout
                col_a, col_b = st.columns(2)
                col_a.dataframe(metrics_df.iloc[:5], hide_index=True, use_container_width=True)
                col_b.dataframe(metrics_df.iloc[5:], hide_index=True, use_container_width=True)
            
            with col2:
                # Investment thesis with AI
                st.subheader("AI Investment Thesis")
                
                thesis = generate_investment_thesis(st.session_state.ticker, info, fin)
                st.markdown(thesis)
                
                # Price chart preview
                if not hist.empty:
                    st.subheader("Price Chart (1 Year)")
                    fig = px.line(
                        hist.iloc[-252:], y='Close',  # Approx. 1 year of trading days
                        labels={'Close': 'Price ($)', 'Date': ''},
                        template="plotly_white"
                    )
                    fig.update_layout(
                        showlegend=False,
                        margin=dict(l=0, r=0, t=0, b=0),
                        height=250
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Financials Tab
        with tabs[1]:
            if fin is None or fin.empty:
                st.warning("No financial data available for this company")
            else:
                st.subheader("Quarterly Financial Performance")
                
                # Plot revenue and growth
                plot_revenue_and_growth(fin)
                
                # Display financial metrics with enhanced styling
                fin_metrics = ["Total Revenue", "Gross Profit", "Operating Income", "Net Income", "EPS"]
                if all(m in fin.columns for m in fin_metrics):
                    # Format columns for display
                    display_fin = fin[fin_metrics].copy()
                    
                    # Highlight table with gradient color based on QoQ growth
                    def highlight_growth(s):
                        if s.name == "Net Income":
                            is_profit = s > 0
                            return ['background-color: rgba(118, 184, 82, 0.2)' if v else 'background-color: rgba(255, 107, 107, 0.2)' for v in is_profit]
                        return [''] * len(s)
                    
                    # Format financial values (in millions)
                    for col in fin_metrics:
                        if col != "EPS":
                            display_fin[col] = display_fin[col] / 1e6
                    
                    st.subheader("Key Financial Metrics (in millions USD)")
                      # Format percentages
                            st.dataframe(
                                growth_metrics.style.format("{:.2f}%"), 
                                use_container_width=True
                            )
                
                # Balance sheet overview
                st.subheader("Balance Sheet Overview")
                
                balance_sheet_items = ["Total Assets", "Total Liabilities", "Total Stockholder Equity", "Cash", "Total Debt"]
                bs_metrics = [col for col in fin.columns if col in balance_sheet_items]
                
                if bs_metrics:
                    # Plot key balance sheet items
                    bs_data = fin[bs_metrics].copy() / 1e6  # Convert to millions
                    
                    fig = px.bar(
                        bs_data.reset_index().melt(id_vars='index', value_vars=bs_metrics),
                        x='index', y='value', color='variable',
                        title="Balance Sheet Trends",
                        labels={'index': 'Quarter', 'value': 'USD (Millions)', 'variable': 'Metric'},
                        template="plotly_white",
                        barmode='group'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate and display key ratios
                    with st.expander("Financial Ratios"):
                        if "Total Assets" in fin.columns and "Total Liabilities" in fin.columns:
                            fin["Debt-to-Assets"] = fin["Total Liabilities"] / fin["Total Assets"]
                        
                        if "Total Debt" in fin.columns and "Total Stockholder Equity" in fin.columns:
                            fin["Debt-to-Equity"] = fin["Total Debt"] / fin["Total Stockholder Equity"]
                        
                        if "Cash" in fin.columns and "Total Liabilities" in fin.columns:
                            fin["Cash-to-Debt"] = fin["Cash"] / fin["Total Liabilities"]
                        
                        ratio_cols = ["Debt-to-Assets", "Debt-to-Equity", "Cash-to-Debt"]
                        ratio_df = fin[[col for col in ratio_cols if col in fin.columns]].copy()
                        
                        if not ratio_df.empty:
                            st.dataframe(ratio_df.style.format("{:.2f}"), use_container_width=True)
        
        # Technical Analysis Tab
        with tabs[2]:
            if hist.empty:
                st.warning("No historical data available for technical analysis")
            else:
                st.subheader("Technical Analysis Dashboard")
                
                # Time period selector for technical analysis
                tech_timeframe = st.selectbox(
                    "Select Analysis Timeframe",
                    options=["1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "All Data"],
                    index=3
                )
                
                # Map selection to number of days
                timeframe_days = {
                    "1 Month": 21,
                    "3 Months": 63,
                    "6 Months": 126,
                    "1 Year": 252,
                    "2 Years": 504,
                    "All Data": len(hist)
                }
                
                # Filter data based on selected timeframe
                days = min(timeframe_days[tech_timeframe], len(hist))
                tech_data = hist.iloc[-days:].copy()
                
                # Add technical indicators
                tech_data = add_technical_indicators(tech_data)
                
                # Price with volume chart
                price_volume_chart(tech_data)
                
                # Technical indicators visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    # MACD Chart
                    create_macd_chart(tech_data)
                    
                with col2:
                    # RSI Chart
                    create_rsi_chart(tech_data)
                
                # Moving averages
                create_ma_chart(tech_data)
                
                # Support/Resistance levels
                support_resistance = calculate_support_resistance(tech_data)
                create_sr_chart(tech_data, support_resistance)
                
                # Signals table
                st.subheader("Technical Signals")
                signals = generate_technical_signals(tech_data)
                
                # Use more detailed signals with explanation
                signal_df = pd.DataFrame(signals)
                
                # Color-code signals
                def color_signal(val):
                    if val == "Bullish":
                        return "background-color: rgba(0, 255, 0, 0.2)"
                    elif val == "Bearish":
                        return "background-color: rgba(255, 0, 0, 0.2)"
                    elif val == "Neutral":
                        return "background-color: rgba(255, 255, 0, 0.1)"
                    return ""
                
                st.dataframe(
                    signal_df.style.applymap(color_signal, subset=["Signal"]),
                    use_container_width=True
                )
        
        # Peer Comparison Tab
        with tabs[3]:
            st.subheader("Peer Comparison Analysis")
            
            peers = get_peer_companies(st.session_state.ticker, info, count=st.session_state.settings["peer_count"])
            
            if not peers:
                st.warning("No peer companies found for comparison")
            else:
                # Create peer metrics dataframe with caching
                with st.spinner("Loading peer company data..."):
                    peer_metrics = get_peer_metrics([st.session_state.ticker] + peers)
                
                if peer_metrics is not None:
                    # Highlight the target company
                    def highlight_company(s):
                        return ['background-color: rgba(255, 255, 0, 0.2)' if s.name == st.session_state.ticker else '' for _ in s]
                    
                    # Apply formatting and show metrics
                    formatted_metrics = peer_metrics.copy()
                    for col in peer_metrics.columns:
                        if "P/E" in col or "Ratio" in col or "Beta" in col:
                            formatted_metrics[col] = formatted_metrics[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
                        elif "Market Cap" in col:
                            formatted_metrics[col] = formatted_metrics[col].apply(lambda x: f"${x/1e9:.2f}B" if pd.notnull(x) and x > 0 else "N/A")
                        elif "%" in col:
                            formatted_metrics[col] = formatted_metrics[col].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
                        elif "Price" in col:
                            formatted_metrics[col] = formatted_metrics[col].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "N/A")
                    
                    st.dataframe(
                        formatted_metrics.style.apply(highlight_company, axis=1),
                        height=min(len(peer_metrics) * 35 + 40, 400),
                        use_container_width=True
                    )
                    
                    # Create comparison charts
                    st.subheader("Comparative Analysis")
                    
                    # Valuation Metrics Chart
                    metric_to_plot = st.selectbox(
                        "Select Metric for Comparison",
                        options=[
                            "P/E Ratio", "Forward P/E", "PEG Ratio", "Price/Book",
                            "EV/EBITDA", "Profit Margin %", "Revenue Growth %", "Dividend Yield %"
                        ]
                    )
                    
                    # Custom color scheme with the target company highlighted
                    bar_colors = ['rgba(66, 135, 245, 0.8)' if company == st.session_state.ticker else 'rgba(192, 192, 192, 0.6)' 
                                 for company in peer_metrics.index]
                    
                    # Create bar chart for the selected metric
                    if metric_to_plot in peer_metrics.columns:
                        fig = go.Figure()
                        
                        fig.add_trace(go.Bar(
                            x=peer_metrics.index,
                            y=peer_metrics[metric_to_plot],
                            marker_color=bar_colors
                        ))
                        
                        # Add industry average line
                        avg_value = peer_metrics[metric_to_plot].mean()
                        fig.add_trace(go.Scatter(
                            x=[peer_metrics.index[0], peer_metrics.index[-1]],
                            y=[avg_value, avg_value],
                            mode='lines',
                            name='Industry Average',
                            line=dict(color='red', width=2, dash='dash')
                        ))
                        
                        fig.update_layout(
                            title=f"{metric_to_plot} Comparison",
                            xaxis_title="Company",
                            yaxis_title=metric_to_plot,
                            template="plotly_white",
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Radar chart for comprehensive comparison
                    with st.expander("Comprehensive Multi-Metric Comparison"):
                        radar_metrics = [
                            "P/E Ratio", "Forward P/E", "PEG Ratio", 
                            "Profit Margin %", "Revenue Growth %", "Dividend Yield %"
                        ]
                        
                        # Filter available metrics and companies with data
                        available_metrics = [m for m in radar_metrics if m in peer_metrics.columns]
                        
                        if available_metrics:
                            # Select companies for radar chart (limit to 5 for readability)
                            companies_to_plot = [st.session_state.ticker] + peers[:4]
                            
                            # Normalize data for radar chart
                            radar_data = peer_metrics.loc[companies_to_plot, available_metrics].copy()
                            
                            # Replace NaN with 0
                            radar_data = radar_data.fillna(0)
                            
                            # Create radar chart
                            fig = go.Figure()
                            
                            for company in radar_data.index:
                                fig.add_trace(go.Scatterpolar(
                                    r=radar_data.loc[company],
                                    theta=available_metrics,
                                    fill='toself',
                                    name=company
                                ))
                            
                            fig.update_layout(
                                polar=dict(
                                    radialaxis=dict(
                                        visible=True,
                                    )
                                ),
                                showlegend=True,
                                height=500
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
        
        # Price Forecast Tab
        with tabs[4]:
            st.subheader("Price Forecast Analysis")
            
            forecast_days = st.session_state.settings["forecast_days"]
            
            with st.spinner("Generating price forecast..."):
                forecast = generate_price_forecast(hist, forecast_days)
            
            if forecast is not None:
                # Plotting forecast
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=hist.index,
                    y=hist['Close'],
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue')
                ))
                
                # Forecast
                fig.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='green')
                ))
                
                # Upper and lower bounds
                fig.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat_upper'],
                    mode='lines',
                    fill=None,
                    line=dict(color='rgba(0, 255, 0, 0)'),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat_lower'],
                    mode='lines',
                    fill='tonexty',
                    fillcolor='rgba(0, 255, 0, 0.1)',
                    line=dict(color='rgba(0, 255, 0, 0)'),
                    name='Confidence Interval'
                ))
                
                fig.update_layout(
                    title=f"{st.session_state.ticker} Price Forecast ({forecast_days} Days)",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    hovermode="x unified",
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display forecast metrics
                col1, col2, col3 = st.columns(3)
                
                # Current price
                current_price = hist['Close'].iloc[-1]
                with col1:
                    st.metric("Current Price", f"${current_price:.2f}")
                
                # Forecasted price (end of period)
                forecast_end = forecast['yhat'].iloc[-1]
                forecast_change = forecast_end - current_price
                forecast_change_pct = (forecast_change / current_price) * 100
                
                with col2:
                    st.metric(
                        f"Forecasted Price ({forecast_days} days)", 
                        f"${forecast_end:.2f}",
                        f"{forecast_change:.2f} ({forecast_change_pct:.2f}%)"
                    )
                
                # Price range
                forecast_min = forecast['yhat_lower'].min()
                forecast_max = forecast['yhat_upper'].max()
                
                with col3:
                    st.metric(
                        "Forecasted Range", 
                        f"${forecast_min:.2f} - ${forecast_max:.2f}"
                    )
                
                # Show detailed forecast data in expander
                with st.expander("Detailed Forecast Data"):
                    # Format the forecast dataframe for display
                    display_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                    display_forecast.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
                    display_forecast['Date'] = display_forecast['Date'].dt.strftime('%Y-%m-%d')
                    
                    st.dataframe(
                        display_forecast.style.format({
                            'Forecast': '${:.2f}',
                            'Lower Bound': '${:.2f}',
                            'Upper Bound': '${:.2f}'
                        }),
                        use_container_width=True
                    )
                
                # Forecast drivers and components
                with st.expander("Forecast Components"):
                    # Try to get components if available
                    try:
                        components = plot_forecast_components(forecast)
                        if components:
                            st.plotly_chart(components, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not generate forecast components: {e}")
        
        # News & Sentiment Tab
        with tabs[5]:
            st.subheader("News & Sentiment Analysis")
            
            # Get news with sentiment analysis
            with st.spinner("Analyzing latest news..."):
                news_data = get_company_news(st.session_state.ticker, days=14)
            
            if news_data is not None and not news_data.empty:
                # Calculate aggregated sentiment
                sentiment_df = analyze_news_sentiment(news_data)
                
                # Visualize sentiment
                visualize_sentiment(sentiment_df)
                
                # Word cloud of news headlines
                if len(sentiment_df) > 5:  # Only show if we have enough headlines
                    with st.expander("News Headline Word Cloud"):
                        try:
                            word_cloud = generate_news_wordcloud(sentiment_df['Headline'])
                            if word_cloud:
                                st.image(word_cloud)
                        except Exception as e:
                            st.warning(f"Could not generate word cloud: {e}")
            else:
                st.warning(f"No recent news found for {st.session_state.ticker}")
                
                # Show Market Context when no company news
                st.subheader("Market Context")
                show_economic_dashboard()
        
        # Reports Tab
        with tabs[6]:
            st.subheader("Analysis Reports")
            
            # Generate report options
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Export data to Excel")
                
                if st.button("Generate Excel Report"):
                    excel_data = export_to_excel(
                        st.session_state.ticker,
                        info,
                        hist,
                        fin
                    )
                    
                    # Provide download button for the generated excel
                    st.download_button(
                        label="Download Excel Report",
                        data=excel_data,
                        file_name=f"{st.session_state.ticker}_financial_report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
            with col2:
                st.write("Generate PDF Analysis Report")
                
                # Get investment thesis for the report
                thesis = st.session_state.get("thesis", generate_investment_thesis(st.session_state.ticker, info, fin))
                
                # Allow editing the thesis
                with st.expander("Edit Investment Thesis"):
                    edited_thesis = st.text_area(
                        "Investment Thesis",
                        value=thesis,
                        height=200
                    )
                    
                    if edited_thesis != thesis:
                        st.session_state.thesis = edited_thesis
                        thesis = edited_thesis
                
                if st.button("Generate PDF Report"):
                    with st.spinner("Generating PDF report..."):
                        pdf_data = generate_pdf_report(
                            st.session_state.ticker,
                            info,
                            hist,
                            fin,
                            thesis
                        )
                        
                        if pdf_data:
                            st.download_button(
                                label="Download PDF Report",
                                data=pdf_data,
                                file_name=f"{st.session_state.ticker}_analysis_report.pdf",
                                mime="application/pdf"
                            )
                        else:
                            st.error("PDF generation failed. Please check if wkhtmltopdf is installed correctly.")
            
            # Additional report template options
            st.subheader("Report Templates")
            
            report_type = st.selectbox(
                "Select Report Type",
                options=[
                    "Executive Summary",
                    "Technical Analysis Report",
                    "Value Investment Analysis",
                    "Growth Potential Analysis",
                    "Risk Assessment"
                ]
            )
            
            # Generate specialized report content based on selection
            with st.spinner(f"Generating {report_type}..."):
                report_content = generate_specialized_report(
                    st.session_state.ticker,
                    report_type,
                    info,
                    hist,
                    fin
                )
                
                # Display report in styled container
                st.markdown(
                    f"""
                    <div style="border:1px solid #ddd; padding:20px; border-radius:5px; background-color:#f9f9f9;">
                    {report_content}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Download as markdown option
                report_md = report_content.replace("<br>", "\n")
                
                st.download_button(
                    label=f"Download {report_type}",
                    data=report_md,
                    file_name=f"{st.session_state.ticker}_{report_type.lower().replace(' ', '_')}.md",
                    mime="text/markdown"
                )

# ── Enhanced Report Generation Functions ─────────────────────────────
def generate_specialized_report(ticker: str, report_type: str, info: dict, hist: pd.DataFrame, fin: pd.DataFrame) -> str:
    """Generate specialized financial report based on selected type"""
    try:
        company_name = info.get('shortName', ticker)
        current_price = info.get('currentPrice', hist['Close'].iloc[-1] if not hist.empty else 0)
        market_cap = info.get('marketCap', 0)
        market_cap_str = f"${market_cap/1e9:.2f}B" if market_cap >= 1e9 else f"${market_cap/1e6:.2f}M"
        
        # Helper function to get financial growth metrics
        def get_growth_metrics(dataframe, column, periods=4):
            if dataframe is None or dataframe.empty or column not in dataframe.columns:
                return "N/A"
            values = dataframe[column].values
            if len(values) < periods:
                return "N/A"
            growth = (values[0] / values[min(periods, len(values)-1)] - 1) * 100
            return f"{growth:.2f}%"
        
        # Report templates based on type
        if report_type == "Executive Summary":
            sector = info.get('sector', 'N/A')
            industry = info.get('industry', 'N/A')
            beta = info.get('beta', 0)
            pe_ratio = info.get('trailingPE', 0)
            
            # Calculate year-to-date performance
            ytd_perf = "N/A"
            if not hist.empty:
                start_of_year = hist.index[0].replace(month=1, day=1)
                start_idx = hist.index.searchsorted(start_of_year)
                if start_idx < len(hist):
                    start_price = hist['Close'].iloc[start_idx]
                    ytd_perf = f"{(current_price / start_price - 1) * 100:.2f}%"
            
            revenue_growth = get_growth_metrics(fin, "Total Revenue")
            income_growth = get_growth_metrics(fin, "Net Income")
            
            return f"""
            <h2>{company_name} ({ticker}) - Executive Summary</h2>
            <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d')}</p>
            <br>
            <h3>Company Overview</h3>
            <p><strong>Sector:</strong> {sector} | <strong>Industry:</strong> {industry}</p>
            <p><strong>Market Cap:</strong> {market_cap_str} | <strong>Current Price:</strong> ${current_price:.2f}</p>
            <p><strong>YTD Performance:</strong> {ytd_perf} | <strong>Beta:</strong> {beta:.2f}</p>
            <br>
            <h3>Financial Highlights</h3>
            <p><strong>P/E Ratio:</strong> {pe_ratio:.2f}</p>
            <p><strong>Revenue Growth (YoY):</strong> {revenue_growth}</p>
            <p><strong>Net Income Growth (YoY):</strong> {income_growth}</p>
            <br>
            <h3>Investment Considerations</h3>
            <p>{generate_investment_thesis(ticker, info, fin)}</p>
            <br>
            <h3>Key Risks</h3>
            <p>{generate_risk_analysis(ticker, info, hist, fin)}</p>
            """
            
        elif report_type == "Technical Analysis Report":
            # Calculate technical indicators
            if hist.empty:
                return "<p>Insufficient historical data for technical analysis</p>"
            
            tech_data = add_technical_indicators(hist)
            
            # Get last values of key indicators
            last_rsi = tech_data['RSI'].iloc[-1] if 'RSI' in tech_data.columns else 0
            last_macd = tech_data['MACD'].iloc[-1] if 'MACD' in tech_data.columns else 0
            last_signal = tech_data['Signal'].iloc[-1] if 'Signal' in tech_data.columns else 0
            
            # Calculate moving averages
            ma50 = tech_data['MA50'].iloc[-1] if 'MA50' in tech_data.columns else 0
            ma200 = tech_data['MA200'].iloc[-1] if 'MA200' in tech_data.columns else 0
            
            # Determine if golden/death cross
            ma_cross = "No recent cross"
            if 'MA50' in tech_data.columns and 'MA200' in tech_data.columns:
                if tech_data['MA50'].iloc[-1] > tech_data['MA200'].iloc[-1] and tech_data['MA50'].iloc[-2] <= tech_data['MA200'].iloc[-2]:
                    ma_cross = "Recent <span style='color:green'>Golden Cross</span> (MA50 crossed above MA200)"
                elif tech_data['MA50'].iloc[-1] < tech_data['MA200'].iloc[-1] and tech_data['MA50'].iloc[-2] >= tech_data['MA200'].iloc[-2]:
                    ma_cross = "Recent <span style='color:red'>Death Cross</span> (MA50 crossed below MA200)"
            
            # Calculate support/resistance levels
            support_resistance = calculate_support_resistance(tech_data)
            support_levels = ", ".join([f"${level:.2f}" for level in support_resistance["support"][:2]])
            resistance_levels = ", ".join([f"${level:.2f}" for level in support_resistance["resistance"][:2]])
            
            # Get trend direction
            trend = "Uptrend" if current_price > ma50 > ma200 else "Downtrend" if current_price < ma50 < ma200 else "Mixed/Sideways"
            trend_color = "green" if trend == "Uptrend" else "red" if trend == "Downtrend" else "orange"
            
            # RSI interpretation
            rsi_interp = "Oversold" if last_rsi < 30 else "Overbought" if last_rsi > 70 else "Neutral"
            rsi_color = "green" if rsi_interp == "Oversold" else "red" if rsi_interp == "Overbought" else "black"
            
            # MACD interpretation
            macd_signal = "Bullish" if last_macd > last_signal else "Bearish"
            macd_color = "green" if macd_signal == "Bullish" else "red"
            
            return f"""
            <h2>{company_name} ({ticker}) - Technical Analysis Report</h2>
            <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d')} | <strong>Current Price:</strong> ${current_price:.2f}</p>
            <br>
            <h3>Technical Indicators</h3>
            <p><strong>Trend:</strong> <span style='color:{trend_color}'>{trend}</span></p>
            <p><strong>Moving Averages:</strong> MA50 = ${ma50:.2f}, MA200 = ${ma200:.2f}</p>
            <p><strong>MA Status:</strong> {ma_cross}</p>
            <p><strong>RSI (14):</strong> {last_rsi:.2f} - <span style='color:{rsi_color}'>{rsi_interp}</span></p>
            <p><strong>MACD:</strong> {last_macd:.4f} vs Signal {last_signal:.4f} - <span style='color:{macd_color}'>{macd_signal}</span></p>
            <br>
            <h3>Support & Resistance</h3>
            <p><strong>Key Support Levels:</strong> {support_levels}</p>
            <p><strong>Key Resistance Levels:</strong> {resistance_levels}</p>
            <br>
            <h3>Technical Outlook</h3>
            <p>{generate_technical_outlook(ticker, tech_data)}</p>
            """
            
        elif report_type == "Value Investment Analysis":
            # Get valuation metrics
            pe_ratio = info.get('trailingPE', 0)
            pb_ratio = info.get('priceToBook', 0)
            ev_ebitda = info.get('enterpriseToEbitda', 0)
            dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
            
            # Calculate intrinsic value if financials available
            intrinsic_value = "N/A"
            discount_percent = "N/A"
            
            if fin is not None and not fin.empty and "EPS" in fin.columns:
                # Simple DCF calculation for demonstration
                last_eps = fin["EPS"].iloc[0] if not fin["EPS"].empty else 0
                growth_rate = 0.05  # Assumed growth rate
                discount_rate = 0.10  # Assumed discount rate
                terminal_multiple = 15  # Terminal P/E multiple
                
                # 5-year simple DCF
                projected_eps = [last_eps * (1 + growth_rate) ** i for i in range(1, 6)]
                terminal_value = projected_eps[-1] * terminal_multiple
                
                # Discount cash flows
                dcf_value = sum([eps / ((1 + discount_rate) ** i) for i, eps in enumerate(projected_eps, 1)])
                dcf_value += terminal_value / ((1 + discount_rate) ** 5)
                
                intrinsic_value = f"${dcf_value:.2f}"
                discount_percent = f"{((dcf_value / current_price - 1) * 100):.2f}%"
            
            return f"""
            <h2>{company_name} ({ticker}) - Value Investment Analysis</h2>
            <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d')} | <strong>Current Price:</strong> ${current_price:.2f}</p>
            <br>
            <h3>Valuation Metrics</h3>
            <p><strong>P/E Ratio:</strong> {pe_ratio:.2f}</p>
            <p><strong>P/B Ratio:</strong> {pb_ratio:.2f}</p>
            <p><strong>EV/EBITDA:</strong> {ev_ebitda:.2f}</p>
            <p><strong>Dividend Yield:</strong> {dividend_yield:.2f}%</p>
            <br>
            <h3>Intrinsic Value Estimate</h3>
            <p><strong>Estimated Intrinsic Value:</strong> {intrinsic_value}</p
