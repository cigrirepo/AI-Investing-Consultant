# app.py  ── Senzu Financial Insights Dashboard
# ================================================
import os
import sqlite3
from typing import List, Tuple

import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go        # needed for Scatter traces
from prophet import Prophet
from textblob import TextBlob
import requests
import openai
from openai import OpenAI
import streamlit as st
from PIL import Image  # for logo display

# ── session defaults ────────────────────────────────────────────────
if "loaded" not in st.session_state:
    st.session_state.loaded = False
    st.session_state.info   = None
    st.session_state.fin    = None
    st.session_state.hist   = None

# ── Streamlit Page Config ───────────────────────────────────────────
st.set_page_config(page_title="Senzu Financial Insights", layout="wide")

# ── Branding & Logo ─────────────────────────────────────────────────
# Centered logo (no text title)
logo = Image.open("senzu_logo.png")
col1, col2, col3 = st.columns([1, 0, 1])
with col2:
    st.image(logo, width=300)

# Optional short description underneath, still centered
with col2:
    st.markdown(
        """
        <p style="font-size:16px; color:gray; margin-top:10px;">
            An AI-driven financial analysis platform designed for investors and analysts.<br>
            Quickly explore revenue trends, margin profiles, valuation metrics, peer benchmarks,<br>
            30-day forecasts, and engage in an interactive analyst Q&A.
        </p>
        """,
        unsafe_allow_html=True
    )

# ── Keys / Secrets ─────────────────────────────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY")
news_api_key   = "166012e1c17248b8b0ff75d114420a72"

# ── SQLite Watchlist ───────────────────────────────────────────────
conn = sqlite3.connect("watchlist.db", check_same_thread=False)
cur  = conn.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS watchlist(ticker TEXT PRIMARY KEY)")
conn.commit()

def add_to_watchlist(ticker: str) -> None:
    cur.execute("INSERT OR IGNORE INTO watchlist VALUES (?)", (ticker.upper(),))
    conn.commit()

def get_watchlist() -> List[str]:
    return [r[0] for r in cur.execute("SELECT ticker FROM watchlist")]

# ── Data Loader (cached) ───────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_company_data(ticker: str) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    stock = yf.Ticker(ticker)
    info  = stock.info
    fin   = stock.quarterly_financials.T.copy()
    hist  = stock.history(period="5y")

    fin["YoY Rev %"]      = fin["Total Revenue"].pct_change(4) * 100
    fin["QoQ Rev %"]      = fin["Total Revenue"].pct_change(1) * 100
    fin["Gross Margin %"] = fin["Gross Profit"] / fin["Total Revenue"] * 100
    fin["Op Margin %"]    = fin["Operating Income"] / fin["Total Revenue"] * 100
    if "Free Cash Flow" in fin.columns:
        fin["FCF Margin %"] = fin["Free Cash Flow"] / fin["Total Revenue"] * 100

    return info, fin, hist

# ── LLM Investment Thesis ──────────────────────────────────────────
def generate_investment_thesis(ticker: str, info: dict) -> str:
    client = OpenAI(api_key=openai.api_key)
    prompt = f"""
You are a financial analyst. Write a concise 2-paragraph investment thesis for {ticker},
incorporating current revenue, margins, and valuation metrics.
Follow with 3 quantitative bullet points and a text matrix of recent YoY revenue growth.

Business Summary: {info.get('longBusinessSummary')}
Market Cap: {info.get('marketCap')}
Revenue (TTM): {info.get('totalRevenue')}
EBITDA: {info.get('ebitda')}
EBITDA Margin: {round((info.get('ebitda') or 0)/(info.get('totalRevenue') or 1)*100,2)}%
Trailing P/E: {info.get('trailingPE')}
Forward  P/E: {info.get('forwardPE')}
EPS (TTM): {info.get('trailingEps')}
Sector: {info.get('sector')} | Industry: {info.get('industry')}
"""
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role":"system","content":"You are a seasoned investment analyst."},
            {"role":"user",  "content":prompt}
        ],
        temperature=0.7,
        max_tokens=600
    )
    return resp.choices[0].message.content

# ── Plot Helpers ───────────────────────────────────────────────────
def plot_revenue_and_growth(fin_df: pd.DataFrame) -> None:
    df = fin_df[["Total Revenue"]].dropna().copy()
    if isinstance(df.index, pd.PeriodIndex):
        df.index = df.index.to_timestamp(how="end")
    df = df.sort_index()
    df["Quarter"]   = df.index.to_period("Q").astype(str)
    df["YoY Rev %"] = df["Total Revenue"].pct_change(4) * 100

    fig = px.bar(
        df,
        x="Quarter",
        y="Total Revenue",
        labels={"Total Revenue": "Revenue ($)"},
        title="Quarterly Revenue & YoY Growth"
    )

    mask = df["YoY Rev %"].notna()
    fig.add_trace(
        go.Scatter(
            x=df["Quarter"][mask],
            y=df["YoY Rev %"][mask],
            name="YoY Rev %",
            mode="lines+markers",
            yaxis="y2",
            line=dict(color="#FFA500")
        )
    )

    fig.update_layout(
        yaxis2=dict(
            overlaying="y",
            side="right",
            title="YoY %",
            tickformat=".0f",
            ticksuffix="%"
        ),
        bargap=0.25,
        xaxis_title=""
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_stock_price_with_sma(hist_df: pd.DataFrame) -> None:
    df = hist_df.copy()
    df["50SMA"]  = df["Close"].rolling(50).mean()
    df["200SMA"] = df["Close"].rolling(200).mean()
    fig = px.line(
        df,
        x=df.index,
        y=["Close", "50SMA", "200SMA"],
        labels={"value": "Price (USD)", "variable": "Series"},
        title="Price with 50/200-day SMA"
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Dynamic Peer Comparables ───────────────────────────────────────
SP500_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
FALLBACK_PEERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

@st.cache_data(show_spinner=False)
def get_sp500() -> pd.DataFrame:
    try:
        df = pd.read_csv(SP500_URL)
        df.columns = df.columns.str.title()
        return df
    except Exception:
        return pd.DataFrame()

def show_peer_comparison(focus_ticker: str, n_peers: int = 5) -> None:
    try:
        sector = yf.Ticker(focus_ticker).info.get("sector")
    except Exception:
        sector = None

    sp_df = get_sp500()
    if sector and not sp_df.empty and {"Sector","Symbol"}.issubset(sp_df.columns):
        peers = sp_df.loc[sp_df["Sector"] == sector, "Symbol"].tolist()
        peers = [t for t in peers if t != focus_ticker][:n_peers]
    else:
        peers = FALLBACK_PEERS[:n_peers]

    tickers = [focus_ticker] + peers
    rows = []
    for t in tickers:
        try:
            inf = yf.Ticker(t).info
            mc  = inf.get("marketCap") or 0
            ebt = inf.get("ebitda")    or 0
            rows.append({
                "Ticker": t,
                "Market Cap ($B)": f"{mc/1e9:,.2f}",
                "P/E (LTM)":       f"{inf.get('trailingPE',0):.2f}" if inf.get("trailingPE") else "N/A",
                "P/E (NTM)":       f"{inf.get('forwardPE',0):.2f}" if inf.get("forwardPE") else "N/A",
                "EV/EBITDA":       f"{inf.get('enterpriseValue',0)/ebt:,.2f}" if ebt else "N/A",
                "FCF Yield (%)":   (
                    f"{inf.get('freeCashflow',0)/mc*100:,.2f}"
                    if inf.get("freeCashflow") and mc else "N/A"
                )
            })
        except Exception:
            continue
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ── Prophet Forecast ───────────────────────────────────────────────
def forecast_stock_price(hist_df: pd.DataFrame) -> None:
    df = hist_df.reset_index()[["Date", "Close"]].rename(columns={"Date":"ds","Close":"y"})
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    fc     = model.predict(future)
    st.pyplot(model.plot(fc))

# ── News Sentiment ────────────────────────────────────────────────
def get_news_sentiment(ticker: str) -> pd.DataFrame:
    url  = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={news_api_key}"
    arts = requests.get(url).json().get("articles", [])[:5]
    rows = []
    for art in arts:
        title = art.get("title","")
        pol   = TextBlob(title).sentiment.polarity
        rows.append({
            "Headline": title,
            "Polarity": round(pol,2),
            "Label": "Positive" if pol>0.1 else "Neutral" if pol>= -0.1 else "Negative"
        })
    return pd.DataFrame(rows)

# ── Analyst Q&A ───────────────────────────────────────────────────
def ask_analyst_question(question: str, info: dict) -> str:
    client = OpenAI(api_key=openai.api_key)
    context = (
        f"Business Summary: {info.get('longBusinessSummary')}\n"
        f"Revenue: {info.get('totalRevenue')}\n" 
        f"EBITDA: {info.get('ebitda')}\n"
        f"Cash: {info.get('totalCash')}\n"
        f"Debt: {info.get('totalDebt')}"
    )
    prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role":"system","content":"You are a helpful financial analyst assistant."},
            {"role":"user",  "content":prompt}
        ],
        temperature=0.4,
        max_tokens=300
    )
    return resp.choices[0].message.content

# ── Main UI ────────────────────────────────────────────────────────
ticker = st.text_input("Enter ticker (e.g., AAPL):", value="AAPL").upper()

if st.button("Load / Refresh Data"):
    try:
        with st.spinner("Fetching company data…"):
            info, fin, hist = get_company_data(ticker)
        st.session_state.loaded = True
        st.session_state.info   = info
        st.session_state.fin    = fin
        st.session_state.hist   = hist
    except Exception as e:
        st.session_state.loaded = False
        st.error(f"Data fetch failed: {e}")

if st.session_state.loaded:
    info = st.session_state.info
    fin  = st.session_state.fin
    hist = st.session_state.hist

    st.subheader("Investment Thesis")
    with st.spinner("Generating thesis…"):
        st.write(generate_investment_thesis(ticker, info))

    st.subheader("Revenue & YoY Growth")
    plot_revenue_and_growth(fin)

    st.subheader("Price with 50/200-day SMA")
    plot_stock_price_with_sma(hist)

    st.subheader("Peer Comparables")
    show_peer_comparison(ticker)

    st.subheader("30-Day Price Forecast")
    forecast_stock_price(hist)

    st.subheader("Latest News Sentiment")
    st.dataframe(get_news_sentiment(ticker), use_container_width=True)

    st.subheader("Ask the Analyst")
    user_q = st.text_input("Type a financial question:", key="analyst_q")
    if user_q:
        with st.spinner("Analyzing…"):
            answer = ask_analyst_question(user_q, info)
            st.markdown(answer)
