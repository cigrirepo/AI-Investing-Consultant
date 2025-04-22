# app.py  ── AI‑Powered Financial Insights Dashboard
# ================================================
import os
import sqlite3
from datetime import datetime

import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from textblob import TextBlob
import requests
import openai
from openai import OpenAI
import streamlit as st

# ── Streamlit Page Config ──────────────────────────────────────────────
st.set_page_config(page_title="AI Financial Insights Dashboard", layout="wide")
st.title("AI‑Powered Financial Insights Dashboard")

# ── API KEYS (set securely in code or via Streamlit → Secrets) ─────────
openai.api_key = os.getenv("OPENAI_API_KEY")                # expects secret
news_api_key   = "166012e1c17248b8b0ff75d114420a72"         # ← YOUR NewsAPI key

# ── Persistent Watchlist (SQLite) ──────────────────────────────────────
conn = sqlite3.connect("watchlist.db", check_same_thread=False)
cur  = conn.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS watchlist(ticker TEXT PRIMARY KEY)")
conn.commit()

def add_to_watchlist(t: str):         # helper
    cur.execute("INSERT OR IGNORE INTO watchlist VALUES (?)", (t.upper(),))
    conn.commit()

def get_watchlist() -> list[str]:     # helper
    return [r[0] for r in cur.execute("SELECT ticker FROM watchlist")]

# ── Data Fetch & Pre‑processing (cached) ───────────────────────────────
@st.cache_data(show_spinner=False)
def get_company_data(ticker: str):
    stock = yf.Ticker(ticker)
    info  = stock.info
    fin   = stock.quarterly_financials.T.copy()
    hist  = stock.history(period="5y")

    fin["YoY Rev %"]      = fin["Total Revenue"].pct_change(4) * 100
    fin["QoQ Rev %"]      = fin["Total Revenue"].pct_change(1) * 100
    fin["Gross Margin %"] = fin["Gross Profit"] / fin["Total Revenue"] * 100
    fin["Op Margin %"]    = fin["Operating Income"] / fin["Total Revenue"] * 100
    if "Free Cash Flow" in fin:
        fin["FCF Margin %"] = fin["Free Cash Flow"] / fin["Total Revenue"] * 100

    return info, fin, hist

# ── LLM Investment Thesis ─────────────────────────────────────────────
def generate_investment_thesis(ticker: str, info: dict) -> str:
    client = OpenAI(api_key=openai.api_key)
    prompt = f"""
You are a financial analyst. Write a concise 2‑paragraph investment thesis for {ticker},
incorporating current revenue figures, margin profiles, and valuation metrics.
After the thesis, add three quantitative bullet points and a text matrix of recent YoY revenue growth.
Business Summary: {info.get('longBusinessSummary')}
Market Cap: {info.get('marketCap')}
Revenue (TTM): {info.get('totalRevenue')}
EBITDA: {info.get('ebitda')}
EBITDA Margin: {round((info.get('ebitda') or 0)/(info.get('totalRevenue') or 1)*100,2)}%
Trailing P/E: {info.get('trailingPE')}
Forward P/E: {info.get('forwardPE')}
EPS (TTM): {info.get('trailingEps')}
Sector: {info.get('sector')}
Industry: {info.get('industry')}
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

# ── Plot Helpers ──────────────────────────────────────────────────────
def plot_revenue_and_growth(fin_df: pd.DataFrame):
    df = fin_df.copy()
    df.index = df.index.astype(str)
    fig = px.bar(df, x=df.index, y="Total Revenue", labels={"Total Revenue":"Revenue ($)"})
    fig.add_trace(go.Scatter(
        x=df.index, y=df["YoY Rev %"], name="YoY Rev %", yaxis="y2", line=dict(color="orange")
    ))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="YoY %"))
    st.plotly_chart(fig, use_container_width=True)

def plot_stock_price_with_sma(hist_df: pd.DataFrame):
    df = hist_df.copy()
    df["50SMA"]  = df["Close"].rolling(50).mean()
    df["200SMA"] = df["Close"].rolling(200).mean()
    fig = px.line(df, x=df.index, y=["Close","50SMA","200SMA"],
                  labels={"value":"Price (USD)", "variable":"Series"})
    st.plotly_chart(fig, use_container_width=True)

# ── Peer Comparables ──────────────────────────────────────────────────
def show_peer_comparison(t
