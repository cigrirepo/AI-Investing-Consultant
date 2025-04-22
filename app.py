import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from textblob import TextBlob
import requests
import openai
import streamlit as st
import os

# --- API KEYS ---
openai.api_key = os.getenv("OPENAI_API_KEY")
news_api_key = "166012e1c17248b8b0ff75d114420a72"

# --- Helper Functions ---

def get_company_data(ticker):
    stock = yf.Ticker(ticker)
    return {
        'info': stock.info,
        'financials': stock.financials,
        'balance_sheet': stock.balance_sheet,
        'cashflow': stock.cashflow,
        'price_history': stock.history(period="1y")
    }

def generate_investment_thesis(ticker, info):
    client = openai.OpenAI()
    prompt = f"""
You are a financial analyst. Write a concise 2-paragraph investment thesis for {ticker}.
Business Summary: {info.get('longBusinessSummary')}
Market Cap: {info.get('marketCap')}
Revenue: {info.get('totalRevenue')}
EBITDA: {info.get('ebitda')}
P/E: {info.get('trailingPE')}
Industry: {info.get('industry')}
    """
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a seasoned investment analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=400
    )
    return response.choices[0].message.content

def plot_revenue(financials):
    revenue = financials.loc['Total Revenue'][::-1]
    revenue.index = revenue.index.strftime('%Y-%m')
    st.line_chart(revenue)

def plot_stock_price(price_history):
    st.line_chart(price_history['Close'])

def get_peer_comparison(tickers):
    data = []
    for t in tickers:
        try:
            stock = yf.Ticker(t)
            info = stock.info
            data.append({
                "Ticker": t,
                "Market Cap": info.get("marketCap"),
                "P/E Ratio": info.get("trailingPE"),
                "Forward P/E": info.get("forwardPE"),
                "Revenue": info.get("totalRevenue"),
                "EBITDA": info.get("ebitda"),
                "Sector": info.get("sector")
            })
        except:
            continue
    return pd.DataFrame(data)

def forecast_stock_price(price_history):
    df = price_history.reset_index()[["Date", "Close"]]
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df.columns = ["ds", "y"]
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    fig = model.plot(forecast)
    st.pyplot(fig)

def get_news_sentiment(ticker):
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={news_api_key}"
    response = requests.get(url).json()
    articles = response.get("articles", [])[:5]
    sentiment_data = []
    for a in articles:
        title = a['title']
        sentiment = TextBlob(title).sentiment.polarity
        label = "🟢 Positive" if sentiment > 0.1 else "🟡 Neutral" if sentiment >= -0.1 else "🔴 Negative"
        sentiment_data.append({"Headline": title, "Sentiment": round(sentiment, 2), "Label": label})
    return pd.DataFrame(sentiment_data)

def ask_analyst_question(question, info):
    client = openai.OpenAI()
    context = f"""
You are a financial analyst assistant. Use the data below to answer:
Business Summary: {info.get('longBusinessSummary')}
Revenue: {info.get('totalRevenue')}
EBITDA: {info.get('ebitda')}
Cash: {info.get('totalCash')}
Debt: {info.get('totalDebt')}
    """
    prompt = context + f"\n\nQuestion: {question}\nAnswer:"
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful financial analyst assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=300
    )
    return response.choices[0].message.content

# --- Streamlit Interface ---
st.set_page_config(page_title="AI Financial Analyst Dashboard", layout="wide")
st.title("📊 AI-Powered Financial Insights Dashboard")

ticker = st.text_input("Enter a stock ticker (e.g., SNOW, AAPL, MSFT):", value="SNOW")
if ticker:
    company = get_company_data(ticker)
    info = company['info']

    st.subheader("📌 Investment Thesis")
    with st.spinner("Generating thesis..."):
        st.write(generate_investment_thesis(ticker, info))

    st.subheader("💸 Revenue Trend")
    plot_revenue(company['financials'])

    st.subheader("📈 Stock Price (1Y)")
    plot_stock_price(company['price_history'])

    st.subheader("📊 Competitor Comparables")
    st.dataframe(get_peer_comparison([ticker, "CRM", "DDOG", "MDB", "ZS"]))

    st.subheader("🔮 Forecasted Price")
    forecast_stock_price(company['price_history'])

    st.subheader("📰 News Sentiment")
    st.dataframe(get_news_sentiment(ticker))

    st.subheader("🤖 Ask the Analyst")
    user_q = st.text_input("Ask a financial question about this company:")
    if user_q:
        with st.spinner("Analyzing..."):
            st.write(ask_analyst_question(user_q, info))
