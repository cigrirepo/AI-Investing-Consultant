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

openai.api\_key = os.getenv("OPENAI\_API\_KEY")
news\_api\_key = "166012e1c17248b8b0ff75d114420a72"

# --- Helper Functions ---

def get\_company\_data(ticker):
stock = yf.Ticker(ticker)
return {
'info': stock.info,
'financials': stock.financials,
'balance\_sheet': stock.balance\_sheet,
'cashflow': stock.cashflow,
'quarterly\_financials': stock.quarterly\_financials,
'price\_history': stock.history(period="5y")
}

def generate\_investment\_thesis(ticker, info):
client = openai.OpenAI(api\_key=os.getenv("OPENAI\_API\_KEY"))
prompt = f"""
You are a financial analyst. Write a concise 2-paragraph investment thesis for {ticker}, using current revenue, margins, and valuation metrics.
Business Summary: {info.get('longBusinessSummary')}
Market Cap: {info.get('marketCap')}
Revenue: {info.get('totalRevenue')}
EBITDA: {info.get('ebitda')}
EBITDA Margin: {round((info.get('ebitda') or 0) / (info.get('totalRevenue') or 1) \* 100, 2)}%
P/E: {info.get('trailingPE')}
EPS: {info.get('trailingEps')}
Sector: {info.get('sector')} | Industry: {info.get('industry')}
"""
response = client.chat.completions.create(
model="gpt-4",
messages=[
{"role": "system", "content": "You are a seasoned investment analyst."},
{"role": "user", "content": prompt}
],
temperature=0.7,
max\_tokens=400
)
return response.choices[0].message.content

def plot\_revenue\_and\_growth(quarterly\_financials):
df = quarterly\_financials.loc[['Total Revenue']].T
df.index = pd.to\_datetime(df.index).to\_period("Q")
df = df.sort\_index()
df['YoY Growth (%)'] = df['Total Revenue'].pct\_change(4) \* 100

```
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.bar(df.index.astype(str), df['Total Revenue'] / 1e9, label='Revenue ($B)', color='#1f77b4')
ax1.set_ylabel('Revenue ($B)')
ax1.tick_params(axis='x', rotation=45)

ax2 = ax1.twinx()
ax2.plot(df.index.astype(str), df['YoY Growth (%)'], label='YoY Growth (%)', color='#ff7f0e', linewidth=2)
ax2.set_ylabel('YoY Growth (%)')

fig.tight_layout()
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
st.pyplot(fig)
```

def plot\_stock\_price\_with\_sma(price\_history):
df = price\_history.copy()
df['50SMA'] = df['Close'].rolling(window=50).mean()
df['200SMA'] = df['Close'].rolling(window=200).mean()

```
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df['Close'], label='Price', linewidth=1.5)
ax.plot(df['50SMA'], label='50-Day SMA', linestyle='--')
ax.plot(df['200SMA'], label='200-Day SMA', linestyle='--')
ax.set_ylabel('Price (USD)')
ax.set_title('Stock Price + Moving Averages')
ax.legend()
st.pyplot(fig)
```

def get\_peer\_comparison(tickers):
data = []
for t in tickers:
try:
stock = yf.Ticker(t)
info = stock.info
revenue = info.get("totalRevenue") or 1
ebitda = info.get("ebitda") or 0
data.append({
"Ticker": t,
"Market Cap": info.get("marketCap"),
"P/E Ratio": info.get("trailingPE"),
"Forward P/E": info.get("forwardPE"),
"Revenue (\$B)": round(revenue / 1e9, 2),
"EBITDA (\$M)": round(ebitda / 1e6, 2),
"EBITDA Margin (%)": round((ebitda / revenue) \* 100, 2),
"Sector": info.get("sector")
})
except:
continue
return pd.DataFrame(data).fillna("N/A")

def forecast\_stock\_price(price\_history):
df = price\_history.reset\_index()[["Date", "Close"]]
df["Date"] = pd.to\_datetime(df["Date"]).dt.tz\_localize(None)
df.columns = ["ds", "y"]
model = Prophet(daily\_seasonality=True)
model.fit(df)
future = model.make\_future\_dataframe(periods=30)
forecast = model.predict(future)
fig = model.plot(forecast)
st.pyplot(fig)

def get\_news\_sentiment(ticker):
url = f"[https://newsapi.org/v2/everything?q={ticker}&apiKey={news\_api\_key}](https://newsapi.org/v2/everything?q={ticker}\&apiKey={news_api_key})"
response = requests.get(url).json()
articles = response.get("articles", [])[:5]
sentiment\_data = []
for a in articles:
title = a['title']
sentiment = TextBlob(title).sentiment.polarity
label = "Positive" if sentiment > 0.1 else "Neutral" if sentiment >= -0.1 else "Negative"
sentiment\_data.append({"Headline": title, "Sentiment": round(sentiment, 2), "Label": label})
return pd.DataFrame(sentiment\_data)

def ask\_analyst\_question(question, info):
client = openai.OpenAI(api\_key=os.getenv("OPENAI\_API\_KEY"))
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
max\_tokens=300
)
return response.choices[0].message.content

# --- Streamlit Interface ---

st.set\_page\_config(page\_title="AI Financial Insights Dashboard", layout="wide")
st.title("AI-Powered Financial Insights Dashboard")

ticker = st.text\_input("Enter a stock ticker (e.g., SNOW, AAPL, MSFT):", value="SNOW")
if ticker:
company = get\_company\_data(ticker)
info = company['info']

```
st.subheader("Investment Thesis")
with st.spinner("Generating thesis..."):
    st.write(generate_investment_thesis(ticker, info))

st.subheader("Revenue + YoY Growth")
plot_revenue_and_growth(company['quarterly_financials'])

st.subheader("Stock Price with SMA (5Y)")
plot_stock_price_with_sma(company['price_history'])

st.subheader("Competitor Comparables")
st.dataframe(get_peer_comparison([ticker, "CRM", "DDOG", "MDB", "ZS"]))

st.subheader("Forecasted Price (Next 30 Days)")
forecast_stock_price(company['price_history'])

st.subheader("News Sentiment")
st.dataframe(get_news_sentiment(ticker))

st.subheader("Ask the Analyst")
user_q = st.text_input("Ask a financial question about this company:")
if user_q:
    with st.spinner("Analyzing..."):
        st.write(ask_analyst_question(user_q, info))
```

how do i implement these changes?


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
