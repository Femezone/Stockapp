import math
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from yfinance import EquityQuery
from streamlit.errors import StreamlitSecretNotFoundError
from urllib.parse import urlsplit


st.set_page_config(page_title="India Stock Dashboard", page_icon="📈", layout="wide")
st.title("India Stock Market Dashboard")
st.caption(
    "NSE/BSE watchlist, portfolio tracker, technical charts, company financials, "
    "news context, and IPO calendar."
)
st.warning(
    "For research and education only. Yahoo Finance data can be delayed, incomplete, "
    "or unavailable. Signals and scenario ranges are not investment advice or predictions."
)

if "custom_holdings" not in st.session_state:
    st.session_state.custom_holdings = []
if "selected_stock" not in st.session_state:
    st.session_state.selected_stock = None
if "realized_pnl" not in st.session_state:
    st.session_state.realized_pnl = 0.0
if "trade_history" not in st.session_state:
    st.session_state.trade_history = []
if "market_ticker_map" not in st.session_state:
    st.session_state.market_ticker_map = {}
if "personal_watchlist" not in st.session_state:
    st.session_state.personal_watchlist = []
if "stock_search_results" not in st.session_state:
    st.session_state.stock_search_results = []
if "stock_search_error" not in st.session_state:
    st.session_state.stock_search_error = ""
if "stock_logged_in" not in st.session_state:
    st.session_state.stock_logged_in = False
if "stock_user_email" not in st.session_state:
    st.session_state.stock_user_email = ""
if "stock_profile_loaded_for" not in st.session_state:
    st.session_state.stock_profile_loaded_for = ""

DATA_DIR = os.environ.get("UNISYNC_DATA_DIR", "data")
AUTH_DB = os.environ.get("UNISYNC_AUTH_DB", os.path.join(DATA_DIR, "accounts.sqlite3"))
STOCK_DB = os.environ.get("UNISYNC_STOCK_DB", os.path.join(DATA_DIR, "stockapp.sqlite3"))


@contextmanager
def database_connection(path):
    connection = sqlite3.connect(path, timeout=10)
    try:
        with connection:
            yield connection
    finally:
        connection.close()


@st.cache_data(ttl=60, show_spinner=False)
def search_live(query):
    if len(query) < 3:
        return []
    response = requests.get(
        "https://query2.finance.yahoo.com/v1/finance/search",
        params={"q": query, "quotesCount": 100, "newsCount": 0},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=10,
    )
    response.raise_for_status()
    matches = []
    for quote in response.json().get("quotes", []):
        symbol = quote.get("symbol", "")
        if symbol.endswith((".NS", ".BO")):
            name = quote.get("longname") or quote.get("shortname") or symbol
            exchange = "NSE" if symbol.endswith(".NS") else "BSE"
            matches.append({"symbol": symbol, "name": name, "exch": exchange})
    unique = {item["symbol"]: item for item in matches}
    query_lower = query.lower()
    return sorted(
        unique.values(),
        key=lambda item: (
            0 if item["name"].lower().startswith(query_lower)
            or item["symbol"].lower().startswith(query_lower) else 1,
            item["name"],
        ),
    )


def update_stock_search():
    query = st.session_state.stock_search.strip()
    st.session_state.selected_stock = None
    st.session_state.stock_search_results = []
    st.session_state.stock_search_error = ""
    if "stock_search_result" in st.session_state:
        del st.session_state["stock_search_result"]
    if len(query) < 3:
        if query:
            st.session_state.stock_search_error = (
                "Type at least 3 characters to see matching NSE/BSE stocks."
            )
        return
    try:
        st.session_state.stock_search_results = search_live(query)
        if not st.session_state.stock_search_results:
            st.session_state.stock_search_error = (
                "No NSE/BSE stocks found. Try another company name or ticker."
            )
    except requests.RequestException as exc:
        st.session_state.stock_search_error = (
            f"Stock search is temporarily unavailable ({type(exc).__name__})."
        )


def normalize_account_email(email):
    normalized = email.strip().casefold()
    if (len(normalized) > 254 or normalized.count("@") != 1
            or any(character.isspace() for character in normalized)):
        return None
    local, domain = normalized.rsplit("@", 1)
    if not local or "." not in domain or domain.startswith(".") or domain.endswith("."):
        return None
    return normalized


def password_digest(password, salt):
    return hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=1 << 14,
        r=8,
        p=1,
        dklen=64,
        maxmem=64 * 1024 * 1024,
    )


def initialize_auth_db():
    os.makedirs(os.path.dirname(os.path.abspath(AUTH_DB)), exist_ok=True)
    with database_connection(AUTH_DB) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS password_accounts (
                email TEXT PRIMARY KEY,
                salt BLOB NOT NULL,
                password_hash BLOB NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )


def create_password_account(email, password):
    normalized = normalize_account_email(email)
    if not normalized:
        return False, "Enter a valid email address."
    if len(password) < 12:
        return False, "Use a password with at least 12 characters."
    if len(password) > 1024:
        return False, "Password must be 1024 characters or fewer."

    initialize_auth_db()
    salt = secrets.token_bytes(16)
    password_hash = password_digest(password, salt)
    with database_connection(AUTH_DB) as connection:
        cursor = connection.execute(
            "INSERT OR IGNORE INTO password_accounts "
            "(email, salt, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (normalized, salt, password_hash, datetime.now().isoformat()),
        )
    if cursor.rowcount == 0:
        return False, "An account with that email already exists. Try logging in."
    return True, normalized


def authenticate_password_account(email, password):
    normalized = normalize_account_email(email)
    if not normalized or not password or len(password) > 1024:
        return None

    initialize_auth_db()
    with database_connection(AUTH_DB) as connection:
        account = connection.execute(
            "SELECT salt, password_hash FROM password_accounts WHERE email = ?",
            (normalized,),
        ).fetchone()

    salt = account[0] if account else b"\0" * 16
    expected_hash = account[1] if account else b"\0" * 64
    actual_hash = password_digest(password, salt)
    if account and hmac.compare_digest(actual_hash, expected_hash):
        return normalized
    return None


def initialize_stock_db():
    os.makedirs(os.path.dirname(os.path.abspath(STOCK_DB)), exist_ok=True)
    with database_connection(STOCK_DB) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS stock_profiles (
                email TEXT PRIMARY KEY,
                profile_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )"""
        )


def load_stock_profile(email):
    initialize_stock_db()
    with database_connection(STOCK_DB) as connection:
        result = connection.execute(
            "SELECT profile_json FROM stock_profiles WHERE email = ?",
            (email,),
        ).fetchone()
    if not result:
        return {}
    profile = json.loads(result[0])
    if not isinstance(profile, dict):
        raise ValueError("Saved stock profile has an invalid format.")
    return profile


def save_stock_profile(email, profile):
    initialize_stock_db()
    with database_connection(STOCK_DB) as connection:
        connection.execute(
            """INSERT INTO stock_profiles (email, profile_json, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(email) DO UPDATE SET
                   profile_json = excluded.profile_json,
                   updated_at = excluded.updated_at""",
            (
                email,
                json.dumps(profile, ensure_ascii=False),
                datetime.now().isoformat(),
            ),
        )


def persist_stock_profile():
    email = st.session_state.stock_user_email
    if not email:
        raise RuntimeError("Cannot save stock data without an authenticated account.")
    save_stock_profile(email, {
        "custom_holdings": st.session_state.custom_holdings,
        "personal_watchlist": st.session_state.personal_watchlist,
        "realized_pnl": st.session_state.realized_pnl,
        "trade_history": st.session_state.trade_history,
    })


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _normalize_news(raw_news):
    normalized = []
    for article in raw_news or []:
        content = article.get("content", article)
        link = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
        if isinstance(link, dict):
            link = link.get("url", "")
        provider = content.get("provider") or {}
        normalized.append({
            "Title": content.get("title") or article.get("title") or "News article",
            "Publisher": provider.get("displayName", "") if isinstance(provider, dict) else str(provider),
            "Published": content.get("pubDate") or article.get("providerPublishTime"),
            "Link": link or article.get("link", ""),
        })
    return normalized


def _statement_value(statement, row_names):
    if statement is None or statement.empty:
        return None
    expected = {"".join(character.lower() for character in name if character.isalnum())
                for name in row_names}
    for row_name in statement.index:
        normalized = "".join(character.lower() for character in str(row_name) if character.isalnum())
        if normalized in expected:
            return _number(statement.loc[row_name].iloc[0])
    return None


def fundamental_feedback(info, statements):
    positive = []
    risks = []
    debt_to_equity = _number(info.get("debtToEquity"))
    current_ratio = _number(info.get("currentRatio"))
    profit_margin = _number(info.get("profitMargins"))
    operating_cash_flow = _statement_value(
        statements.get("Cash flow"), ("Operating Cash Flow", "Operating Cashflow"),
    )
    net_income = _statement_value(
        statements.get("Income statement"), ("Net Income", "Net Income Common Stockholders"),
    )
    if debt_to_equity is not None:
        if debt_to_equity < 100:
            positive.append(f"Reported debt/equity is {debt_to_equity:.1f}%, below 100%.")
        elif debt_to_equity > 200:
            risks.append(f"Reported debt/equity is elevated at {debt_to_equity:.1f}%.")
    if current_ratio is not None:
        if current_ratio >= 1.5:
            positive.append(f"Reported current ratio is {current_ratio:.2f}.")
        elif current_ratio < 1:
            risks.append(f"Reported current ratio is below 1 ({current_ratio:.2f}).")
    if profit_margin is not None:
        if profit_margin > 0:
            positive.append(f"Reported profit margin is positive ({profit_margin:.1%}).")
        else:
            risks.append(f"Reported profit margin is negative ({profit_margin:.1%}).")
    if net_income is not None and operating_cash_flow is not None:
        if net_income > 0 and operating_cash_flow > 0:
            positive.append("Latest reported net income and operating cash flow are positive.")
        elif net_income > 0 >= operating_cash_flow:
            risks.append("Net income is positive but operating cash flow is not.")
        elif net_income <= 0:
            risks.append("Latest reported net income is not positive.")
    if not positive and not risks:
        risks.append("Insufficient reported data to create fundamental flags.")
    return positive, risks


@st.cache_data(ttl=180, show_spinner=False)
def get_stock_full(ticker, include_details=True):
    stock = yf.Ticker(ticker)
    history = stock.history(period="2y", interval="1d", auto_adjust=False)
    if history.empty or history["Close"].dropna().empty:
        return {"error": "No price history was returned for this ticker."}

    daily_close = history["Close"].dropna()
    latest = float(daily_close.iloc[-1])
    intraday = pd.DataFrame()
    try:
        intraday = stock.history(period="5d", interval="5m", auto_adjust=False)
    except Exception as exc:
        print(f"Intraday quote unavailable for {ticker}: {type(exc).__name__}")
    if not intraday.empty and intraday["Close"].dropna().size:
        latest = float(intraday["Close"].dropna().iloc[-1])
        quote_timestamp = intraday["Close"].dropna().index[-1]
        quote_label = "Latest available 5-minute Yahoo Finance bar"
    else:
        quote_timestamp = daily_close.index[-1]
        quote_label = "Latest available daily Yahoo Finance close"

    previous_close = float(daily_close.iloc[-2]) if len(daily_close) > 1 else latest
    change = latest - previous_close
    change_pct = change / previous_close * 100 if previous_close else 0
    close = daily_close
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    relative_strength = gain / loss.replace(0, np.nan)
    rsi_series = 100 - 100 / (1 + relative_strength)
    rsi_value = _number(rsi_series.iloc[-1])
    if rsi_value is None:
        rsi_value = 50.0
    macd_line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    sma50_series = close.rolling(50).mean()
    sma200_series = close.rolling(200).mean()
    sma50 = _number(sma50_series.iloc[-1])
    sma200 = _number(sma200_series.iloc[-1])
    macd_value = _number(macd_line.iloc[-1])
    macd_signal_value = _number(macd_signal.iloc[-1])
    macd_bullish = macd_value is not None and macd_signal_value is not None and macd_value > macd_signal_value
    positives = [
        label for condition, label in (
            (sma50 is not None and latest > sma50, "Price is above the 50-day average"),
            (sma200 is not None and latest > sma200, "Price is above the 200-day average"),
            (macd_bullish, "MACD is above its signal line"),
            (40 <= rsi_value <= 65, f"RSI is in a non-extreme range ({rsi_value:.1f})"),
        ) if condition
    ]
    negatives = [
        label for condition, label in (
            (sma50 is not None and latest < sma50, "Price is below the 50-day average"),
            (sma200 is not None and latest < sma200, "Price is below the 200-day average"),
            (macd_value is not None and macd_signal_value is not None and macd_value < macd_signal_value,
             "MACD is below its signal line"),
            (rsi_value >= 70, f"RSI is elevated ({rsi_value:.1f})"),
            (rsi_value <= 30, f"RSI is weak ({rsi_value:.1f})"),
        ) if condition
    ]
    if len(positives) >= 3 and not negatives:
        signal = "Bullish technical setup"
    elif len(negatives) >= 3 and not positives:
        signal = "Bearish technical risk"
    else:
        signal = "Mixed / wait for confirmation"

    info = {}
    info_error = ""
    news = []
    news_error = ""
    statements = {
        "Income statement": pd.DataFrame(),
        "Balance sheet": pd.DataFrame(),
        "Cash flow": pd.DataFrame(),
    }
    statement_errors = []
    if include_details:
        try:
            info = stock.info or {}
        except Exception as exc:
            info_error = f"Company profile unavailable ({type(exc).__name__})."
        try:
            news = _normalize_news(stock.news)
        except Exception as exc:
            news_error = f"News unavailable ({type(exc).__name__})."
        for label, attribute in (
            ("Income statement", "financials"),
            ("Balance sheet", "balance_sheet"),
            ("Cash flow", "cashflow"),
        ):
            try:
                statements[label] = getattr(stock, attribute)
            except Exception as exc:
                statement_errors.append(f"{label} unavailable ({type(exc).__name__}).")

    chart_history = history.copy()
    chart_history["SMA 50"] = sma50_series
    chart_history["SMA 200"] = sma200_series
    return {
        "ticker": ticker,
        "name": info.get("longName") or info.get("shortName") or ticker,
        "sector": info.get("sector") or "Not reported",
        "live": latest,
        "chg": change,
        "pct": change_pct,
        "quote_timestamp": quote_timestamp,
        "quote_label": quote_label,
        "history": chart_history,
        "rsi": rsi_value,
        "macd": macd_value,
        "macd_signal": macd_signal_value,
        "sma50": sma50,
        "sma200": sma200,
        "signal": signal,
        "positives": positives,
        "negatives": negatives,
        "info": info,
        "info_error": info_error,
        "news": news,
        "news_error": news_error,
        "statements": statements,
        "statement_errors": statement_errors,
    }


@st.cache_data(ttl=45, show_spinner=False)
def get_market_snapshot(ticker):
    history = yf.Ticker(ticker).history(period="5d", interval="5m", auto_adjust=False)
    if history.empty or history["Close"].dropna().empty:
        history = yf.Ticker(ticker).history(period="5d", interval="1d", auto_adjust=False)
        if history.empty or history["Close"].dropna().empty:
            return None
        close = history["Close"].dropna()
        latest = float(close.iloc[-1])
        previous = float(close.iloc[-2]) if len(close) > 1 else latest
        quote_timestamp = close.index[-1]
        interval = "Latest available daily close"
    else:
        close = history["Close"].dropna()
        session_dates = pd.Index(close.index.date).unique()
        latest_session = session_dates[-1]
        latest_session_close = close[pd.Index(close.index.date) == latest_session]
        latest = float(latest_session_close.iloc[-1])
        earlier_sessions = close[pd.Index(close.index.date) < latest_session]
        previous = float(earlier_sessions.iloc[-1]) if not earlier_sessions.empty else latest
        quote_timestamp = latest_session_close.index[-1]
        interval = "Latest available 5-minute bar"
    return {
        "price": latest,
        "change": latest - previous,
        "change_pct": (latest - previous) / previous * 100 if previous else 0,
        "timestamp": quote_timestamp,
        "interval": interval,
    }


@st.cache_data(ttl=30, show_spinner=False)
def get_live_quotes(tickers):
    tickers = tuple(dict.fromkeys(tickers))
    quotes = {}
    for start in range(0, len(tickers), 50):
        batch = list(tickers[start:start + 50])
        if not batch:
            continue
        history = yf.download(
            batch, period="5d", interval="5m", group_by="ticker",
            auto_adjust=False, progress=False, threads=True, timeout=20,
            multi_level_index=True,
        )
        for ticker in batch:
            try:
                if isinstance(history.columns, pd.MultiIndex):
                    if ticker in history.columns.get_level_values(0):
                        ticker_history = history[ticker]
                    elif ticker in history.columns.get_level_values(1):
                        ticker_history = history.xs(ticker, axis=1, level=1)
                    else:
                        continue
                elif len(batch) == 1:
                    ticker_history = history
                else:
                    continue
                close = ticker_history["Close"].dropna()
                if close.empty:
                    continue
                session_dates = pd.Index(close.index.date).unique()
                latest_session = session_dates[-1]
                latest_session_close = close[pd.Index(close.index.date) == latest_session]
                prior_sessions = close[pd.Index(close.index.date) < latest_session]
                latest = float(latest_session_close.iloc[-1])
                previous = float(prior_sessions.iloc[-1]) if not prior_sessions.empty else latest
                quotes[ticker] = {
                    "price": latest,
                    "change": latest - previous,
                    "change_pct": (latest - previous) / previous * 100 if previous else 0,
                    "timestamp": latest_session_close.index[-1],
                    "interval": "Latest available 5-minute bar",
                }
            except (KeyError, IndexError, TypeError, ValueError):
                continue
    return quotes


@st.cache_data(ttl=30, show_spinner=False)
def get_intraday_history(ticker):
    history = yf.Ticker(ticker).history(period="5d", interval="5m", auto_adjust=False)
    if history.empty or history["Close"].dropna().empty:
        return pd.DataFrame()
    return history[["Close"]].dropna()


def format_india_time(timestamp):
    if timestamp is None or pd.isna(timestamp):
        return "Unavailable"
    if isinstance(timestamp, (int, float, np.number)):
        value = pd.to_datetime(timestamp, unit="s", utc=True)
    else:
        value = pd.Timestamp(timestamp)
    if value.tzinfo is None:
        value = value.tz_localize("UTC")
    return value.tz_convert("Asia/Kolkata").strftime("%d %b %Y, %I:%M %p IST")


@st.cache_data(ttl=60, show_spinner=False)
def get_top_market_caps():
    query = EquityQuery("and", [
        EquityQuery("eq", ["region", "in"]),
        EquityQuery("eq", ["exchange", "NSI"]),
        EquityQuery("gt", ["intradaymarketcap", 0]),
    ])
    results = []
    for offset in (0, 250):
        page = yf.screen(
            query, offset=offset, size=250,
            sortField="intradaymarketcap", sortAsc=False,
        )
        quotes = page.get("quotes", [])
        if not isinstance(quotes, list):
            raise ValueError("Yahoo Finance returned an invalid NSE screener response.")
        results.extend(quotes)
        if len(quotes) < 250:
            break
    by_symbol = {}
    for quote in results:
        ticker = quote.get("symbol", "")
        market_cap = _number(quote.get("marketCap"))
        if not ticker.endswith(".NS") or not market_cap:
            continue
        by_symbol[ticker] = {
            "Rank": 0,
            "Company": quote.get("longName") or quote.get("shortName") or ticker,
            "Ticker": ticker,
            "Market cap (₹ Cr)": market_cap / 10_000_000,
            "Price (₹)": _number(quote.get("regularMarketPrice")),
            "Change (₹)": _number(quote.get("regularMarketChange")),
            "Change (%)": _number(quote.get("regularMarketChangePercent")),
            "Market cap": market_cap,
            "Market time": quote.get("regularMarketTime"),
        }
    ranked = sorted(by_symbol.values(), key=lambda row: row["Market cap"], reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row["Rank"] = rank
    groups = {
        "Large-cap": ranked[:50],
        "Mid-cap": ranked[100:150],
        "Small-cap": ranked[250:300],
    }
    if not any(groups.values()):
        raise ValueError("Yahoo Finance returned no NSE market-cap results.")
    return groups


def select_market_ticker(widget_key):
    ticker = st.session_state.get(widget_key)
    stock = st.session_state.market_ticker_map.get(ticker)
    if stock:
        st.session_state.selected_stock = stock
        st.session_state.stock_search_results = []


@st.fragment(run_every="60s")
def render_market_cap_watchlists():
    try:
        groups = get_top_market_caps()
    except Exception as exc:
        st.error(f"NSE market-cap rankings are unavailable ({type(exc).__name__}).")
        return

    st.caption(
        "Yahoo Finance NSE rankings by market capitalization (large: ranks 1–50, "
        "mid: 101–150, small: 251–300). Intraday quotes refresh every 60 seconds "
        "when this page is open; data may be delayed."
    )
    all_tickers = [row["Ticker"] for rows in groups.values() for row in rows]
    quote_warning = ""
    try:
        live_quotes = get_live_quotes(tuple(all_tickers))
    except Exception as exc:
        live_quotes = {}
        quote_warning = (
            f"Fresh 5-minute quotes are unavailable ({type(exc).__name__}); "
            "showing the latest prices returned by the NSE screener."
        )
    if quote_warning:
        st.warning(quote_warning)
    elif not live_quotes:
        st.warning("No fresh 5-minute quotes were returned; prices below are Yahoo screener snapshots.")
    elif len(live_quotes) < len(all_tickers):
        st.warning(
            f"Fresh 5-minute quotes are unavailable for "
            f"{len(all_tickers) - len(live_quotes)} stocks; those rows use screener snapshots."
        )

    watch_tabs = st.tabs(list(groups))
    for (group_name, rows), group_tab in zip(groups.items(), watch_tabs):
        with group_tab:
            if len(rows) < 50:
                st.warning(f"Yahoo Finance returned only {len(rows)} of 50 {group_name} results.")
            if not rows:
                st.error(f"No {group_name} results are available from the NSE screener.")
                continue

            display_rows = []
            for row in rows:
                display_row = {
                    "Rank": row["Rank"],
                    "Company": row["Company"],
                    "Ticker": row["Ticker"],
                    "Market cap (₹ Cr)": row["Market cap (₹ Cr)"],
                    "Price (₹)": row["Price (₹)"],
                    "Change (₹)": row["Change (₹)"],
                    "Change (%)": row["Change (%)"],
                    "Quote time (IST)": format_india_time(row["Market time"]),
                }
                quote = live_quotes.get(row["Ticker"])
                if quote:
                    display_row.update({
                        "Price (₹)": quote["price"],
                        "Change (₹)": quote["change"],
                        "Change (%)": quote["change_pct"],
                        "Quote time (IST)": format_india_time(quote["timestamp"]),
                    })
                display_rows.append(display_row)

            st.dataframe(
                pd.DataFrame(display_rows).style.format({
                    "Market cap (₹ Cr)": "₹{:,.0f}",
                    "Price (₹)": "₹{:,.2f}",
                    "Change (₹)": "{:+,.2f}",
                    "Change (%)": "{:+.2f}%",
                }, na_rep="N/A"),
                width="stretch",
                hide_index=True,
                alt=f"Top {group_name} NSE stocks by market capitalization and latest prices",
            )
            selection_options = {
                "Select a stock": None,
                **{
                    f"{row['Company']} ({row['Ticker']})": {
                        "symbol": row["Ticker"],
                        "name": row["Company"],
                        "exch": "NSE",
                    }
                    for row in rows
                },
            }
            st.session_state.market_ticker_map.update({
                label: stock for label, stock in selection_options.items() if stock
            })
            selected_label = st.selectbox(
                f"Choose a {group_name} stock",
                list(selection_options),
                key=f"market_cap_selection_{group_name}",
            )
            if selection_options[selected_label]:
                if st.button(
                    "Open chart and portfolio controls",
                    key=f"open_market_stock_{group_name}",
                ):
                    select_market_ticker(f"market_cap_selection_{group_name}")
                    st.rerun(scope="app")


@st.cache_data(ttl=3600, show_spinner=False)
def get_upcoming_ipos(start_date, end_date):
    calendar = yf.Calendars(start=start_date, end=end_date)
    return calendar.get_ipo_info_calendar(start=start_date, end=end_date, limit=100)


POSITIVE_WORDS = {
    "beats", "beat", "growth", "profit", "record", "upgrade", "surge", "strong",
    "wins", "win", "deal", "approval", "expansion", "dividend", "buyback",
    "raises", "raised", "positive", "partnership", "launch",
}
NEGATIVE_WORDS = {
    "miss", "misses", "loss", "decline", "downgrade", "fall", "falls", "weak",
    "fraud", "probe", "investigation", "debt", "cut", "cuts", "layoff",
    "warning", "lawsuit", "default", "negative", "resigns", "resignation",
}


def classify_headline(title):
    words = set("".join(c.lower() if c.isalnum() else " " for c in title).split())
    positive = len(words & POSITIVE_WORDS)
    negative = len(words & NEGATIVE_WORDS)
    if positive > negative:
        return "Positive keywords"
    if negative > positive:
        return "Risk keywords"
    return "Mixed / no clear keywords"


def apply_portfolio_trade(holdings, ticker, name, quantity, price, side):
    if quantity <= 0 or price <= 0:
        return holdings, 0.0, "Quantity and price must be greater than zero."
    updated = [dict(holding) for holding in holdings]
    index = next((i for i, holding in enumerate(updated) if holding["ticker"] == ticker), None)
    realized = 0.0
    if side == "Sell":
        if index is None or quantity > updated[index]["shares"]:
            return holdings, 0.0, "Sell quantity cannot exceed the shares in this portfolio."
        position = updated[index]
        realized = (price - position["avg_cost"]) * quantity
        position["shares"] -= quantity
        if position["shares"] == 0:
            updated.pop(index)
    elif side == "Buy":
        if index is None:
            updated.append({
                "ticker": ticker, "shares": quantity, "avg_cost": price,
                "asset_type": "Stock", "name": name,
            })
        else:
            position = updated[index]
            old_cost = position["shares"] * position["avg_cost"]
            position["shares"] += quantity
            position["avg_cost"] = (old_cost + quantity * price) / position["shares"]
    else:
        return holdings, 0.0, "Choose Buy or Sell."
    return updated, realized, ""


def portfolio_df(holdings):
    rows = []
    quote_warning = ""
    try:
        live_quotes = get_live_quotes(tuple(asset["ticker"] for asset in holdings))
    except Exception as exc:
        live_quotes = {}
        quote_warning = f"Fresh portfolio quotes unavailable ({type(exc).__name__})."
        print(quote_warning)
    if holdings and not live_quotes and not quote_warning:
        quote_warning = "No fresh portfolio quotes were returned; cached prices are shown."
    elif len(live_quotes) < len(holdings) and not quote_warning:
        quote_warning = (
            f"Fresh quotes unavailable for {len(holdings) - len(live_quotes)} positions; "
            "cached prices are shown for those positions."
        )
    for asset in holdings:
        try:
            data = get_stock_full(asset["ticker"], include_details=False)
        except Exception as exc:
            print(f"Portfolio quote unavailable for {asset['ticker']}: {type(exc).__name__}")
            data = {"error": f"Quote request failed ({type(exc).__name__})."}
        if data and not data.get("error"):
            quote = live_quotes.get(asset["ticker"])
            latest = quote["price"] if quote else data["live"]
            change_pct = quote["change_pct"] if quote else data["pct"]
            quote_time = quote["timestamp"] if quote else data["quote_timestamp"]
            market_value = asset["shares"] * latest
            cost = asset["shares"] * asset["avg_cost"]
            rows.append({
                "Company": asset["name"], "Ticker": asset["ticker"],
                "Shares": asset["shares"], "Average cost": asset["avg_cost"],
                "Latest price": latest, "Day change %": change_pct,
                "Quote time": format_india_time(quote_time),
                "Market value": market_value, "Unrealized P&L": market_value - cost,
                "Return %": (market_value - cost) / cost * 100 if cost else 0,
                "Technical view": data["signal"] if quote else data["quote_label"],
            })
        else:
            rows.append({
                "Company": asset["name"], "Ticker": asset["ticker"],
                "Shares": asset["shares"], "Average cost": asset["avg_cost"],
                "Latest price": np.nan, "Day change %": np.nan,
                "Quote time": "Unavailable",
                "Market value": np.nan, "Unrealized P&L": np.nan,
                "Return %": np.nan, "Technical view": data.get("error", "Price unavailable"),
            })
    result = pd.DataFrame(rows)
    result.attrs["quote_warning"] = quote_warning
    return result


def show_candlestick_chart(data, period_label):
    periods = {"1 month": 22, "3 months": 66, "6 months": 132, "1 year": 252, "2 years": 504}
    history = data["history"].tail(periods[period_label])
    chart = go.Figure()
    chart.add_trace(go.Candlestick(
        x=history.index, open=history["Open"], high=history["High"],
        low=history["Low"], close=history["Close"], name=data["ticker"],
    ))
    chart.add_trace(go.Scatter(x=history.index, y=history["SMA 50"], name="50-day average"))
    chart.add_trace(go.Scatter(x=history.index, y=history["SMA 200"], name="200-day average"))
    chart.update_layout(
        height=520, xaxis_rangeslider_visible=False, yaxis_title="Price",
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        legend={"orientation": "h", "y": 1.02},
    )
    st.plotly_chart(chart, width="stretch")


def show_intraday_chart(ticker):
    try:
        history = get_intraday_history(ticker)
    except Exception as exc:
        st.warning(f"Intraday chart unavailable ({type(exc).__name__}).")
        return
    if history.empty:
        st.info("No 5-minute price history is currently available for this stock.")
        return

    session_dates = pd.Index(history.index.date).unique()
    latest_session = session_dates[-1]
    latest_day = history[pd.Index(history.index.date) == latest_session]
    earlier = history[pd.Index(history.index.date) < latest_session]
    chart = go.Figure()
    chart.add_trace(go.Scatter(
        x=latest_day.index,
        y=latest_day["Close"],
        mode="lines",
        name="5-minute price",
    ))
    if not earlier.empty:
        chart.add_hline(
            y=float(earlier["Close"].iloc[-1]),
            line_dash="dash",
            annotation_text="Previous session close",
        )
    chart.update_layout(
        height=340,
        xaxis_title="Time",
        yaxis_title="Price (₹)",
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        showlegend=False,
    )
    st.plotly_chart(chart, width="stretch")
    st.caption(
        f"Latest session shown: {format_india_time(latest_day.index[-1])}. "
        "Yahoo Finance 5-minute data may be delayed."
    )


def show_statements(data):
    if data["info_error"]:
        st.caption(data["info_error"])
    info = data["info"]
    ratio_cols = st.columns(4)
    for column, label, key, suffix in (
        (ratio_cols[0], "P/E", "trailingPE", ""),
        (ratio_cols[1], "P/B", "priceToBook", ""),
        (ratio_cols[2], "Debt / equity", "debtToEquity", ""),
        (ratio_cols[3], "Return on equity", "returnOnEquity", "%"),
    ):
        value = _number(info.get(key))
        column.metric(label, f"{value * 100:.1f}{suffix}" if value is not None and key == "returnOnEquity"
                      else f"{value:.2f}{suffix}" if value is not None else "Not reported")

    positive, risks = fundamental_feedback(info, data["statements"])
    positive_col, risk_col = st.columns(2)
    with positive_col:
        st.markdown("**Positive reported fundamentals**")
        for item in positive:
            st.write(f"- {item}")
    with risk_col:
        st.markdown("**Potential fundamental risks**")
        for item in risks:
            st.write(f"- {item}")
    st.caption(
        "These basic flags use reported figures when available. They are not comparable "
        "across every sector and are not a complete assessment of a company."
    )

    for label, statement in data["statements"].items():
        with st.expander(label):
            if statement is None or statement.empty:
                st.info(f"{label} is not available from this data source.")
            else:
                display = statement.iloc[:20, :4].copy()
                display.columns = [str(column.date()) if hasattr(column, "date") else str(column)
                                   for column in display.columns]
                st.dataframe(display, width="stretch")
    for message in data["statement_errors"]:
        st.caption(message)


oidc_user = getattr(st, "user", None)
if (
    not st.session_state.stock_logged_in
    and oidc_user is not None
    and getattr(oidc_user, "is_logged_in", False)
):
    oidc_email = normalize_account_email(getattr(oidc_user, "email", ""))
    if oidc_email:
        st.session_state.stock_logged_in = True
        st.session_state.stock_user_email = oidc_email
    else:
        st.error("Your Google account did not provide a valid email address.")

if not st.session_state.stock_logged_in:
    st.subheader("Sign in to save your portfolio")
    st.write(
        "Your portfolio, watchlist, and transaction history are saved to your account "
        "and restored when you sign in again."
    )
    st.caption(
        "Stock data is stored in SQLite at data/stockapp.sqlite3 by default. "
        "For hosted deployments, set UNISYNC_STOCK_DB to a persistent disk path."
    )
    login_col, google_col = st.columns([1.5, 1])
    with login_col:
        auth_mode = st.radio(
            "Account",
            ["Log in", "Create account"],
            horizontal=True,
            key="stock_auth_mode",
        )
        with st.form("stock_password_auth"):
            email_input = st.text_input("Email address", key="stock_auth_email")
            password_input = st.text_input(
                "Password",
                type="password",
                key="stock_auth_password",
            )
            confirm_password = ""
            if auth_mode == "Create account":
                confirm_password = st.text_input(
                    "Confirm password",
                    type="password",
                    key="stock_auth_password_confirm",
                )
                st.caption(
                    "Use at least 12 characters. Passwords are stored as salted hashes."
                )
            auth_submitted = st.form_submit_button(
                auth_mode,
                type="primary",
                width="stretch",
            )
        if auth_submitted:
            try:
                if auth_mode == "Create account":
                    if password_input != confirm_password:
                        st.error("Passwords do not match.")
                    else:
                        created, result = create_password_account(email_input, password_input)
                        if created:
                            st.session_state.stock_user_email = result
                            st.session_state.stock_logged_in = True
                            st.rerun()
                        else:
                            st.error(result)
                else:
                    account_email = authenticate_password_account(
                        email_input, password_input,
                    )
                    if account_email:
                        st.session_state.stock_user_email = account_email
                        st.session_state.stock_logged_in = True
                        st.rerun()
                    else:
                        st.error("Email or password is incorrect.")
            except (sqlite3.Error, OSError) as exc:
                print(f"Stock account storage error: {type(exc).__name__}")
                st.error("Account service is unavailable. Please try again.")

    with google_col:
        st.markdown("#### Or use Google")
        try:
            configured_redirect = st.secrets.get("auth", {}).get("redirect_uri", "")
        except StreamlitSecretNotFoundError:
            configured_redirect = ""
        active_url = st.context.url or ""
        configured_origin = urlsplit(configured_redirect).netloc
        active_origin = urlsplit(active_url).netloc
        redirect_mismatch = bool(
            configured_origin and active_origin and configured_origin != active_origin
        )
        if redirect_mismatch:
            st.error(
                f"Google is configured for {configured_origin}, but stockapp is open "
                f"at {active_origin}. Launch with run_stockapp.ps1 and use the stockapp URL."
            )
        elif not configured_redirect:
            st.warning(
                "Stockapp Google OAuth is not configured. Launch with run_stockapp.ps1 "
                "to prepare its separate callback."
            )
        if st.button(
            "Sign in with Google",
            key="stock_google_login",
            width="stretch",
            disabled=redirect_mismatch or not configured_redirect,
        ):
            try:
                st.login()
            except Exception as exc:
                st.error(
                    f"Google sign-in is unavailable ({type(exc).__name__}). "
                    "Configure Streamlit OIDC to enable it."
                )
        st.caption(
            "For local use, add http://localhost:8502/oauth2callback as an authorized "
            "redirect URI in Google Cloud, then start stockapp with run_stockapp.ps1. "
            "Password accounts work without Google setup."
        )
    st.stop()

if st.session_state.stock_profile_loaded_for != st.session_state.stock_user_email:
    try:
        saved_profile = load_stock_profile(st.session_state.stock_user_email)
        holdings = saved_profile.get("custom_holdings", [])
        watchlist = saved_profile.get("personal_watchlist", [])
        trade_history = saved_profile.get("trade_history", [])
        realized_pnl = _number(saved_profile.get("realized_pnl", 0.0))
        if (
            not isinstance(holdings, list)
            or not all(isinstance(item, dict) for item in holdings)
            or not isinstance(watchlist, list)
            or not all(isinstance(item, dict) for item in watchlist)
            or not isinstance(trade_history, list)
            or not all(isinstance(item, dict) for item in trade_history)
            or realized_pnl is None
        ):
            raise ValueError("Saved stock profile has invalid fields.")
    except (sqlite3.Error, OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"Stock profile load error: {type(exc).__name__}")
        st.error("Could not load your saved portfolio. Please retry or contact support.")
        st.stop()
    st.session_state.custom_holdings = holdings
    st.session_state.personal_watchlist = watchlist
    st.session_state.realized_pnl = realized_pnl
    st.session_state.trade_history = trade_history
    st.session_state.stock_profile_loaded_for = st.session_state.stock_user_email

account_col, logout_col = st.columns([5, 1], vertical_alignment="center")
account_col.caption(f"Signed in as {st.session_state.stock_user_email} · portfolio is saved to your account")
if logout_col.button("Log out", key="stock_logout"):
    st.session_state.stock_logged_in = False
    st.session_state.stock_user_email = ""
    st.session_state.stock_profile_loaded_for = ""
    st.session_state.custom_holdings = []
    st.session_state.personal_watchlist = []
    st.session_state.realized_pnl = 0.0
    st.session_state.trade_history = []
    if oidc_user is not None and getattr(oidc_user, "is_logged_in", False):
        st.logout()
    st.rerun()

# ========= Main-page stock search and portfolio controls =========
st.subheader("Find stocks")
st.caption("Type at least 3 characters to find matching NSE/BSE stocks.")
st.text_input(
    "Company or ticker",
    placeholder="e.g. Reliance, TCS, INFY",
    key="stock_search",
    on_change=update_stock_search,
)

if st.session_state.stock_search_error:
    st.info(st.session_state.stock_search_error)

search_results = st.session_state.stock_search_results
if search_results:
    result_labels = {
        f"{item['name']} · {item['symbol']} · {item['exch']}": item
        for item in search_results
    }
    chosen_label = st.selectbox(
        "Search results",
        list(result_labels),
        key="stock_search_result",
    )
    st.session_state.selected_stock = result_labels[chosen_label]

selected = st.session_state.selected_stock
selected_data = None
if selected:
    ticker = selected["symbol"]
    try:
        selected_data = get_live_quotes((ticker,)).get(ticker)
    except Exception as exc:
        st.warning(f"Fresh quote unavailable ({type(exc).__name__}).")
    if not selected_data:
        try:
            selected_data = get_market_snapshot(ticker)
        except Exception as exc:
            st.warning(f"Latest market price unavailable ({type(exc).__name__}).")

    with st.container(border=True):
        st.markdown(f"### {selected['name']}")
        st.caption(f"{ticker} · {selected['exch']}")
        is_saved = any(
            item["symbol"] == ticker
            for item in st.session_state.personal_watchlist
        )
        button_label = "Remove from watchlist" if is_saved else "Add to watchlist"
        if st.button(button_label, key=f"add_watchlist_{ticker}"):
            if is_saved:
                st.session_state.personal_watchlist = [
                    item for item in st.session_state.personal_watchlist
                    if item["symbol"] != ticker
                ]
            else:
                st.session_state.personal_watchlist.append(selected)
            try:
                persist_stock_profile()
            except (sqlite3.Error, OSError) as exc:
                print(f"Stock profile save error: {type(exc).__name__}")
                st.error("Could not save your watchlist. Please retry.")
            else:
                st.rerun()

        if selected_data:
            quote_col, change_col, action_col = st.columns([1, 1, 1.5])
            quote_col.metric("Latest available price", f"₹{selected_data['price']:,.2f}")
            change_col.metric("Today", f"{selected_data['change_pct']:+.2f}%",
                              f"₹{selected_data['change']:+,.2f}")
            action_col.caption(
                f"{selected_data['interval']} · "
                f"{format_india_time(selected_data['timestamp'])}"
            )

            trade_price_key = f"trade_price_{ticker}"
            if trade_price_key not in st.session_state:
                st.session_state[trade_price_key] = float(selected_data["price"])
            with st.form(f"trade_form_{ticker}"):
                st.markdown("#### Add a portfolio transaction")
                side = st.radio(
                    "Action",
                    ["Buy", "Sell"],
                    horizontal=True,
                    key=f"trade_side_{ticker}",
                )
                quantity_col, price_col, submit_col = st.columns(
                    [1, 1.3, 1], vertical_alignment="bottom",
                )
                quantity = quantity_col.number_input(
                    "Shares",
                    min_value=1,
                    max_value=100000,
                    value=1,
                    step=1,
                    key=f"trade_qty_{ticker}",
                )
                trade_price = price_col.number_input(
                    "Price per share (₹)",
                    min_value=0.01,
                    step=0.05,
                    key=trade_price_key,
                )
                trade_submitted = submit_col.form_submit_button(
                    f"Record {side.lower()}",
                    type="primary",
                    width="stretch",
                )
            st.caption(
                "This records a manual portfolio transaction only; it does not place a broker order."
            )
            if trade_submitted:
                updated, realized, error = apply_portfolio_trade(
                    st.session_state.custom_holdings,
                    ticker,
                    selected["name"],
                    int(quantity),
                    float(trade_price),
                    side,
                )
                if error:
                    st.error(error)
                else:
                    st.session_state.custom_holdings = updated
                    st.session_state.realized_pnl += realized
                    st.session_state.trade_history.insert(0, {
                        "Time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "Side": side,
                        "Ticker": ticker,
                        "Quantity": int(quantity),
                        "Price": float(trade_price),
                        "Realized P&L": realized,
                    })
                    try:
                        persist_stock_profile()
                    except (sqlite3.Error, OSError) as exc:
                        print(f"Stock profile save error: {type(exc).__name__}")
                        st.error("Transaction recorded for this session, but could not be saved to your account.")
                    else:
                        st.rerun()
        else:
            st.warning(
                "No market quote is available for this stock right now. "
                "Try refreshing before recording a transaction."
            )


st.subheader("Market snapshot")


@st.fragment(run_every="60s")
def render_market_snapshot():
    indices = [
        ("NIFTY 50", "^NSEI"),
        ("SENSEX", "^BSESN"),
        ("NIFTY BANK", "^NSEBANK"),
    ]
    market_cols = st.columns(len(indices))
    for column, (label, ticker) in zip(market_cols, indices):
        try:
            snapshot = get_market_snapshot(ticker)
        except Exception as exc:
            snapshot = None
            column.caption(f"{label}: unavailable ({type(exc).__name__})")
        if snapshot:
            column.metric(label, f"{snapshot['price']:,.2f}", f"{snapshot['change_pct']:+.2f}%")
        else:
            column.metric(label, "Unavailable")


render_market_snapshot()
st.caption("Snapshot refreshes every 60 seconds when the app is open. Quotes may be delayed; check your broker for tradable prices.")
if st.button("Refresh all market data", key="refresh_market_data"):
    get_stock_full.clear()
    get_market_snapshot.clear()
    get_live_quotes.clear()
    get_intraday_history.clear()
    get_top_market_caps.clear()
    get_upcoming_ipos.clear()
    st.rerun()

tab_portfolio, tab_watchlist, tab_market, tab_analysis, tab_outlook, tab_news = st.tabs(
    ["Portfolio", "My watchlist", "Market leaders", "Company analysis", "Scenarios & IPOs", "News"],
    on_change="rerun",
    key="stock_dashboard_tabs",
)


@st.fragment(run_every="60s")
def render_portfolio():
    holdings_df = portfolio_df(st.session_state.custom_holdings)
    if holdings_df.attrs.get("quote_warning"):
        st.warning(holdings_df.attrs["quote_warning"])
    if not holdings_df.empty:
        st.caption("Portfolio values refresh every 60 seconds while this page is open.")
    if holdings_df.empty:
        st.info("Search for a stock above and record a Buy to start tracking your portfolio.")
    else:
        priced = holdings_df.dropna(subset=["Market value"])
        total_value = priced["Market value"].sum()
        unrealized = priced["Unrealized P&L"].sum()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Market value", f"₹{total_value:,.2f}")
        col2.metric("Unrealized P&L", f"₹{unrealized:,.2f}")
        col3.metric("Realized P&L", f"₹{st.session_state.realized_pnl:,.2f}")
        col4.metric("Positions", str(len(holdings_df)))
        st.dataframe(
            holdings_df.style.format({
                "Average cost": "₹{:,.2f}", "Latest price": "₹{:,.2f}",
                "Day change %": "{:+.2f}%", "Market value": "₹{:,.2f}",
                "Unrealized P&L": "₹{:,.2f}", "Return %": "{:+.2f}%",
            }, na_rep="N/A"),
            width="stretch", hide_index=True, alt="Portfolio positions with latest prices and returns",
        )
    if st.session_state.trade_history:
        with st.expander("Recent portfolio entries"):
            st.dataframe(
                pd.DataFrame(st.session_state.trade_history),
                width="stretch",
                hide_index=True,
                alt="Recent manually recorded portfolio transactions",
            )


with tab_portfolio:
    st.subheader("Your portfolio")
    st.caption("Portfolio and trade history are held in this browser session; no orders are sent to a broker.")
    if tab_portfolio.open:
        render_portfolio()

with tab_watchlist:
    st.subheader("My watchlist")
    st.caption("Save stocks to follow them. Watchlist and portfolio are kept for this browser session.")
    if tab_watchlist.open and not st.session_state.personal_watchlist:
        st.info("Find a stock above and add it to your watchlist to keep it handy.")
    elif tab_watchlist.open:
        watchlist_rows = []
        try:
            watchlist_quotes = get_live_quotes(tuple(
                stock["symbol"] for stock in st.session_state.personal_watchlist
            ))
        except Exception as exc:
            watchlist_quotes = {}
            st.warning(f"Watchlist quotes are unavailable ({type(exc).__name__}).")
        if not watchlist_quotes:
            st.warning("No current watchlist quotes are available.")
        elif len(watchlist_quotes) < len(st.session_state.personal_watchlist):
            st.warning("Some watchlist quotes are unavailable; those rows show N/A.")
        for stock in st.session_state.personal_watchlist:
            quote = watchlist_quotes.get(stock["symbol"], {})
            watchlist_rows.append({
                "Company": stock["name"],
                "Symbol": stock["symbol"],
                "Exchange": stock["exch"],
                "Price (₹)": quote.get("price", np.nan),
                "Change (%)": quote.get("change_pct", np.nan),
                "Quote time (IST)": format_india_time(quote.get("timestamp")),
            })
        st.dataframe(
            pd.DataFrame(watchlist_rows).style.format({
                "Price (₹)": "₹{:,.2f}",
                "Change (%)": "{:+.2f}%",
            }, na_rep="N/A"),
            width="stretch",
            hide_index=True,
            alt="Your saved stock watchlist with recent quotes",
        )
        watchlist_labels = {
            f"{stock['name']} ({stock['symbol']})": stock
            for stock in st.session_state.personal_watchlist
        }
        selected_watchlist_label = st.selectbox(
            "Choose a saved stock to open",
            list(watchlist_labels),
            key="watchlist_stock_selection",
        )
        if st.button("Open this stock", key="open_watchlist_stock"):
            st.session_state.selected_stock = watchlist_labels[selected_watchlist_label]
            st.session_state.stock_search_results = []
            st.rerun()
        if st.button("Remove selected stock", key="remove_watchlist_stock"):
            to_remove = watchlist_labels[selected_watchlist_label]["symbol"]
            st.session_state.personal_watchlist = [
                stock for stock in st.session_state.personal_watchlist
                if stock["symbol"] != to_remove
            ]
            try:
                persist_stock_profile()
            except (sqlite3.Error, OSError) as exc:
                print(f"Stock profile save error: {type(exc).__name__}")
                st.error("Could not save your updated watchlist. Please retry.")
            else:
                st.rerun()

with tab_market:
    st.subheader("Market leaders")
    st.caption("Browse market-cap groups and open a stock to view its quote and record a portfolio transaction.")
    if tab_market.open:
        render_market_cap_watchlists()

with tab_analysis:
    st.subheader("Candlestick chart and company fundamentals")
    available_tickers = list(dict.fromkeys(
        [item["ticker"] for item in st.session_state.custom_holdings]
        + ([selected["symbol"]] if selected else [])
    ))
    if not available_tickers:
        st.info("Search for an NSE/BSE stock above to inspect its chart and financials.")
    else:
        analysis_ticker = st.selectbox("Company", available_tickers, key="analysis_ticker")
        if tab_analysis.open:
            try:
                analysis = get_stock_full(analysis_ticker)
            except Exception as exc:
                analysis = {"error": f"Data request failed ({type(exc).__name__})."}
        else:
            analysis = {"error": "Open this tab to load company analysis."}
        if analysis.get("error"):
            st.error(analysis["error"])
        else:
            quote_col, signal_col, rsi_col = st.columns(3)
            quote_col.metric("Latest available price", f"₹{analysis['live']:,.2f}",
                             f"{analysis['pct']:+.2f}%")
            signal_col.metric("Technical summary", analysis["signal"])
            rsi_col.metric("14-day RSI", f"{analysis['rsi']:.1f}")
            st.caption(
                f"{analysis['name']} · {analysis['ticker']} · {analysis['sector']} · "
                f"{analysis['quote_label']} · {analysis['quote_timestamp']}"
            )
            st.markdown("#### Live intraday price movement")
            show_intraday_chart(analysis_ticker)
            chart_period = st.selectbox(
                "Candlestick range", ["1 month", "3 months", "6 months", "1 year", "2 years"],
                index=2, key="candle_period",
            )
            show_candlestick_chart(analysis, chart_period)
            st.markdown("#### What the indicators show")
            positive_col, negative_col = st.columns(2)
            with positive_col:
                st.markdown("**Positive technical observations**")
                if analysis["positives"]:
                    for observation in analysis["positives"]:
                        st.write(f"- {observation}")
                else:
                    st.write("No positive checks from the displayed indicators.")
            with negative_col:
                st.markdown("**Risk / negative technical observations**")
                if analysis["negatives"]:
                    for observation in analysis["negatives"]:
                        st.write(f"- {observation}")
                else:
                    st.write("No negative checks from the displayed indicators.")
            st.caption(
                "Signal rules summarize four historical indicators (50/200-day averages, "
                "MACD and RSI). They do not account for your finances, valuation, or risk tolerance."
            )
            st.markdown("#### Fundamentals and reported statements")
            show_statements(analysis)

with tab_outlook:
    st.subheader("Short-term historical-volatility scenarios")
    st.caption(
        "Illustrative range only: today's price plus/minus one standard deviation of recent "
        "20-session returns, scaled to five sessions. It does not predict direction."
    )
    scenario_tickers = list(dict.fromkeys(
        ["^NSEI"]
        + [item["ticker"] for item in st.session_state.custom_holdings]
        + ([selected["symbol"]] if selected else [])
    ))
    if scenario_tickers:
        scenario_ticker = st.selectbox("Scenario ticker", scenario_tickers, key="scenario_ticker")
        if tab_outlook.open:
            try:
                scenario = get_stock_full(scenario_ticker)
            except Exception as exc:
                scenario = {"error": f"Data request failed ({type(exc).__name__})."}
        else:
            scenario = {"error": "Open this tab to load market scenarios."}
        if scenario.get("error"):
            st.error(scenario["error"])
        else:
            returns = scenario["history"]["Close"].pct_change().dropna().tail(20)
            if len(returns) >= 5:
                move = float(returns.std()) * math.sqrt(5)
                low = scenario["live"] * (1 - move)
                high = scenario["live"] * (1 + move)
                low_col, base_col, high_col = st.columns(3)
                low_col.metric("Illustrative lower range", f"₹{low:,.2f}", f"{-move:.1%}")
                base_col.metric("Latest available price", f"₹{scenario['live']:,.2f}")
                high_col.metric("Illustrative upper range", f"₹{high:,.2f}", f"{move:.1%}")
                st.caption(
                    f"Based on {len(returns)} historical daily returns. Actual prices can move "
                    "outside this range, and past volatility does not predict future returns."
                )
            else:
                st.info("Not enough recent price history to calculate a scenario range.")
    else:
        st.info("Search for a stock to view a historical-volatility scenario.")

    st.divider()
    st.subheader("Upcoming IPO calendar")
    st.caption(
        "Calendar coverage depends on Yahoo Finance and may omit Indian issues. Verify dates, "
        "price bands and subscription details on NSE/BSE before acting."
    )
    today = date.today()
    if tab_outlook.open:
        try:
            ipo_data = get_upcoming_ipos(today.isoformat(), (today + timedelta(days=90)).isoformat())
        except Exception as exc:
            ipo_data = None
            st.warning(f"IPO calendar unavailable ({type(exc).__name__}); use the official exchange pages below.")
    else:
        ipo_data = None
    if ipo_data is not None and not ipo_data.empty:
        ipo_frame = ipo_data.reset_index()
        company_column = next((column for column in ("Company", "Company Name") if column in ipo_frame), None)
        exchange_column = next((column for column in ("Exchange", "Exchange Short Name") if column in ipo_frame), None)
        symbol_column = next((column for column in ("Symbol", "Ticker") if column in ipo_frame), None)
        india_rows = pd.Series(False, index=ipo_frame.index)
        if exchange_column:
            india_rows |= ipo_frame[exchange_column].astype(str).str.contains(r"NSE|BSE|India", case=False, na=False)
        if symbol_column:
            india_rows |= ipo_frame[symbol_column].astype(str).str.endswith((".NS", ".BO"))
        india_ipos = ipo_frame[india_rows]
        if not india_ipos.empty:
            st.dataframe(india_ipos, width="stretch", hide_index=True)
        else:
            st.info("No NSE/BSE IPO listings were identified in Yahoo's 90-day calendar.")
    else:
        st.info("No upcoming IPO entries were returned by the public calendar.")
    st.markdown(
        "- [NSE issue information](https://www.nseindia.com/market-data/issue-information)\n"
        "- [BSE IPO notices](https://www.bseindia.com/markets/MarketInfo/DispNewNoticesCirculars.aspx?page=ipo)\n"
        "- [SEBI public issues](https://www.sebi.gov.in/)"
    )

with tab_news:
    st.subheader("Company news and explainable technical signals")
    news_tickers = list(dict.fromkeys(
        [item["ticker"] for item in st.session_state.custom_holdings]
        + ([selected["symbol"]] if selected else [])
    ))
    if not news_tickers:
        st.info("Search for an NSE/BSE stock to view its news and technical summary.")
    else:
        news_ticker = st.selectbox("Company news", news_tickers, key="news_ticker")
        if tab_news.open:
            try:
                news_data = get_stock_full(news_ticker)
            except Exception as exc:
                news_data = {"error": f"Data request failed ({type(exc).__name__})."}
        else:
            news_data = {"error": "Open this tab to load company news."}
        if news_data.get("error"):
            st.error(news_data["error"])
        else:
            st.metric("Technical summary", news_data["signal"])
            st.caption(
                "Headline labels below are simple keyword matches, not a language model or "
                "verified assessment of whether a story is good or bad."
            )
            if news_data["news"]:
                news_frame = pd.DataFrame(news_data["news"])
                news_frame["Headline context"] = news_frame["Title"].map(classify_headline)
                st.dataframe(
                    news_frame, width="stretch", hide_index=True,
                    column_config={"Link": st.column_config.LinkColumn("Article", display_text="Open")},
                )
            else:
                st.info(news_data["news_error"] or "No company headlines were returned.")

st.caption(
    "Quotes, headlines, fundamentals and IPO dates are sourced from Yahoo Finance where "
    "available. Verify against official exchange filings and your broker; this dashboard "
    "does not place trades."
)
