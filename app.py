import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="EGX Stock Intelligence PRO",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# STYLE
# =========================================================

st.markdown("""
<style>
.main-title {
    font-size: 36px;
    font-weight: 800;
    margin-bottom: 5px;
}

.sub-title {
    color: #777;
    font-size: 16px;
    margin-bottom: 25px;
}

.metric-card {
    padding: 15px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,.25);
    background: rgba(128,128,128,.05);
    text-align: center;
    margin-bottom: 10px;
}

.metric-title {
    font-size: 13px;
    color: #777;
}

.metric-value {
    font-size: 23px;
    font-weight: 700;
}

.good {
    color: #16a34a;
    font-weight: 700;
}

.warning {
    color: #d97706;
    font-weight: 700;
}

.bad {
    color: #dc2626;
    font-weight: 700;
}

.neutral {
    color: #64748b;
    font-weight: 700;
}

.section {
    font-size: 24px;
    font-weight: 800;
    margin-top: 25px;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# CONSTANTS
# =========================================================

DEFAULT_PERIOD = "2y"

TECH_SCORE_MAX = 35
FUND_SCORE_MAX = 35
VALUATION_SCORE_MAX = 15
LIQUIDITY_SCORE_MAX = 10
NEWS_SCORE_MAX = 5

# =========================================================
# HELPERS
# =========================================================

def safe_float(value, default=np.nan):
    try:
        if value is None:
            return default

        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) == 0:
                return default
            value = value[0]

        value = float(value)

        if np.isfinite(value):
            return value

        return default

    except Exception:
        return default


def fmt_number(value, decimals=2):
    value = safe_float(value)

    if pd.isna(value):
        return "غير متاح"

    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.{decimals}f} B"

    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.{decimals}f} M"

    if abs(value) >= 1_000:
        return f"{value / 1_000:.{decimals}f} K"

    return f"{value:.{decimals}f}"


def fmt_price(value):
    value = safe_float(value)

    if pd.isna(value):
        return "غير متاح"

    return f"{value:.2f}"


def fmt_pct(value):
    value = safe_float(value)

    if pd.isna(value):
        return "غير متاح"

    return f"{value:.2f}%"


def clean_symbol(symbol):
    symbol = str(symbol).strip().upper()

    if not symbol:
        return ""

    if symbol.endswith(".CA"):
        return symbol

    return symbol + ".CA"


def get_raw_symbol(symbol):
    return symbol.replace(".CA", "")


# =========================================================
# DATA FETCH
# =========================================================

@st.cache_data(ttl=900, show_spinner=False)
def load_price_data(symbol, period="2y"):

    try:

        ticker = yf.Ticker(symbol)

        df = ticker.history(
            period=period,
            interval="1d",
            auto_adjust=False
        )

        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy()

        df.columns = [str(c).title() for c in df.columns]

        required = ["Open", "High", "Low", "Close", "Volume"]

        for col in required:
            if col not in df.columns:
                return pd.DataFrame()

        df = df[required].dropna(subset=["Close"])

        return df

    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def load_info(symbol):

    try:

        ticker = yf.Ticker(symbol)

        info = ticker.info

        if not isinstance(info, dict):
            return {}

        return info

    except Exception:
        return {}


@st.cache_data(ttl=900, show_spinner=False)
def load_financials(symbol):

    result = {}

    try:

        ticker = yf.Ticker(symbol)

        try:
            result["income"] = ticker.income_stmt
        except Exception:
            result["income"] = pd.DataFrame()

        try:
            result["balance"] = ticker.balance_sheet
        except Exception:
            result["balance"] = pd.DataFrame()

        try:
            result["cashflow"] = ticker.cashflow
        except Exception:
            result["cashflow"] = pd.DataFrame()

    except Exception:
        result = {
            "income": pd.DataFrame(),
            "balance": pd.DataFrame(),
            "cashflow": pd.DataFrame()
        }

    return result


# =========================================================
# NEWS
# =========================================================

@st.cache_data(ttl=900, show_spinner=False)
def load_news(symbol):

    try:

        ticker = yf.Ticker(symbol)

        news = ticker.news

        if not news:
            return []

        cleaned = []

        for item in news[:15]:

            content = item.get("content", item)

            title = (
                content.get("title")
                or item.get("title")
                or ""
            )

            publisher = (
                content.get("provider", {}).get("displayName")
                if isinstance(content.get("provider"), dict)
                else item.get("publisher", "")
            )

            link = (
                content.get("canonicalUrl", {}).get("url")
                if isinstance(content.get("canonicalUrl"), dict)
                else item.get("link", "")
            )

            if title:
                cleaned.append({
                    "title": title,
                    "publisher": publisher or "Unknown",
                    "link": link or ""
                })

        return cleaned

    except Exception:
        return []


# =========================================================
# TECHNICAL INDICATORS
# =========================================================

def add_indicators(df):

    df = df.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # EMA
    df["EMA20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()
    df["EMA100"] = close.ewm(span=100, adjust=False).mean()
    df["EMA200"] = close.ewm(span=200, adjust=False).mean()

    # RSI
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

    # ATR
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["ATR"] = true_range.rolling(14).mean()

    # ADX
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where(
        (plus_dm > minus_dm) & (plus_dm > 0),
        0
    )

    minus_dm = minus_dm.where(
        (minus_dm > plus_dm) & (minus_dm > 0),
        0
    )

    atr14 = true_range.rolling(14).mean()

    plus_di = 100 * (
        plus_dm.rolling(14).mean() /
        atr14.replace(0, np.nan)
    )

    minus_di = 100 * (
        minus_dm.rolling(14).mean() /
        atr14.replace(0, np.nan)
    )

    dx = (
        (plus_di - minus_di).abs() /
        (plus_di + minus_di).replace(0, np.nan)
    ) * 100

    df["ADX"] = dx.rolling(14).mean()

    # Volume
    df["Volume_MA20"] = volume.rolling(20).mean()

    df["Volume_Ratio"] = (
        volume /
        df["Volume_MA20"].replace(0, np.nan)
    )

    # OBV
    direction = np.sign(close.diff()).fillna(0)

    df["OBV"] = (
        direction * volume
    ).cumsum()

    # VWAP
    typical_price = (high + low + close) / 3

    cumulative_volume = volume.cumsum()

    df["VWAP"] = (
        typical_price * volume
    ).cumsum() / cumulative_volume.replace(0, np.nan)

    # ROC
    df["ROC20"] = close.pct_change(20) * 100

    return df


# =========================================================
# SUPPORT / RESISTANCE
# =========================================================

def calculate_support_resistance(df):

    if df.empty:
        return np.nan, np.nan

    recent = df.tail(120)

    current = safe_float(recent["Close"].iloc[-1])

    lows = recent["Low"].rolling(5, center=True).min()
    highs = recent["High"].rolling(5, center=True).max()

    supports = recent.loc[
        lows == recent["Low"],
        "Low"
    ].dropna()

    resistances = recent.loc[
        highs == recent["High"],
        "High"
    ].dropna()

    support_candidates = [
        x for x in supports.tolist()
        if x < current
    ]

    resistance_candidates = [
        x for x in resistances.tolist()
        if x > current
    ]

    support = (
        max(support_candidates)
        if support_candidates
        else recent["Low"].min()
    )

    resistance = (
        min(resistance_candidates)
        if resistance_candidates
        else recent["High"].max()
    )

    return safe_float(support), safe_float(resistance)


# =========================================================
# FIBONACCI
# =========================================================

def calculate_fibonacci(df):

    if df.empty:
        return {}

    recent = df.tail(180)

    swing_high = safe_float(recent["High"].max())
    swing_low = safe_float(recent["Low"].min())

    if pd.isna(swing_high) or pd.isna(swing_low):
        return {}

    diff = swing_high - swing_low

    if diff <= 0:
        return {}

    return {
        "0.236": swing_high - diff * 0.236,
        "0.382": swing_high - diff * 0.382,
        "0.500": swing_high - diff * 0.500,
        "0.618": swing_high - diff * 0.618,
        "0.786": swing_high - diff * 0.786
    }


# =========================================================
# TREND
# =========================================================

def detect_trend(row):

    close = safe_float(row.get("Close"))
    ema20 = safe_float(row.get("EMA20"))
    ema50 = safe_float(row.get("EMA50"))
    ema200 = safe_float(row.get("EMA200"))

    if pd.isna(close):
        return "غير معروف"

    score = 0

    if not pd.isna(ema20) and close > ema20:
        score += 1

    if not pd.isna(ema50) and close > ema50:
        score += 1

    if not pd.isna(ema200) and close > ema200:
        score += 1

    if (
        not pd.isna(ema20)
        and not pd.isna(ema50)
        and ema20 > ema50
    ):
        score += 1

    if (
        not pd.isna(ema50)
        and not pd.isna(ema200)
        and ema50 > ema200
    ):
        score += 1

    if score >= 4:
        return "🟢 صاعد قوي"

    if score >= 3:
        return "🟢 صاعد"

    if score == 2:
        return "🟡 محايد"

    if score == 1:
        return "🟠 ضعيف"

    return "🔴 هابط"


# =========================================================
# FINANCIAL HELPERS
# =========================================================

def find_statement_value(df, possible_names):

    if df is None or df.empty:
        return np.nan

    for name in possible_names:

        if name in df.index:

            series = df.loc[name]

            if isinstance(series, pd.Series):

                series = series.dropna()

                if not series.empty:
                    return safe_float(series.iloc[0])

    return np.nan


def extract_financial_data(financials, info):

    income = financials.get("income", pd.DataFrame())
    balance = financials.get("balance", pd.DataFrame())
    cashflow = financials.get("cashflow", pd.DataFrame())

    revenue = find_statement_value(
        income,
        [
            "Total Revenue",
            "Operating Revenue",
            "Revenue"
        ]
    )

    net_income = find_statement_value(
        income,
        [
            "Net Income",
            "Net Income Common Stockholders",
            "Net Income Including Noncontrolling Interests"
        ]
    )

    operating_income = find_statement_value(
        income,
        [
            "Operating Income"
        ]
    )

    ebitda = safe_float(
        info.get("ebitda")
    )

    total_debt = find_statement_value(
        balance,
        [
            "Total Debt",
            "Long Term Debt",
            "Long Term Debt And Capital Lease Obligation"
        ]
    )

    cash = find_statement_value(
        balance,
        [
            "Cash Cash Equivalents And Short Term Investments",
            "Cash And Cash Equivalents",
            "Cash Financial"
        ]
    )

    total_assets = find_statement_value(
        balance,
        [
            "Total Assets"
        ]
    )

    total_liabilities = find_statement_value(
        balance,
        [
            "Total Liabilities Net Minority Interest",
            "Total Liabilities"
        ]
    )

    equity = find_statement_value(
        balance,
        [
            "Stockholders Equity",
            "Total Equity Gross Minority Interest",
            "Common Stock Equity"
        ]
    )

    current_assets = find_statement_value(
        balance,
        [
            "Current Assets"
        ]
    )

    current_liabilities = find_statement_value(
        balance,
        [
            "Current Liabilities"
        ]
    )

    operating_cashflow = find_statement_value(
        cashflow,
        [
            "Operating Cash Flow",
            "Total Cash From Operating Activities"
        ]
    )

    capex = find_statement_value(
        cashflow,
        [
            "Capital Expenditure",
            "Capital Expenditures"
        ]
    )

    if not pd.isna(operating_cashflow) and not pd.isna(capex):
        free_cashflow = operating_cashflow + capex
    else:
        free_cashflow = safe_float(
            info.get("freeCashflow")
        )

    return {
        "revenue": revenue,
        "net_income": net_income,
        "operating_income": operating_income,
        "ebitda": ebitda,
        "total_debt": total_debt,
        "cash": cash,
        "total_assets": total_assets,
        "total_liabilities": total_liabilities,
        "equity": equity,
        "current_assets": current_assets,
        "current_liabilities": current_liabilities,
        "operating_cashflow": operating_cashflow,
        "free_cashflow": free_cashflow
    }


# =========================================================
# GROWTH
# =========================================================

def calculate_growth(financials):

    income = financials.get(
        "income",
        pd.DataFrame()
    )

    result = {
        "revenue_growth": np.nan,
        "profit_growth": np.nan,
        "revenue_cagr": np.nan,
        "profit_cagr": np.nan
    }

    if income is None or income.empty:
        return result

    try:

        revenue_row = None

        for name in [
            "Total Revenue",
            "Operating Revenue",
            "Revenue"
        ]:
            if name in income.index:
                revenue_row = income.loc[name]
                break

        profit_row = None

        for name in [
            "Net Income",
            "Net Income Common Stockholders"
        ]:
            if name in income.index:
                profit_row = income.loc[name]
                break

        if revenue_row is not None:

            revenue = revenue_row.dropna()

            if len(revenue) >= 2:

                newest = safe_float(revenue.iloc[0])
                oldest = safe_float(revenue.iloc[-1])

                if oldest != 0:

                    result["revenue_growth"] = (
                        (newest / oldest) - 1
                    ) * 100

                years = len(revenue) - 1

                if (
                    oldest > 0
                    and newest > 0
                    and years > 0
                ):
                    result["revenue_cagr"] = (
                        (newest / oldest) **
                        (1 / years) - 1
                    ) * 100

        if profit_row is not None:

            profit = profit_row.dropna()

            if len(profit) >= 2:

                newest = safe_float(profit.iloc[0])
                oldest = safe_float(profit.iloc[-1])

                if oldest != 0:

                    result["profit_growth"] = (
                        (newest / oldest) - 1
                    ) * 100

                years = len(profit) - 1

                if (
                    oldest > 0
                    and newest > 0
                    and years > 0
                ):
                    result["profit_cagr"] = (
                        (newest / oldest) **
                        (1 / years) - 1
                    ) * 100

    except Exception:
        pass

    return result


# =========================================================
# RATIOS
# =========================================================

def calculate_ratios(financial_data, info):

    debt = safe_float(
        financial_data["total_debt"]
    )

    cash = safe_float(
        financial_data["cash"]
    )

    equity = safe_float(
        financial_data["equity"]
    )

    current_assets = safe_float(
        financial_data["current_assets"]
    )

    current_liabilities = safe_float(
        financial_data["current_liabilities"]
    )

    revenue = safe_float(
        financial_data["revenue"]
    )

    net_income = safe_float(
        financial_data["net_income"]
    )

    ebitda = safe_float(
        financial_data["ebitda"]
    )

    ratios = {}

    ratios["debt_equity"] = (
        debt / equity
        if not pd.isna(debt)
        and not pd.isna(equity)
        and equity != 0
        else np.nan
    )

    ratios["debt_ebitda"] = (
        debt / ebitda
        if not pd.isna(debt)
        and not pd.isna(ebitda)
        and ebitda > 0
        else np.nan
    )

    ratios["cash_debt"] = (
        cash / debt
        if not pd.isna(cash)
        and not pd.isna(debt)
        and debt > 0
        else np.nan
    )

    ratios["current_ratio"] = (
        current_assets / current_liabilities
        if not pd.isna(current_assets)
        and not pd.isna(current_liabilities)
        and current_liabilities != 0
        else safe_float(
            info.get("currentRatio")
        )
    )

    ratios["quick_ratio"] = safe_float(
        info.get("quickRatio")
    )

    ratios["profit_margin"] = (
        net_income / revenue * 100
        if not pd.isna(net_income)
        and not pd.isna(revenue)
        and revenue != 0
        else safe_float(
            info.get("profitMargins")
        ) * 100
    )

    ratios["roe"] = safe_float(
        info.get("returnOnEquity")
    )

    if not pd.isna(ratios["roe"]):
        ratios["roe"] *= 100

    ratios["roa"] = safe_float(
        info.get("returnOnAssets")
    )

    if not pd.isna(ratios["roa"]):
        ratios["roa"] *= 100

    ratios["pe"] = safe_float(
        info.get("trailingPE")
    )

    ratios["forward_pe"] = safe_float(
        info.get("forwardPE")
    )

    ratios["pb"] = safe_float(
        info.get("priceToBook")
    )

    ratios["ps"] = safe_float(
        info.get("priceToSalesTrailing12Months")
    )

    ratios["ev_ebitda"] = safe_float(
        info.get("enterpriseToEbitda")
    )

    ratios["peg"] = safe_float(
        info.get("pegRatio")
    )

    ratios["dividend_yield"] = safe_float(
        info.get("dividendYield")
    )

    return ratios


# =========================================================
# LIQUIDITY
# =========================================================

def calculate_liquidity(df, info):

    if df.empty:
        return {}

    close = safe_float(df["Close"].iloc[-1])

    volume = safe_float(df["Volume"].iloc[-1])

    avg_volume = safe_float(
        df["Volume"].tail(20).mean()
    )

    avg_value = safe_float(
        (df["Close"] * df["Volume"])
        .tail(20)
        .mean()
    )

    volume_ratio = (
        volume / avg_volume
        if avg_volume > 0
        else np.nan
    )

    return {
        "last_volume": volume,
        "avg_volume": avg_volume,
        "avg_trading_value": avg_value,
        "volume_ratio": volume_ratio,
        "market_cap": safe_float(
            info.get("marketCap")
        )
    }


# =========================================================
# RELATIVE STRENGTH
# =========================================================

def calculate_relative_strength(df):

    if df.empty or len(df) < 60:
        return np.nan

    close = df["Close"]

    return (
        (close.iloc[-1] / close.iloc[-60]) - 1
    ) * 100


# =========================================================
# TARGET ENGINE
# =========================================================

def calculate_targets(df, support, resistance, fib):

    if df.empty:
        return []

    current = safe_float(
        df["Close"].iloc[-1]
    )

    atr = safe_float(
        df["ATR"].iloc[-1]
    )

    candidates = []

    # Resistance
    if not pd.isna(resistance):
        if resistance > current:
            candidates.append(
                (resistance, "مقاومة حقيقية")
            )

    # Fibonacci
    for level, price in fib.items():

        price = safe_float(price)

        if (
            not pd.isna(price)
            and price > current
        ):
            candidates.append(
                (
                    price,
                    f"Fibonacci {level}"
                )
            )

    # Recent swing highs
    for window in [20, 60, 120]:

        if len(df) >= window:

            swing = safe_float(
                df["High"].tail(window).max()
            )

            if swing > current:
                candidates.append(
                    (
                        swing,
                        f"Swing High {window}"
                    )
                )

    # ATR fallback
    if not pd.isna(atr) and atr > 0:

        for multiplier in [2, 3, 4, 5]:

            candidates.append(
                (
                    current + atr * multiplier,
                    f"ATR × {multiplier}"
                )
            )

    # Remove duplicates / unrealistic targets
    cleaned = []

    for price, reason in candidates:

        if price <= current:
            continue

        distance = (
            (price - current) /
            current
        ) * 100

        if distance < 1:
            continue

        if distance > 100:
            continue

        cleaned.append(
            (
                round(price, 4),
                reason,
                distance
            )
        )

    cleaned.sort(key=lambda x: x[0])

    targets = []

    for price, reason, distance in cleaned:

        if not targets:

            targets.append(
                (price, reason, distance)
            )

        else:

            previous = targets[-1][0]

            # Minimum separation
            if price > previous * 1.02:

                targets.append(
                    (price, reason, distance)
                )

        if len(targets) == 4:
            break

    return targets


# =========================================================
# ENTRY ENGINE
# =========================================================

def calculate_entries(df, support, resistance):

    if df.empty:
        return {}

    current = safe_float(
        df["Close"].iloc[-1]
    )

    ema20 = safe_float(
        df["EMA20"].iloc[-1]
    )

    ema50 = safe_float(
        df["EMA50"].iloc[-1]
    )

    atr = safe_float(
        df["ATR"].iloc[-1]
    )

    pullback = np.nan

    if not pd.isna(ema20):
        pullback = ema20

    if (
        not pd.isna(support)
        and support < current
    ):

        if pd.isna(pullback):
            pullback = support
        else:
            pullback = max(
                support,
                min(pullback, current)
            )

    breakout = np.nan

    if not pd.isna(resistance):
        breakout = resistance * 1.01

    stop = np.nan

    if not pd.isna(support):
        stop = support * 0.98

    elif not pd.isna(atr):
        stop = current - atr * 2

    return {
        "current": current,
        "pullback": pullback,
        "breakout": breakout,
        "stop": stop
    }


# =========================================================
# TECHNICAL SCORE
# =========================================================

def calculate_technical_score(df):

    if df.empty:
        return 0, []

    row = df.iloc[-1]

    score = 0
    reasons = []

    close = safe_float(row["Close"])
    ema20 = safe_float(row["EMA20"])
    ema50 = safe_float(row["EMA50"])
    ema200 = safe_float(row["EMA200"])

    rsi = safe_float(row["RSI"])
    macd = safe_float(row["MACD"])
    signal = safe_float(row["MACD_SIGNAL"])
    adx = safe_float(row["ADX"])
    volume_ratio = safe_float(row["Volume_Ratio"])

    # EMA structure
    if (
        not pd.isna(ema20)
        and close > ema20
    ):
        score += 4
        reasons.append(
            "السعر فوق EMA20"
        )

    if (
        not pd.isna(ema50)
        and close > ema50
    ):
        score += 5
        reasons.append(
            "السعر فوق EMA50"
        )

    if (
        not pd.isna(ema200)
        and close > ema200
    ):
        score += 6
        reasons.append(
            "السعر فوق EMA200"
        )

    if (
        not pd.isna(ema20)
        and not pd.isna(ema50)
        and ema20 > ema50
    ):
        score += 4
        reasons.append(
            "EMA20 أعلى من EMA50"
        )

    if (
        not pd.isna(ema50)
        and not pd.isna(ema200)
        and ema50 > ema200
    ):
        score += 5
        reasons.append(
            "EMA50 أعلى من EMA200"
        )

    # RSI
    if not pd.isna(rsi):

        if 50 <= rsi <= 70:
            score += 4
            reasons.append(
                "RSI في منطقة إيجابية"
            )

        elif 40 <= rsi < 50:
            score += 2

        elif rsi > 75:
            score -= 2
            reasons.append(
                "RSI مرتفع جدًا"
            )

    # MACD
    if (
        not pd.isna(macd)
        and not pd.isna(signal)
        and macd > signal
    ):
        score += 4
        reasons.append(
            "MACD إيجابي"
        )

    # ADX
    if not pd.isna(adx):

        if adx >= 25:
            score += 2
            reasons.append(
                "الاتجاه مدعوم بـ ADX"
            )

    # Volume
    if not pd.isna(volume_ratio):

        if volume_ratio >= 1.5:
            score += 3
            reasons.append(
                "ارتفاع قوي في حجم التداول"
            )

        elif volume_ratio >= 1.2:
            score += 1

    score = max(
        0,
        min(
            TECH_SCORE_MAX,
            score
        )
    )

    return score, reasons


# =========================================================
# FUNDAMENTAL SCORE
# =========================================================

def calculate_fundamental_score(
    financial_data,
    growth,
    ratios
):

    score = 0
    reasons = []

    revenue_growth = safe_float(
        growth["revenue_growth"]
    )

    profit_growth = safe_float(
        growth["profit_growth"]
    )

    roe = safe_float(
        ratios["roe"]
    )

    margin = safe_float(
        ratios["profit_margin"]
    )

    current_ratio = safe_float(
        ratios["current_ratio"]
    )

    debt_equity = safe_float(
        ratios["debt_equity"]
    )

    fcf = safe_float(
        financial_data["free_cashflow"]
    )

    # Revenue growth
    if not pd.isna(revenue_growth):

        if revenue_growth > 30:
            score += 7
            reasons.append(
                "نمو الإيرادات قوي"
            )

        elif revenue_growth > 10:
            score += 5
            reasons.append(
                "نمو الإيرادات جيد"
            )

        elif revenue_growth > 0:
            score += 2

        else:
            score -= 2
            reasons.append(
                "الإيرادات في تراجع"
            )

    # Profit growth
    if not pd.isna(profit_growth):

        if profit_growth > 30:
            score += 7
            reasons.append(
                "نمو الأرباح قوي"
            )

        elif profit_growth > 10:
            score += 5
            reasons.append(
                "نمو الأرباح جيد"
            )

        elif profit_growth > 0:
            score += 2

        else:
            score -= 3
            reasons.append(
                "الأرباح في تراجع"
            )

    # ROE
    if not pd.isna(roe):

        if roe > 20:
            score += 6
            reasons.append(
                "ROE ممتاز"
            )

        elif roe > 12:
            score += 4

        elif roe > 5:
            score += 2

    # Margin
    if not pd.isna(margin):

        if margin > 20:
            score += 5
            reasons.append(
                "هامش الربح قوي"
            )

        elif margin > 10:
            score += 3

        elif margin > 0:
            score += 1

    # Liquidity
    if not pd.isna(current_ratio):

        if current_ratio >= 2:
            score += 3
            reasons.append(
                "السيولة المالية جيدة"
            )

        elif current_ratio >= 1:
            score += 2

        else:
            score -= 2
            reasons.append(
                "Current Ratio ضعيف"
            )

    # Debt
    if not pd.isna(debt_equity):

        if debt_equity < 0.5:
            score += 5
            reasons.append(
                "الديون منخفضة مقارنة بحقوق الملكية"
            )

        elif debt_equity < 1:
            score += 3

        elif debt_equity > 2:
            score -= 4
            reasons.append(
                "نسبة الديون مرتفعة"
            )

    # FCF
    if not pd.isna(fcf):

        if fcf > 0:
            score += 2
            reasons.append(
                "Free Cash Flow موجب"
            )

        else:
            score -= 2
            reasons.append(
                "Free Cash Flow سلبي"
            )

    score = max(
        0,
        min(
            FUND_SCORE_MAX,
            score
        )
    )

    return score, reasons


# =========================================================
# VALUATION SCORE
# =========================================================

def calculate_valuation_score(ratios):

    score = 0
    reasons = []

    pe = safe_float(
        ratios["pe"]
    )

    pb = safe_float(
        ratios["pb"]
    )

    ps = safe_float(
        ratios["ps"]
    )

    ev_ebitda = safe_float(
        ratios["ev_ebitda"]
    )

    # P/E
    if not pd.isna(pe) and pe > 0:

        if pe < 10:
            score += 5
            reasons.append(
                "P/E منخفض نسبيًا"
            )

        elif pe < 18:
            score += 3

        elif pe > 30:
            score -= 2
            reasons.append(
                "P/E مرتفع"
            )

    # P/B
    if not pd.isna(pb) and pb > 0:

        if pb < 1.5:
            score += 3

        elif pb > 5:
            score -= 1

    # P/S
    if not pd.isna(ps) and ps > 0:

        if ps < 2:
            score += 2

        elif ps > 8:
            score -= 1

    # EV/EBITDA
    if not pd.isna(ev_ebitda) and ev_ebitda > 0:

        if ev_ebitda < 8:
            score += 5
            reasons.append(
                "EV/EBITDA جذاب"
            )

        elif ev_ebitda < 15:
            score += 3

        elif ev_ebitda > 25:
            score -= 2

    score = max(
        0,
        min(
            VALUATION_SCORE_MAX,
            score
        )
    )

    return score, reasons


# =========================================================
# LIQUIDITY SCORE
# =========================================================

def calculate_liquidity_score(liquidity):

    score = 0
    reasons = []

    avg_value = safe_float(
        liquidity.get(
            "avg_trading_value"
        )
    )

    volume_ratio = safe_float(
        liquidity.get(
            "volume_ratio"
        )
    )

    if not pd.isna(avg_value):

        if avg_value >= 50_000_000:
            score += 5
            reasons.append(
                "سيولة تداول قوية جدًا"
            )

        elif avg_value >= 10_000_000:
            score += 4
            reasons.append(
                "سيولة تداول جيدة"
            )

        elif avg_value >= 2_000_000:
            score += 2

        elif avg_value < 500_000:
            score -= 2
            reasons.append(
                "سيولة التداول ضعيفة"
            )

    if not pd.isna(volume_ratio):

        if volume_ratio >= 1.5:
            score += 5
            reasons.append(
                "Volume Spike واضح"
            )

        elif volume_ratio >= 1.2:
            score += 3

        elif volume_ratio < 0.7:
            score -= 1

    return max(
        0,
        min(
            LIQUIDITY_SCORE_MAX,
            score
        )
    ), reasons


# =========================================================
# NEWS SCORE
# =========================================================

def calculate_news_score(news):

    if not news:
        return 0, [
            "لا توجد أخبار كافية للتقييم"
        ]

    positive_words = [
        "profit",
        "growth",
        "dividend",
        "acquisition",
        "contract",
        "revenue",
        "earnings",
        "positive",
        "increase",
        "approval",
        "توزيعات",
        "أرباح",
        "نمو",
        "عقد",
        "زيادة"
    ]

    negative_words = [
        "loss",
        "debt",
        "decline",
        "lawsuit",
        "investigation",
        "warning",
        "negative",
        "decrease",
        "bankruptcy",
        "خسائر",
        "ديون",
        "تراجع",
        "تحقيق",
        "انخفاض"
    ]

    score = 0

    for item in news[:10]:

        title = item["title"].lower()

        pos = sum(
            word.lower() in title
            for word in positive_words
        )

        neg = sum(
            word.lower() in title
            for word in negative_words
        )

        if pos > neg:
            score += 1

        elif neg > pos:
            score -= 1

    score = max(
        -NEWS_SCORE_MAX,
        min(
            NEWS_SCORE_MAX,
            score
        )
    )

    return score, []


# =========================================================
# FINAL SCORE
# =========================================================

def calculate_final_score(
    technical_score,
    fundamental_score,
    valuation_score,
    liquidity_score,
    news_score
):

    # News score is centered around zero
    normalized_news = (
        news_score + NEWS_SCORE_MAX
    )

    raw = (
        technical_score +
        fundamental_score +
        valuation_score +
        liquidity_score +
        normalized_news
    )

    maximum = (
        TECH_SCORE_MAX +
        FUND_SCORE_MAX +
        VALUATION_SCORE_MAX +
        LIQUIDITY_SCORE_MAX +
        NEWS_SCORE_MAX * 2
    )

    final_score = (
        raw / maximum
    ) * 100

    return round(
        max(
            0,
            min(
                100,
                final_score
            )
        ),
        1
    )


def final_rating(score):

    if score >= 85:
        return "🟢 قوي جدًا", "good"

    if score >= 75:
        return "🟢 إيجابي قوي", "good"

    if score >= 65:
        return "🟢 إيجابي", "good"

    if score >= 55:
        return "🟡 محايد يميل للإيجابية", "warning"

    if score >= 45:
        return "🟡 محايد", "neutral"

    if score >= 35:
        return "🟠 ضعيف", "warning"

    return "🔴 سلبي", "bad"


# =========================================================
# RISK
# =========================================================

def calculate_risk(
    ratios,
    liquidity,
    df
):

    risk = 0
    reasons = []

    debt_equity = safe_float(
        ratios.get("debt_equity")
    )

    current_ratio = safe_float(
        ratios.get("current_ratio")
    )

    avg_value = safe_float(
        liquidity.get(
            "avg_trading_value"
        )
    )

    rsi = safe_float(
        df["RSI"].iloc[-1]
    ) if not df.empty else np.nan

    if not pd.isna(debt_equity):

        if debt_equity > 2:
            risk += 3
            reasons.append(
                "مديونية مرتفعة"
            )

        elif debt_equity > 1:
            risk += 1

    if not pd.isna(current_ratio):

        if current_ratio < 1:
            risk += 2
            reasons.append(
                "السيولة قصيرة الأجل ضعيفة"
            )

    if not pd.isna(avg_value):

        if avg_value < 500_000:
            risk += 3
            reasons.append(
                "سيولة تداول منخفضة"
            )

        elif avg_value < 2_000_000:
            risk += 1

    if not pd.isna(rsi):

        if rsi > 80:
            risk += 2
            reasons.append(
                "السهم في تشبع شرائي"
            )

    if risk >= 6:
        label = "🔴 مرتفع"

    elif risk >= 3:
        label = "🟠 متوسط"

    else:
        label = "🟢 منخفض"

    return label, reasons


# =========================================================
# FULL ANALYSIS
# =========================================================

def analyze_stock(symbol):

    df = load_price_data(
        symbol,
        DEFAULT_PERIOD
    )

    if df.empty:
        return {
            "error": "لم يتم العثور على بيانات سعرية للسهم."
        }

    info = load_info(symbol)

    financials = load_financials(symbol)

    news = load_news(symbol)

    df = add_indicators(df)

    financial_data = extract_financial_data(
        financials,
        info
    )

    growth = calculate_growth(
        financials
    )

    ratios = calculate_ratios(
        financial_data,
        info
    )

    liquidity = calculate_liquidity(
        df,
        info
    )

    relative_strength = calculate_relative_strength(
        df
    )

    support, resistance = (
        calculate_support_resistance(df)
    )

    fib = calculate_fibonacci(df)

    entries = calculate_entries(
        df,
        support,
        resistance
    )

    targets = calculate_targets(
        df,
        support,
        resistance,
        fib
    )

    technical_score, technical_reasons = (
        calculate_technical_score(df)
    )

    fundamental_score, fundamental_reasons = (
        calculate_fundamental_score(
            financial_data,
            growth,
            ratios
        )
    )

    valuation_score, valuation_reasons = (
        calculate_valuation_score(
            ratios
        )
    )

    liquidity_score, liquidity_reasons = (
        calculate_liquidity_score(
            liquidity
        )
    )

    news_score, news_reasons = (
        calculate_news_score(news)
    )

    final_score = calculate_final_score(
        technical_score,
        fundamental_score,
        valuation_score,
        liquidity_score,
        news_score
    )

    rating, rating_class = final_rating(
        final_score
    )

    risk_label, risk_reasons = calculate_risk(
        ratios,
        liquidity,
        df
    )

    trend = detect_trend(
        df.iloc[-1]
    )

    current = safe_float(
        df["Close"].iloc[-1]
    )

    return {
        "symbol": symbol,
        "raw_symbol": get_raw_symbol(symbol),
        "name": info.get(
            "longName",
            info.get(
                "shortName",
                symbol
            )
        ),
        "sector": info.get(
            "sector",
            "غير متاح"
        ),
        "industry": info.get(
            "industry",
            "غير متاح"
        ),
        "currency": info.get(
            "currency",
            "EGP"
        ),
        "current": current,
        "df": df,
        "info": info,
        "financials": financials,
        "financial_data": financial_data,
        "growth": growth,
        "ratios": ratios,
        "liquidity": liquidity,
        "relative_strength": relative_strength,
        "support": support,
        "resistance": resistance,
        "fib": fib,
        "entries": entries,
        "targets": targets,
        "news": news,
        "trend": trend,
        "technical_score": technical_score,
        "technical_reasons": technical_reasons,
        "fundamental_score": fundamental_score,
        "fundamental_reasons": fundamental_reasons,
        "valuation_score": valuation_score,
        "valuation_reasons": valuation_reasons,
        "liquidity_score": liquidity_score,
        "liquidity_reasons": liquidity_reasons,
        "news_score": news_score,
        "news_reasons": news_reasons,
        "final_score": final_score,
        "rating": rating,
        "rating_class": rating_class,
        "risk": risk_label,
        "risk_reasons": risk_reasons
    }


# =========================================================
# UI HELPERS
# =========================================================

def metric_card(title, value):

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def score_progress(label, score, maximum):

    percentage = (
        score / maximum * 100
        if maximum > 0
        else 0
    )

    st.write(
        f"**{label}: {score:.1f}/{maximum}**"
    )

    st.progress(
        max(
            0,
            min(
                100,
                int(percentage)
            )
        )
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ إعدادات التحليل")

    symbol_input = st.text_input(
        "كود السهم",
        value="COMI",
        help="مثال: COMI أو SWDY أو EAST"
    )

    auto_ca = st.checkbox(
        "إضافة .CA تلقائيًا",
        value=True
    )

    st.markdown("---")

    st.info(
        """
        البرنامج بيجمع:
        
        • Fundamentals
        • Growth
        • Debt
        • Liquidity
        • Valuation
        • Technical
        • Support/Resistance
        • Fibonacci
        • Targets
        • News
        • Risk
        • Final Score
        """
    )

# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">📊 EGX Stock Intelligence PRO</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">تحليل شامل للسهم: مالي + فني + تقييم + سيولة + أخبار + مخاطر</div>',
    unsafe_allow_html=True
)

# =========================================================
# ANALYZE BUTTON
# =========================================================

if auto_ca:
    symbol = clean_symbol(
        symbol_input
    )
else:
    symbol = symbol_input.strip().upper()

col1, col2, col3 = st.columns(
    [2, 1, 1]
)

with col1:
    analyze = st.button(
        "🔍 تحليل السهم بالكامل",
        type="primary",
        use_container_width=True
    )

with col2:
    st.metric(
        "السهم",
        get_raw_symbol(symbol)
    )

with col3:
    st.metric(
        "تاريخ التحليل",
        datetime.now().strftime(
            "%Y-%m-%d"
        )
    )

# =========================================================
# RUN
# =========================================================

if analyze:

    with st.spinner(
        "⏳ جاري جمع وتحليل بيانات السهم..."
    ):

        result = analyze_stock(
            symbol
        )

    if "error" in result:

        st.error(
            result["error"]
        )

        st.warning(
            "تأكد إن كود السهم صحيح وإن Yahoo Finance عنده بيانات للسهم."
        )

        st.stop()

    # =====================================================
    # COMPANY HEADER
    # =====================================================

    st.markdown(
        '<div class="section">🏢 معلومات الشركة</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        metric_card(
            "الشركة",
            result["name"]
        )

    with c2:
        metric_card(
            "السعر الحالي",
            fmt_price(result["current"])
        )

    with c3:
        metric_card(
            "القطاع",
            result["sector"]
        )

    with c4:
        metric_card(
            "الصناعة",
            result["industry"]
        )

    with c5:
        metric_card(
            "القيمة السوقية",
            fmt_number(
                result["liquidity"].get(
                    "market_cap"
                )
            )
        )

    # =====================================================
    # FINAL SCORE
    # =====================================================

    st.markdown(
        '<div class="section">⭐ التقييم النهائي</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Final Score",
            f'{result["final_score"]}/100'
        )

    with c2:
        st.markdown(
            f"### {result['rating']}"
        )

    with c3:
        st.markdown(
            f"### مخاطر: {result['risk']}"
        )

    # =====================================================
    # SCORE BREAKDOWN
    # =====================================================

    st.markdown(
        "### توزيع التقييم"
    )

    s1, s2 = st.columns(2)

    with s1:

        score_progress(
            "التحليل الفني",
            result["technical_score"],
            TECH_SCORE_MAX
        )

        score_progress(
            "التحليل المالي",
            result["fundamental_score"],
            FUND_SCORE_MAX
        )

        score_progress(
            "التقييم السعري",
            result["valuation_score"],
            VALUATION_SCORE_MAX
        )

    with s2:

        score_progress(
            "السيولة",
            result["liquidity_score"],
            LIQUIDITY_SCORE_MAX
        )

        news_display = (
            result["news_score"] +
            NEWS_SCORE_MAX
        )

        score_progress(
            "الأخبار",
            news_display,
            NEWS_SCORE_MAX * 2
        )

        st.write(
            f"**الاتجاه الحالي:** {result['trend']}"
        )

    # =====================================================
    # PRICE CHART
    # =====================================================

    st.markdown(
        '<div class="section">📈 حركة السهم</div>',
        unsafe_allow_html=True
    )

    chart_df = result["df"].copy()

    st.line_chart(
        chart_df[
            [
                "Close",
                "EMA20",
                "EMA50",
                "EMA200"
            ]
        ].tail(180)
    )

    # =====================================================
    # FINANCIALS
    # =====================================================

    st.markdown(
        '<div class="section">💰 البيانات المالية</div>',
        unsafe_allow_html=True
    )

    fd = result["financial_data"]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "الإيرادات",
            fmt_number(fd["revenue"])
        )

    with c2:
        metric_card(
            "صافي الأرباح",
            fmt_number(fd["net_income"])
        )

    with c3:
        metric_card(
            "EBITDA",
            fmt_number(fd["ebitda"])
        )

    with c4:
        metric_card(
            "Operating Cash Flow",
            fmt_number(
                fd["operating_cashflow"]
            )
        )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Free Cash Flow",
            fmt_number(
                fd["free_cashflow"]
            )
        )

    with c2:
        metric_card(
            "إجمالي الديون",
            fmt_number(
                fd["total_debt"]
            )
        )

    with c3:
        metric_card(
            "النقدية",
            fmt_number(
                fd["cash"]
            )
        )

    with c4:
        metric_card(
            "حقوق الملكية",
            fmt_number(
                fd["equity"]
            )
        )

    # =====================================================
    # GROWTH
    # =====================================================

    st.markdown(
        '<div class="section">📈 النمو</div>',
        unsafe_allow_html=True
    )

    growth = result["growth"]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "نمو الإيرادات",
            fmt_pct(
                growth["revenue_growth"]
            )
        )

    with c2:
        metric_card(
            "نمو الأرباح",
            fmt_pct(
                growth["profit_growth"]
            )
        )

    with c3:
        metric_card(
            "Revenue CAGR",
            fmt_pct(
                growth["revenue_cagr"]
            )
        )

    with c4:
        metric_card(
            "Profit CAGR",
            fmt_pct(
                growth["profit_cagr"]
            )
        )

    # =====================================================
    # DEBT & LIQUIDITY
    # =====================================================

    st.markdown(
        '<div class="section">🏦 الديون والسيولة المالية</div>',
        unsafe_allow_html=True
    )

    ratios = result["ratios"]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Debt / Equity",
            fmt_number(
                ratios["debt_equity"]
            )
        )

    with c2:
        metric_card(
            "Debt / EBITDA",
            fmt_number(
                ratios["debt_ebitda"]
            )
        )

    with c3:
        metric_card(
            "Cash / Debt",
            fmt_number(
                ratios["cash_debt"]
            )
        )

    with c4:
        metric_card(
            "Current Ratio",
            fmt_number(
                ratios["current_ratio"]
            )
        )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Quick Ratio",
            fmt_number(
                ratios["quick_ratio"]
            )
        )

    with c2:
        metric_card(
            "ROE",
            fmt_pct(
                ratios["roe"]
            )
        )

    with c3:
        metric_card(
            "ROA",
            fmt_pct(
                ratios["roa"]
            )
        )

    with c4:
        metric_card(
            "Profit Margin",
            fmt_pct(
                ratios["profit_margin"]
            )
        )

    # =====================================================
    # VALUATION
    # =====================================================

    st.markdown(
        '<div class="section">💵 التقييم السعري</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "P/E",
            fmt_number(
                ratios["pe"]
            )
        )

    with c2:
        metric_card(
            "Forward P/E",
            fmt_number(
                ratios["forward_pe"]
            )
        )

    with c3:
        metric_card(
            "P/B",
            fmt_number(
                ratios["pb"]
            )
        )

    with c4:
        metric_card(
            "P/S",
            fmt_number(
                ratios["ps"]
            )
        )

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "EV/EBITDA",
            fmt_number(
                ratios["ev_ebitda"]
            )
        )

    with c2:
        metric_card(
            "PEG",
            fmt_number(
                ratios["peg"]
            )
        )

    with c3:
        metric_card(
            "Dividend Yield",
            fmt_pct(
                ratios["dividend_yield"]
            )
        )

    # =====================================================
    # TECHNICAL
    # =====================================================

    st.markdown(
        '<div class="section">📊 التحليل الفني</div>',
        unsafe_allow_html=True
    )

    latest = result["df"].iloc[-1]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "RSI",
            fmt_number(
                latest["RSI"]
            )
        )

    with c2:
        metric_card(
            "MACD",
            fmt_number(
                latest["MACD"]
            )
        )

    with c3:
        metric_card(
            "ADX",
            fmt_number(
                latest["ADX"]
            )
        )

    with c4:
        metric_card(
            "ATR",
            fmt_number(
                latest["ATR"]
            )
        )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "EMA20",
            fmt_price(
                latest["EMA20"]
            )
        )

    with c2:
        metric_card(
            "EMA50",
            fmt_price(
                latest["EMA50"]
            )
        )

    with c3:
        metric_card(
            "EMA200",
            fmt_price(
                latest["EMA200"]
            )
        )

    with c4:
        metric_card(
            "VWAP",
            fmt_price(
                latest["VWAP"]
            )
        )

    # =====================================================
    # SUPPORT / RESISTANCE
    # =====================================================

    st.markdown(
        '<div class="section">🎯 الدعم والمقاومة</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "السعر الحالي",
            fmt_price(
                result["current"]
            )
        )

    with c2:
        metric_card(
            "الدعم الأقرب",
            fmt_price(
                result["support"]
            )
        )

    with c3:
        metric_card(
            "المقاومة الأقرب",
            fmt_price(
                result["resistance"]
            )
        )

    # =====================================================
    # ENTRIES
    # =====================================================

    st.markdown(
        '<div class="section">🚦 مناطق الدخول</div>',
        unsafe_allow_html=True
    )

    entries = result["entries"]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "دخول فوري",
            fmt_price(
                entries["current"]
            )
        )

    with c2:
        metric_card(
            "دخول Pullback",
            fmt_price(
                entries["pullback"]
            )
        )

    with c3:
        metric_card(
            "دخول Breakout",
            fmt_price(
                entries["breakout"]
            )
        )

    with c4:
        metric_card(
            "Stop Loss",
            fmt_price(
                entries["stop"]
            )
        )

    # =====================================================
    # TARGETS
    # =====================================================

    st.markdown(
        '<div class="section">🎯 الأهداف</div>',
        unsafe_allow_html=True
    )

    if result["targets"]:

        target_rows = []

        for i, (
            price,
            reason,
            profit_pct
        ) in enumerate(
            result["targets"],
            start=1
        ):

            target_rows.append({
                "الهدف": f"TP{i}",
                "السعر": round(price, 2),
                "الربح المتوقع": f"{profit_pct:.2f}%",
                "سبب الهدف": reason
            })

        target_df = pd.DataFrame(
            target_rows
        )

        st.dataframe(
            target_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "لم يتم العثور على أهداف موثوقة فوق السعر الحالي."
        )

    # =====================================================
    # FIBONACCI
    # =====================================================

    st.markdown(
        '<div class="section">📐 Fibonacci</div>',
        unsafe_allow_html=True
    )

    if result["fib"]:

        fib_df = pd.DataFrame(
            [
                {
                    "المستوى": level,
                    "السعر": round(
                        price,
                        2
                    )
                }
                for level, price
                in result["fib"].items()
            ]
        )

        st.dataframe(
            fib_df,
            use_container_width=True,
            hide_index=True
        )

    # =====================================================
    # LIQUIDITY
    # =====================================================

    st.markdown(
        '<div class="section">💧 سيولة التداول</div>',
        unsafe_allow_html=True
    )

    liq = result["liquidity"]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "متوسط حجم التداول",
            fmt_number(
                liq.get(
                    "avg_volume"
                )
            )
        )

    with c2:
        metric_card(
            "متوسط قيمة التداول",
            fmt_number(
                liq.get(
                    "avg_trading_value"
                )
            )
        )

    with c3:
        metric_card(
            "Volume Ratio",
            fmt_number(
                liq.get(
                    "volume_ratio"
                )
            )
        )

    with c4:
        metric_card(
            "Relative Strength 60D",
            fmt_pct(
                result["relative_strength"]
            )
        )

    # =====================================================
    # REASONS
    # =====================================================

    st.markdown(
        '<div class="section">🧠 أسباب التقييم</div>',
        unsafe_allow_html=True
    )

    tabs = st.tabs([
        "📊 فني",
        "💰 مالي",
        "💵 تقييم",
        "💧 سيولة",
        "⚠️ مخاطر"
    ])

    with tabs[0]:

        if result["technical_reasons"]:

            for reason in result["technical_reasons"]:
                st.success(
                    f"✓ {reason}"
                )

        else:
            st.info(
                "لا توجد أسباب فنية كافية."
            )

    with tabs[1]:

        if result["fundamental_reasons"]:

            for reason in result["fundamental_reasons"]:
                st.success(
                    f"✓ {reason}"
                )

        else:
            st.info(
                "البيانات المالية غير كافية."
            )

    with tabs[2]:

        if result["valuation_reasons"]:

            for reason in result["valuation_reasons"]:
                st.success(
                    f"✓ {reason}"
                )

        else:
            st.info(
                "لا توجد بيانات تقييم كافية."
            )

    with tabs[3]:

        if result["liquidity_reasons"]:

            for reason in result["liquidity_reasons"]:
                st.success(
                    f"✓ {reason}"
                )

        else:
            st.info(
                "لا توجد بيانات سيولة كافية."
            )

    with tabs[4]:

        if result["risk_reasons"]:

            for reason in result["risk_reasons"]:
                st.warning(
                    f"⚠️ {reason}"
                )

        else:
            st.success(
                "✓ لا توجد مخاطر واضحة حسب البيانات المتاحة."
            )

    # =====================================================
    # NEWS
    # =====================================================

    st.markdown(
        '<div class="section">📰 آخر الأخبار</div>',
        unsafe_allow_html=True
    )

    if result["news"]:

        for item in result["news"]:

            title = item["title"]
            publisher = item["publisher"]
            link = item["link"]

            if link:

                st.markdown(
                    f"### [{title}]({link})"
                )

            else:

                st.markdown(
                    f"### {title}"
                )

            st.caption(
                f"المصدر: {publisher}"
            )

            st.divider()

    else:

        st.info(
            "لم يتم العثور على أخبار متاحة من المصدر الحالي."
        )

    # =====================================================
    # RAW DATA
    # =====================================================

    with st.expander(
        "🔎 عرض البيانات الخام"
    ):

        st.dataframe(
            result["df"].tail(100),
            use_container_width=True
        )

    # =====================================================
    # DOWNLOAD
    # =====================================================

    export = pd.DataFrame({
        "Metric": [
            "Symbol",
            "Company",
            "Current Price",
            "Final Score",
            "Rating",
            "Risk",
            "Trend",
            "Technical Score",
            "Fundamental Score",
            "Valuation Score",
            "Liquidity Score",
            "News Score",
            "Support",
            "Resistance",
            "Immediate Entry",
            "Pullback Entry",
            "Breakout Entry",
            "Stop Loss",
            "RSI",
            "MACD",
            "ADX",
            "EMA20",
            "EMA50",
            "EMA200",
            "Revenue",
            "Net Income",
            "Total Debt",
            "Cash",
            "Revenue Growth",
            "Profit Growth",
            "Debt/Equity",
            "Current Ratio",
            "ROE",
            "P/E",
            "P/B",
            "EV/EBITDA",
            "Average Trading Value",
            "Relative Strength 60D"
        ],
        "Value": [
            result["raw_symbol"],
            result["name"],
            result["current"],
            result["final_score"],
            result["rating"],
            result["risk"],
            result["trend"],
            result["technical_score"],
            result["fundamental_score"],
            result["valuation_score"],
            result["liquidity_score"],
            result["news_score"],
            result["support"],
            result["resistance"],
            result["entries"]["current"],
            result["entries"]["pullback"],
            result["entries"]["breakout"],
            result["entries"]["stop"],
            latest["RSI"],
            latest["MACD"],
            latest["ADX"],
            latest["EMA20"],
            latest["EMA50"],
            latest["EMA200"],
            fd["revenue"],
            fd["net_income"],
            fd["total_debt"],
            fd["cash"],
            growth["revenue_growth"],
            growth["profit_growth"],
            ratios["debt_equity"],
            ratios["current_ratio"],
            ratios["roe"],
            ratios["pe"],
            ratios["pb"],
            ratios["ev_ebitda"],
            liq["avg_trading_value"],
            result["relative_strength"]
        ]
    })

    csv = export.to_csv(
        index=False,
        encoding="utf-8-sig"
    )

    st.download_button(
        "📥 تحميل تقرير السهم CSV",
        data=csv,
        file_name=f"{result['raw_symbol']}_analysis.csv",
        mime="text/csv",
        use_container_width=True
    )

else:

    st.info(
        "👆 اكتب كود السهم واضغط «تحليل السهم بالكامل»."
    )

    st.markdown("""
    ### أمثلة

    `COMI` — البنك التجاري الدولي

    `SWDY` — السويدي إليكتريك

    `EAST` — الشرقية للدخان

    `MFPC` — مصر لإنتاج الأسمدة

    `PHDC` — بالم هيلز

    **ملاحظة:** البرنامج بيضيف `.CA` تلقائيًا عند استخدام Yahoo Finance.
    """)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "EGX Stock Intelligence PRO — البيانات تعتمد على المصادر المتاحة، "
    "وأي بيانات غير متاحة يتم عرضها كغير متاحة بدل اختلاق قيمة."
)
