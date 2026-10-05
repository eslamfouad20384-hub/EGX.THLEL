import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================================================
# EGX STOCK INTELLIGENCE PRO
# Technical + Fundamental + Valuation + Investment Engine
# =========================================================

st.set_page_config(
    page_title="EGX Stock Intelligence PRO",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main {direction: rtl;}
[data-testid="stSidebar"] {direction: rtl;}
h1,h2,h3,h4 {direction: rtl;}
.metric-card {
    padding: 14px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,.25);
    margin-bottom: 8px;
}
.small-note {font-size: 13px; opacity: .8;}
.good {font-weight:700;}
.bad {font-weight:700;}
</style>
""", unsafe_allow_html=True)

# =========================
# CONFIG
# =========================
DEFAULT_PERIOD = "2y"
TECH_SCORE_MAX = 35.0
FUND_SCORE_MAX = 35.0
VALUATION_SCORE_MAX = 15.0
LIQUIDITY_SCORE_MAX = 10.0
NEWS_SCORE_MAX = 5.0

CACHE_TTL = 900
REQUEST_TIMEOUT = 12

RISK_FREE_RATE = 0.18       # configurable assumption for Egypt
EQUITY_RISK_PREMIUM = 0.08
TERMINAL_GROWTH = 0.05
DEFAULT_MARGIN_OF_SAFETY = 0.20

# =========================================================
# HELPERS
# =========================================================
def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def clean_symbol(symbol):
    if not symbol:
        return ""
    s = str(symbol).strip().upper()
    if s.endswith(".CA"):
        return s
    return s + ".CA"


def get_raw_symbol(symbol):
    return clean_symbol(symbol).replace(".CA", "")


def fmt_number(x, decimals=2):
    v = safe_float(x)
    if not np.isfinite(v):
        return "غير متاح"
    return f"{v:,.{decimals}f}"


def fmt_price(x):
    v = safe_float(x)
    if not np.isfinite(v):
        return "غير متاح"
    return f"{v:,.2f} ج"


def fmt_pct(x, decimals=2):
    v = safe_float(x)
    if not np.isfinite(v):
        return "غير متاح"
    return f"{v:.{decimals}f}%"


def first_valid(*values):
    for x in values:
        v = safe_float(x)
        if np.isfinite(v):
            return v
    return np.nan


def clamp(x, lo, hi):
    if not np.isfinite(safe_float(x)):
        return np.nan
    return max(lo, min(hi, float(x)))


def is_positive(x):
    v = safe_float(x)
    return np.isfinite(v) and v > 0


# =========================================================
# YAHOO DATA - CACHED
# =========================================================
@st.cache_resource(ttl=CACHE_TTL, show_spinner=False)
def get_ticker(symbol):
    return yf.Ticker(symbol)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_price_data(symbol, period=DEFAULT_PERIOD):
    symbol = clean_symbol(symbol)
    try:
        t = get_ticker(symbol)
        df = t.history(
            period=period,
            interval="1d",
            auto_adjust=False,
            actions=False,
            timeout=REQUEST_TIMEOUT,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        needed = ["Open", "High", "Low", "Close", "Volume"]
        for c in needed:
            if c not in df.columns:
                return pd.DataFrame()
        df = df[needed].dropna(subset=["Close"])
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_info(symbol):
    symbol = clean_symbol(symbol)
    try:
        t = get_ticker(symbol)
        info = t.info
        return info if isinstance(info, dict) else {}
    except Exception:
        return {}


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_financials(symbol):
    symbol = clean_symbol(symbol)
    try:
        t = get_ticker(symbol)
        out = {}
        try:
            out["income"] = t.income_stmt
        except Exception:
            out["income"] = pd.DataFrame()
        try:
            out["balance"] = t.balance_sheet
        except Exception:
            out["balance"] = pd.DataFrame()
        try:
            out["cashflow"] = t.cashflow
        except Exception:
            out["cashflow"] = pd.DataFrame()
        return out
    except Exception:
        return {"income": pd.DataFrame(), "balance": pd.DataFrame(), "cashflow": pd.DataFrame()}


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_news(symbol):
    symbol = clean_symbol(symbol)
    try:
        t = get_ticker(symbol)
        raw = getattr(t, "news", []) or []
        rows = []
        for item in raw[:15]:
            content = item.get("content", {}) if isinstance(item, dict) else {}
            title = (
                item.get("title")
                or content.get("title")
                or ""
            )
            publisher = (
                item.get("publisher")
                or content.get("provider", {}).get("displayName")
                or ""
            )
            link = item.get("link") or content.get("canonicalUrl", {}).get("url") or ""
            if title:
                rows.append({
                    "العنوان": title,
                    "المصدر": publisher,
                    "الرابط": link,
                })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame(columns=["العنوان", "المصدر", "الرابط"])


# =========================================================
# TECHNICAL INDICATORS
# =========================================================
def ema(series, span):
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df, period=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()


def adx(df, period=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    up = high.diff()
    down = -low.diff()

    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    atrv = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1/period, adjust=False, min_periods=period
    ).mean() / atrv.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1/period, adjust=False, min_periods=period
    ).mean() / atrv.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1/period, adjust=False, min_periods=period).mean()


def add_indicators(df):
    if df.empty:
        return df

    x = df.copy()
    close = x["Close"]

    for p in [20, 50, 100, 200]:
        x[f"EMA{p}"] = ema(close, p)

    x["RSI"] = rsi(close, 14)

    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    x["MACD"] = ema12 - ema26
    x["MACD_Signal"] = ema(x["MACD"], 9)
    x["MACD_Hist"] = x["MACD"] - x["MACD_Signal"]

    x["ATR"] = atr(x, 14)
    x["ADX"] = adx(x, 14)

    x["Volume_MA20"] = x["Volume"].rolling(20, min_periods=5).mean()
    x["Volume_Ratio"] = x["Volume"] / x["Volume_MA20"].replace(0, np.nan)

    direction = np.sign(close.diff()).fillna(0)
    x["OBV"] = (direction * x["Volume"]).cumsum()

    typical = (x["High"] + x["Low"] + x["Close"]) / 3
    cumulative_volume = x["Volume"].cumsum().replace(0, np.nan)
    x["VWAP"] = (typical * x["Volume"]).cumsum() / cumulative_volume

    x["ROC20"] = close.pct_change(20) * 100

    return x


# =========================================================
# SUPPORT / RESISTANCE / FIBONACCI
# =========================================================
def get_support_resistance(df, lookback=120):
    if df.empty:
        return np.nan, np.nan

    x = df.tail(lookback).copy()
    lows = x["Low"].rolling(5, center=True).min()
    highs = x["High"].rolling(5, center=True).max()

    current = safe_float(x["Close"].iloc[-1])

    supports = lows.dropna().unique()
    resistances = highs.dropna().unique()

    support_candidates = [v for v in supports if v < current]
    resistance_candidates = [v for v in resistances if v > current]

    support = max(support_candidates) if support_candidates else safe_float(x["Low"].min())
    resistance = min(resistance_candidates) if resistance_candidates else safe_float(x["High"].max())

    return support, resistance


def get_fibonacci(df, lookback=180):
    if df.empty:
        return {}

    x = df.tail(lookback)
    high = safe_float(x["High"].max())
    low = safe_float(x["Low"].min())

    if not np.isfinite(high) or not np.isfinite(low) or high <= low:
        return {}

    diff = high - low
    return {
        "23.6%": high - diff * 0.236,
        "38.2%": high - diff * 0.382,
        "50.0%": high - diff * 0.500,
        "61.8%": high - diff * 0.618,
        "78.6%": high - diff * 0.786,
    }


def get_trend(row):
    c = safe_float(row.get("Close"))
    e20 = safe_float(row.get("EMA20"))
    e50 = safe_float(row.get("EMA50"))
    e200 = safe_float(row.get("EMA200"))

    if not all(np.isfinite(v) for v in [c, e20, e50]):
        return "بيانات غير كافية"

    if np.isfinite(e200) and c > e20 > e50 > e200:
        return "اتجاه صاعد قوي"
    if c > e20 > e50:
        return "اتجاه صاعد"
    if c < e20 < e50 and np.isfinite(e200) and e50 < e200:
        return "اتجاه هابط قوي"
    if c < e20 < e50:
        return "اتجاه هابط"
    return "عرضي / يحتاج تأكيد"


# =========================================================
# FINANCIAL EXTRACTION
# =========================================================
def find_statement_value(df, names):
    if df is None or df.empty:
        return np.nan

    lookup = {str(idx).lower(): idx for idx in df.index}

    for name in names:
        key = str(name).lower()
        if key in lookup:
            row = df.loc[lookup[key]]
            if isinstance(row, pd.Series):
                vals = row.dropna()
                if len(vals):
                    return safe_float(vals.iloc[0])

    for idx in df.index:
        s = str(idx).lower()
        for name in names:
            if str(name).lower() in s:
                row = df.loc[idx]
                if isinstance(row, pd.Series):
                    vals = row.dropna()
                    if len(vals):
                        return safe_float(vals.iloc[0])

    return np.nan


def statement_series(df, names):
    if df is None or df.empty:
        return pd.Series(dtype=float)

    for idx in df.index:
        s = str(idx).lower()
        if any(str(n).lower() in s for n in names):
            row = pd.to_numeric(df.loc[idx], errors="coerce").dropna()
            if len(row):
                return row
    return pd.Series(dtype=float)


def extract_financials(financials, info):
    income = financials.get("income", pd.DataFrame())
    balance = financials.get("balance", pd.DataFrame())
    cashflow = financials.get("cashflow", pd.DataFrame())

    revenue = find_statement_value(income, [
        "Total Revenue", "Operating Revenue", "Revenue"
    ])
    net_income = find_statement_value(income, [
        "Net Income", "Net Income Common Stockholders"
    ])
    operating_income = find_statement_value(income, [
        "Operating Income"
    ])
    ebitda = find_statement_value(income, [
        "EBITDA", "Normalized EBITDA"
    ])

    debt = find_statement_value(balance, [
        "Total Debt", "Long Term Debt And Capital Lease Obligation",
        "Long Term Debt", "Current Debt"
    ])
    cash = find_statement_value(balance, [
        "Cash Cash Equivalents And Short Term Investments",
        "Cash And Cash Equivalents", "Cash Financial"
    ])
    assets = find_statement_value(balance, [
        "Total Assets"
    ])
    liabilities = find_statement_value(balance, [
        "Total Liabilities Net Minority Interest", "Total Liabilities"
    ])
    equity = find_statement_value(balance, [
        "Stockholders Equity", "Common Stock Equity",
        "Total Equity Gross Minority Interest"
    ])
    current_assets = find_statement_value(balance, [
        "Current Assets"
    ])
    current_liabilities = find_statement_value(balance, [
        "Current Liabilities"
    ])

    operating_cf = find_statement_value(cashflow, [
        "Operating Cash Flow", "Total Cash From Operating Activities"
    ])
    capex = find_statement_value(cashflow, [
        "Capital Expenditure", "Capital Expenditures"
    ])

    fcf = find_statement_value(cashflow, [
        "Free Cash Flow"
    ])
    if not np.isfinite(fcf) and np.isfinite(operating_cf) and np.isfinite(capex):
        fcf = operating_cf + capex if capex < 0 else operating_cf - capex

    market_cap = first_valid(
        info.get("marketCap"),
        info.get("enterpriseValue")
    )

    shares = first_valid(
        info.get("sharesOutstanding"),
        info.get("impliedSharesOutstanding")
    )

    eps = first_valid(
        info.get("trailingEps"),
        info.get("epsTrailingTwelveMonths")
    )

    book_value = first_valid(
        info.get("bookValue")
    )

    price = first_valid(
        info.get("currentPrice"),
        info.get("regularMarketPrice"),
        info.get("previousClose")
    )

    return {
        "Revenue": revenue,
        "NetIncome": net_income,
        "OperatingIncome": operating_income,
        "EBITDA": ebitda,
        "Debt": debt,
        "Cash": cash,
        "Assets": assets,
        "Liabilities": liabilities,
        "Equity": equity,
        "CurrentAssets": current_assets,
        "CurrentLiabilities": current_liabilities,
        "OperatingCF": operating_cf,
        "Capex": capex,
        "FCF": fcf,
        "MarketCap": market_cap,
        "Shares": shares,
        "EPS": eps,
        "BookValue": book_value,
        "Price": price,
    }


# =========================================================
# GROWTH / RATIOS
# =========================================================
def calculate_growth(financials):
    income = financials.get("income", pd.DataFrame())

    revenue_series = statement_series(income, ["Total Revenue", "Operating Revenue", "Revenue"])
    profit_series = statement_series(income, ["Net Income", "Net Income Common Stockholders"])

    def growth(series):
        if len(series) < 2:
            return np.nan, np.nan
        newest = safe_float(series.iloc[0])
        previous = safe_float(series.iloc[1])
        if not np.isfinite(newest) or not np.isfinite(previous) or previous == 0:
            return np.nan, np.nan
        yoy = (newest / previous - 1) * 100

        cagr = np.nan
        if len(series) >= 3:
            oldest = safe_float(series.iloc[-1])
            if np.isfinite(oldest) and oldest > 0 and newest > 0:
                years = max(1, len(series) - 1)
                cagr = ((newest / oldest) ** (1 / years) - 1) * 100
        return yoy, cagr

    rg, rcagr = growth(revenue_series)
    pg, pcagr = growth(profit_series)

    return {
        "RevenueGrowth": rg,
        "RevenueCAGR": rcagr,
        "ProfitGrowth": pg,
        "ProfitCAGR": pcagr,
    }


def calculate_ratios(f, info):
    debt = safe_float(f["Debt"])
    equity = safe_float(f["Equity"])
    cash = safe_float(f["Cash"])
    ebitda = safe_float(f["EBITDA"])
    ca = safe_float(f["CurrentAssets"])
    cl = safe_float(f["CurrentLiabilities"])
    revenue = safe_float(f["Revenue"])
    net_income = safe_float(f["NetIncome"])
    assets = safe_float(f["Assets"])
    price = safe_float(f["Price"])

    de = debt / equity if np.isfinite(debt) and np.isfinite(equity) and equity != 0 else np.nan
    debt_ebitda = debt / ebitda if np.isfinite(debt) and np.isfinite(ebitda) and ebitda > 0 else np.nan
    cash_debt = cash / debt if np.isfinite(cash) and np.isfinite(debt) and debt > 0 else np.nan
    current_ratio = ca / cl if np.isfinite(ca) and np.isfinite(cl) and cl != 0 else np.nan
    quick_ratio = current_ratio  # fallback when inventory isn't reliably available
    margin = net_income / revenue * 100 if np.isfinite(net_income) and np.isfinite(revenue) and revenue != 0 else np.nan
    roe = net_income / equity * 100 if np.isfinite(net_income) and np.isfinite(equity) and equity != 0 else safe_float(info.get("returnOnEquity")) * 100
    roa = net_income / assets * 100 if np.isfinite(net_income) and np.isfinite(assets) and assets != 0 else safe_float(info.get("returnOnAssets")) * 100

    return {
        "DebtEquity": de,
        "DebtEBITDA": debt_ebitda,
        "CashDebt": cash_debt,
        "CurrentRatio": current_ratio,
        "QuickRatio": quick_ratio,
        "ProfitMargin": margin,
        "ROE": roe,
        "ROA": roa,
        "PE": first_valid(info.get("trailingPE"), info.get("forwardPE")),
        "ForwardPE": safe_float(info.get("forwardPE")),
        "PB": safe_float(info.get("priceToBook")),
        "PS": safe_float(info.get("priceToSalesTrailing12Months")),
        "EVEBITDA": safe_float(info.get("enterpriseToEbitda")),
        "PEG": safe_float(info.get("pegRatio")),
        "DividendYield": first_valid(
            info.get("dividendYield"),
            safe_float(info.get("trailingAnnualDividendYield"))
        ),
    }


# =========================================================
# BUSINESS TYPE / VALUATION
# =========================================================
def classify_business(symbol, info, f):
    name = str(info.get("longName") or info.get("shortName") or "").lower()
    sector = str(info.get("sector") or "").lower()
    industry = str(info.get("industry") or "").lower()
    text = f"{name} {sector} {industry} {symbol.lower()}"

    bank_keys = ["bank", "بنك", "banking", "credit", "financial services"]
    financial_keys = ["insurance", "financial", "broker", "leasing", "mortgage"]

    if any(k in text for k in bank_keys):
        return "BANK"
    if any(k in text for k in financial_keys):
        return "FINANCIAL"
    return "NON_FINANCIAL"


def estimate_cost_of_equity():
    return RISK_FREE_RATE + EQUITY_RISK_PREMIUM


def normalize_growth(g, default=0.08):
    v = safe_float(g)
    if not np.isfinite(v):
        return default
    return clamp(v / 100.0, -0.10, 0.35)


def dcf_equity_value(f, growths, discount_rate=0.24, terminal_growth=TERMINAL_GROWTH):
    """
    Simplified FCFE-style equity DCF.
    Uses FCF as the available cash flow proxy.
    """
    fcf = safe_float(f["FCF"])
    shares = safe_float(f["Shares"])
    if not np.isfinite(fcf) or fcf <= 0 or not np.isfinite(shares) or shares <= 0:
        return np.nan

    g1, g2, g3 = growths
    cash = safe_float(f["Cash"], 0)
    debt = safe_float(f["Debt"], 0)

    if discount_rate <= terminal_growth:
        return np.nan

    current = fcf
    pv = 0.0
    for year, g in enumerate([g1, g2, g3], start=1):
        current *= (1 + g)
        pv += current / ((1 + discount_rate) ** year)

    terminal = current * (1 + terminal_growth) / (discount_rate - terminal_growth)
    pv_terminal = terminal / ((1 + discount_rate) ** 3)

    equity_value = pv + pv_terminal + cash - debt
    return equity_value / shares if equity_value > 0 else np.nan


def pe_fair_value(f, ratios, growths):
    eps = safe_float(f["EPS"])
    if not np.isfinite(eps) or eps <= 0:
        ni = safe_float(f["NetIncome"])
        shares = safe_float(f["Shares"])
        if np.isfinite(ni) and np.isfinite(shares) and shares > 0:
            eps = ni / shares

    if not np.isfinite(eps) or eps <= 0:
        return np.nan

    growth = max(0.05, min(0.25, normalize_growth(growths.get("ProfitGrowth"), 0.10)))
    justified_pe = 8 + growth * 100 * 0.45
    justified_pe = clamp(justified_pe, 8, 22)

    return eps * justified_pe


def pb_fair_value(f, ratios, business_type):
    bvps = safe_float(f["BookValue"])
    if not np.isfinite(bvps) or bvps <= 0:
        equity = safe_float(f["Equity"])
        shares = safe_float(f["Shares"])
        if np.isfinite(equity) and np.isfinite(shares) and shares > 0:
            bvps = equity / shares

    if not np.isfinite(bvps) or bvps <= 0:
        return np.nan

    roe = safe_float(ratios.get("ROE"))
    if np.isfinite(roe):
        target_pb = clamp(0.7 + roe / 100 * 4.0, 0.7, 3.0)
    else:
        target_pb = 1.3

    if business_type == "BANK":
        target_pb = clamp(target_pb, 0.8, 2.5)

    return bvps * target_pb


def ev_ebitda_fair_value(f, ratios):
    ebitda = safe_float(f["EBITDA"])
    debt = safe_float(f["Debt"], 0)
    cash = safe_float(f["Cash"], 0)
    shares = safe_float(f["Shares"])

    if not all(np.isfinite(v) for v in [ebitda, shares]) or ebitda <= 0 or shares <= 0:
        return np.nan

    multiple = 7.0
    current_ev_ebitda = safe_float(ratios.get("EVEBITDA"))
    if np.isfinite(current_ev_ebitda) and current_ev_ebitda > 0:
        multiple = clamp(current_ev_ebitda * 0.90, 5.0, 12.0)

    ev = ebitda * multiple
    equity_value = ev - debt + cash
    return equity_value / shares if equity_value > 0 else np.nan


def bank_fair_value(f, ratios, growths):
    """
    Bank-friendly model:
    combines P/B and P/E because debt/EV/EBITDA are not meaningful
    in the same way as industrial companies.
    """
    pb = pb_fair_value(f, ratios, "BANK")
    pe = pe_fair_value(f, ratios, growths)

    vals = [v for v in [pb, pe] if np.isfinite(v) and v > 0]
    if not vals:
        return np.nan

    if len(vals) == 1:
        return vals[0]

    return 0.55 * pb + 0.45 * pe


def valuation_engine(symbol, f, ratios, growths, info):
    business_type = classify_business(symbol, info, f)

    pe = pe_fair_value(f, ratios, growths)
    pb = pb_fair_value(f, ratios, business_type)

    if business_type == "BANK":
        dcf = np.nan
        ev = np.nan
        base = bank_fair_value(f, ratios, growths)
    elif business_type == "FINANCIAL":
        dcf = dcf_equity_value(
            f,
            (
                normalize_growth(growths.get("ProfitGrowth"), 0.08),
                normalize_growth(growths.get("ProfitGrowth"), 0.08) * 0.85,
                normalize_growth(growths.get("ProfitGrowth"), 0.08) * 0.70,
            ),
        )
        ev = ev_ebitda_fair_value(f, ratios)
        candidates = [x for x in [dcf, pe, pb, ev] if np.isfinite(x) and x > 0]
        base = float(np.median(candidates)) if candidates else np.nan
    else:
        dcf = dcf_equity_value(
            f,
            (
                normalize_growth(growths.get("ProfitGrowth"), 0.08),
                normalize_growth(growths.get("ProfitGrowth"), 0.08) * 0.85,
                normalize_growth(growths.get("ProfitGrowth"), 0.08) * 0.70,
            ),
        )
        ev = ev_ebitda_fair_value(f, ratios)
        candidates = [x for x in [dcf, pe, pb, ev] if np.isfinite(x) and x > 0]
        if candidates:
            # Prefer DCF when valid, but keep multiples in the blend.
            if np.isfinite(dcf):
                base = 0.40 * dcf + 0.30 * np.nanmedian([x for x in [pe, pb] if np.isfinite(x)]) if any(np.isfinite(x) for x in [pe, pb]) else dcf
                if np.isfinite(ev):
                    base = 0.30 * dcf + 0.25 * pe if np.isfinite(pe) else 0.55 * dcf
                    base += 0.20 * pb if np.isfinite(pb) else 0
                    base += 0.25 * ev
            else:
                base = float(np.median(candidates))
        else:
            base = np.nan

    if not np.isfinite(base) or base <= 0:
        base = first_valid(f.get("Price"))

    # Scenarios are deliberately different.
    conservative = base * 0.80
    optimistic = base * 1.25

    # Fair-value band based on dispersion of usable models.
    model_values = [pe, pb, ev]
    if business_type != "BANK":
        model_values.append(dcf)
    model_values = [v for v in model_values if np.isfinite(v) and v > 0]

    if model_values:
        low_band = float(np.percentile(model_values, 25))
        high_band = float(np.percentile(model_values, 75))
    else:
        low_band = conservative
        high_band = optimistic

    return {
        "BusinessType": business_type,
        "DCF": dcf,
        "PEFairValue": pe,
        "PBFairValue": pb,
        "EVEBITDAFairValue": ev,
        "FairValue": base,
        "ConservativeFairValue": min(conservative, low_band if np.isfinite(low_band) else conservative),
        "BaseFairValue": base,
        "OptimisticFairValue": max(optimistic, high_band if np.isfinite(high_band) else optimistic),
    }


# =========================================================
# INVESTMENT ENGINE
# =========================================================
def investment_engine(f, ratios, growths, valuation):
    current = safe_float(f["Price"])
    fair = safe_float(valuation.get("FairValue"))

    if not np.isfinite(current) or current <= 0:
        current = np.nan

    if not np.isfinite(fair) or fair <= 0:
        return {
            "SafeBuy": np.nan,
            "ExcellentBuy": np.nan,
            "Upside": np.nan,
            "ThreeYearTarget": np.nan,
            "ThreeYearCAGR": np.nan,
            "DividendYield": safe_float(ratios.get("DividendYield")),
            "Dividend3Y": np.nan,
        }

    safe_buy = fair * (1 - DEFAULT_MARGIN_OF_SAFETY)
    excellent_buy = fair * 0.70

    profit_growth = normalize_growth(growths.get("ProfitGrowth"), 0.08)
    revenue_growth = normalize_growth(growths.get("RevenueGrowth"), 0.08)

    sustainable_growth = clamp(
        0.60 * profit_growth + 0.40 * revenue_growth,
        -0.05, 0.30
    )

    # 3-year multiple-growth target.
    target = fair * ((1 + sustainable_growth) ** 3)

    # Avoid absurd targets while still allowing genuine upside.
    target = clamp(target, fair * 0.75, fair * 2.50)

    cagr = ((target / current) ** (1 / 3) - 1) * 100 if np.isfinite(current) and current > 0 else np.nan

    dy = safe_float(ratios.get("DividendYield"))
    if np.isfinite(dy):
        dividend_3y = current * (dy / 100.0) * 3
    else:
        dividend_3y = np.nan

    upside = ((fair / current) - 1) * 100 if np.isfinite(current) and current > 0 else np.nan

    return {
        "SafeBuy": safe_buy,
        "ExcellentBuy": excellent_buy,
        "Upside": upside,
        "ThreeYearTarget": target,
        "ThreeYearCAGR": cagr,
        "DividendYield": dy,
        "Dividend3Y": dividend_3y,
    }


def scenario_engine(valuation, growths):
    base = safe_float(valuation.get("FairValue"))
    if not np.isfinite(base):
        return {
            "Conservative": np.nan,
            "Base": np.nan,
            "Optimistic": np.nan
        }

    pg = normalize_growth(growths.get("ProfitGrowth"), 0.08)

    # Explicitly different scenarios.
    conservative_factor = clamp(0.80 + min(pg, 0.15) * 0.20, 0.75, 0.85)
    base_factor = 1.00
    optimistic_factor = clamp(1.20 + max(pg, 0.05) * 0.30, 1.20, 1.35)

    return {
        "Conservative": base * conservative_factor,
        "Base": base * base_factor,
        "Optimistic": base * optimistic_factor,
    }


# =========================================================
# LIQUIDITY
# =========================================================
def liquidity_analysis(df, info):
    if df.empty:
        return {
            "LastVolume": np.nan,
            "AvgVolume": np.nan,
            "AvgTradingValue": np.nan,
            "VolumeRatio": np.nan,
            "MarketCap": safe_float(info.get("marketCap")),
        }

    close = df["Close"]
    volume = df["Volume"]

    last_volume = safe_float(volume.iloc[-1])
    avg_volume = safe_float(volume.tail(20).mean())
    avg_trading_value = safe_float((close.tail(20) * volume.tail(20)).mean())
    volume_ratio = last_volume / avg_volume if np.isfinite(last_volume) and np.isfinite(avg_volume) and avg_volume > 0 else np.nan

    return {
        "LastVolume": last_volume,
        "AvgVolume": avg_volume,
        "AvgTradingValue": avg_trading_value,
        "VolumeRatio": volume_ratio,
        "MarketCap": safe_float(info.get("marketCap")),
    }


# =========================================================
# RELATIVE STRENGTH
# =========================================================
def relative_strength(df):
    if df.empty or len(df) < 61:
        return np.nan
    c0 = safe_float(df["Close"].iloc[-61])
    c1 = safe_float(df["Close"].iloc[-1])
    if np.isfinite(c0) and np.isfinite(c1) and c0 > 0:
        return (c1 / c0 - 1) * 100
    return np.nan


# =========================================================
# TARGETS / ENTRIES
# =========================================================
def target_engine(df, support, resistance, fib):
    if df.empty:
        return []

    current = safe_float(df["Close"].iloc[-1])
    atrv = safe_float(df["ATR"].iloc[-1])

    candidates = []

    if np.isfinite(resistance) and resistance > current:
        candidates.append(resistance)

    for v in fib.values():
        v = safe_float(v)
        if np.isfinite(v) and v > current:
            candidates.append(v)

    for n in [20, 60, 120]:
        if len(df) >= n:
            v = safe_float(df["High"].tail(n).max())
            if np.isfinite(v) and v > current:
                candidates.append(v)

    if np.isfinite(atrv) and atrv > 0:
        for mult in [2, 3, 4, 5]:
            v = current + atrv * mult
            if v > current:
                candidates.append(v)

    candidates = sorted(set(round(float(x), 4) for x in candidates))

    final = []
    for v in candidates:
        gain = (v / current - 1) * 100 if current > 0 else 0
        if gain < 1 or gain > 100:
            continue
        if not final or abs(v - final[-1]) / final[-1] >= 0.02:
            final.append(v)
        if len(final) >= 4:
            break

    return final


def entry_engine(df, support, resistance):
    if df.empty:
        return {}

    row = df.iloc[-1]
    current = safe_float(row["Close"])
    e20 = safe_float(row.get("EMA20"))
    atrv = safe_float(row.get("ATR"))

    pullback = np.nan
    if np.isfinite(support) and np.isfinite(e20):
        pullback = (support + e20) / 2
    elif np.isfinite(e20):
        pullback = e20
    elif np.isfinite(support):
        pullback = support

    breakout = resistance * 1.01 if np.isfinite(resistance) else np.nan

    if np.isfinite(support) and support > 0:
        stop = support * 0.98
    elif np.isfinite(atrv):
        stop = current - 2 * atrv
    else:
        stop = np.nan

    return {
        "Current": current,
        "Pullback": pullback,
        "Breakout": breakout,
        "Stop": stop,
    }


# =========================================================
# SCORES
# =========================================================
def technical_score(df):
    if df.empty:
        return 0.0, []

    row = df.iloc[-1]
    score = 0.0
    reasons = []

    c = safe_float(row.get("Close"))
    e20 = safe_float(row.get("EMA20"))
    e50 = safe_float(row.get("EMA50"))
    e200 = safe_float(row.get("EMA200"))
    r = safe_float(row.get("RSI"))
    macd = safe_float(row.get("MACD"))
    macd_signal = safe_float(row.get("MACD_Signal"))
    adxv = safe_float(row.get("ADX"))
    vr = safe_float(row.get("Volume_Ratio"))

    if np.isfinite(c) and np.isfinite(e20) and c > e20:
        score += 5
        reasons.append("السعر فوق EMA20")
    if np.isfinite(c) and np.isfinite(e50) and c > e50:
        score += 5
        reasons.append("السعر فوق EMA50")
    if np.isfinite(c) and np.isfinite(e200) and c > e200:
        score += 5
        reasons.append("السعر فوق EMA200")

    if np.isfinite(e20) and np.isfinite(e50) and e20 > e50:
        score += 4
        reasons.append("EMA20 أعلى EMA50")

    if np.isfinite(r):
        if 50 <= r <= 70:
            score += 4
            reasons.append("RSI إيجابي بدون تشبع قوي")
        elif 70 < r <= 78:
            score += 2
            reasons.append("RSI قوي مع مراقبة التشبع")
        elif r < 35:
            score += 1
            reasons.append("RSI منخفض")

    if np.isfinite(macd) and np.isfinite(macd_signal) and macd > macd_signal:
        score += 4
        reasons.append("MACD إيجابي")

    if np.isfinite(adxv):
        if adxv >= 25:
            score += 4
            reasons.append("ADX يؤكد قوة الاتجاه")
        elif adxv >= 18:
            score += 2

    if np.isfinite(vr):
        if vr >= 1.20:
            score += 4
            reasons.append("حجم أعلى من المتوسط")
        elif vr >= 0.90:
            score += 2

    return min(score, TECH_SCORE_MAX), reasons


def fundamental_score(f, ratios, growths):
    score = 0.0
    reasons = []

    rg = safe_float(growths.get("RevenueGrowth"))
    pg = safe_float(growths.get("ProfitGrowth"))
    roe = safe_float(ratios.get("ROE"))
    margin = safe_float(ratios.get("ProfitMargin"))
    cr = safe_float(ratios.get("CurrentRatio"))
    de = safe_float(ratios.get("DebtEquity"))
    fcf = safe_float(f.get("FCF"))

    if np.isfinite(rg):
        if rg >= 20:
            score += 7
            reasons.append("نمو الإيرادات قوي")
        elif rg >= 10:
            score += 5
        elif rg > 0:
            score += 3

    if np.isfinite(pg):
        if pg >= 25:
            score += 8
            reasons.append("نمو الأرباح قوي")
        elif pg >= 10:
            score += 6
        elif pg > 0:
            score += 3

    if np.isfinite(roe):
        if roe >= 20:
            score += 6
            reasons.append("ROE قوي")
        elif roe >= 12:
            score += 4
        elif roe > 0:
            score += 2

    if np.isfinite(margin):
        if margin >= 15:
            score += 4
        elif margin >= 8:
            score += 3
        elif margin > 0:
            score += 1

    if np.isfinite(cr):
        if cr >= 1.5:
            score += 3
        elif cr >= 1:
            score += 2

    if np.isfinite(de):
        if de <= 0.50:
            score += 4
            reasons.append("مديونية منخفضة")
        elif de <= 1.5:
            score += 2
    elif np.isfinite(fcf) and fcf > 0:
        score += 3

    if np.isfinite(fcf) and fcf > 0:
        score += 3
        reasons.append("تدفق نقدي حر إيجابي")

    return min(score, FUND_SCORE_MAX), reasons


def valuation_score(current, valuation):
    if not np.isfinite(current) or current <= 0:
        return 0.0, []

    fair = safe_float(valuation.get("FairValue"))
    if not np.isfinite(fair) or fair <= 0:
        return 0.0, []

    upside = (fair / current - 1) * 100
    score = 0.0
    reasons = []

    if upside >= 40:
        score = 15
        reasons.append("السعر أقل بكثير من القيمة العادلة")
    elif upside >= 25:
        score = 13
        reasons.append("خصم جيد عن القيمة العادلة")
    elif upside >= 10:
        score = 10
    elif upside >= 0:
        score = 7
    elif upside >= -10:
        score = 4
    else:
        score = 1
        reasons.append("السعر أعلى من القيمة العادلة")

    return min(score, VALUATION_SCORE_MAX), reasons


def liquidity_score(liq):
    av = safe_float(liq.get("AvgTradingValue"))
    vr = safe_float(liq.get("VolumeRatio"))

    score = 0.0
    if np.isfinite(av):
        if av >= 10_000_000:
            score += 6
        elif av >= 3_000_000:
            score += 4
        elif av >= 1_000_000:
            score += 2
        elif av > 0:
            score += 1

    if np.isfinite(vr):
        if vr >= 1.2:
            score += 4
        elif vr >= 0.8:
            score += 2
        elif vr > 0:
            score += 1

    return min(score, LIQUIDITY_SCORE_MAX)


def news_score(news_df):
    if news_df is None or news_df.empty:
        return 0.0, []

    positive = [
        "profit", "growth", "record", "upgrade", "dividend",
        "award", "contract", "expansion", "increase", "positive",
        "أرباح", "نمو", "توزيعات", "عقد", "توسعات", "إيجابي"
    ]
    negative = [
        "loss", "downgrade", "debt", "lawsuit", "decline",
        "warning", "negative", "investigation", "default",
        "خسائر", "ديون", "تراجع", "تحقيق", "تحذير", "سلبي"
    ]

    score = 0
    reasons = []

    for title in news_df["العنوان"].astype(str).head(10):
        text = title.lower()
        p = sum(k.lower() in text for k in positive)
        n = sum(k.lower() in text for k in negative)
        if p > n:
            score += 1
        elif n > p:
            score -= 1

    score = clamp(score, -NEWS_SCORE_MAX, NEWS_SCORE_MAX)

    if score > 0:
        reasons.append("الأخبار تميل للإيجابية")
    elif score < 0:
        reasons.append("الأخبار تميل للسلبية")

    return score, reasons


def final_score(tech, fund, val, liq, news):
    # Convert news from [-5,+5] to [0,10].
    news_component = news + NEWS_SCORE_MAX
    total = tech + fund + val + liq + news_component
    return clamp(total, 0, 100)


def rating(score):
    if score >= 85:
        return "ممتاز جدًا"
    if score >= 75:
        return "قوي"
    if score >= 65:
        return "جيد"
    if score >= 55:
        return "متوسط"
    if score >= 45:
        return "ضعيف نسبيًا"
    return "ضعيف"


def risk_score(ratios, liq):
    risk = 0.0
    de = safe_float(ratios.get("DebtEquity"))
    cr = safe_float(ratios.get("CurrentRatio"))
    av = safe_float(liq.get("AvgTradingValue"))
    rsi_v = np.nan

    if np.isfinite(de):
        risk += 25 if de > 2 else 15 if de > 1 else 5
    if np.isfinite(cr):
        risk += 20 if cr < 0.8 else 10 if cr < 1 else 3
    if np.isfinite(av):
        risk += 25 if av < 500_000 else 15 if av < 1_000_000 else 5

    return clamp(risk, 0, 100)


# =========================================================
# DATA QUALITY
# =========================================================
def data_quality_score(df, f, ratios, growths, valuation):
    checks = []

    checks += [
        not df.empty,
        len(df) >= 60 if not df.empty else False,
        np.isfinite(safe_float(f.get("Revenue"))),
        np.isfinite(safe_float(f.get("NetIncome"))),
        np.isfinite(safe_float(f.get("Equity"))),
        np.isfinite(safe_float(f.get("FCF"))),
        np.isfinite(safe_float(ratios.get("ROE"))),
        np.isfinite(safe_float(growths.get("RevenueGrowth"))),
        np.isfinite(safe_float(growths.get("ProfitGrowth"))),
        np.isfinite(safe_float(valuation.get("FairValue"))),
    ]

    return sum(checks) / len(checks) * 100


# =========================================================
# ANALYSIS
# =========================================================
def analyze_stock(symbol, period=DEFAULT_PERIOD):
    symbol = clean_symbol(symbol)

    df_raw = load_price_data(symbol, period)
    info = load_info(symbol)
    financials = load_financials(symbol)
    news = load_news(symbol)

    if df_raw.empty:
        return {
            "error": "لم يتم الحصول على بيانات سعرية من Yahoo Finance.",
            "Symbol": symbol
        }

    df = add_indicators(df_raw)

    # Prefer the latest actual candle close over info.currentPrice.
    current = safe_float(df["Close"].iloc[-1])
    f = extract_financials(financials, info)
    f["Price"] = current

    growths = calculate_growth(financials)
    ratios = calculate_ratios(f, info)

    support, resistance = get_support_resistance(df)
    fib = get_fibonacci(df)
    trend = get_trend(df.iloc[-1])

    valuation = valuation_engine(symbol, f, ratios, growths, info)
    scenarios = scenario_engine(valuation, growths)
    investment = investment_engine(f, ratios, growths, valuation)

    liq = liquidity_analysis(df, info)
    rs60 = relative_strength(df)
    targets = target_engine(df, support, resistance, fib)
    entries = entry_engine(df, support, resistance)

    tech, tech_reasons = technical_score(df)
    fund, fund_reasons = fundamental_score(f, ratios, growths)
    val_score, val_reasons = valuation_score(current, valuation)
    liq_score = liquidity_score(liq)
    news_sc, news_reasons = news_score(news)

    total = final_score(tech, fund, val_score, liq_score, news_sc)
    risk = risk_score(ratios, liq)
    quality = data_quality_score(df, f, ratios, growths, valuation)

    last_date = df.index[-1]

    return {
        "Symbol": symbol,
        "RawSymbol": get_raw_symbol(symbol),
        "Name": info.get("longName") or info.get("shortName") or get_raw_symbol(symbol),
        "Sector": info.get("sector") or "غير متاح",
        "Industry": info.get("industry") or "غير متاح",
        "Currency": info.get("currency") or "EGP",
        "CurrentPrice": current,
        "LastCandleDate": last_date,
        "DataRows": len(df),
        "DataQuality": quality,
        "Data": df,
        "Financials": f,
        "Growth": growths,
        "Ratios": ratios,
        "Liquidity": liq,
        "Trend": trend,
        "Support": support,
        "Resistance": resistance,
        "Fibonacci": fib,
        "Entries": entries,
        "TechnicalTargets": targets,
        "Valuation": valuation,
        "Scenarios": scenarios,
        "Investment": investment,
        "RelativeStrength60": rs60,
        "Scores": {
            "Technical": tech,
            "Fundamental": fund,
            "Valuation": val_score,
            "Liquidity": liq_score,
            "News": news_sc,
            "Final": total,
            "Risk": risk,
        },
        "Reasons": {
            "Technical": tech_reasons,
            "Fundamental": fund_reasons,
            "Valuation": val_reasons,
            "News": news_reasons,
        },
        "News": news,
        "Info": info,
    }


# =========================================================
# UI HELPERS
# =========================================================
def show_metric(label, value, help_text=None):
    st.metric(label, value, help=help_text)


def safe_date(x):
    try:
        return pd.to_datetime(x).strftime("%Y-%m-%d")
    except Exception:
        return "غير متاح"


# =========================================================
# APP
# =========================================================
st.title("📈 EGX Stock Intelligence PRO")
st.caption("محرك فني + مالي + تقييم + استثمار — مع كاش وتقليل طلبات Yahoo Finance")

with st.sidebar:
    st.header("⚙️ إعداد التحليل")

    symbol_input = st.text_input(
        "رمز السهم",
        value="DAPH",
        help="مثال: DAPH أو DAPH.CA"
    )

    period = st.selectbox(
        "الفترة السعرية",
        ["1y", "2y", "5y", "10y"],
        index=1
    )

    st.markdown("---")
    st.write("**افتراضات التقييم**")
    st.write(f"معدل خالٍ من المخاطر: {RISK_FREE_RATE*100:.1f}%")
    st.write(f"علاوة مخاطر الأسهم: {EQUITY_RISK_PREMIUM*100:.1f}%")
    st.write(f"النمو النهائي: {TERMINAL_GROWTH*100:.1f}%")
    st.write("هامش الأمان: 20%")

    analyze_btn = st.button("🔎 تحليل السهم", type="primary", use_container_width=True)

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if analyze_btn:
    with st.spinner("جاري تحميل وتحليل البيانات مرة واحدة..."):
        st.session_state.analysis_result = analyze_stock(symbol_input, period)

result = st.session_state.analysis_result

if result is None:
    st.info("اكتب رمز السهم واضغط «تحليل السهم».")
    st.stop()

if result.get("error"):
    st.error(result["error"])
    st.stop()

# =========================================================
# HEADER
# =========================================================
current = result["CurrentPrice"]
scores = result["Scores"]
valuation = result["Valuation"]
investment = result["Investment"]
scenarios = result["Scenarios"]

st.markdown(f"## {result['RawSymbol']} — {result['Name']}")
st.caption(
    f"آخر شمعة: {safe_date(result['LastCandleDate'])} | "
    f"القطاع: {result['Sector']} | "
    f"الصناعة: {result['Industry']}"
)

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    show_metric("السعر الحالي", fmt_price(current))
with c2:
    show_metric("القيمة العادلة", fmt_price(valuation["FairValue"]))
with c3:
    show_metric("الشراء الآمن", fmt_price(investment["SafeBuy"]))
with c4:
    show_metric("هدف 3 سنوات", fmt_price(investment["ThreeYearTarget"]))
with c5:
    show_metric("الدرجة النهائية", f"{scores['Final']:.1f}/100")

st.success(
    f"التقييم: **{rating(scores['Final'])}** | "
    f"الاتجاه الفني: **{result['Trend']}** | "
    f"جودة البيانات: **{result['DataQuality']:.1f}%**"
)

# =========================================================
# INVESTMENT SNAPSHOT
# =========================================================
st.subheader("💰 الخلاصة الاستثمارية")

cols = st.columns(6)
metrics = [
    ("القيمة العادلة", fmt_price(valuation["FairValue"])),
    ("محافظ", fmt_price(scenarios["Conservative"])),
    ("أساسي", fmt_price(scenarios["Base"])),
    ("متفائل", fmt_price(scenarios["Optimistic"])),
    ("شراء ممتاز", fmt_price(investment["ExcellentBuy"])),
    ("Upside", fmt_pct(investment["Upside"])),
]
for col, (label, value) in zip(cols, metrics):
    with col:
        st.metric(label, value)

cols2 = st.columns(4)
with cols2[0]:
    st.metric("CAGR 3 سنوات", fmt_pct(investment["ThreeYearCAGR"]))
with cols2[1]:
    st.metric("العائد النقدي السنوي", fmt_pct(investment["DividendYield"]))
with cols2[2]:
    st.metric("تقدير توزيعات 3 سنوات", fmt_price(investment["Dividend3Y"]))
with cols2[3]:
    st.metric("مخاطر", f"{scores['Risk']:.0f}/100")

st.caption(
    "القيمة العادلة تقديرية وليست سعرًا مضمونًا. نماذج البنوك والمؤسسات المالية تختلف عن الشركات التشغيلية."
)

# =========================================================
# SCORE BREAKDOWN
# =========================================================
st.subheader("🎯 توزيع الدرجة")

score_df = pd.DataFrame({
    "المحور": ["فني", "مالي", "تقييم", "سيولة", "أخبار"],
    "الدرجة": [
        scores["Technical"],
        scores["Fundamental"],
        scores["Valuation"],
        scores["Liquidity"],
        scores["News"] + NEWS_SCORE_MAX,
    ],
    "الحد الأقصى": [
        TECH_SCORE_MAX,
        FUND_SCORE_MAX,
        VALUATION_SCORE_MAX,
        LIQUIDITY_SCORE_MAX,
        NEWS_SCORE_MAX * 2,
    ]
})
st.dataframe(score_df, use_container_width=True, hide_index=True)

# =========================================================
# CHART
# =========================================================
st.subheader("📊 السعر والمتوسطات")

chart_df = result["Data"][["Close", "EMA20", "EMA50", "EMA200"]].tail(400).copy()
chart_df.columns = ["السعر", "EMA20", "EMA50", "EMA200"]
st.line_chart(chart_df, use_container_width=True)

# =========================================================
# TABS
# =========================================================
tabs = st.tabs([
    "📈 الفني",
    "💼 المالي",
    "💰 التقييم",
    "🎯 الأهداف والدخول",
    "🧮 السيناريوهات",
    "📰 الأخبار",
    "📋 البيانات"
])

# =========================================================
# TECH TAB
# =========================================================
with tabs[0]:
    st.subheader("التحليل الفني")

    row = result["Data"].iloc[-1]

    tech_metrics = st.columns(6)
    technical_values = [
        ("RSI", fmt_number(row.get("RSI"))),
        ("MACD", fmt_number(row.get("MACD"))),
        ("ADX", fmt_number(row.get("ADX"))),
        ("ATR", fmt_price(row.get("ATR"))),
        ("Volume Ratio", fmt_number(row.get("Volume_Ratio"))),
        ("RS 60D", fmt_pct(result["RelativeStrength60"])),
    ]

    for col, (label, value) in zip(tech_metrics, technical_values):
        with col:
            st.metric(label, value)

    st.write(f"**الاتجاه:** {result['Trend']}")
    st.write(f"**الدعم:** {fmt_price(result['Support'])}")
    st.write(f"**المقاومة:** {fmt_price(result['Resistance'])}")

    st.markdown("### أسباب القوة الفنية")
    if result["Reasons"]["Technical"]:
        for r in result["Reasons"]["Technical"]:
            st.write("✅", r)
    else:
        st.write("لا توجد إشارات قوية كافية.")

    st.markdown("### Fibonacci")
    fib_df = pd.DataFrame(
        [{"المستوى": k, "السعر": v} for k, v in result["Fibonacci"].items()]
    )
    if not fib_df.empty:
        fib_df["السعر"] = fib_df["السعر"].map(lambda x: fmt_price(x))
        st.dataframe(fib_df, use_container_width=True, hide_index=True)

# =========================================================
# FUNDAMENTAL TAB
# =========================================================
with tabs[1]:
    st.subheader("التحليل المالي")

    f = result["Financials"]
    g = result["Growth"]
    r = result["Ratios"]

    a, b, c, d = st.columns(4)
    with a:
        st.metric("نمو الإيرادات", fmt_pct(g["RevenueGrowth"]))
    with b:
        st.metric("CAGR الإيرادات", fmt_pct(g["RevenueCAGR"]))
    with c:
        st.metric("نمو الأرباح", fmt_pct(g["ProfitGrowth"]))
    with d:
        st.metric("CAGR الأرباح", fmt_pct(g["ProfitCAGR"]))

    fin_table = pd.DataFrame([
        ["الإيرادات", f["Revenue"]],
        ["صافي الربح", f["NetIncome"]],
        ["EBITDA", f["EBITDA"]],
        ["التدفق التشغيلي", f["OperatingCF"]],
        ["FCF", f["FCF"]],
        ["النقد", f["Cash"]],
        ["الديون", f["Debt"]],
        ["الأصول", f["Assets"]],
        ["حقوق الملكية", f["Equity"]],
    ], columns=["البند", "القيمة"])

    fin_table["القيمة"] = fin_table["القيمة"].map(fmt_number)
    st.dataframe(fin_table, use_container_width=True, hide_index=True)

    ratios_table = pd.DataFrame([
        ["Debt / Equity", r["DebtEquity"]],
        ["Debt / EBITDA", r["DebtEBITDA"]],
        ["Cash / Debt", r["CashDebt"]],
        ["Current Ratio", r["CurrentRatio"]],
        ["ROE", r["ROE"]],
        ["ROA", r["ROA"]],
        ["Profit Margin", r["ProfitMargin"]],
        ["P/E", r["PE"]],
        ["P/B", r["PB"]],
        ["P/S", r["PS"]],
        ["EV/EBITDA", r["EVEBITDA"]],
        ["PEG", r["PEG"]],
        ["Dividend Yield", r["DividendYield"]],
    ], columns=["المؤشر", "القيمة"])

    def ratio_format(row):
        name = row["المؤشر"]
        value = row["القيمة"]
        if name in ["ROE", "ROA", "Profit Margin", "Dividend Yield"]:
            return fmt_pct(value)
        return fmt_number(value)

    ratios_table["القيمة"] = ratios_table.apply(ratio_format, axis=1)
    st.dataframe(ratios_table, use_container_width=True, hide_index=True)

    st.markdown("### أسباب القوة المالية")
    for rr in result["Reasons"]["Fundamental"]:
        st.write("✅", rr)

# =========================================================
# VALUATION TAB
# =========================================================
with tabs[2]:
    st.subheader("💰 محرك القيمة العادلة")

    st.info(
        f"نوع الشركة المستخدم في التقييم: **{valuation['BusinessType']}**. "
        "البنوك تستخدم P/B + P/E بدل تطبيق EV/EBITDA بالطريقة التقليدية."
    )

    val_table = pd.DataFrame([
        ["DCF / FCFE", valuation["DCF"]],
        ["P/E Fair Value", valuation["PEFairValue"]],
        ["P/B Fair Value", valuation["PBFairValue"]],
        ["EV/EBITDA Fair Value", valuation["EVEBITDAFairValue"]],
        ["القيمة العادلة النهائية", valuation["FairValue"]],
        ["القيمة المحافظة", valuation["ConservativeFairValue"]],
        ["القيمة الأساسية", valuation["BaseFairValue"]],
        ["القيمة المتفائلة", valuation["OptimisticFairValue"]],
    ], columns=["النموذج", "القيمة"])

    val_table["القيمة"] = val_table["القيمة"].map(fmt_price)
    st.dataframe(val_table, use_container_width=True, hide_index=True)

    st.markdown("### قراءة التقييم")
    if np.isfinite(investment["Upside"]):
        if investment["Upside"] >= 25:
            st.success(f"السهم يتداول بخصم تقديري {investment['Upside']:.1f}% عن القيمة العادلة.")
        elif investment["Upside"] >= 0:
            st.warning(f"الهامش التقديري محدود: {investment['Upside']:.1f}%.")
        else:
            st.error(f"السعر أعلى من القيمة العادلة التقديرية بنحو {abs(investment['Upside']):.1f}%.")

# =========================================================
# TARGETS TAB
# =========================================================
with tabs[3]:
    st.subheader("🎯 الدخول والأهداف")

    e = result["Entries"]

    entry_table = pd.DataFrame([
        ["السعر الحالي", e.get("Current")],
        ["دخول Pullback", e.get("Pullback")],
        ["دخول Breakout", e.get("Breakout")],
        ["Stop Loss", e.get("Stop")],
    ], columns=["المستوى", "السعر"])
    entry_table["السعر"] = entry_table["السعر"].map(fmt_price)
    st.dataframe(entry_table, use_container_width=True, hide_index=True)

    st.markdown("### أهداف فنية قصيرة/متوسطة")
    targets = result["TechnicalTargets"]
    if targets:
        target_table = pd.DataFrame([
            [i + 1, x, ((x / current) - 1) * 100]
            for i, x in enumerate(targets)
        ], columns=["الهدف", "السعر", "العائد"])
        target_table["السعر"] = target_table["السعر"].map(fmt_price)
        target_table["العائد"] = target_table["العائد"].map(fmt_pct)
        st.dataframe(target_table, use_container_width=True, hide_index=True)
    else:
        st.write("لا توجد أهداف فنية موثوقة كافية.")

    st.markdown("### هدف الاستثمار 3 سنوات")
    st.metric("هدف 3 سنوات", fmt_price(investment["ThreeYearTarget"]))
    st.metric("CAGR المتوقع من السعر الحالي", fmt_pct(investment["ThreeYearCAGR"]))

# =========================================================
# SCENARIO TAB
# =========================================================
with tabs[4]:
    st.subheader("🧮 السيناريوهات")

    scenario_table = pd.DataFrame([
        ["محافظ", scenarios["Conservative"], -20],
        ["أساسي", scenarios["Base"], 0],
        ["متفائل", scenarios["Optimistic"], 25],
    ], columns=["السيناريو", "القيمة", "تعديل تقريبي"])

    scenario_table["القيمة"] = scenario_table["القيمة"].map(fmt_price)
    scenario_table["تعديل تقريبي"] = scenario_table["تعديل تقريبي"].map(lambda x: f"{x:+.0f}%")
    st.dataframe(scenario_table, use_container_width=True, hide_index=True)

    st.markdown("""
**المحافظ:** نمو أضعف وتقييم أكثر تحفظًا.  
**الأساسي:** استمرار الاتجاه الحالي مع نمو معقول.  
**المتفائل:** نمو أعلى وتحسن في التقييم.
""")

# =========================================================
# NEWS TAB
# =========================================================
with tabs[5]:
    st.subheader("📰 الأخبار")

    st.metric("درجة الأخبار", f"{scores['News']:+.1f}/5")
    for rr in result["Reasons"]["News"]:
        st.write(rr)

    news = result["News"]
    if news.empty:
        st.info("لا توجد أخبار متاحة من Yahoo Finance حاليًا.")
    else:
        for _, n in news.iterrows():
            title = str(n.get("العنوان", ""))
            source = str(n.get("المصدر", ""))
            link = str(n.get("الرابط", ""))
            st.markdown(f"**{title}**")
            if source:
                st.caption(source)
            if link:
                st.markdown(f"[فتح الخبر]({link})")
            st.divider()

# =========================================================
# DATA TAB
# =========================================================
with tabs[6]:
    st.subheader("📋 جودة البيانات والبيانات الخام")

    q1, q2, q3 = st.columns(3)
    with q1:
        st.metric("جودة البيانات", f"{result['DataQuality']:.1f}%")
    with q2:
        st.metric("عدد الشموع", f"{result['DataRows']:,}")
    with q3:
        st.metric("آخر شمعة", safe_date(result["LastCandleDate"]))

    raw = result["Data"].tail(250).copy()
    st.dataframe(raw, use_container_width=True)

    csv = raw.to_csv(index=True).encode("utf-8-sig")
    st.download_button(
        "⬇️ تحميل CSV",
        data=csv,
        file_name=f"{result['RawSymbol']}_analysis.csv",
        mime="text/csv",
        use_container_width=True,
    )

# =========================================================
# FINAL SUMMARY
# =========================================================
st.markdown("---")
st.subheader("🧠 الحكم النهائي")

final = scores["Final"]

summary_points = []

if final >= 75:
    summary_points.append("الدرجة النهائية قوية.")
elif final >= 65:
    summary_points.append("السهم جيد لكنه يحتاج انتقاء سعر الدخول.")
else:
    summary_points.append("السهم يحتاج حذرًا وانتظار تأكيدات أقوى.")

if np.isfinite(investment["Upside"]):
    if investment["Upside"] >= 25:
        summary_points.append("يوجد هامش أمان/قيمة محتملة جيدة مقارنة بالسعر الحالي.")
    elif investment["Upside"] < 0:
        summary_points.append("السعر الحالي أعلى من القيمة العادلة التقديرية.")

if result["Trend"] in ["اتجاه صاعد قوي", "اتجاه صاعد"]:
    summary_points.append("الاتجاه الفني داعم.")
else:
    summary_points.append("الاتجاه الفني ليس داعمًا بالكامل حاليًا.")

for s in summary_points:
    st.write("•", s)

st.caption(
    "تنبيه: هذا التطبيق أداة تحليلية تعليمية وليس توصية شراء أو بيع. "
    "القيم العادلة والسيناريوهات تعتمد على البيانات المتاحة والافتراضات، وقد تختلف جذريًا للشركات ذات البيانات الناقصة."
)
