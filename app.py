import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import re
import math
import time
from io import StringIO
from urllib.parse import quote
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# EGX FINANCIAL INTELLIGENCE PRO MAX V5.2 — DATA INTEGRITY EDITION
# Fundamental + Quality + Valuation + Risk + 3Y Scenarios
#
# NO technical indicators.
#
# Main improvements:
# - Robust data layer
# - Freshness / completeness scoring
# - Sector-aware valuation
# - FCFF/WACC DCF
# - FCFE fallback
# - Correct Residual Income logic
# - Dynamic valuation model weighting
# - Bear/Base/Bull 3Y targets
# - Dividend + Total Return + CAGR
# - Conservative scoring under missing data
# - Error logging
# - CSV export
# ============================================================

st.set_page_config(
    page_title="EGX Financial Intelligence PRO MAX V5",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        direction: rtl;
    }

    .block-container {
        max-width: 1650px;
        padding-top: 1rem;
    }

    /* إخفاء القائمة الجانبية بالكامل */
    [data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
    }

    [data-testid="stDataFrame"] {
        direction: rtl;
    }

    .small-note {
        font-size: 0.85rem;
        opacity: 0.8;
    }

    .good-box {
        padding: 12px;
        border-radius: 10px;
        border: 1px solid rgba(0,128,0,.25);
        background: rgba(0,128,0,.05);
    }

    .warning-box {
        padding: 12px;
        border-radius: 10px;
        border: 1px solid rgba(200,120,0,.25);
        background: rgba(200,120,0,.05);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# APP CONFIG
# ============================================================

APP_VERSION = "5.2 MULTI-SOURCE + DATA INTEGRITY + SECTOR MODELS"
YAHOO_SUFFIX = ".CA"

DEFAULT_EGX_SYMBOLS = list(
    dict.fromkeys(
        """
        ABUK ACGC ADIB AIVC ALCN AMER ARCC ARPI ASCM ATLC AUTO AXPH
        BINV BIOC BTFH CICH CIRA CIEB COMI COPR COSG CPCI CRST CSAG
        DAPH DOMT EAST ECAP EFID EFIH EGAS EGCH EGTS ELEC EMFD ENGC
        ETEL ETRS FAIT FWRY GDWA GBCO HELI HDBK HRHO ICFC IDHC IEEC
        IFAP IRON ISMA JUFO KABO KZPC LCID MAAL MCQE MFPC MICH MISR
        MNHD MOIL MPBS MPCO MPCI MTIE NAHO NCCW NIPH NILE OCIC OCPH
        ODIN ORAS ORHD ORWE PHAR PHDC PRCL PRMH QNBE RACC RAYA RDFI
        REAC RMDA ROTO SAIB SAUD SCEM SDTI SKPC SMFR SPIN SPMD SUGR
        SWDY TALM TMGH TORA UASG UNIT UNIP UPMS VERT WCDF WEAS WKOL
        ZECO ZAHI AFMC ARAB CCRS CLHO CNFN EASB ELSH EXPA FARE GEMA
        GSSC HITP IDBE INFI LEDA MENA MEPA NEDA OBOU PHTV PION RREI
        SIPC TAQA TATW TRTO UNBE VTMN WADI ZMID BICC BODA BTMN CITI
        EGBE MOBG NBEG QNBA UBEE AMIA APPC CERA GISS GOLD ICID ISPH
        PACH PRDC SCTS SCFM TMMT TOWN
        """
        .split()
    )
)


# Extra symbols found across EGX market directories and current/legacy watchlists.
# These are candidates, not a claim that every symbol is currently active.
EXTRA_EGX_SYMBOLS = """
EGAL CANA GPPL VLMR VLMRA OCDI FAITA FERC VALU MBSC POUL EGSA MASR EFIC ATQA
KORA MHOT ISPH NAPR OIH MIPH MOIN MPRC OLFI BONY ISMQ ZMID PRDC GOUR SPHT
NINH MCRO SVCE DSCW MFSC OFH ACTF UEFM ADCI ELKA LCSW CFGH ALRA ZEOT ACAMD
EDFM AALR AFDI AIDC AIHC ALEX ALUM AMES AMII AMOC AMPI APSW AREH ASPI CCAP
CERA CNFN DICE EALR FERC GPIM ISMQ MASR NAPR OFH POUL SCTS UEFM VALU VLMR
VLMRA WCDF ZEOT ADCI AJWA ALRA AMES ARCC ASPI ATQA BONY CANA CFHG CLHO
CPCI DSCW EDFM EGAL EGSA ELSH FERC GEMA GPIM GPPL GOUR INFI KORA LCSW
MBSC MFSC MHOT MIPH MOIN MPRC NAPR NINH OCDI OFH OIH OLFI POUL PRDC
RREI SPHT SVCE TALA TORA UEFM VLMR VLMRA ALEX ALUM AMPI GTEX RTVC ACAMD
ACAP ACTF ADCI AFDI AFMC AJWA ALRA AMES AMII AMOC AMPI APSW ARCC AREH
ASPI BICC BODA BONY CANA CCAP CERA CITI CFGH CNFN DSCW EALR EDFM EGAL
EGSA ELKA ELSH FERC GEMA GPIM GPPL GOUR INFI KORA LCSW MASR MBSC MFSC
MHOT MIPH MOIN MPRC NAPR NINH OCDI OFH OIH OLFI POUL PRDC SCTS SPHT
SVCE UEFM VALU VLMR VLMRA WCDF ZMID GTWL NARE CEFM MILS PHGC SNFC IDRE GGRN DTPP EHDR KRDI MOSC MBEG GGCC DEIN CAED NDRL LUTS
""".split()
DEFAULT_EGX_SYMBOLS = list(dict.fromkeys(DEFAULT_EGX_SYMBOLS + EXTRA_EGX_SYMBOLS))


# ============================================================
# SECTOR MAP
# ============================================================

SECTOR_MAP = {
    # Banks
    "COMI": "بنوك",
    "CIEB": "بنوك",
    "ADIB": "بنوك",
    "HDBK": "بنوك",
    "QNBE": "بنوك",
    "SAIB": "بنوك",
    "FAIT": "بنوك",
    "EGBE": "بنوك",
    "UBEE": "بنوك",
    "EXPA": "بنوك",
    "CICH": "بنوك",
    "BTFH": "بنوك",
    "NBEG": "بنوك",
    "QNBA": "بنوك",
    "CITI": "بنوك",
    "BODA": "بنوك",
    "BTMN": "بنوك",

    # Real Estate
    "TMGH": "عقارات",
    "HELI": "عقارات",
    "ORHD": "عقارات",
    "MNHD": "عقارات",
    "PHDC": "عقارات",
    "TALA": "عقارات",
    "EMFD": "عقارات",
    "MENA": "عقارات",
    "ARAB": "عقارات",

    # Healthcare / Pharma
    "DAPH": "رعاية صحية/دواء",
    "NIPH": "رعاية صحية/دواء",
    "PHAR": "رعاية صحية/دواء",
    "AXPH": "رعاية صحية/دواء",
    "IDHC": "رعاية صحية/دواء",
    "RMDA": "رعاية صحية/دواء",
    "MPCI": "رعاية صحية/دواء",

    # Chemicals / Fertilizers
    "MFPC": "كيماويات/أسمدة",
    "ABUK": "كيماويات/أسمدة",
    "SKPC": "كيماويات/أسمدة",
    "KZPC": "كيماويات/أسمدة",
    "MICH": "كيماويات/أسمدة",
    "EGCH": "كيماويات/أسمدة",

    # Food
    "DOMT": "أغذية",
    "JUFO": "أغذية",
    "EAST": "أغذية/تبغ",
    "EFID": "أغذية",
    "UASG": "أغذية",

    # Telecom / Technology
    "ETEL": "اتصالات",
    "FWRY": "مدفوعات/تكنولوجيا",
    "EFIH": "مدفوعات/تكنولوجيا",
    "RAYA": "تكنولوجيا/خدمات",
    "MTIE": "تكنولوجيا/توزيع",
    "CIRA": "تعليم/خدمات",

    # Financial services
    "HRHO": "خدمات مالية",
    "EFG": "خدمات مالية",
    "BINV": "خدمات مالية",
    "MOBG": "خدمات مالية",
    "NILE": "خدمات مالية",
    "INFI": "خدمات مالية",

    # Energy
    "EGAS": "طاقة/غاز",
    "TAQA": "طاقة",
    "MOIL": "طاقة",
    "AMOC": "طاقة",
    "GASCO": "طاقة",

    # Metals
    "ESRS": "معادن/حديد",
    "IRCC": "معادن/حديد",
    "IRON": "معادن/حديد",
    "GOLD": "معادن/تعدين",

    # Industrials
    "SWDY": "صناعة/كابلات",
    "ARCC": "مواد بناء",
    "TORA": "مواد بناء",
    "SPMD": "مواد بناء",
    "SCEM": "مواد بناء",
    "WCDF": "مواد بناء",

    # Consumer
    "AMER": "خدمات استهلاكية",
    "AUTO": "سيارات",
    "MPCO": "استهلاكي",
    "LCID": "استهلاكي",
    "ORWE": "منسوجات",
}

# ============================================================
# SECTOR CONFIGURATION
# ============================================================

SECTOR_DEFAULTS = {
    "بنوك": {
        "ke": 0.19,
        "pe": 8.5,
        "pb": 1.00,
        "growth_cap": 0.16,
        "terminal": 0.05,
        "model": "bank",
    },

    "عقارات": {
        "ke": 0.19,
        "pe": 10.0,
        "pb": 0.85,
        "growth_cap": 0.15,
        "terminal": 0.04,
        "model": "realestate",
    },

    "رعاية صحية/دواء": {
        "ke": 0.20,
        "pe": 13.0,
        "pb": 1.25,
        "growth_cap": 0.18,
        "terminal": 0.05,
        "model": "standard",
    },

    "كيماويات/أسمدة": {
        "ke": 0.20,
        "pe": 8.5,
        "pb": 1.00,
        "growth_cap": 0.12,
        "terminal": 0.04,
        "model": "cyclical",
    },

    "أغذية": {
        "ke": 0.19,
        "pe": 11.0,
        "pb": 1.15,
        "growth_cap": 0.12,
        "terminal": 0.04,
        "model": "standard",
    },

    "أغذية/تبغ": {
        "ke": 0.19,
        "pe": 10.0,
        "pb": 1.10,
        "growth_cap": 0.10,
        "terminal": 0.04,
        "model": "standard",
    },

    "اتصالات": {
        "ke": 0.18,
        "pe": 10.0,
        "pb": 1.10,
        "growth_cap": 0.10,
        "terminal": 0.04,
        "model": "telecom",
    },

    "مدفوعات/تكنولوجيا": {
        "ke": 0.21,
        "pe": 18.0,
        "pb": 2.00,
        "growth_cap": 0.25,
        "terminal": 0.06,
        "model": "growth",
    },

    "تكنولوجيا/خدمات": {
        "ke": 0.21,
        "pe": 16.0,
        "pb": 1.80,
        "growth_cap": 0.22,
        "terminal": 0.06,
        "model": "growth",
    },

    "خدمات مالية": {
        "ke": 0.20,
        "pe": 11.0,
        "pb": 1.20,
        "growth_cap": 0.16,
        "terminal": 0.05,
        "model": "financial",
    },

    "طاقة/غاز": {
        "ke": 0.19,
        "pe": 8.0,
        "pb": 1.00,
        "growth_cap": 0.10,
        "terminal": 0.04,
        "model": "cyclical",
    },

    "طاقة": {
        "ke": 0.19,
        "pe": 8.0,
        "pb": 1.00,
        "growth_cap": 0.10,
        "terminal": 0.04,
        "model": "cyclical",
    },

    "معادن/حديد": {
        "ke": 0.21,
        "pe": 8.0,
        "pb": 0.90,
        "growth_cap": 0.10,
        "terminal": 0.04,
        "model": "cyclical",
    },

    "معادن/تعدين": {
        "ke": 0.22,
        "pe": 9.0,
        "pb": 1.00,
        "growth_cap": 0.10,
        "terminal": 0.04,
        "model": "cyclical",
    },

    "مواد بناء": {
        "ke": 0.20,
        "pe": 9.0,
        "pb": 1.00,
        "growth_cap": 0.10,
        "terminal": 0.04,
        "model": "cyclical",
    },

    "صناعة/كابلات": {
        "ke": 0.20,
        "pe": 11.0,
        "pb": 1.15,
        "growth_cap": 0.12,
        "terminal": 0.04,
        "model": "standard",
    },

    "استهلاكي": {
        "ke": 0.20,
        "pe": 11.0,
        "pb": 1.10,
        "growth_cap": 0.12,
        "terminal": 0.04,
        "model": "standard",
    },

    "خدمات استهلاكية": {
        "ke": 0.21,
        "pe": 12.0,
        "pb": 1.20,
        "growth_cap": 0.14,
        "terminal": 0.05,
        "model": "standard",
    },

    "سيارات": {
        "ke": 0.21,
        "pe": 10.0,
        "pb": 1.00,
        "growth_cap": 0.10,
        "terminal": 0.04,
        "model": "standard",
    },

    "منسوجات": {
        "ke": 0.21,
        "pe": 9.0,
        "pb": 0.90,
        "growth_cap": 0.10,
        "terminal": 0.04,
        "model": "cyclical",
    },

    "تعليم/خدمات": {
        "ke": 0.20,
        "pe": 14.0,
        "pb": 1.40,
        "growth_cap": 0.18,
        "terminal": 0.05,
        "model": "growth",
    },

    "عام": {
        "ke": 0.21,
        "pe": 10.0,
        "pb": 1.00,
        "growth_cap": 0.12,
        "terminal": 0.04,
        "model": "standard",
    },
}


# Sector-specific assumptions for the newly classified industries.
# These are explicit valuation priors, not guarantees or analyst forecasts.
SECTOR_DEFAULTS.update({
    "سياحة وفنادق": {"ke": 0.24, "pe": 9.0, "pb": 0.80, "growth_cap": 0.12, "terminal": 0.035, "model": "cyclical"},
    "زراعة": {"ke": 0.24, "pe": 8.5, "pb": 0.85, "growth_cap": 0.12, "terminal": 0.035, "model": "cyclical"},
    "مواد بناء/أسمنت": {"ke": 0.23, "pe": 8.0, "pb": 0.85, "growth_cap": 0.12, "terminal": 0.035, "model": "cyclical"},
    "منسوجات/ملابس": {"ke": 0.24, "pe": 8.0, "pb": 0.80, "growth_cap": 0.11, "terminal": 0.035, "model": "cyclical"},
    "طاقة/مرافق": {"ke": 0.23, "pe": 8.5, "pb": 0.90, "growth_cap": 0.12, "terminal": 0.04, "model": "cyclical"},
    "نقل وشحن": {"ke": 0.23, "pe": 9.0, "pb": 0.90, "growth_cap": 0.12, "terminal": 0.04, "model": "cyclical"},
    "اتصالات/تكنولوجيا": {"ke": 0.22, "pe": 13.0, "pb": 1.60, "growth_cap": 0.18, "terminal": 0.045, "model": "growth"},
})


# ============================================================
# ALIASES
# ============================================================

ALIASES = {
    "revenue": [
        "Total Revenue",
        "Operating Revenue",
        "Revenue",
    ],
    "net_income": [
        "Net Income",
        "Net Income Common Stockholders",
        "Net Income Including Noncontrolling Interests",
    ],
    "pretax": [
        "Pretax Income",
    ],
    "ebit": [
        "EBIT",
        "Operating Income",
    ],
    "ebitda": [
        "EBITDA",
        "Normalized EBITDA",
    ],
    "eps": [
        "Diluted EPS",
        "Basic EPS",
        "Diluted EPS from Continuing Operations",
    ],
    "equity": [
        "Stockholders Equity",
        "Common Stock Equity",
        "Total Equity Gross Minority Interest",
        "Total Equity",
    ],
    "assets": [
        "Total Assets",
    ],
    "debt": [
        "Total Debt",
        "Long Term Debt And Capital Lease Obligation",
        "Long Term Debt",
    ],
    "cash": [
        "Cash Cash Equivalents And Short Term Investments",
        "Cash And Cash Equivalents",
        "Cash Financial",
        "Cash Equivalents",
    ],
    "ocf": [
        "Operating Cash Flow",
        "Total Cash From Operating Activities",
        "Cash Flow From Continuing Operating Activities",
    ],
    "capex": [
        "Capital Expenditure",
        "Capital Expenditures",
    ],
    "interest": [
        "Interest Expense Non Operating",
        "Interest Expense",
    ],
    "current_assets": [
        "Current Assets",
    ],
    "current_liabilities": [
        "Current Liabilities",
    ],
    "shares": [
        "Ordinary Shares Number",
        "Share Issued",
    ],
    "gross_profit": [
        "Gross Profit",
    ],
    "operating_income": [
        "Operating Income",
    ],
    "tax": [
        "Tax Provision",
    ],
    "dividends": [
        "Cash Dividends Paid",
        "Common Stock Dividend Paid",
    ],
    "retained_earnings": [
        "Retained Earnings",
        "Retained Earnings Common Stockholders",
    ],
    "depreciation": [
        "Depreciation",
        "Depreciation And Amortization",
        "Depreciation Amortization Depletion",
    ],
}

# ============================================================
# SAFE HELPERS
# ============================================================

def finite(x):
    try:
        return x is not None and np.isfinite(float(x))
    except Exception:
        return False


def sf(x, default=np.nan):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def clean(symbol):
    return re.sub(
        r"[^A-Z0-9]",
        "",
        str(symbol).upper().strip().replace(".CA", "")
    )


def ys(symbol):
    return clean(symbol) + YAHOO_SUFFIX


def safe_div(a, b, default=np.nan):
    a = sf(a)
    b = sf(b)

    if not finite(a) or not finite(b):
        return default

    if abs(b) < 1e-12:
        return default

    return a / b


def clamp(x, low, high):
    if not finite(x):
        return np.nan

    return float(np.clip(x, low, high))


def medianv(values):
    vals = [
        sf(x)
        for x in values
        if finite(x)
    ]

    return float(np.median(vals)) if vals else np.nan


def meanv(values):
    vals = [
        sf(x)
        for x in values
        if finite(x)
    ]

    return float(np.mean(vals)) if vals else np.nan


def pct_change(a, b):
    if not finite(a) or not finite(b) or b == 0:
        return np.nan

    return a / b - 1


def normalize_ratio(x):
    x = sf(x)

    if not finite(x):
        return np.nan

    if abs(x) > 2:
        return x / 100

    return x


# ============================================================
# SERIES EXTRACTION
# ============================================================

def normalize_index(value):
    return re.sub(
        r"[^a-z0-9]",
        "",
        str(value).lower()
    )


def series(df, names):
    if (
        df is None
        or not isinstance(df, pd.DataFrame)
        or df.empty
    ):
        return pd.Series(dtype=float)

    # Exact
    for name in names:
        if name in df.index:
            s = pd.to_numeric(
                df.loc[name],
                errors="coerce"
            ).dropna()

            if not s.empty:
                return s.sort_index()

    # Normalized
    norm = {
        normalize_index(i): i
        for i in df.index
    }

    for name in names:
        key = normalize_index(name)

        if key in norm:
            s = pd.to_numeric(
                df.loc[norm[key]],
                errors="coerce"
            ).dropna()

            if not s.empty:
                return s.sort_index()

    return pd.Series(dtype=float)


def last_value(df, key):
    s = series(df, ALIASES.get(key, []))

    if s.empty:
        return np.nan

    return sf(s.iloc[-1])


# ============================================================
# GROWTH
# ============================================================

def cagr(s, years=3):
    if s is None or len(s) < 2:
        return np.nan

    s = pd.to_numeric(
        s,
        errors="coerce"
    ).dropna().sort_index()

    if len(s) < 2:
        return np.nan

    n = min(years, len(s) - 1)

    start = sf(s.iloc[-1 - n])
    end = sf(s.iloc[-1])

    if (
        not finite(start)
        or not finite(end)
        or start <= 0
        or end <= 0
    ):
        return np.nan

    return (end / start) ** (1 / n) - 1


def growth_blend(*values):
    valid = [
        sf(x)
        for x in values
        if finite(x)
        and x > -0.50
        and x < 1.00
    ]

    if not valid:
        return np.nan

    # Median is intentionally used to reduce the effect
    # of one extraordinary year.
    return float(np.median(valid))


# ============================================================
# SECTOR
# ============================================================

def sector(symbol, yahoo_sector="", yahoo_industry=""):
    symbol = clean(symbol)

    if symbol in SECTOR_MAP:
        return SECTOR_MAP[symbol]

    s = (
        f"{yahoo_sector} {yahoo_industry}"
    ).lower()

    if "bank" in s:
        return "بنوك"

    if "real estate" in s or "reit" in s:
        return "عقارات"

    if any(
        x in s
        for x in [
            "health",
            "drug",
            "biotech",
            "pharmaceutical"
        ]
    ):
        return "رعاية صحية/دواء"

    if any(
        x in s
        for x in [
            "technology",
            "software",
            "information"
        ]
    ):
        return "تكنولوجيا/خدمات"

    if "financial" in s:
        return "خدمات مالية"

    if any(
        x in s
        for x in [
            "energy",
            "oil",
            "gas"
        ]
    ):
        return "طاقة"

    if any(
        x in s
        for x in [
            "communication",
            "telecom"
        ]
    ):
        return "اتصالات"

    if any(
        x in s
        for x in [
            "consumer",
            "food"
        ]
    ):
        return "أغذية"

    if any(
        x in s
        for x in [
            "chemical",
            "fertilizer"
        ]
    ):
        return "كيماويات/أسمدة"

    if any(
        x in s
        for x in [
            "metal",
            "steel",
            "mining"
        ]
    ):
        return "معادن/حديد"

    if "building" in s:
        return "مواد بناء"

    return "عام"


# ============================================================
# UNIVERSE
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def discover_universe():
    """Discover EGX candidates from independent directories, then add a stable fallback.

    The requested scan universe is capped at 246 unique symbols. Source order prioritizes
    the actively traded StockAnalysis directory, then official EGX market-watch symbols,
    then our locally maintained fallback universe. Each symbol is still checked for data;
    unpriced/unverified symbols are shown in the errors/data-quality table, not silently hidden.
    """
    stockanalysis_symbols = []
    egx_symbols = []
    headers = {"User-Agent": "Mozilla/5.0 (compatible; EGXFinancialScanner/5.0)"}

    # Source 1: StockAnalysis EGX directory (229 active tickers at the time this version was built).
    urls = [
        "https://stockanalysis.com/list/egyptian-stock-exchange/",
        "https://beta.egx.com.eg/en/market/market-watch",
    ]
    for idx, url in enumerate(urls):
        try:
            response = requests.get(url, headers=headers, timeout=14)
            if not response.ok:
                continue
            html = response.text
            if idx == 0:
                # Prefer symbols linked from EGX quote pages; handles current URL formats.
                patterns = [
                    r'/quote/egx/([A-Z0-9]{3,6})/',
                    r'/quote/egx/([A-Z0-9]{3,6})(?:\?|"|\')',
                    r'\b([A-Z]{3,6})\.CA\b',
                ]
                for pattern in patterns:
                    stockanalysis_symbols.extend(re.findall(pattern, html.upper()))
                # Pandas table parser is a second extraction route if markup changes.
                try:
                    for table in pd.read_html(StringIO(html)):
                        for col in table.columns:
                            if str(col).strip().lower() in {"symbol", "ticker", "ticker symbol"}:
                                stockanalysis_symbols.extend(table[col].astype(str).tolist())
                except Exception:
                    pass
            else:
                # Official EGX page exposes Reuters codes in its market-watch data/table.
                for pattern in [
                    r'"reutersCode"\s*:\s*"([A-Z0-9]{3,6})"',
                    r'"symbol"\s*:\s*"([A-Z0-9]{3,6})"',
                    r'\b([A-Z]{3,6})\b(?=.{0,120}(?:Reuters Code|ISIN))',
                ]:
                    egx_symbols.extend(re.findall(pattern, html.upper()))
        except Exception:
            continue

    def valid_symbol(x):
        x = clean(x)
        return bool(re.fullmatch(r"[A-Z0-9]{3,6}", x)) and x not in {
            "EGX", "EGP", "ISIN", "HTTP", "HTTPS", "HTML", "TRUE", "FALSE",
            "STOCK", "PRICE", "CLOSE", "VOLUME", "VALUE", "CHANGE", "NAME"
        }

    def dedupe(values):
        out = []
        seen = set()
        for value in values:
            if valid_symbol(value):
                sym = clean(value)
                if sym not in seen:
                    seen.add(sym)
                    out.append(sym)
        return out

    active = dedupe(stockanalysis_symbols)
    official = dedupe(egx_symbols)
    fallback = dedupe(DEFAULT_EGX_SYMBOLS)
    combined = dedupe(active + official + fallback)

    # The user requested a 246-symbol scan. Prefer the freshest active directory order,
    # and use locally maintained candidates only to fill the gap. Never duplicate a symbol.
    target = 246
    if len(combined) >= target:
        return combined[:target]

    # In case every web source is blocked, preserve the local list and add only symbols
    # that passed the ticker-format validation. No synthetic symbols are generated.
    return combined


# ============================================================
# DATA FETCH
# ============================================================

def _stockanalysis_number(value, money_in_millions=True, per_share=False):
    """Parse StockAnalysis compact values. Source financial statements are usually in millions EGP."""
    if value is None:
        return np.nan
    raw = str(value).strip().replace(",", "").replace("−", "-").replace("–", "-")
    if raw.lower() in {"", "-", "—", "n/a", "na", "nan", "none", "null", "nm", "not available"}:
        return np.nan
    negative = raw.startswith("(") and raw.endswith(")")
    if negative:
        raw = raw[1:-1].strip()
    # Ignore common currency marks while preserving suffixes and percentages.
    raw = re.sub(r"^(?:EGP|LE|USD|\$|£)\s*", "", raw, flags=re.I)
    raw = raw.replace("EGP", "").replace("LE", "").strip()
    is_pct = raw.endswith("%")
    raw = raw.replace("%", "")
    mult = 1.0
    if raw[-1:].upper() in {"B", "M", "K"}:
        suffix = raw[-1:].upper()
        raw = raw[:-1]
        mult = {"B": 1e9, "M": 1e6, "K": 1e3}[suffix]
        money_in_millions = False
    try:
        number = float(raw) * mult
        if negative:
            number = -abs(number)
        if is_pct:
            return number / 100.0
        if money_in_millions and not per_share:
            number *= 1e6
        return number
    except Exception:
        return np.nan


def _stockanalysis_frame_from_tables(tables, statement_kind):
    """Convert StockAnalysis HTML tables into Yahoo-compatible annual statement frames."""
    if not tables:
        return pd.DataFrame()
    mapping = {
        "income": {
            "revenue": ("Total Revenue", False), "total revenue": ("Total Revenue", False),
            "revenue revenue growth": ("Total Revenue", False), "net income net income growth": ("Net Income", False),
            "net income": ("Net Income", False), "earnings per share eps growth": ("Diluted EPS", True),
            "earnings per share": ("Diluted EPS", True), "diluted eps": ("Diluted EPS", True),
            "operating income": ("Operating Income", False), "gross profit": ("Gross Profit", False),
            "pretax income": ("Pretax Income", False), "ebitda": ("EBITDA", False),
            "tax provision": ("Tax Provision", False),
        },
        "balance": {
            "total assets": ("Total Assets", False), "total equity": ("Total Equity", False),
            "stockholders equity": ("Stockholders Equity", False), "total debt": ("Total Debt", False),
            "cash and cash equivalents": ("Cash And Cash Equivalents", False),
            "cash & equivalents": ("Cash And Cash Equivalents", False),
            "cash cash equivalents and short term investments": ("Cash Cash Equivalents And Short Term Investments", False),
            "current assets": ("Current Assets", False), "current liabilities": ("Current Liabilities", False),
            "retained earnings": ("Retained Earnings", False),
        },
        "cashflow": {
            "operating cash flow": ("Operating Cash Flow", False), "capital expenditures": ("Capital Expenditures", False),
            "capex": ("Capital Expenditures", False), "depreciation & amortization": ("Depreciation And Amortization", False),
            "cash dividends paid": ("Cash Dividends Paid", False),
        },
    }
    selected = mapping.get(statement_kind, {})
    rows = {}
    for table in tables:
        if not isinstance(table, pd.DataFrame) or table.empty:
            continue
        df = table.copy()
        # Flatten multi-row column headers produced by pandas.read_html.
        df.columns = [" ".join(str(x) for x in c if str(x) != "nan") if isinstance(c, tuple) else str(c) for c in df.columns]
        label_col = df.columns[0]
        for _, row in df.iterrows():
            label = str(row.get(label_col, "")).strip()
            normalized = re.sub(r"[^a-z0-9& ]", "", label.lower()).strip()
            target = selected.get(normalized)
            if not target:
                # Tolerate label suffixes such as "Revenue Revenue Growth".
                for key, val in selected.items():
                    if normalized.startswith(key) or key.startswith(normalized):
                        target = val
                        break
            if not target:
                continue
            target_label, per_share = target
            values = {}
            for col in df.columns[1:]:
                coltext = str(col)
                years = re.findall(r"(?:FY\s*)?(20\d{2})", coltext)
                if not years:
                    continue
                year = int(years[-1])
                val = _stockanalysis_number(row.get(col), money_in_millions=True, per_share=per_share)
                if finite(val):
                    values[pd.Timestamp(year=year, month=12, day=31)] = val
            if values and target_label not in rows:
                rows[target_label] = values
            elif values:
                rows[target_label].update(values)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).T
    return frame.reindex(sorted(frame.columns), axis=1)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_stockanalysis_bundle(symbol, needed=("income", "balance", "cashflow")):
    """Independent fundamentals/quote fallback from StockAnalysis (S&P Global data attribution on pages)."""
    symbol = clean(symbol)
    result = {"info": {}, "income": pd.DataFrame(), "balance": pd.DataFrame(), "cashflow": pd.DataFrame(), "price": np.nan, "price_date": "", "source": "StockAnalysis"}
    base = f"https://stockanalysis.com/quote/egx/{quote(symbol)}/"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; EGXFinancialScanner/5.0)"}
    pages = {
        "income": base + "financials/",
        "balance": base + "financials/balance-sheet/",
        "cashflow": base + "financials/cash-flow-statement/",
    }
    for kind, url in pages.items():
        if kind not in set(needed or ()):
            continue
        try:
            resp = requests.get(url, headers=headers, timeout=8)
            if resp.ok:
                try:
                    tables = pd.read_html(StringIO(resp.text))
                except Exception:
                    tables = []
                result[kind] = _stockanalysis_frame_from_tables(tables, kind)
                # Capture price from page metadata where available.
                if not finite(result.get("price")):
                    text = resp.text
                    for pattern in [r'"price"\s*:\s*([0-9]+(?:\.[0-9]+)?)', r'"lastPrice"\s*:\s*([0-9]+(?:\.[0-9]+)?)', r'"regularMarketPrice"\s*:\s*([0-9]+(?:\.[0-9]+)?)']:
                        match = re.search(pattern, text, re.I)
                        if match:
                            candidate = sf(match.group(1))
                            if finite(candidate) and candidate > 0:
                                result["price"] = candidate
                                break
        except Exception:
            continue
    # Overview page has price/ratios/sector in human-readable text or embedded JSON.
    try:
        resp = requests.get(base, headers=headers, timeout=8)
        if resp.ok:
            html = resp.text
            plain = re.sub(r"<[^>]+>", " ", html)
            plain = re.sub(r"\s+", " ", plain)
            if not finite(result["price"]):
                match = re.search(r"(?:Delayed Price)[^0-9]{0,100}([0-9]+(?:\.[0-9]+)?)|(?:^|\s)Price\s+([0-9]+(?:\.[0-9]+)?)", plain, re.I)
                if match:
                    result["price"] = sf(match.group(1) or match.group(2))
            sector_match = re.search(r"Industry\s+([^|<]{3,80})", plain, re.I)
            if sector_match:
                result["info"]["industry"] = sector_match.group(1).strip()
            marketcap = re.search(r"Market Cap\s+([0-9,.]+)\s*([BMK])", plain, re.I)
            if marketcap:
                result["info"]["marketCap"] = _stockanalysis_number(marketcap.group(1) + marketcap.group(2), False)
            eps_match = re.search(r"EPS\s+([0-9,.]+)", plain, re.I)
            if eps_match:
                result["info"]["trailingEps"] = sf(eps_match.group(1).replace(",", ""))
            pe_match = re.search(r"PE Ratio\s+([0-9,.]+)", plain, re.I)
            if pe_match:
                result["info"]["trailingPE"] = sf(pe_match.group(1).replace(",", ""))
            div_match = re.search(r"Dividend\s+([0-9,.]+)\s*\(([0-9,.]+)%\)", plain, re.I)
            if div_match:
                result["info"]["dividendRate"] = sf(div_match.group(1).replace(",", ""))
                result["info"]["dividendYield"] = sf(div_match.group(2).replace(",", "")) / 100.0
    except Exception:
        pass
    return result


def fetch_price_fallbacks(symbol):
    """Try independent public quote sources after Yahoo has no recent close."""
    symbol = clean(symbol)
    headers = {"User-Agent": "Mozilla/5.0 (compatible; EGXFinancialScanner/5.0)"}
    # Source A: StockAnalysis delayed EGX quote.
    try:
        data = fetch_stockanalysis_bundle(symbol, ())
        if finite(data.get("price")) and data["price"] > 0:
            return data["price"], data.get("price_date", ""), "StockAnalysis"
    except Exception:
        pass
    # Source B: Stooq CSV, if the symbol is available there. Validate the date and price.
    for suffix in (".eg", ".ca"):
        try:
            url = f"https://stooq.com/q/d/l/?s={symbol.lower()}{suffix}&i=d"
            resp = requests.get(url, headers=headers, timeout=8)
            if resp.ok and "Date,Open,High,Low,Close,Volume" in resp.text[:200]:
                hist = pd.read_csv(StringIO(resp.text))
                if not hist.empty and "Close" in hist.columns:
                    hist["Close"] = pd.to_numeric(hist["Close"], errors="coerce")
                    hist = hist.dropna(subset=["Close"]).sort_values("Date")
                    if not hist.empty:
                        last = hist.iloc[-1]
                        dt = pd.to_datetime(last["Date"], errors="coerce")
                        age = (pd.Timestamp.utcnow().tz_localize(None) - dt).days if pd.notna(dt) else 9999
                        if 0 < float(last["Close"]) and age <= 30:
                            return float(last["Close"]), str(last["Date"]), "Stooq"
        except Exception:
            continue
    # Source C: Mubasher quote page. Use only an explicit numeric quote pattern, not arbitrary page numbers.
    try:
        url = f"https://english.mubasher.info/markets/EGX/stocks/{quote(symbol)}/"
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.ok:
            html = resp.text
            patterns = [r'"lastPrice"\s*:\s*"?([0-9]+(?:\.[0-9]+)?)', r'"last_price"\s*:\s*"?([0-9]+(?:\.[0-9]+)?)', r'"price"\s*:\s*"?([0-9]+(?:\.[0-9]+)?)']
            for pattern in patterns:
                match = re.search(pattern, html, re.I)
                if match:
                    val = sf(match.group(1))
                    if finite(val) and val > 0:
                        return val, "", "Mubasher"
    except Exception:
        pass
    return np.nan, "", "Unavailable"


@st.cache_data(
    ttl=1800,
    show_spinner=False
)
def fetch_bundle(symbol):

    symbol = clean(symbol)

    result = {
        "symbol": symbol,
        "ok": False,
        "error": "",
        "info": {},
        "income": pd.DataFrame(),
        "balance": pd.DataFrame(),
        "cashflow": pd.DataFrame(),
        "history": pd.DataFrame(),
        "price": np.nan,
        "price_date": "",
        "price_source": "Yahoo Finance",
        "financial_source": "Yahoo Finance",
        "sources_used": [],
    }

    try:

        ticker = yf.Ticker(
            ys(symbol)
        )

        # ----------------------------------------------------
        # INFO
        # ----------------------------------------------------

        try:
            info = ticker.info or {}
        except Exception:
            info = {}

        result["info"] = info

        # ----------------------------------------------------
        # FINANCIAL STATEMENTS
        # ----------------------------------------------------

        try:
            income = ticker.get_income_stmt(
                freq="yearly"
            )
        except Exception:
            try:
                income = ticker.income_stmt
            except Exception:
                income = pd.DataFrame()

        try:
            balance = ticker.get_balance_sheet(
                freq="yearly"
            )
        except Exception:
            try:
                balance = ticker.balance_sheet
            except Exception:
                balance = pd.DataFrame()

        try:
            cashflow = ticker.get_cash_flow(
                freq="yearly"
            )
        except Exception:
            try:
                cashflow = ticker.cashflow
            except Exception:
                cashflow = pd.DataFrame()

        result["income"] = income
        result["balance"] = balance
        result["cashflow"] = cashflow

        # ----------------------------------------------------
        # PRICE
        # ----------------------------------------------------

        history = pd.DataFrame()

        try:
            history = ticker.history(
                period="15d",
                interval="1d",
                auto_adjust=False,
                actions=True
            )
        except Exception:
            history = pd.DataFrame()

        result["history"] = history

        price = np.nan
        price_date = ""

        if (
            history is not None
            and not history.empty
            and "Close" in history.columns
        ):

            closes = pd.to_numeric(
                history["Close"],
                errors="coerce"
            ).dropna()

            if not closes.empty:
                price = sf(closes.iloc[-1])

                try:
                    price_date = str(
                        history.index[-1].date()
                    )
                except Exception:
                    price_date = ""

        if not finite(price):

            price = sf(
                info.get(
                    "currentPrice",
                    info.get(
                        "regularMarketPrice"
                    )
                )
            )

        result["price"] = price
        result["price_date"] = price_date
        has_yahoo_any = finite(price) or any(isinstance(df, pd.DataFrame) and not df.empty for df in [income, balance, cashflow])
        result["sources_used"] = ["Yahoo Finance"] if has_yahoo_any else []

        # Independent financial fallback: request only missing statement pages to reduce
        # load on public websites. Never overwrite Yahoo data with a conflicting source silently.
        missing_statements = [key for key, df in [("income", income), ("balance", balance), ("cashflow", cashflow)] if df is None or not isinstance(df, pd.DataFrame) or df.empty]
        need_sa = bool(missing_statements) or not finite(price)
        sa = {}
        if need_sa:
            try:
                sa = fetch_stockanalysis_bundle(symbol, tuple(missing_statements))
                for key in ["income", "balance", "cashflow"]:
                    existing = result.get(key)
                    alternate = sa.get(key)
                    if (existing is None or not isinstance(existing, pd.DataFrame) or existing.empty) and isinstance(alternate, pd.DataFrame) and not alternate.empty:
                        result[key] = alternate
                        result["financial_source"] = "Yahoo Finance + StockAnalysis" if result["sources_used"] else "StockAnalysis"
                yahoo_age = 9999
                if price_date:
                    try:
                        yahoo_age = (datetime.now().date() - datetime.strptime(price_date, "%Y-%m-%d").date()).days
                    except Exception:
                        yahoo_age = 9999
                # Replace a missing or clearly stale Yahoo close only when an independent
                # quote source returns a positive, explicit price.
                if finite(sa.get("price")) and sa["price"] > 0 and (not finite(result.get("price")) or result.get("price", 0) <= 0 or yahoo_age > 5):
                    result["price"] = sa["price"]
                    # Never label a replacement quote with the stale Yahoo candle date.
                    result["price_date"] = sa.get("price_date", "") or ""
                    result["price_source"] = "StockAnalysis (date not independently verified)"
                if isinstance(sa.get("info"), dict):
                    merged = dict(sa["info"])
                    merged.update({k: v for k, v in result["info"].items() if v not in (None, "", np.nan)})
                    result["info"] = merged
                if sa:
                    result["sources_used"] = list(dict.fromkeys(result["sources_used"] + ["StockAnalysis"]))
            except Exception:
                pass

        # Last-resort quote-only sources: Stooq then Mubasher.
        if not finite(result.get("price")) or result.get("price", np.nan) <= 0:
            fallback_price, fallback_date, fallback_source = fetch_price_fallbacks(symbol)
            if finite(fallback_price) and fallback_price > 0:
                result["price"] = fallback_price
                result["price_date"] = fallback_date
                result["price_source"] = fallback_source + (" (date unknown)" if not fallback_date else "")
                result["sources_used"] = list(dict.fromkeys(result["sources_used"] + [fallback_source]))

        # Do not trust a stale Yahoo close if a second source provides a newer explicit quote;
        # avoid replacing a valid current Yahoo price without a validated alternative.
        has_financials = any(
            isinstance(x, pd.DataFrame)
            and not x.empty
            for x in [
                income,
                balance,
                cashflow
            ]
        )

        result["ok"] = (
            (finite(result.get("price")) and result.get("price") > 0)
            or any(isinstance(result.get(k), pd.DataFrame) and not result[k].empty for k in ["income", "balance", "cashflow"])
        )

        if not result["ok"]:
            result["error"] = (
                "No usable price or financial data"
            )

        return result

    except Exception as exc:

        result["error"] = str(exc)[:500]

        return result


# ============================================================
# FCF
# ============================================================

def calculate_fcf(ocf, capex):
    if not finite(ocf) or not finite(capex):
        return np.nan

    return ocf - abs(capex)


def calculate_fcf_series(ocf_series, capex_series):

    if (
        ocf_series is None
        or capex_series is None
        or ocf_series.empty
        or capex_series.empty
    ):
        return pd.Series(dtype=float)

    rows = []

    common = sorted(
        set(ocf_series.index)
        & set(capex_series.index)
    )

    for idx in common:

        ocf = sf(ocf_series.loc[idx])
        capex = sf(capex_series.loc[idx])

        fcf = calculate_fcf(
            ocf,
            capex
        )

        if finite(fcf):
            rows.append(
                (idx, fcf)
            )

    if not rows:
        return pd.Series(dtype=float)

    return pd.Series(
        dict(rows)
    ).sort_index()


# ============================================================
# TAX RATE
# ============================================================

def effective_tax_rate(income):

    pretax = series(
        income,
        ALIASES["pretax"]
    )

    tax = series(
        income,
        ALIASES["tax"]
    )

    if pretax.empty or tax.empty:
        return np.nan

    pre = sf(pretax.iloc[-1])
    taxv = sf(tax.iloc[-1])

    if not finite(pre) or pre <= 0:
        return np.nan

    rate = taxv / pre

    return clamp(
        rate,
        0.0,
        0.40
    )


# ============================================================
# COST OF DEBT / WACC
# ============================================================

def estimate_cost_of_debt(
    interest_expense,
    debt,
    sector_cfg
):

    if (
        finite(interest_expense)
        and finite(debt)
        and debt > 0
    ):

        kd = abs(
            interest_expense
            / debt
        )

        if 0.01 <= kd <= 0.30:
            return kd

    return max(
        0.06,
        sector_cfg["ke"] - 0.04
    )


def calculate_wacc(
    ke,
    debt,
    equity_market,
    interest_expense,
    tax_rate,
    sector_cfg
):

    if not finite(ke):
        return np.nan

    tax = (
        tax_rate
        if finite(tax_rate)
        else 0.20
    )

    debt = max(
        sf(debt, 0),
        0
    )

    equity_market = max(
        sf(equity_market, 0),
        0
    )

    if debt <= 0 or equity_market <= 0:
        return ke

    kd = estimate_cost_of_debt(
        interest_expense,
        debt,
        sector_cfg
    )

    total = debt + equity_market

    we = equity_market / total
    wd = debt / total

    wacc = (
        we * ke
        + wd * kd * (1 - tax)
    )

    return clamp(
        wacc,
        0.07,
        0.35
    )


# ============================================================
# DCF FCFF
# ============================================================

def dcf_fcff(
    fcff,
    shares,
    wacc,
    growth,
    terminal_growth,
    years=5,
    debt=0.0,
    cash_balance=0.0
):

    if not all(
        finite(x)
        for x in [
            fcff,
            shares,
            wacc,
            growth,
            terminal_growth
        ]
    ):
        return np.nan

    if (
        fcff <= 0
        or shares <= 0
        or wacc <= terminal_growth
    ):
        return np.nan

    g = clamp(
        growth,
        -0.03,
        min(0.20, wacc - 0.02)
    )

    gt = clamp(
        terminal_growth,
        0.02,
        min(0.05, wacc - 0.02)
    )

    if not finite(g) or not finite(gt):
        return np.nan

    pv = 0.0

    for year in range(1, years + 1):

        forecast_cash = (
            fcff
            * (1 + g) ** year
        )

        pv += (
            forecast_cash
            / (1 + wacc) ** year
        )

    terminal = (
        fcff
        * (1 + g) ** years
        * (1 + gt)
        / (wacc - gt)
    )

    pv += (
        terminal
        / (1 + wacc) ** years
    )

    # FCFF estimates enterprise value, not equity value.
    # Convert to equity value by subtracting debt and adding cash.
    net_debt = max(sf(debt, 0.0), 0.0) - max(sf(cash_balance, 0.0), 0.0)
    equity_value = pv - net_debt
    return max(0.0, equity_value / shares)


# ============================================================
# FCFE
# ============================================================

def dcf_fcfe(
    fcfe,
    shares,
    ke,
    growth,
    terminal_growth,
    years=5
):

    if not all(
        finite(x)
        for x in [
            fcfe,
            shares,
            ke,
            growth,
            terminal_growth
        ]
    ):
        return np.nan

    if (
        fcfe <= 0
        or shares <= 0
        or ke <= terminal_growth
    ):
        return np.nan

    g = clamp(
        growth,
        -0.03,
        min(0.20, ke - 0.02)
    )

    gt = clamp(
        terminal_growth,
        0.02,
        min(0.05, ke - 0.02)
    )

    pv = 0.0

    for year in range(1, years + 1):

        cash = (
            fcfe
            * (1 + g) ** year
        )

        pv += (
            cash
            / (1 + ke) ** year
        )

    terminal = (
        fcfe
        * (1 + g) ** years
        * (1 + gt)
        / (ke - gt)
    )

    pv += (
        terminal
        / (1 + ke) ** years
    )

    return pv / shares


# ============================================================
# RESIDUAL INCOME
# ============================================================

def residual_income_model(
    bvps,
    eps,
    roe,
    payout,
    ke,
    growth,
    years=5
):

    if not all(
        finite(x)
        for x in [
            bvps,
            eps,
            ke,
            growth
        ]
    ):
        return np.nan

    if (
        bvps <= 0
        or eps <= 0
        or ke <= 0
    ):
        return np.nan

    g = clamp(
        growth,
        -0.03,
        min(0.18, ke - 0.02)
    )

    if not finite(g):
        return np.nan

    if finite(payout):
        payout = clamp(
            payout,
            0.0,
            0.90
        )
    else:
        payout = 0.35

    retention = 1 - payout

    current_book = bvps
    value = bvps

    forecast_eps = eps

    for year in range(1, years + 1):

        forecast_eps *= (
            1 + g
        )

        forecast_roe = (
            roe
            if finite(roe)
            else ke + 0.04
        )

        implied_book_growth = (
            forecast_roe
            * retention
        )

        implied_book_growth = clamp(
            implied_book_growth,
            -0.05,
            0.25
        )

        previous_book = current_book

        current_book = (
            current_book
            * (1 + implied_book_growth)
        )

        residual_income = (
            forecast_eps
            - ke * previous_book
        )

        value += (
            residual_income
            / (1 + ke) ** year
        )

    terminal_roe = (
        ke + 0.02
    )

    terminal_ri = (
        forecast_eps
        * (1 + min(g, 0.05))
        - ke * current_book
    )

    terminal_value = (
        terminal_ri
        / max(
            0.01,
            ke - min(g, 0.05)
        )
    )

    value += (
        terminal_value
        / (1 + ke) ** years
    )

    return max(
        0,
        value
    )


# ============================================================
# MULTIPLE VALUATIONS
# ============================================================

def pe_value(
    eps,
    multiple
):

    if (
        not finite(eps)
        or eps <= 0
        or not finite(multiple)
        or multiple <= 0
    ):
        return np.nan

    return eps * multiple


def pb_value(
    bvps,
    multiple
):

    if (
        not finite(bvps)
        or bvps <= 0
        or not finite(multiple)
        or multiple <= 0
    ):
        return np.nan

    return bvps * multiple


def ev_ebitda_value(
    ebitda,
    debt,
    cash,
    shares,
    multiple
):

    if not all(
        finite(x)
        for x in [
            ebitda,
            debt,
            cash,
            shares,
            multiple
        ]
    ):
        return np.nan

    if (
        ebitda <= 0
        or shares <= 0
        or multiple <= 0
    ):
        return np.nan

    enterprise_value = (
        ebitda * multiple
    )

    equity_value = (
        enterprise_value
        - max(debt, 0)
        + max(cash, 0)
    )

    return max(
        0,
        equity_value / shares
    )


# ============================================================
# PIOTROSKI
# ============================================================

def piotroski_f_score(
    income,
    balance,
    cashflow
):

    ni = series(
        income,
        ALIASES["net_income"]
    )

    ocf = series(
        cashflow,
        ALIASES["ocf"]
    )

    assets = series(
        balance,
        ALIASES["assets"]
    )

    debt = series(
        balance,
        ALIASES["debt"]
    )

    current_assets = series(
        balance,
        ALIASES["current_assets"]
    )

    current_liabilities = series(
        balance,
        ALIASES["current_liabilities"]
    )

    shares = series(
        balance,
        ALIASES["shares"]
    )

    gross_profit = series(
        income,
        ALIASES["gross_profit"]
    )

    revenue = series(
        income,
        ALIASES["revenue"]
    )

    if any(
        len(x) < 2
        for x in [
            ni,
            ocf,
            assets
        ]
    ):
        return np.nan

    score = 0

    # 1 ROA positive
    roa_now = safe_div(
        ni.iloc[-1],
        assets.iloc[-1]
    )

    if finite(roa_now) and roa_now > 0:
        score += 1

    # 2 OCF positive
    if (
        finite(ocf.iloc[-1])
        and ocf.iloc[-1] > 0
    ):
        score += 1

    # 3 ROA improving
    roa_prev = safe_div(
        ni.iloc[-2],
        assets.iloc[-2]
    )

    if (
        finite(roa_now)
        and finite(roa_prev)
        and roa_now > roa_prev
    ):
        score += 1

    # 4 CFO > NI
    accrual = (
        ocf.iloc[-1]
        - ni.iloc[-1]
    )

    if finite(accrual) and accrual > 0:
        score += 1

    # 5 Lower leverage
    if len(debt) >= 2:

        if (
            finite(debt.iloc[-1])
            and finite(debt.iloc[-2])
            and debt.iloc[-1] <= debt.iloc[-2]
        ):
            score += 1

    # 6 Better liquidity
    if (
        len(current_assets) >= 2
        and len(current_liabilities) >= 2
    ):

        cr_now = safe_div(
            current_assets.iloc[-1],
            current_liabilities.iloc[-1]
        )

        cr_prev = safe_div(
            current_assets.iloc[-2],
            current_liabilities.iloc[-2]
        )

        if (
            finite(cr_now)
            and finite(cr_prev)
            and cr_now > cr_prev
        ):
            score += 1

    # 7 No dilution
    if len(shares) >= 2:

        if (
            finite(shares.iloc[-1])
            and finite(shares.iloc[-2])
            and shares.iloc[-1] <= shares.iloc[-2] * 1.01
        ):
            score += 1

    # 8 Gross margin improving
    if (
        len(gross_profit) >= 2
        and len(revenue) >= 2
    ):

        gm_now = safe_div(
            gross_profit.iloc[-1],
            revenue.iloc[-1]
        )

        gm_prev = safe_div(
            gross_profit.iloc[-2],
            revenue.iloc[-2]
        )

        if (
            finite(gm_now)
            and finite(gm_prev)
            and gm_now > gm_prev
        ):
            score += 1

        # 9 Asset turnover improving
        at_now = safe_div(
            revenue.iloc[-1],
            assets.iloc[-1]
        )

        at_prev = safe_div(
            revenue.iloc[-2],
            assets.iloc[-2]
        )

        if (
            finite(at_now)
            and finite(at_prev)
            and at_now > at_prev
        ):
            score += 1

    return float(score)


# ============================================================
# ALTMAN Z
# ============================================================

def altman_z(r):

    if r.get("sector") == "بنوك":
        return np.nan

    A = r.get("assets")
    WC = np.nan

    if (
        finite(r.get("current_assets"))
        and finite(r.get("current_liabilities"))
    ):
        WC = (
            r["current_assets"]
            - r["current_liabilities"]
        )

    RE = r.get("retained_earnings")
    EBIT = r.get("ebit")
    MC = r.get("market_cap")
    TL = r.get("debt")
    SALES = r.get("revenue")

    if not all(
        finite(x)
        for x in [
            A,
            WC,
            RE,
            EBIT,
            MC,
            TL,
            SALES
        ]
    ):
        return np.nan

    if A <= 0 or TL <= 0:
        return np.nan

    return (
        1.2 * WC / A
        + 1.4 * RE / A
        + 3.3 * EBIT / A
        + 0.6 * MC / TL
        + SALES / A
    )


# ============================================================
# QUALITY METRICS
# ============================================================

def percentile_score(
    x,
    low,
    high
):

    if not finite(x):
        return np.nan

    if high == low:
        return 50.0

    return float(
        np.clip(
            (x - low)
            / (high - low)
            * 100,
            0,
            100
        )
    )


def quality_components(r):

    # Growth
    growth = percentile_score(
        r.get("normalized_growth"),
        -0.05,
        0.25
    )

    # ROE
    roe = percentile_score(
        r.get("roe"),
        -0.05,
        0.30
    )

    # Margin
    margin = percentile_score(
        r.get("net_margin"),
        -0.05,
        0.30
    )

    profitability = meanv(
        [
            roe,
            margin
        ]
    )

    # Financial strength
    if r.get("sector") == "بنوك":

        strength = percentile_score(
            r.get("roe"),
            -0.02,
            0.30
        )

    else:

        de = r.get("debt_equity")
        cr = r.get("current_ratio")
        ic = r.get("interest_coverage")

        if finite(de):
            debt_score = percentile_score(
                1 / (1 + max(de, 0)),
                0,
                1
            )
        else:
            debt_score = np.nan

        current_score = percentile_score(
            cr,
            0.3,
            3.0
        )

        interest_score = percentile_score(
            ic,
            1,
            15
        )

        strength = meanv(
            [
                debt_score,
                current_score,
                interest_score
            ]
        )

    # Cash quality
    cash_conversion = r.get("ocf_ni")

    cashq = percentile_score(
        cash_conversion,
        0.3,
        1.5
    )

    if model not in ("bank", "financial") and (
        finite(r.get("fcf_normalized"))
        and r["fcf_normalized"] > 0
        and finite(cashq)
    ):
        cashq = min(
            100,
            cashq + 15
        )

    # Valuation
    upside = r.get("upside")

    valuation = percentile_score(
        upside,
        -0.30,
        0.50
    )

    # Dividend
    dividend = percentile_score(
        r.get("dividend_yield"),
        0,
        0.10
    )

    # Piotroski
    piotroski = percentile_score(
        r.get("piotroski"),
        2,
        9
    )

    return {
        "valuation": valuation,
        "growth": growth,
        "profitability": profitability,
        "strength": strength,
        "cash_quality": cashq,
        "dividend": dividend,
        "piotroski": piotroski,
    }


# ============================================================
# DATA QUALITY
# ============================================================

def financial_integrity_audit(r):
    """Flag material unit/accounting inconsistencies without fabricating replacement values."""
    flags = []
    price = sf(r.get("price"))
    shares = sf(r.get("shares"))
    market_cap = sf(r.get("market_cap"))
    net_income = sf(r.get("net_income"))
    eps = sf(r.get("eps"))
    equity = sf(r.get("equity"))
    assets = sf(r.get("assets"))
    revenue = sf(r.get("revenue"))
    debt = sf(r.get("debt"))
    cash = sf(r.get("cash"))

    # A reported market cap should be broadly compatible with price x shares.
    if all(finite(x) and x > 0 for x in (price, shares, market_cap)):
        implied = price * shares
        ratio = max(implied, market_cap) / max(min(implied, market_cap), 1e-12)
        if ratio > 5:
            flags.append("عدم اتساق كبير بين القيمة السوقية والسعر وعدد الأسهم؛ راجع الوحدات")

    # EPS and net income/share count can differ, but not normally by orders of magnitude.
    if all(finite(x) for x in (net_income, eps, shares)) and shares > 0 and abs(net_income) > 0 and abs(eps) > 0:
        implied_eps = net_income / shares
        ratio = max(abs(implied_eps), abs(eps)) / max(min(abs(implied_eps), abs(eps)), 1e-12)
        if ratio > 20:
            flags.append("احتمال اختلاف وحدات صافي الربح أو عدد الأسهم أو EPS")

    if finite(assets) and assets > 0 and finite(equity) and equity > assets * 1.10:
        flags.append("حقوق الملكية أكبر من الأصول بأكثر من 10%؛ راجع المصدر والوحدات")
    if finite(revenue) and revenue > 0 and finite(net_income) and abs(net_income) > revenue * 3:
        flags.append("صافي الربح يتجاوز الإيرادات بأكثر من 3 مرات؛ يحتاج تحققًا")
    if finite(debt) and debt < 0:
        flags.append("قيمة الدين سالبة؛ تحتاج مراجعة")
    if finite(cash) and cash < 0:
        flags.append("قيمة النقدية سالبة؛ تحتاج مراجعة")

    age = sf(r.get("price_age_days"))
    if finite(age) and age > 10:
        flags.append("السعر أقدم من 10 أيام")
    elif not finite(age) and any(token in str(r.get("price_source", "")).lower() for token in ("date unknown", "date not independently verified")):
        flags.append("تاريخ السعر غير متاح أو غير متحقق منه")
    return flags


def calculate_data_quality(r):

    checks = []

    # Price
    checks.append(
        1 if finite(r.get("price")) else 0
    )

    # Revenue
    checks.append(
        1 if finite(r.get("revenue")) else 0
    )

    # Net income
    checks.append(
        1 if finite(r.get("net_income")) else 0
    )

    # EPS
    checks.append(
        1 if finite(r.get("eps")) else 0
    )

    # Equity
    checks.append(
        1 if finite(r.get("equity")) else 0
    )

    # Assets
    checks.append(
        1 if finite(r.get("assets")) else 0
    )

    # Debt
    checks.append(
        1 if finite(r.get("debt")) else 0
    )

    # Cash
    checks.append(
        1 if finite(r.get("cash")) else 0
    )

    # OCF
    checks.append(
        1 if finite(r.get("ocf")) else 0
    )

    # FCF
    checks.append(
        1 if finite(r.get("fcf")) else 0
    )

    # Shares
    checks.append(
        1 if finite(r.get("shares")) else 0
    )

    # Historical growth
    checks.append(
        1 if finite(r.get("revenue_growth_3y")) else 0
    )

    completeness = (
        np.mean(checks) * 100
        if checks
        else 0
    )

    # Financial history depth
    history_years = r.get(
        "financial_years",
        0
    )

    history_score = min(
        100,
        history_years / 5 * 100
    )

    # Price freshness
    freshness = r.get(
        "price_age_days",
        np.nan
    )

    if finite(freshness):

        if freshness <= 2:
            freshness_score = 100

        elif freshness <= 5:
            freshness_score = 90

        elif freshness <= 10:
            freshness_score = 75

        elif freshness <= 20:
            freshness_score = 50

        else:
            freshness_score = 20

    else:
        freshness_score = 40

    quality = (
        0.55 * completeness
        + 0.25 * history_score
        + 0.20 * freshness_score
    )

    # Missing dates are already reflected in freshness. Material unit/accounting flags
    # receive an additional penalty; values remain visible for human review.
    integrity_flags = r.get("integrity_warnings", []) or []
    quality -= min(25, 8 * len(integrity_flags))

    return round(
        float(
            np.clip(
                quality,
                0,
                100
            )
        ),
        1
    )


# ============================================================
# VALUATION ENGINE
# ============================================================

def valuation_engine(r):

    cfg = SECTOR_DEFAULTS.get(
        r["sector"],
        SECTOR_DEFAULTS["عام"]
    )

    model = cfg["model"]

    growth = r.get(
        "normalized_growth"
    )

    if not finite(growth):
        growth = 0.06

    growth = clamp(
        growth,
        -0.03,
        cfg["growth_cap"]
    )

    ke = cfg["ke"]

    # --------------------------------------------------------
    # DCF
    # --------------------------------------------------------

    dcf = np.nan
    dcf_type = "غير متاح"

    if model != "bank" and model != "financial" and (
        finite(r.get("fcff_normalized"))
        and r["fcff_normalized"] > 0
        and finite(r.get("shares"))
        and r["shares"] > 0
    ):

        wacc = calculate_wacc(
            ke=ke,
            debt=r.get("debt"),
            equity_market=r.get("market_cap"),
            interest_expense=r.get("interest_expense"),
            tax_rate=r.get("tax_rate"),
            sector_cfg=cfg
        )

        dcf = dcf_fcff(
            r["fcff_normalized"],
            r["shares"],
            wacc,
            growth,
            cfg["terminal"],
            debt=r.get("debt", 0.0),
            cash_balance=r.get("cash", 0.0)
        )

        if finite(dcf):
            dcf_type = "FCFF / WACC"

    # FCFE fallback
    if not finite(dcf) and model not in ("bank", "financial"):

        if (
            finite(r.get("fcf_normalized"))
            and r["fcf_normalized"] > 0
            and finite(r.get("shares"))
            and r["shares"] > 0
        ):

            dcf = dcf_fcfe(
                r["fcf_normalized"],
                r["shares"],
                ke,
                growth,
                cfg["terminal"]
            )

            if finite(dcf):
                dcf_type = "FCFE / Ke"

    # --------------------------------------------------------
    # Residual Income
    # --------------------------------------------------------

    residual = np.nan

    if model == "bank":

        residual = residual_income_model(
            bvps=r.get("bvps"),
            eps=r.get("eps_normalized"),
            roe=r.get("roe"),
            payout=r.get("payout"),
            ke=ke,
            growth=growth
        )

    # --------------------------------------------------------
    # Multiples
    # --------------------------------------------------------

    pe = pe_value(
        r.get("eps_normalized"),
        cfg["pe"]
    )

    pb = pb_value(
        r.get("bvps"),
        cfg["pb"]
    )

    evm = ev_ebitda_value(
        r.get("ebitda_normalized"),
        r.get("debt"),
        r.get("cash"),
        r.get("shares"),
        max(
            5,
            cfg["pe"] - 1
        )
    )

    fcf_yield_value = np.nan

    if (
        finite(r.get("fcf_normalized"))
        and r["fcf_normalized"] > 0
        and finite(r.get("shares"))
        and r["shares"] > 0
    ):

        fcf_yield_value = (
            r["fcf_normalized"]
            / r["shares"]
            / max(
                0.07,
                ke
            )
        )

    # --------------------------------------------------------
    # Candidate models
    # --------------------------------------------------------

    candidates = []

    if model == "bank":

        candidate_specs = [
            ("Residual Income", residual, 0.45),
            ("P/B", pb, 0.35),
            ("P/E", pe, 0.20),
        ]

    elif model == "financial":

        candidate_specs = [
            ("P/B", pb, 0.55),
            ("P/E", pe, 0.45),
        ]

    elif model == "realestate":

        candidate_specs = [
            ("DCF", dcf, 0.30),
            ("P/B", pb, 0.25),
            ("P/E", pe, 0.20),
            ("EV/EBITDA", evm, 0.15),
            ("FCF Yield", fcf_yield_value, 0.10),
        ]

    else:

        candidate_specs = [
            ("DCF", dcf, 0.35),
            ("P/E", pe, 0.20),
            ("EV/EBITDA", evm, 0.20),
            ("FCF Yield", fcf_yield_value, 0.15),
            ("P/B", pb, 0.10),
        ]

    for name, value, weight in candidate_specs:

        if (
            finite(value)
            and value > 0
            and value < 100000
        ):
            candidates.append(
                (
                    name,
                    float(value),
                    float(weight)
                )
            )

    if not candidates:

        return {
            "fair_value": np.nan,
            "dcf_value": dcf,
            "dcf_type": dcf_type,
            "residual_value": residual,
            "pe_value": pe,
            "pb_value": pb,
            "ev_ebitda_value": evm,
            "fcf_yield_value": fcf_yield_value,
            "valuation_confidence": 0,
            "model_count": 0,
            "model_dispersion": np.nan,
            "model_agreement": 0,
        }

    values = np.array([x[1] for x in candidates], dtype=float)
    weights = np.array([x[2] for x in candidates], dtype=float)
    weights = weights / weights.sum()

    # Robust consensus: reject extreme model outputs relative to the median.
    # This is a guardrail, not a claim that a model is automatically wrong.
    initial_median = float(np.median(values))
    if len(values) >= 3 and initial_median > 0:
        keep = (values >= initial_median / 4.0) & (values <= initial_median * 4.0)
        if int(keep.sum()) >= 2:
            values = values[keep]
            weights = weights[keep]
            candidates = [c for c, k in zip(candidates, keep) if k]
            weights = weights / weights.sum()

    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    fair = float(sorted_values[np.searchsorted(np.cumsum(sorted_weights), 0.5)])

    # Median absolute deviation is less sensitive to a single extreme model.
    median_value = float(np.median(values))
    mad = float(np.median(np.abs(values - median_value)))
    dispersion = mad / max(abs(median_value), 1e-9)
    agreement = float(np.clip(100.0 - dispersion * 150.0, 0.0, 100.0))

    coverage = float(np.clip(sf(r.get("coverage", 0), 0), 0, 100))
    data_quality = float(np.clip(sf(r.get("data_quality", 0), 0), 0, 100))
    confidence = 0.45 * agreement + 0.30 * coverage + 0.25 * data_quality

    # A disagreement in financial units is a reason to lower confidence, not to
    # silently alter the fair value or replace inputs with guessed numbers.
    confidence -= min(30, 10 * len(r.get("integrity_warnings", []) or []))

    # Additional penalty when only one model exists
    if len(candidates) == 1:
        confidence *= 0.65

    elif len(candidates) == 2:
        confidence *= 0.82

    confidence = float(
        np.clip(
            confidence,
            0,
            100
        )
    )

    return {
        "fair_value": fair,
        "dcf_value": dcf,
        "dcf_type": dcf_type,
        "residual_value": residual,
        "pe_value": pe,
        "pb_value": pb,
        "ev_ebitda_value": evm,
        "fcf_yield_value": fcf_yield_value,
        "valuation_confidence": confidence,
        "model_count": len(candidates),
        "model_dispersion": dispersion,
        "model_agreement": agreement,
    }


# ============================================================
# 3Y SCENARIOS
# ============================================================

def scenario_growths(r):

    cfg = SECTOR_DEFAULTS.get(
        r["sector"],
        SECTOR_DEFAULTS["عام"]
    )

    base = r.get(
        "normalized_growth"
    )

    if not finite(base):
        base = 0.06

    base = clamp(
        base,
        -0.05,
        cfg["growth_cap"]
    )

    conservative = clamp(
        base - 0.05,
        -0.08,
        cfg["growth_cap"] - 0.01
    )

    optimistic = clamp(
        base + 0.05,
        -0.03,
        cfg["growth_cap"] + 0.05
    )

    return (
        conservative,
        base,
        optimistic
    )


def target_multiple(
    cfg,
    scenario
):

    if scenario == "conservative":
        factor = 0.85

    elif scenario == "optimistic":
        factor = 1.15

    else:
        factor = 1.00

    return {
        "pe": cfg["pe"] * factor,
        "pb": cfg["pb"] * factor,
    }


def targets_3y(r):

    cfg = SECTOR_DEFAULTS.get(
        r["sector"],
        SECTOR_DEFAULTS["عام"]
    )

    g_cons, g_base, g_opt = (
        scenario_growths(r)
    )

    eps = r.get(
        "eps_normalized"
    )

    bvps = r.get(
        "bvps"
    )

    fair = r.get(
        "fair_value"
    )

    model = cfg["model"]

    def calculate_target(
        growth,
        scenario
    ):

        mult = target_multiple(
            cfg,
            scenario
        )

        eps_target = np.nan
        pb_target = np.nan

        if (
            finite(eps)
            and eps > 0
        ):

            eps_future = (
                eps
                * (1 + growth) ** 3
            )

            eps_target = (
                eps_future
                * mult["pe"]
            )

        if (
            finite(bvps)
            and bvps > 0
        ):

            bv_future = (
                bvps
                * (1 + max(
                    -0.02,
                    growth * 0.70
                )) ** 3
            )

            pb_target = (
                bv_future
                * mult["pb"]
            )

        if model == "bank":

            target = medianv(
                [
                    pb_target,
                    eps_target
                ]
            )

        elif model in [
            "realestate",
            "financial"
        ]:

            target = medianv(
                [
                    pb_target,
                    eps_target,
                    fair
                ]
            )

        else:

            target = medianv(
                [
                    eps_target,
                    fair
                ]
            )

            if not finite(target):
                target = pb_target

        return target

    conservative = calculate_target(
        g_cons,
        "conservative"
    )

    base = calculate_target(
        g_base,
        "base"
    )

    optimistic = calculate_target(
        g_opt,
        "optimistic"
    )

    # --------------------------------------------------------
    # Dividend contribution
    # --------------------------------------------------------

    dividend_yield = r.get(
        "dividend_yield"
    )

    if not finite(dividend_yield):
        dividend_yield = 0

    payout = r.get(
        "payout"
    )

    if not finite(payout):
        payout = 0.30

    payout = clamp(
        payout,
        0,
        0.90
    )

    dividend_growth = clamp(
        g_base * 0.60,
        -0.03,
        0.10
    )

    annual_dividend_yield = dividend_yield

    total_dividend_per_share = 0

    if finite(
        annual_dividend_yield
    ):

        for year in range(1, 4):

            total_dividend_per_share += (
                r.get("price", 0)
                * annual_dividend_yield
                * (1 + dividend_growth)
                ** (year - 1)
            )

    current_price = r.get(
        "price"
    )

    base_price_return = np.nan
    base_total_return = np.nan
    base_cagr = np.nan

    if (
        finite(base)
        and finite(current_price)
        and current_price > 0
    ):

        base_price_return = (
            base / current_price
            - 1
        )

        base_total_return = (
            base_price_return
            + total_dividend_per_share
            / current_price
        )

        if base_total_return > -1:
            base_cagr = (
                (1 + base_total_return)
                ** (1 / 3)
                - 1
            )

    return {
        "target_3y_cons": conservative,
        "target_3y_base": base,
        "target_3y_opt": optimistic,
        "dividend_3y_per_share": (
            total_dividend_per_share
        ),
        "total_return_3y_base": (
            base_total_return
        ),
        "return_3y_base": base_cagr,
        "growth_3y_cons": g_cons,
        "growth_3y_base": g_base,
        "growth_3y_opt": g_opt,
    }


# ============================================================
# DCF SENSITIVITY
# ============================================================

def dcf_sensitivity(r):

    fcff = r.get(
        "fcff_normalized"
    )

    shares = r.get(
        "shares"
    )

    cfg = SECTOR_DEFAULTS.get(
        r["sector"],
        SECTOR_DEFAULTS["عام"]
    )

    ke = cfg["ke"]

    if not all(
        finite(x)
        for x in [
            fcff,
            shares,
            ke
        ]
    ):
        return pd.DataFrame()

    rows = []

    growth_values = [
        max(
            -0.02,
            r.get(
                "normalized_growth",
                0.06
            ) - 0.03
        ),
        r.get(
            "normalized_growth",
            0.06
        ),
        min(
            cfg["growth_cap"],
            r.get(
                "normalized_growth",
                0.06
            ) + 0.03
        )
    ]

    wacc_values = [
        max(0.08, ke - 0.02),
        ke,
        ke + 0.02
    ]

    for wacc in wacc_values:

        row = {
            "WACC": wacc
        }

        for growth in growth_values:

            if wacc <= cfg["terminal"]:
                value = np.nan
            else:
                value = dcf_fcff(
                    fcff,
                    shares,
                    wacc,
                    growth,
                    cfg["terminal"],
                    debt=r.get("debt", 0.0),
                    cash_balance=r.get("cash", 0.0)
                )

            row[
                f"Growth {growth:.1%}"
            ] = value

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# FINAL SCORE
# ============================================================

def score_engine(r):

    components = quality_components(r)

    # Fill unavailable components conservatively,
    # not optimistically.
    valuation = (
        components["valuation"]
        if finite(
            components["valuation"]
        )
        else 40
    )

    growth = (
        components["growth"]
        if finite(
            components["growth"]
        )
        else 40
    )

    profitability = (
        components["profitability"]
        if finite(
            components["profitability"]
        )
        else 40
    )

    strength = (
        components["strength"]
        if finite(
            components["strength"]
        )
        else 45
    )

    cash_quality = (
        components["cash_quality"]
        if finite(
            components["cash_quality"]
        )
        else 40
    )

    dividend = (
        components["dividend"]
        if finite(
            components["dividend"]
        )
        else 40
    )

    piotroski = (
        components["piotroski"]
        if finite(
            components["piotroski"]
        )
        else 45
    )

    raw = (
        0.24 * valuation
        + 0.18 * growth
        + 0.18 * profitability
        + 0.15 * strength
        + 0.10 * cash_quality
        + 0.05 * dividend
        + 0.05 * piotroski
        + 0.05 * r["coverage"]
    )

    # --------------------------------------------------------
    # Confidence penalties
    # --------------------------------------------------------

    coverage = r.get(
        "coverage",
        0
    )

    valuation_confidence = r.get(
        "valuation_confidence",
        0
    )

    model_count = r.get(
        "model_count",
        0
    )

    # Data penalty
    if coverage >= 80:
        data_penalty = 0

    elif coverage >= 65:
        data_penalty = 2

    elif coverage >= 50:
        data_penalty = 6

    elif coverage >= 40:
        data_penalty = 12

    else:
        data_penalty = 25

    # Valuation penalty
    if valuation_confidence >= 80:
        valuation_penalty = 0

    elif valuation_confidence >= 65:
        valuation_penalty = 2

    elif valuation_confidence >= 50:
        valuation_penalty = 5

    elif valuation_confidence > 0:
        valuation_penalty = 9

    else:
        valuation_penalty = 15

    # Model-count penalty
    if model_count >= 4:
        model_penalty = 0

    elif model_count == 3:
        model_penalty = 1

    elif model_count == 2:
        model_penalty = 3

    elif model_count == 1:
        model_penalty = 7

    else:
        model_penalty = 15

    final = (
        raw
        - data_penalty
        - valuation_penalty
        - model_penalty
    )

    return round(
        float(
            np.clip(
                final,
                0,
                100
            )
        ),
        2
    )


# ============================================================
# RATING
# ============================================================

def rating(score):

    if not finite(score):
        return "غير متاح"

    if score >= 90:
        return "استثنائي"

    if score >= 85:
        return "ممتاز جدًا"

    if score >= 78:
        return "ممتاز"

    if score >= 70:
        return "قوي"

    if score >= 62:
        return "جيد"

    if score >= 52:
        return "متوسط"

    if score >= 42:
        return "ضعيف"

    return "تجنب"


# ============================================================
# INVESTMENT STATUS
# ============================================================

def investment_status(r):

    price = r.get("price")
    fair = r.get("fair_value")
    score = r.get("score")

    if not finite(price) or not finite(fair):
        return "بيانات غير كافية"

    if price <= fair * 0.70 and score >= 70:
        return "شراء ممتاز"

    if price <= fair * 0.80 and score >= 65:
        return "شراء قوي"

    if price <= fair * 0.90 and score >= 60:
        return "شراء مقبول"

    if price <= fair * 1.05:
        return "احتفاظ"

    if price <= fair * 1.20:
        return "مبالغ فيه نسبيًا"

    return "مبالغ فيه"


# ============================================================
# BUILD ANALYSIS
# ============================================================

def build(bundle):

    info = bundle.get(
        "info",
        {}
    ) or {}

    income = bundle.get(
        "income"
    )

    balance = bundle.get(
        "balance"
    )

    cashflow = bundle.get(
        "cashflow"
    )

    symbol = clean(
        bundle["symbol"]
    )

    price = sf(
        bundle.get("price")
    )

    yahoo_sector = info.get(
        "sector",
        ""
    )

    yahoo_industry = info.get(
        "industry",
        ""
    )

    sec = sector(
        symbol,
        yahoo_sector,
        yahoo_industry
    )

    # --------------------------------------------------------
    # Series
    # --------------------------------------------------------

    revenue_s = series(
        income,
        ALIASES["revenue"]
    )

    ni_s = series(
        income,
        ALIASES["net_income"]
    )

    eps_s = series(
        income,
        ALIASES["eps"]
    )

    equity_s = series(
        balance,
        ALIASES["equity"]
    )

    assets_s = series(
        balance,
        ALIASES["assets"]
    )

    debt_s = series(
        balance,
        ALIASES["debt"]
    )

    cash_s = series(
        balance,
        ALIASES["cash"]
    )

    ocf_s = series(
        cashflow,
        ALIASES["ocf"]
    )

    capex_s = series(
        cashflow,
        ALIASES["capex"]
    )

    current_assets_s = series(
        balance,
        ALIASES["current_assets"]
    )

    current_liabilities_s = series(
        balance,
        ALIASES["current_liabilities"]
    )

    ebit_s = series(
        income,
        ALIASES["ebit"]
    )

    ebitda_s = series(
        income,
        ALIASES["ebitda"]
    )

    gross_s = series(
        income,
        ALIASES["gross_profit"]
    )

    interest_s = series(
        income,
        ALIASES["interest"]
    )

    retained_s = series(
        balance,
        ALIASES["retained_earnings"]
    )

    shares_s = series(
        balance,
        ALIASES["shares"]
    )

    # --------------------------------------------------------
    # Latest values
    # --------------------------------------------------------

    shares = sf(
        info.get(
            "sharesOutstanding"
        )
    )

    if not finite(shares) and not shares_s.empty:
        shares = sf(
            shares_s.iloc[-1]
        )

    market_cap = sf(
        info.get(
            "marketCap"
        )
    )

    if (
        not finite(shares)
        and finite(market_cap)
        and finite(price)
        and price > 0
    ):
        shares = (
            market_cap
            / price
        )

    if (
        not finite(market_cap)
        and finite(shares)
        and finite(price)
    ):
        market_cap = (
            shares
            * price
        )

    revenue = (
        sf(revenue_s.iloc[-1])
        if not revenue_s.empty
        else sf(
            info.get(
                "totalRevenue"
            )
        )
    )

    net_income = (
        sf(ni_s.iloc[-1])
        if not ni_s.empty
        else sf(
            info.get(
                "netIncomeToCommon"
            )
        )
    )

    eps = (
        sf(eps_s.iloc[-1])
        if not eps_s.empty
        else sf(
            info.get(
                "trailingEps"
            )
        )
    )

    equity = (
        sf(equity_s.iloc[-1])
        if not equity_s.empty
        else np.nan
    )

    assets = (
        sf(assets_s.iloc[-1])
        if not assets_s.empty
        else np.nan
    )

    debt = (
        sf(debt_s.iloc[-1])
        if not debt_s.empty
        else sf(
            info.get(
                "totalDebt"
            )
        )
    )

    cash = (
        sf(cash_s.iloc[-1])
        if not cash_s.empty
        else sf(
            info.get(
                "totalCash"
            )
        )
    )

    ocf = (
        sf(ocf_s.iloc[-1])
        if not ocf_s.empty
        else np.nan
    )

    capex = (
        sf(capex_s.iloc[-1])
        if not capex_s.empty
        else np.nan
    )

    fcf = calculate_fcf(
        ocf,
        capex
    )

    bvps = safe_div(
        equity,
        shares
    )

    if (
        not finite(bvps)
        or bvps <= 0
    ):
        bvps = sf(
            info.get(
                "bookValue"
            )
        )

    revenue_per_share = safe_div(
        revenue,
        shares
    )

    roe = safe_div(
        net_income,
        equity
    )

    if not finite(roe):
        roe = normalize_ratio(
            info.get(
                "returnOnEquity"
            )
        )

    roa = safe_div(
        net_income,
        assets
    )

    net_margin = safe_div(
        net_income,
        revenue
    )

    debt_equity = safe_div(
        debt,
        equity
    )

    current_assets = (
        sf(current_assets_s.iloc[-1])
        if not current_assets_s.empty
        else np.nan
    )

    current_liabilities = (
        sf(current_liabilities_s.iloc[-1])
        if not current_liabilities_s.empty
        else np.nan
    )

    current_ratio = safe_div(
        current_assets,
        current_liabilities
    )

    if not finite(current_ratio):
        current_ratio = sf(
            info.get(
                "currentRatio"
            )
        )

    interest_expense = (
        abs(
            sf(
                interest_s.iloc[-1]
            )
        )
        if not interest_s.empty
        else np.nan
    )

    ebit = (
        sf(ebit_s.iloc[-1])
        if not ebit_s.empty
        else np.nan
    )

    interest_coverage = safe_div(
        ebit,
        interest_expense
    )

    ebitda = (
        sf(ebitda_s.iloc[-1])
        if not ebitda_s.empty
        else np.nan
    )

    retained_earnings = (
        sf(retained_s.iloc[-1])
        if not retained_s.empty
        else np.nan
    )

    # --------------------------------------------------------
    # Growth
    # --------------------------------------------------------

    revenue_growth = cagr(
        revenue_s
    )

    earnings_growth = cagr(
        ni_s
    )

    eps_growth = cagr(
        eps_s
    )

    fcf_s = calculate_fcf_series(
        ocf_s,
        capex_s
    )

    fcf_growth = cagr(
        fcf_s
    )

    normalized_growth = growth_blend(
        revenue_growth,
        earnings_growth,
        eps_growth,
        fcf_growth
    )

    if not finite(
        normalized_growth
    ):

        info_growth = normalize_ratio(
            info.get(
                "earningsGrowth"
            )
        )

        if finite(info_growth):
            normalized_growth = info_growth

        else:
            normalized_growth = 0.06

    # --------------------------------------------------------
    # Normalized values
    # --------------------------------------------------------

    eps_normalized = np.nan

    if (
        finite(eps)
        and eps > 0
    ):

        recent_eps = []

        if not eps_s.empty:

            recent_eps = [
                sf(x)
                for x in eps_s.tail(3).values
                if finite(x)
                and x > 0
            ]

        eps_normalized = medianv(
            recent_eps + [
                eps
            ]
        )

    fcf_normalized = np.nan

    if not fcf_s.empty:

        positive_fcfs = [
            sf(x)
            for x in fcf_s.tail(3).values
            if finite(x)
            and x > 0
        ]

        if positive_fcfs:
            fcf_normalized = float(
                np.median(
                    positive_fcfs
                )
            )

    if (
        not finite(fcf_normalized)
        and finite(fcf)
        and fcf > 0
    ):
        fcf_normalized = fcf

    # FCFF approximation:
    # OCF is after interest.
    # Add back after-tax interest.
    tax_rate = effective_tax_rate(
        income
    )

    if not finite(tax_rate):
        tax_rate = 0.20

    fcff = np.nan

    if finite(fcf_normalized):

        after_tax_interest = 0

        if finite(
            interest_expense
        ):
            after_tax_interest = (
                interest_expense
                * (1 - tax_rate)
            )

        fcff = (
            fcf_normalized
            + after_tax_interest
        )

    # EBITDA normalized
    ebitda_normalized = np.nan

    if not ebitda_s.empty:

        vals = [
            sf(x)
            for x in ebitda_s.tail(3).values
            if finite(x)
            and x > 0
        ]

        if vals:
            ebitda_normalized = float(
                np.median(vals)
            )

    if (
        not finite(ebitda_normalized)
        and finite(ebitda)
        and ebitda > 0
    ):
        ebitda_normalized = ebitda

    # --------------------------------------------------------
    # Dividend
    # --------------------------------------------------------

    dividend_yield = normalize_ratio(
        info.get(
            "dividendYield"
        )
    )

    payout = normalize_ratio(
        info.get(
            "payoutRatio"
        )
    )

    # Fallback dividend per share
    dividend_per_share = sf(
        info.get(
            "dividendRate"
        )
    )

    if (
        not finite(dividend_yield)
        and finite(dividend_per_share)
        and finite(price)
        and price > 0
    ):
        dividend_yield = (
            dividend_per_share
            / price
        )

    # --------------------------------------------------------
    # Price freshness
    # --------------------------------------------------------

    price_date = bundle.get(
        "price_date",
        ""
    )

    price_age_days = np.nan

    if price_date:

        try:

            pdate = datetime.strptime(
                price_date,
                "%Y-%m-%d"
            ).date()

            today = datetime.now().date()

            price_age_days = (
                today - pdate
            ).days

        except Exception:
            price_age_days = np.nan

    # --------------------------------------------------------
    # Base record
    # --------------------------------------------------------

    financial_years = max(
        len(revenue_s),
        len(ni_s),
        len(equity_s),
        len(ocf_s)
    )

    r = {
        "symbol": symbol,
        "name": info.get(
            "longName",
            info.get(
                "shortName",
                symbol
            )
        ),
        "sector": sec,
        "price_source": bundle.get("price_source", "Yahoo Finance"),
        "financial_source": bundle.get("financial_source", "Yahoo Finance"),
        "sources_used": ", ".join(bundle.get("sources_used", [])) if bundle.get("sources_used") else "غير محدد",

        "price": price,
        "price_date": price_date,
        "price_age_days": price_age_days,

        "shares": shares,
        "market_cap": market_cap,

        "revenue": revenue,
        "net_income": net_income,
        "eps": eps,
        "equity": equity,
        "assets": assets,
        "debt": debt,
        "cash": cash,

        "ocf": ocf,
        "capex": capex,
        "fcf": fcf,
        "fcf_normalized": fcf_normalized,
        "fcff_normalized": fcff,

        "bvps": bvps,
        "revenue_per_share": revenue_per_share,

        "roe": roe,
        "roa": roa,
        "net_margin": net_margin,
        "debt_equity": debt_equity,
        "current_ratio": current_ratio,
        "interest_coverage": interest_coverage,

        "revenue_growth_3y": revenue_growth,
        "earnings_growth_3y": earnings_growth,
        "eps_growth_3y": eps_growth,
        "fcf_growth_3y": fcf_growth,

        "normalized_growth": normalized_growth,

        "eps_normalized": eps_normalized,
        "ebitda_normalized": ebitda_normalized,

        "current_assets": current_assets,
        "current_liabilities": current_liabilities,
        "retained_earnings": retained_earnings,
        "ebit": ebit,

        "interest_expense": interest_expense,
        "tax_rate": tax_rate,

        "dividend_yield": dividend_yield,
        "payout": payout,
        "dividend_per_share": dividend_per_share,

        "financial_years": financial_years,
    }

    # --------------------------------------------------------
    # Quality / risk
    # --------------------------------------------------------

    r["piotroski"] = piotroski_f_score(
        income,
        balance,
        cashflow
    )

    r["ocf_ni"] = safe_div(
        ocf,
        net_income
    )

    r["altman_z"] = altman_z(
        r
    )

    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    coverage_keys = [
        "price",
        "revenue",
        "net_income",
        "eps",
        "equity",
        "assets",
        "debt",
        "cash",
        "ocf",
        "fcf",
        "shares",
        "bvps",
        "roe",
        "revenue_growth_3y",
        "earnings_growth_3y",
        "normalized_growth",
    ]

    available = sum(
        finite(
            r.get(key)
        )
        for key in coverage_keys
    )

    r["coverage"] = round(
        100
        * available
        / len(coverage_keys),
        1
    )

    r["integrity_warnings"] = financial_integrity_audit(r)

    r["data_quality"] = (
        calculate_data_quality(r)
    )

    r["data_quality_label"] = (
        "عالية"
        if r["data_quality"] >= 80
        else "جيدة"
        if r["data_quality"] >= 65
        else "متوسطة"
        if r["data_quality"] >= 50
        else "منخفضة"
    )

    # --------------------------------------------------------
    # Valuation
    # --------------------------------------------------------

    val = valuation_engine(
        r
    )

    r.update(val)

    # Extreme fair-value/current-price ratios are surfaced explicitly and reduce
    # confidence. The calculated value is preserved for inspection, never silently capped.
    if finite(r.get("fair_value")) and finite(r.get("price")) and r.get("price", 0) > 0:
        fv_ratio = r["fair_value"] / r["price"]
        if fv_ratio > 5 or fv_ratio < 0.20:
            r["valuation_confidence"] = max(0.0, sf(r.get("valuation_confidence"), 0) - 20.0)
            r.setdefault("integrity_warnings", []).append(
                "فرق شديد بين القيمة العادلة والسعر الحالي (>5x أو <0.2x)؛ لا تعتمد على التقييم قبل المراجعة"
            )

    # --------------------------------------------------------
    # Upside / Buy Zones
    # --------------------------------------------------------

    fair = r.get(
        "fair_value"
    )

    if (
        finite(fair)
        and fair > 0
        and finite(price)
        and price > 0
    ):

        r["upside"] = (
            fair / price - 1
        )

        r["buy_excellent"] = (
            fair * 0.70
        )

        r["buy_strong"] = (
            fair * 0.80
        )

        r["buy_acceptable"] = (
            fair * 0.90
        )

    else:

        r["upside"] = np.nan
        r["buy_excellent"] = np.nan
        r["buy_strong"] = np.nan
        r["buy_acceptable"] = np.nan

    # --------------------------------------------------------
    # 3Y
    # --------------------------------------------------------

    r.update(
        targets_3y(r)
    )

    # --------------------------------------------------------
    # Score
    # --------------------------------------------------------

    r["score"] = score_engine(
        r
    )

    r["rating"] = rating(
        r["score"]
    )

    r["investment_status"] = (
        investment_status(r)
    )

    # --------------------------------------------------------
    # Data warning
    # --------------------------------------------------------

    warnings = []

    if r["coverage"] < 60:
        warnings.append(
            "بيانات أساسية ناقصة"
        )

    if (
        finite(r["price_age_days"])
        and r["price_age_days"] > 5
    ):
        warnings.append(
            "السعر قديم نسبيًا"
        )

    if r["model_count"] < 2:
        warnings.append(
            "عدد نماذج التقييم قليل"
        )

    if not finite(
        r["fair_value"]
    ):
        warnings.append(
            "لا توجد قيمة عادلة موثوقة"
        )

    if finite(r.get("fair_value")) and finite(r.get("price")) and r.get("price", 0) > 0:
        ratio = r["fair_value"] / r["price"]
        if ratio > 5 or ratio < 0.20:
            warnings.append("القيمة العادلة بعيدة جدًا عن السعر؛ راجع المدخلات والنماذج")

    warnings.extend(r.get("integrity_warnings", []) or [])

    r["warnings"] = (
        " | ".join(warnings)
        if warnings
        else "لا توجد تحذيرات رئيسية"
    )

    return r


# ============================================================
# ANALYZE
# ============================================================

def analyze(symbol):

    symbol = clean(symbol)

    bundle = fetch_bundle(
        symbol
    )

    if not bundle.get("ok"):
        return {
            "symbol": symbol,
            "name": symbol,
            "sector": SECTOR_MAP.get(
                symbol,
                "عام"
            ),
            "price": np.nan,
            "price_source": "غير متاح",
            "financial_source": "غير متاح",
            "sources_used": "غير متاح",
            "score": np.nan,
            "coverage": 0,
            "valuation_confidence": 0,
            "data_quality": 0,
            "error": bundle.get(
                "error",
                "Unknown error"
            ),
        }

    try:

        return build(
            bundle
        )

    except Exception as exc:

        return {
            "symbol": symbol,
            "name": symbol,
            "sector": SECTOR_MAP.get(
                symbol,
                "عام"
            ),
            "price": np.nan,
            "score": np.nan,
            "coverage": 0,
            "valuation_confidence": 0,
            "data_quality": 0,
            "error": str(exc)[:500],
        }


# ============================================================
# SCAN
# ============================================================

def scan(
    symbols,
    workers=5
):

    rows = []

    errors = []

    symbols = list(
        dict.fromkeys(
            clean(x)
            for x in symbols
            if clean(x)
        )
    )

    with ThreadPoolExecutor(
        max_workers=workers
    ) as executor:

        futures = {
            executor.submit(
                analyze,
                symbol
            ): symbol
            for symbol in symbols
        }

        for future in as_completed(
            futures
        ):

            symbol = futures[
                future
            ]

            try:

                result = future.result()

                if result:

                    rows.append(
                        result
                    )

                    if result.get(
                        "error"
                    ):
                        errors.append(
                            {
                                "symbol": symbol,
                                "error": result[
                                    "error"
                                ],
                            }
                        )

            except Exception as exc:

                errors.append(
                    {
                        "symbol": symbol,
                        "error": str(exc)[:500],
                    }
                )

    df = pd.DataFrame(
        rows
    )

    if df.empty:

        return (
            pd.DataFrame(
                columns=[
                    "symbol",
                    "name",
                    "sector",
                    "score",
                    "valuation_confidence",
                    "coverage",
                    "data_quality",
                ]
            ),
            pd.DataFrame(
                errors
            )
        )

    for col in [
        "score",
        "valuation_confidence",
        "coverage",
        "data_quality",
        "upside",
    ]:

        if col not in df.columns:
            df[col] = np.nan

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.sort_values(
        [
            "score",
            "valuation_confidence",
            "data_quality",
            "coverage",
        ],
        ascending=False,
        na_position="last"
    ).reset_index(
        drop=True
    )

    df["rank"] = (
        np.arange(
            1,
            len(df) + 1
        )
    )

    return (
        df,
        pd.DataFrame(errors)
    )


# ============================================================
# FORMATTING
# ============================================================

def money(x):

    if not finite(x):
        return "—"

    return f"{x:,.2f}"


def pct(x):

    if not finite(x):
        return "—"

    return f"{x * 100:.1f}%"


def number(x):

    if not finite(x):
        return "—"

    return f"{x:,.2f}"


@st.cache_data(
    show_spinner=False
)
def csv_bytes(df):

    return df.to_csv(
        index=False
    ).encode(
        "utf-8-sig"
    )


# ============================================================
# DISPLAY HELPERS
# ============================================================

def financial_summary_table(r):

    rows = [
        ("الإيرادات", r.get("revenue")),
        ("صافي الربح", r.get("net_income")),
        ("EPS", r.get("eps")),
        ("حقوق الملكية", r.get("equity")),
        ("الأصول", r.get("assets")),
        ("الدين", r.get("debt")),
        ("النقد", r.get("cash")),
        ("OCF", r.get("ocf")),
        ("Capex", r.get("capex")),
        ("FCF", r.get("fcf")),
        ("FCF طبيعي", r.get("fcf_normalized")),
        ("FCFF طبيعي", r.get("fcff_normalized")),
        ("BVPS", r.get("bvps")),
        ("ROE", r.get("roe")),
        ("ROA", r.get("roa")),
        ("هامش الربح", r.get("net_margin")),
        ("D/E", r.get("debt_equity")),
        ("Current Ratio", r.get("current_ratio")),
        ("Interest Coverage", r.get("interest_coverage")),
        ("نمو الإيرادات 3Y", r.get("revenue_growth_3y")),
        ("نمو الأرباح 3Y", r.get("earnings_growth_3y")),
        ("نمو EPS 3Y", r.get("eps_growth_3y")),
        ("نمو FCF 3Y", r.get("fcf_growth_3y")),
        ("النمو الطبيعي", r.get("normalized_growth")),
        ("Piotroski", r.get("piotroski")),
        ("Altman Z", r.get("altman_z")),
        ("Dividend Yield", r.get("dividend_yield")),
        ("Payout", r.get("payout")),
    ]

    data = []

    for key, value in rows:

        if key in [
            "ROE",
            "ROA",
            "هامش الربح",
            "نمو الإيرادات 3Y",
            "نمو الأرباح 3Y",
            "نمو EPS 3Y",
            "نمو FCF 3Y",
            "النمو الطبيعي",
            "Dividend Yield",
            "Payout",
        ]:
            display = pct(value)

        else:
            display = number(value)

        data.append(
            [key, display]
        )

    return pd.DataFrame(
        data,
        columns=[
            "المؤشر",
            "القيمة"
        ]
    )


# ============================================================
# MAIN
# ============================================================

def main():

    st.title(
        "💰 EGX Financial Intelligence PRO MAX"
    )

    st.caption(
        f"""
        Fundamental-only Investment Engine |
        Quality + Valuation + Risk + 3Y Scenarios |
        Version {APP_VERSION}
        """
    )

    st.info(
        """
        المحرك مالي فقط — بدون RSI أو MACD أو مؤشرات فنية.
        الترتيب يعتمد على جودة الشركة، النمو، الربحية،
        القوة المالية، التدفقات النقدية، التقييم،
        جودة البيانات، والقيمة العادلة. الإصدار 5.2 يضيف فحص اتساق الوحدات
        والسعر وعدد الأسهم، ويخفض الثقة عند ظهور تعارضات بدل تخمين بيانات بديلة.
        """
    )

    # ========================================================
    # إعدادات افتراضية بدون القائمة الجانبية
    # ========================================================

    workers = 8
    mincov = 40
    topn = 20
    mode = "اكتشاف + احتياطي"
    custom = ""

    # ========================================================
    # UNIVERSE
    # ========================================================

    if mode == "اكتشاف + احتياطي":

        symbols = discover_universe()

    elif mode == "احتياطي فقط":

        symbols = list(
            DEFAULT_EGX_SYMBOLS
        )

    else:

        symbols = list(
            dict.fromkeys(
                clean(x)
                for x in re.split(
                    r"[,\s]+",
                    custom
                )
                if clean(x)
            )
        )

    st.write(
        f"**الكون:** {len(symbols)} رمز | الهدف 246 رمز | مصادر الكون: EGX + StockAnalysis + قائمة احتياطية"
    )
    st.caption("مصادر التحليل: Yahoo Finance أولًا، ثم StockAnalysis للقوائم/الاقتباس عند النقص، ثم Stooq وMubasher كسعر احتياطي عند توافره. لا يتم اختلاق القيم؛ النقص يقلل جودة البيانات والثقة.")

    # ========================================================
    # TABS
    # ========================================================

    tabs = st.tabs(
        [
            "📊 السوق",
            "🏭 القطاعات",
            "🔎 سهم واحد",
            "📈 التاريخ المالي",
            "🧮 منهجية المحرك",
            "🛠️ أخطاء البيانات",
        ]
    )

    # ========================================================
    # MARKET
    # ========================================================

    with tabs[0]:

        if st.button(
            "🚀 ابدأ PRO MAX Scan",
            type="primary",
            use_container_width=True
        ):

            start = time.time()

            with st.spinner(
                f"جاري تحليل {len(symbols)} سهم..."
            ):

                df, errors = scan(
                    symbols,
                    workers
                )

            st.session_state[
                "df"
            ] = df

            st.session_state[
                "errors"
            ] = errors

            elapsed = time.time() - start

            st.success(
                f"اكتمل التحليل في {elapsed:.1f} ثانية"
            )

        df = st.session_state.get(
            "df",
            pd.DataFrame()
        )

        if df.empty:

            st.warning(
                "ابدأ المسح المالي أولًا."
            )

        else:

            view = df[
                df["coverage"] >= mincov
            ].copy()

            view = view.sort_values(
                "score",
                ascending=False
            ).reset_index(
                drop=True
            )

            view["rank"] = (
                np.arange(
                    1,
                    len(view) + 1
                )
            )

            # ------------------------------------------------
            # KPIs
            # ------------------------------------------------

            a, b, c, d, e = st.columns(5)

            a.metric(
                "الأسهم المؤهلة",
                len(view)
            )

            b.metric(
                "متوسط الدرجة",
                (
                    f"{view['score'].mean():.1f}"
                    if len(view)
                    else "—"
                )
            )

            c.metric(
                "أفضل سهم",
                (
                    view.iloc[0]["symbol"]
                    if len(view)
                    else "—"
                )
            )

            d.metric(
                "وسيط Upside",
                (
                    pct(
                        view["upside"].median()
                    )
                    if len(view)
                    else "—"
                )
            )

            e.metric(
                "متوسط جودة البيانات",
                (
                    f"{view['data_quality'].mean():.1f}%"
                    if len(view)
                    else "—"
                )
            )

            st.subheader(
                "🏆 الترتيب النهائي"
            )

            cols = [
                "rank",
                "symbol",
                "name",
                "sector",
                "price",
                "fair_value",
                "buy_excellent",
                "buy_strong",
                "buy_acceptable",
                "target_3y_cons",
                "target_3y_base",
                "target_3y_opt",
                "normalized_growth",
                "roe",
                "piotroski",
                "altman_z",
                "dividend_yield",
                "upside",
                "valuation_confidence",
                "data_quality",
                "score",
                "rating",
                "investment_status",
                "price_source",
                "price_date",
                "financial_source",
                "sources_used",
                "warnings",
            ]

            cols = [
                c
                for c in cols
                if c in view.columns
            ]

            shown = view[
                cols
            ].copy()

            rename = {
                "rank": "الترتيب",
                "symbol": "الرمز",
                "name": "الشركة",
                "sector": "القطاع",
                "price": "السعر",
                "fair_value": "القيمة العادلة",
                "buy_excellent": "شراء ممتاز",
                "buy_strong": "شراء قوي",
                "buy_acceptable": "شراء مقبول",
                "target_3y_cons": "هدف محافظ 3Y",
                "target_3y_base": "هدف أساسي 3Y",
                "target_3y_opt": "هدف متفائل 3Y",
                "normalized_growth": "النمو الطبيعي",
                "roe": "ROE",
                "piotroski": "Piotroski",
                "altman_z": "Altman Z",
                "dividend_yield": "التوزيعات",
                "upside": "Upside",
                "valuation_confidence": "ثقة التقييم",
                "data_quality": "جودة البيانات",
                "score": "الدرجة",
                "rating": "التقييم",
                "investment_status": "الحالة الاستثمارية",
                "price_source": "مصدر السعر",
                "financial_source": "مصدر القوائم المالية",
                "sources_used": "المصادر المستخدمة",
            }

            shown = shown.rename(
                columns=rename
            )

            st.dataframe(
                shown,
                hide_index=True,
                use_container_width=True
            )

            st.download_button(
                "⬇️ تحميل CSV كامل",
                csv_bytes(shown),
                "EGX_PRO_MAX_Ranking_V5.csv",
                "text/csv",
                use_container_width=True
            )

            st.subheader(
                f"🥇 أفضل {topn}"
            )

            best_cols = [
                "rank",
                "symbol",
                "sector",
                "price",
                "fair_value",
                "buy_strong",
                "target_3y_base",
                "upside",
                "return_3y_base",
                "valuation_confidence",
                "data_quality",
                "score",
                "rating",
                "investment_status",
            ]

            best_cols = [
                c
                for c in best_cols
                if c in view.columns
            ]

            st.dataframe(
                view.head(topn)[
                    best_cols
                ],
                hide_index=True,
                use_container_width=True
            )

    # ========================================================
    # SECTORS
    # ========================================================

    with tabs[1]:

        df = st.session_state.get(
            "df",
            pd.DataFrame()
        )

        if df.empty:

            st.warning(
                "ابدأ المسح أولًا."
            )

        else:

            v = df[
                df["coverage"] >= mincov
            ].copy()

            if v.empty:

                st.warning(
                    "لا توجد أسهم مؤهلة."
                )

            else:

                sector_table = (
                    v.groupby("sector")
                    .agg(
                        عدد=("symbol", "count"),
                        متوسط_الدرجة=("score", "mean"),
                        أفضل_درجة=("score", "max"),
                        وسيط_Upside=("upside", "median"),
                        متوسط_الثقة=(
                            "valuation_confidence",
                            "mean"
                        ),
                        جودة_البيانات=(
                            "data_quality",
                            "mean"
                        )
                    )
                    .reset_index()
                    .sort_values(
                        "متوسط_الدرجة",
                        ascending=False
                    )
                )

                st.dataframe(
                    sector_table.rename(
                        columns={
                            "sector": "القطاع"
                        }
                    ),
                    hide_index=True,
                    use_container_width=True
                )

                sectors = sorted(
                    [
                        x
                        for x in v[
                            "sector"
                        ].dropna().unique()
                    ]
                )

                if sectors:

                    selected_sector = (
                        st.selectbox(
                            "اختار القطاع",
                            sectors
                        )
                    )

                    ss = v[
                        v["sector"]
                        == selected_sector
                    ].sort_values(
                        "score",
                        ascending=False
                    )

                    st.dataframe(
                        ss[
                            [
                                "symbol",
                                "name",
                                "price",
                                "fair_value",
                                "buy_strong",
                                "target_3y_base",
                                "upside",
                                "return_3y_base",
                                "score",
                                "data_quality",
                                "investment_status",
                            ]
                        ],
                        hide_index=True,
                        use_container_width=True
                    )

    # ========================================================
    # SINGLE STOCK
    # ========================================================

    with tabs[2]:

        sym = st.text_input(
            "رمز السهم",
            value="DAPH",
            key="single_symbol"
        ).upper()

        if st.button(
            "🔍 تحليل PRO MAX",
            use_container_width=True
        ):

            st.session_state[
                "single"
            ] = analyze(
                clean(sym)
            )

        r = st.session_state.get(
            "single"
        )

        if r:

            if r.get(
                "error"
            ):

                st.error(
                    r["error"]
                )

            else:

                st.subheader(
                    f"{r['symbol']} — {r['name']}"
                )

                # ------------------------------------------------
                # Main metrics
                # ------------------------------------------------

                m = st.columns(6)

                main_metrics = [
                    (
                        "السعر",
                        money(
                            r.get(
                                "price"
                            )
                        )
                    ),
                    (
                        "Fair Value",
                        money(
                            r.get(
                                "fair_value"
                            )
                        )
                    ),
                    (
                        "شراء قوي",
                        money(
                            r.get(
                                "buy_strong"
                            )
                        )
                    ),
                    (
                        "هدف 3Y",
                        money(
                            r.get(
                                "target_3y_base"
                            )
                        )
                    ),
                    (
                        "Score",
                        (
                            f"{r.get('score', np.nan):.1f}/100"
                            if finite(
                                r.get("score")
                            )
                            else "—"
                        )
                    ),
                    (
                        "Valuation Confidence",
                        (
                            f"{r.get('valuation_confidence', np.nan):.1f}%"
                            if finite(
                                r.get(
                                    "valuation_confidence"
                                )
                            )
                            else "—"
                        )
                    ),
                ]

                for col, (label, value) in zip(
                    m,
                    main_metrics
                ):
                    col.metric(
                        label,
                        value
                    )

                st.markdown(
                    f"""
                    **القطاع:** {r.get('sector', '—')}  
                    **التقييم:** {r.get('rating', '—')}  
                    **الحالة الاستثمارية:** {r.get('investment_status', '—')}  
                    **جودة البيانات:** {r.get('data_quality', 0):.1f}% — {r.get('data_quality_label', '—')}  
                    **تغطية البيانات:** {r.get('coverage', 0):.1f}%  
                    **السعر بتاريخ:** {r.get('price_date', '—')}  
                    **عمر السعر:** {r.get('price_age_days', '—')} يوم
                    """
                )

                if r.get(
                    "warnings"
                ):

                    st.warning(
                        r["warnings"]
                    )

                # ------------------------------------------------
                # Buy zones
                # ------------------------------------------------

                st.subheader(
                    "🎯 مناطق الشراء"
                )

                buy_table = pd.DataFrame(
                    [
                        [
                            "شراء ممتاز",
                            r.get(
                                "buy_excellent"
                            ),
                            "30% تحت Fair Value"
                        ],
                        [
                            "شراء قوي",
                            r.get(
                                "buy_strong"
                            ),
                            "20% تحت Fair Value"
                        ],
                        [
                            "شراء مقبول",
                            r.get(
                                "buy_acceptable"
                            ),
                            "10% تحت Fair Value"
                        ],
                        [
                            "القيمة العادلة",
                            r.get(
                                "fair_value"
                            ),
                            "Fair Value"
                        ],
                    ],
                    columns=[
                        "المستوى",
                        "السعر",
                        "القاعدة"
                    ]
                )

                buy_table["السعر"] = (
                    buy_table["السعر"]
                    .apply(money)
                )

                st.dataframe(
                    buy_table,
                    hide_index=True,
                    use_container_width=True
                )

                # ------------------------------------------------
                # 3Y
                # ------------------------------------------------

                st.subheader(
                    "📌 سيناريوهات 3 سنوات"
                )

                scenario_table = pd.DataFrame(
                    [
                        [
                            "محافظ",
                            r.get(
                                "target_3y_cons"
                            ),
                            r.get(
                                "growth_3y_cons"
                            ),
                        ],
                        [
                            "أساسي",
                            r.get(
                                "target_3y_base"
                            ),
                            r.get(
                                "growth_3y_base"
                            ),
                        ],
                        [
                            "متفائل",
                            r.get(
                                "target_3y_opt"
                            ),
                            r.get(
                                "growth_3y_opt"
                            ),
                        ],
                    ],
                    columns=[
                        "السيناريو",
                        "الهدف",
                        "النمو المفترض"
                    ]
                )

                scenario_table["الهدف"] = (
                    scenario_table["الهدف"]
                    .apply(money)
                )

                scenario_table[
                    "النمو المفترض"
                ] = (
                    scenario_table[
                        "النمو المفترض"
                    ].apply(pct)
                )

                st.dataframe(
                    scenario_table,
                    hide_index=True,
                    use_container_width=True
                )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "توزيعات 3Y / سهم",
                    money(
                        r.get(
                            "dividend_3y_per_share"
                        )
                    )
                )

                c2.metric(
                    "إجمالي عائد 3Y",
                    pct(
                        r.get(
                            "total_return_3y_base"
                        )
                    )
                )

                c3.metric(
                    "CAGR 3Y",
                    pct(
                        r.get(
                            "return_3y_base"
                        )
                    )
                )

                # ------------------------------------------------
                # Financials
                # ------------------------------------------------

                st.subheader(
                    "📊 المؤشرات المالية"
                )

                st.dataframe(
                    financial_summary_table(r),
                    hide_index=True,
                    use_container_width=True
                )

                # ------------------------------------------------
                # Valuation
                # ------------------------------------------------

                st.subheader(
                    "🧮 التقييم متعدد النماذج"
                )

                valuation_table = pd.DataFrame(
                    [
                        [
                            "DCF",
                            r.get(
                                "dcf_value"
                            ),
                            r.get(
                                "dcf_type"
                            )
                        ],
                        [
                            "Residual Income",
                            r.get(
                                "residual_value"
                            ),
                            "للبنوك أساسًا"
                        ],
                        [
                            "P/E",
                            r.get(
                                "pe_value"
                            ),
                            "مضاعف قطاعي"
                        ],
                        [
                            "P/B",
                            r.get(
                                "pb_value"
                            ),
                            "مضاعف قطاعي"
                        ],
                        [
                            "EV/EBITDA",
                            r.get(
                                "ev_ebitda_value"
                            ),
                            "Enterprise Value"
                        ],
                        [
                            "FCF Yield",
                            r.get(
                                "fcf_yield_value"
                            ),
                            "FCF / Ke"
                        ],
                        [
                            "Fair Value",
                            r.get(
                                "fair_value"
                            ),
                            "النتيجة المجمعة"
                        ],
                    ],
                    columns=[
                        "النموذج",
                        "القيمة",
                        "ملاحظة"
                    ]
                )

                valuation_table["القيمة"] = (
                    valuation_table[
                        "القيمة"
                    ].apply(money)
                )

                st.dataframe(
                    valuation_table,
                    hide_index=True,
                    use_container_width=True
                )

                # ------------------------------------------------
                # Model confidence
                # ------------------------------------------------

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "عدد نماذج التقييم",
                    r.get(
                        "model_count",
                        0
                    )
                )

                c2.metric(
                    "اتفاق النماذج",
                    (
                        f"{r.get('model_agreement', 0):.1f}%"
                        if finite(
                            r.get(
                                "model_agreement"
                            )
                        )
                        else "—"
                    )
                )

                c
تم اقتطاع المعاينة لأن الملف كبير
