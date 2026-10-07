import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import re
import io
import math
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# EGX FINANCIAL INTELLIGENCE PRO
# Financial-only: fundamentals + valuation + 3Y scenarios
# No RSI / MACD / technical indicators.
# ============================================================

st.set_page_config(page_title="EGX Financial Intelligence PRO", page_icon="💰", layout="wide")

st.markdown("""
<style>
html, body, [class*="css"] { direction: rtl; }
.block-container { max-width: 1500px; padding-top: 1.2rem; }
[data-testid="stDataFrame"] { direction: rtl; }
.metric-card { padding: 12px; border-radius: 12px; border: 1px solid rgba(128,128,128,.25); }
.small {font-size: .85rem; opacity: .78;}
</style>
""", unsafe_allow_html=True)

APP_VERSION = "2.0 PRO"
YAHOO_SUFFIX = ".CA"

# Best-effort fallback universe. The app also attempts dynamic discovery and
# allows a custom CSV/text universe. Never treats this list as guaranteed exhaustive.
DEFAULT_EGX_SYMBOLS = [
"ABUK","ACGC","ADIB","AIVC","ALCN","AMER","ARCC","ARPI","ASCM","ATLC","AUTO","AXPH","BINV","BIOC","BTFH","CICH","CIRA","COMI","COPR","COSG","CPCI","CRST","CSAG","DAPH","DOMT","EAST","ECAP","EFID","EFIH","EGAS","EGCH","EGTS","ELEC","EMFD","ENGC","ETEL","ETRS","FAIT","FWRY","GDWA","GBCO","HELI","HDBK","HRHO","ICFC","IDHC","IEEC","IFAP","IRON","ISMA","JUFO","KABO","KZPC","LCID","MICH","MCQE","MFPC","MISR","MNHD","MOIL","MPBS","MPCO","MPCI","MTIE","NAHO","NCCW","NIPH","NILE","OCIC","OCPH","ODIN","ORAS","ORHD","ORWE","PHAR","PHDC","PRCL","PRMH","QNBE","RACC","RAYA","RDFI","REAC","RMDA","ROTO","SAUD","SCEM","SDTI","SKPC","SMFR","SPIN","SPMD","SUGR","SWDY","TALM","TMGH","TORA","UASG","UNIT","UNIP","UPMS","VERT","WCDF","WEAS","WKOL","ZECO","ZAHI",
"AFMC","ARAB","CCRS","CLHO","CNFN","EASB","ELSH","EXPA","FARE","GEMA","GSSC","HITP","IDBE","INFI","LEDA","MENA","MEPA","NEDA","OBOU","PHTV","PION","RACC","RREI","SAIB","SIPC","TAQA","TATW","TRTO","UNBE","VTMN","WADI","ZMID",
"BICC","BODA","BTMN","CIRA","CIEB","CITI","EGBE","EXPA","FAIT","HDBK","MOBG","NBEG","QNBA","SAUD","UBEE",
"AMIA","APPC","ARAB","CERA","CIEB","EGAS","ELSH","GISS","GOLD","ICID","ISPH","MAAL","MEPA","NCCW","NCCW","ORWE","PACH","PRDC","SCTS","SCFM","SIPC","SPIN","TAQA","TMMT","TOWN","UASG"
]

SECTOR_MAP = {
    # Banks
    "COMI":"بنوك","CIEB":"بنوك","ADIB":"بنوك","HDBK":"بنوك","QNBE":"بنوك","SAIB":"بنوك","FAIT":"بنوك","EGBE":"بنوك","UBEE":"بنوك","EXPA":"بنوك","CICH":"بنوك","BTFH":"بنوك",
    # Real estate
    "TMGH":"عقارات","HELI":"عقارات","ORAS":"إنشاءات/عقارات","ORHD":"عقارات","MNHD":"عقارات","PHDC":"عقارات","TALA":"عقارات","EMFD":"عقارات","MENA":"عقارات","ARAB":"عقارات",
    # Healthcare/pharma
    "DAPH":"رعاية صحية/دواء","NIPH":"رعاية صحية/دواء","PHAR":"رعاية صحية/دواء","AXPH":"رعاية صحية/دواء","IDHC":"رعاية صحية/دواء","RMDA":"رعاية صحية/دواء","MPCI":"رعاية صحية/دواء",
    # Fertilizers/chemicals
    "MFPC":"كيماويات/أسمدة","ABUK":"كيماويات/أسمدة","SKPC":"كيماويات/أسمدة","KZPC":"كيماويات/أسمدة","MICH":"كيماويات/أسمدة","EGCH":"كيماويات/أسمدة",
    # Food
    "DOMT":"أغذية","JUFO":"أغذية","EAST":"أغذية/تبغ","EFID":"أغذية","UASG":"أغذية","NCCW":"أغذية",
    # Telecom/IT/payments
    "ETEL":"اتصالات","FWRY":"مدفوعات/تكنولوجيا","RAYA":"تكنولوجيا/خدمات","EFIH":"مدفوعات/تكنولوجيا","MTIE":"تكنولوجيا/توزيع","CIRA":"تعليم/خدمات",
    # Financial services
    "HRHO":"خدمات مالية","EFG":"خدمات مالية","BINV":"خدمات مالية","MOBG":"خدمات مالية","NILE":"خدمات مالية","INFI":"خدمات مالية",
    # Energy/oil/gas
    "EGAS":"طاقة/غاز","TAQA":"طاقة","MOIL":"طاقة","GASCO":"طاقة","AMOC":"طاقة",
    # Industrial/materials
    "ESRS":"معادن/حديد","IRCC":"معادن/حديد","IRON":"معادن/حديد","SWDY":"صناعة/كابلات","ARCC":"مواد بناء","TORA":"مواد بناء","SPMD":"مواد بناء","SCEM":"مواد بناء","WCDF":"مواد بناء",
    # Consumer/retail
    "AMER":"خدمات استهلاكية","AUTO":"سيارات","MPCO":"استهلاكي","LCID":"استهلاكي","ORWE":"منسوجات","BTFH":"خدمات مالية",
}

SECTOR_DEFAULTS = {
    "بنوك": {"ke":0.20,"pe":9.0,"pb":1.0,"growth_cap":0.15,"terminal":0.05},
    "عقارات": {"ke":0.19,"pe":10.0,"pb":0.85,"growth_cap":0.15,"terminal":0.04},
    "رعاية صحية/دواء": {"ke":0.20,"pe":13.0,"pb":1.25,"growth_cap":0.18,"terminal":0.05},
    "كيماويات/أسمدة": {"ke":0.20,"pe":8.5,"pb":1.0,"growth_cap":0.12,"terminal":0.04},
    "أغذية": {"ke":0.19,"pe":11.0,"pb":1.15,"growth_cap":0.12,"terminal":0.04},
    "اتصالات": {"ke":0.18,"pe":10.0,"pb":1.1,"growth_cap":0.10,"terminal":0.04},
    "مدفوعات/تكنولوجيا": {"ke":0.21,"pe":18.0,"pb":2.0,"growth_cap":0.25,"terminal":0.06},
    "تكنولوجيا/خدمات": {"ke":0.21,"pe":16.0,"pb":1.8,"growth_cap":0.22,"terminal":0.06},
    "خدمات مالية": {"ke":0.20,"pe":11.0,"pb":1.2,"growth_cap":0.16,"terminal":0.05},
    "طاقة/غاز": {"ke":0.19,"pe":8.0,"pb":1.0,"growth_cap":0.10,"terminal":0.04},
    "طاقة": {"ke":0.19,"pe":8.0,"pb":1.0,"growth_cap":0.10,"terminal":0.04},
    "معادن/حديد": {"ke":0.21,"pe":8.0,"pb":0.9,"growth_cap":0.10,"terminal":0.04},
    "مواد بناء": {"ke":0.20,"pe":9.0,"pb":1.0,"growth_cap":0.10,"terminal":0.04},
    "صناعة/كابلات": {"ke":0.20,"pe":11.0,"pb":1.15,"growth_cap":0.12,"terminal":0.04},
    "استهلاكي": {"ke":0.20,"pe":11.0,"pb":1.1,"growth_cap":0.12,"terminal":0.04},
    "سيارات": {"ke":0.21,"pe":10.0,"pb":1.0,"growth_cap":0.10,"terminal":0.04},
    "منسوجات": {"ke":0.21,"pe":9.0,"pb":0.9,"growth_cap":0.10,"terminal":0.04},
    "تعليم/خدمات": {"ke":0.20,"pe":14.0,"pb":1.4,"growth_cap":0.18,"terminal":0.05},
    "عام": {"ke":0.21,"pe":10.0,"pb":1.0,"growth_cap":0.12,"terminal":0.04},
}

LINE_ALIASES = {
    "revenue":["Total Revenue","Operating Revenue","Revenue"],
    "net_income":["Net Income","Net Income Common Stockholders","Net Income Including Noncontrolling Interests"],
    "pretax":["Pretax Income"],
    "ebit":["EBIT","Operating Income"],
    "ebitda":["EBITDA","Normalized EBITDA"],
    "eps":["Diluted EPS","Basic EPS","Diluted EPS from Continuing Operations"],
    "equity":["Stockholders Equity","Common Stock Equity","Total Equity Gross Minority Interest","Total Equity"],
    "assets":["Total Assets"],
    "debt":["Total Debt","Long Term Debt And Capital Lease Obligation","Long Term Debt"],
    "cash":["Cash Cash Equivalents And Short Term Investments","Cash And Cash Equivalents","Cash Financial"],
    "ocf":["Operating Cash Flow","Total Cash From Operating Activities"],
    "capex":["Capital Expenditure","Capital Expenditures"],
    "dividends":["Cash Dividends Paid","Common Stock Dividend Paid"],
    "interest":["Interest Expense Non Operating","Interest Expense"],
    "current_assets":["Current Assets"],
    "current_liabilities":["Current Liabilities"],
    "shares":["Ordinary Shares Number","Share Issued"],
}


def clean_symbol(s):
    s = str(s).upper().strip().replace(".CA","")
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s


def yahoo_symbol(s):
    return clean_symbol(s) + YAHOO_SUFFIX


def finite(x):
    try:
        return x is not None and np.isfinite(float(x))
    except Exception:
        return False


def safe_float(x, default=np.nan):
    try:
        v=float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def clip(x, lo, hi):
    if not finite(x): return np.nan
    return float(np.clip(x, lo, hi))


def median_valid(vals, default=np.nan):
    a=[float(v) for v in vals if finite(v)]
    return float(np.median(a)) if a else default


def get_sector(sym, yf_sector=None):
    if sym in SECTOR_MAP: return SECTOR_MAP[sym]
    s=(yf_sector or "").lower()
    if any(k in s for k in ["bank"]): return "بنوك"
    if any(k in s for k in ["real estate","reit"]): return "عقارات"
    if any(k in s for k in ["health","drug","biotech"]): return "رعاية صحية/دواء"
    if any(k in s for k in ["technology","software","information"]): return "تكنولوجيا/خدمات"
    if any(k in s for k in ["financial"]): return "خدمات مالية"
    if any(k in s for k in ["energy","oil","gas"]): return "طاقة"
    if any(k in s for k in ["communication","telecom"]): return "اتصالات"
    if any(k in s for k in ["consumer","food"]): return "أغذية"
    if any(k in s for k in ["chemical"]): return "كيماويات/أسمدة"
    return "عام"


@st.cache_data(ttl=3600, show_spinner=False)
def discover_universe():
    urls=[
        "https://stockanalysis.com/list/egyptian-stock-exchange/",
        "https://stockanalysis.com/stocks/egx/",
    ]
    found=[]
    headers={"User-Agent":"Mozilla/5.0"}
    for url in urls:
        try:
            r=requests.get(url,headers=headers,timeout=12)
            if r.ok:
                text=r.text.upper()
                # Capture common EGX ticker patterns, then validate later through Yahoo.
                candidates=re.findall(r'\b[A-Z]{3,5}\.CA\b',text)
                found.extend([c.replace('.CA','') for c in candidates])
        except Exception:
            pass
    merged=list(dict.fromkeys(found + [clean_symbol(x) for x in DEFAULT_EGX_SYMBOLS]))
    return merged


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_symbol_bundle(sym):
    ys=yahoo_symbol(sym)
    out={"symbol":sym,"yf":ys,"ok":False,"error":""}
    try:
        t=yf.Ticker(ys)
        info={}
        try: info=t.info or {}
        except Exception: info={}
        income=t.income_stmt
        balance=t.balance_sheet
        cash=t.cashflow
        # yfinance documents these financial statement properties.
        out.update({"info":info,"income":income,"balance":balance,"cashflow":cash})
        hist=t.history(period="5d", interval="1d", auto_adjust=False, actions=True)
        out["history"]=hist
        if hist is not None and not hist.empty:
            out["price"]=safe_float(hist["Close"].dropna().iloc[-1])
            out["price_date"]=str(hist.index[-1].date())
        else:
            out["price"]=safe_float(info.get("currentPrice", info.get("regularMarketPrice")))
            out["price_date"]=""
        out["ok"]=finite(out.get("price")) or (income is not None and not income.empty)
        return out
    except Exception as e:
        out["error"]=str(e)[:220]
        return out


def series_from_statement(df, aliases):
    if df is None or not isinstance(df,pd.DataFrame) or df.empty:
        return pd.Series(dtype=float)
    for name in aliases:
        if name in df.index:
            s=pd.to_numeric(df.loc[name],errors="coerce").dropna()
            if not s.empty: return s.sort_index()
    # normalized fallback
    norm={re.sub(r"[^a-z0-9]","",str(i).lower()):i for i in df.index}
    for name in aliases:
        key=re.sub(r"[^a-z0-9]","",name.lower())
        if key in norm:
            s=pd.to_numeric(df.loc[norm[key]],errors="coerce").dropna()
            if not s.empty: return s.sort_index()
    return pd.Series(dtype=float)


def latest_series_value(df, key):
    s=series_from_statement(df,LINE_ALIASES[key])
    return safe_float(s.iloc[-1]) if not s.empty else np.nan


def growth_cagr(s, years=3):
    if s is None or len(s)<2: return np.nan
    s=s.sort_index()
    recent=s.iloc[-1]; n=min(years,len(s)-1); old=s.iloc[-1-n]
    if not finite(recent) or not finite(old) or old<=0 or recent<=0: return np.nan
    return (recent/old)**(1/n)-1


def trend_slope_pct(s):
    if s is None or len(s)<2: return np.nan
    vals=pd.to_numeric(s,errors="coerce").dropna().values
    if len(vals)<2: return np.nan
    x=np.arange(len(vals),dtype=float)
    slope=np.polyfit(x,vals,1)[0]
    base=np.mean(np.abs(vals))
    return slope/base if base>0 else np.nan


def dcf_value_per_share(fcf, shares, ke, growth, terminal, years=5):
    if not finite(fcf) or not finite(shares) or fcf<=0 or shares<=0: return np.nan
    if not finite(ke) or not finite(growth) or not finite(terminal): return np.nan
    growth=float(np.clip(growth,-0.05,0.20)); terminal=float(np.clip(terminal,0.02,min(0.05,ke-0.01)))
    pv=0.0
    for y in range(1,years+1):
        cf=fcf*((1+growth)**y)
        pv += cf/((1+ke)**y)
    terminal_value=(fcf*((1+growth)**years)*(1+terminal))/(ke-terminal)
    pv += terminal_value/((1+ke)**years)
    return pv/shares


def residual_income_value(bvps, eps, roe, ke, growth, years=5):
    if not all(finite(x) for x in [bvps,eps,roe,ke,growth]): return np.nan
    if bvps<=0 or ke<=0: return np.nan
    growth=float(np.clip(growth,-0.03,0.18))
    value=bvps
    book=bvps
    for y in range(1,years+1):
        book=book*(1+growth)
        expected_earn=book*ke
        actual_earn=eps*((1+growth)**y)
        ri=actual_earn-expected_earn
        value += ri/((1+ke)**y)
    term_ri=(eps*((1+growth)**years)*(1+growth)-book*ke)
    terminal=term_ri/(ke-max(0.02,min(growth,ke-0.01)))
    value += terminal/((1+ke)**years)
    return max(value,0)


def normalize_growth(g, fallback):
    return float(np.clip(g if finite(g) else fallback, -0.05, 0.20))


def get_fair_values(r):
    sec=r["sector"]; cfg=SECTOR_DEFAULTS.get(sec,SECTOR_DEFAULTS["عام"])
    price=r["price"]; shares=r["shares"]; fcf=r["fcf"]; eps=r["eps"]; bvps=r["bvps"]
    ke=cfg["ke"]
    g=normalize_growth(r["earnings_growth_3y"], 0.08)
    g=min(g,cfg["growth_cap"])
    dcf=np.nan
    if sec not in ["بنوك"]:
        dcf=dcf_value_per_share(fcf,shares,ke,g,cfg["terminal"])
    ri=np.nan
    if sec=="بنوك" and finite(bvps) and finite(eps):
        ri=residual_income_value(bvps,eps,r["roe"],ke,g)
    pe=np.nan
    if finite(eps) and eps>0: pe=eps*cfg["pe"]
    pb=np.nan
    if finite(bvps) and bvps>0: pb=bvps*cfg["pb"]
    ps=np.nan
    if finite(r["revenue_per_share"]) and r["revenue_per_share"]>0:
        ps=r["revenue_per_share"]*(1.2 if sec in ["مدفوعات/تكنولوجيا","تكنولوجيا/خدمات"] else 0.8)
    vals=[]; weights=[]
    if finite(dcf): vals.append(dcf); weights.append(0.45 if sec not in ["عقارات","خدمات مالية"] else 0.35)
    if finite(ri): vals.append(ri); weights.append(0.45)
    if finite(pe): vals.append(pe); weights.append(0.25 if sec=="بنوك" else 0.30)
    if finite(pb): vals.append(pb); weights.append(0.25 if sec=="بنوك" else 0.15)
    if finite(ps) and not finite(dcf): vals.append(ps); weights.append(0.25)
    if not vals: return np.nan,np.nan,np.nan,np.nan,np.nan
    weights=np.array(weights,float); weights/=weights.sum()
    fair=float(np.sum(np.array(vals)*weights))
    return fair,dcf,ri,pe,pb


def build_record(bundle):
    sym=bundle["symbol"]; info=bundle.get("info",{}) or {}; inc=bundle.get("income"); bal=bundle.get("balance"); cf=bundle.get("cashflow")
    price=safe_float(bundle.get("price"))
    yf_sector=info.get("sector","")
    sector=get_sector(sym,yf_sector)
    revs=series_from_statement(inc,LINE_ALIASES["revenue"])
    nis=series_from_statement(inc,LINE_ALIASES["net_income"])
    epss=series_from_statement(inc,LINE_ALIASES["eps"])
    eqs=series_from_statement(bal,LINE_ALIASES["equity"])
    assets=series_from_statement(bal,LINE_ALIASES["assets"])
    debts=series_from_statement(bal,LINE_ALIASES["debt"])
    cashs=series_from_statement(bal,LINE_ALIASES["cash"])
    ocfs=series_from_statement(cf,LINE_ALIASES["ocf"])
    caps=series_from_statement(cf,LINE_ALIASES["capex"])
    ints=series_from_statement(inc,LINE_ALIASES["interest"])
    cur_a=series_from_statement(bal,LINE_ALIASES["current_assets"])
    cur_l=series_from_statement(bal,LINE_ALIASES["current_liabilities"])
    shares=safe_float(info.get("sharesOutstanding"))
    if not finite(shares):
        shares=latest_series_value(bal,"shares")
    if not finite(shares) and finite(price):
        mcap=safe_float(info.get("marketCap")); shares=mcap/price if finite(mcap) and price>0 else np.nan
    revenue=safe_float(revs.iloc[-1]) if not revs.empty else safe_float(info.get("totalRevenue"))
    ni=safe_float(nis.iloc[-1]) if not nis.empty else safe_float(info.get("netIncomeToCommon"))
    eps=safe_float(epss.iloc[-1]) if not epss.empty else safe_float(info.get("trailingEps"))
    equity=safe_float(eqs.iloc[-1])
    debt=safe_float(debts.iloc[-1]) if not debts.empty else safe_float(info.get("totalDebt"))
    cash=safe_float(cashs.iloc[-1]) if not cashs.empty else safe_float(info.get("totalCash"))
    ocf=safe_float(ocfs.iloc[-1])
    capex=safe_float(caps.iloc[-1])
    fcf=ocf+capex if finite(ocf) and finite(capex) else np.nan
    assets_v=safe_float(assets.iloc[-1])
    interest=safe_float(ints.iloc[-1])
    ca=safe_float(cur_a.iloc[-1]); cl=safe_float(cur_l.iloc[-1])
    bvps=equity/shares if finite(equity) and finite(shares) and shares>0 else safe_float(info.get("bookValue"))
    revps=revenue/shares if finite(revenue) and finite(shares) and shares>0 else np.nan
    roe=ni/equity if finite(ni) and finite(equity) and equity!=0 else safe_float(info.get("returnOnEquity"))
    roa=ni/assets_v if finite(ni) and finite(assets_v) and assets_v!=0 else safe_float(info.get("returnOnAssets"))
    net_margin=ni/revenue if finite(ni) and finite(revenue) and revenue!=0 else safe_float(info.get("profitMargins"))
    debt_equity=debt/equity if finite(debt) and finite(equity) and equity!=0 else safe_float(info.get("debtToEquity"))
    current_ratio=ca/cl if finite(ca) and finite(cl) and cl!=0 else safe_float(info.get("currentRatio"))
    ocf_ni=ocf/ni if finite(ocf) and finite(ni) and ni!=0 else np.nan
    interest_cov=(safe_float(series_from_statement(inc,LINE_ALIASES["ebit"]).iloc[-1])/abs(interest)) if not series_from_statement(inc,LINE_ALIASES["ebit"]).empty and finite(interest) and interest!=0 else np.nan
    rev_growth=growth_cagr(revs,3); ni_growth=growth_cagr(nis,3); eps_growth=growth_cagr(epss,3)
    roe_trend=trend_slope_pct(eqs)
    dividend_yield=safe_float(info.get("dividendYield"))
    if finite(dividend_yield) and dividend_yield>1: dividend_yield/=100
    payout=safe_float(info.get("payoutRatio"));
    if finite(payout) and payout>1: payout/=100
    fair_placeholder={}
    r={"symbol":sym,"name":info.get("longName",info.get("shortName",sym)),"sector":sector,"price":price,"price_date":bundle.get("price_date",""),
       "shares":shares,"revenue":revenue,"net_income":ni,"eps":eps,"equity":equity,"assets":assets_v,"debt":debt,"cash":cash,"ocf":ocf,"capex":capex,"fcf":fcf,
       "bvps":bvps,"revenue_per_share":revps,"roe":roe,"roa":roa,"net_margin":net_margin,"debt_equity":debt_equity,"current_ratio":current_ratio,"ocf_ni":ocf_ni,"interest_coverage":interest_cov,
       "revenue_growth_3y":rev_growth,"earnings_growth_3y":ni_growth if finite(ni_growth) else eps_growth,"eps_growth_3y":eps_growth,"roe_trend":roe_trend,
       "dividend_yield":dividend_yield,"payout":payout,"market_cap":safe_float(info.get("marketCap")),"currency":info.get("currency","EGP"),"data_error":bundle.get("error","")}
    fair,dcf,ri,pe,pb=get_fair_values(r)
    r.update({"fair_value":fair,"dcf_value":dcf,"residual_value":ri,"pe_value":pe,"pb_value":pb})
    if finite(fair) and fair>0 and finite(price):
        r["upside"]=(fair/price)-1
        r["buy_excellent"]=fair*0.70; r["buy_strong"]=fair*0.80; r["buy_acceptable"]=fair*0.90
    else:
        r["upside"]=np.nan; r["buy_excellent"]=np.nan; r["buy_strong"]=np.nan; r["buy_acceptable"]=np.nan
    r.update(three_year_targets(r))
    r["coverage"]=data_coverage(r)
    r["score"]=investment_score(r)
    r["rating"]=rating_label(r["score"])
    return r


def data_coverage(r):
    keys=["price","revenue","net_income","eps","equity","debt","cash","ocf","fcf","shares","bvps","roe","revenue_growth_3y","earnings_growth_3y","fair_value"]
    return round(100*sum(finite(r.get(k)) for k in keys)/len(keys),1)


def three_year_targets(r):
    price=r["price"]; eps=r["eps"]; bvps=r["bvps"]; sec=r["sector"]; cfg=SECTOR_DEFAULTS.get(sec,SECTOR_DEFAULTS["عام"])
    g=normalize_growth(r["earnings_growth_3y"],0.08)
    g=min(g+0.01,cfg["growth_cap"]); g_cons=max(-0.03,min(g-0.04,cfg["growth_cap"])); g_opt=min(g+0.05,cfg["growth_cap"]+0.05)
    def target(e, multiple): return e*((1+g)**3)*multiple if finite(e) and e>0 else np.nan
    base_mult=cfg["pe"]
    cons=target(eps,g_cons*0+base_mult*0.85) if finite(eps) else np.nan
    base=target(eps,base_mult) if finite(eps) else np.nan
    opt=target(eps,base_mult*1.15) if finite(eps) else np.nan
    # For banks / low EPS, book-value path is a robust fallback.
    if sec=="بنوك" or not finite(eps) or eps<=0:
        def btarget(mult,growth): return bvps*((1+growth)**3)*mult if finite(bvps) and bvps>0 else np.nan
        cons=btarget(cfg["pb"]*0.85,g_cons); base=btarget(cfg["pb"],g); opt=btarget(cfg["pb"]*1.15,g_opt)
    if finite(r["fair_value"]):
        base=median_valid([base,r["fair_value"]*((1+g)**3)],base)
        cons=median_valid([cons,r["fair_value"]*((1+g_cons)**3)],cons)
        opt=median_valid([opt,r["fair_value"]*((1+g_opt)**3)],opt)
    dy=r.get("dividend_yield",np.nan)
    div3=3*dy if finite(dy) and dy>=0 else 0
    return {"target_3y_cons":cons,"target_3y_base":base,"target_3y_opt":opt,"dividend_yield_annual":dy,"dividend_3y_yield":div3,
            "return_3y_base":((base/price)**(1/3)-1 if finite(base) and finite(price) and base>0 and price>0 else np.nan)}


def investment_score(r):
    # Sector-aware 100-point score. Missing values lower confidence, never fabricate.
    s=0
    # Valuation 25
    up=r.get("upside")
    val=50 if not finite(up) else np.clip(50+up*100,0,100)
    s+=0.25*val
    # Growth 20
    g=r.get("earnings_growth_3y")
    growth=50 if not finite(g) else np.clip(50+g*250,0,100)
    s+=0.20*growth
    # Profitability 20
    roe=r.get("roe"); margin=r.get("net_margin")
    p1=50 if not finite(roe) else np.clip(50+roe*180,0,100)
    p2=50 if not finite(margin) else np.clip(50+margin*250,0,100)
    s+=0.20*(0.65*p1+0.35*p2)
    # Balance sheet 15; banks get less penalty from leverage.
    de=r.get("debt_equity"); cr=r.get("current_ratio")
    if r["sector"]=="بنوك": bs=65 if finite(r.get("roe")) and r.get("roe",0)>0.12 else 50
    else:
        dscore=50 if not finite(de) else np.clip(100-de*40,0,100)
        cscore=50 if not finite(cr) else np.clip(cr/2*100,0,100)
        bs=.65*dscore+.35*cscore
    s+=0.15*bs
    # Cash flow / earnings quality 10
    q=50
    if finite(r.get("ocf_ni")): q=np.clip(50+(r["ocf_ni"]-1)*50,0,100)
    if finite(r.get("fcf")) and r["fcf"]>0: q=min(100,q+15)
    s+=0.10*q
    # Dividend 5
    dy=r.get("dividend_yield"); div=50 if not finite(dy) else np.clip(dy*1000,0,100)
    s+=0.05*div
    # Data quality 5
    s+=0.05*r.get("coverage",50)
    return round(float(np.clip(s,0,100)),2)


def rating_label(score):
    if not finite(score): return "غير متاح"
    if score>=85:return "ممتاز"
    if score>=75:return "قوي"
    if score>=65:return "جيد"
    if score>=55:return "متوسط"
    return "ضعيف"


def analyze_symbol(sym):
    try:
        b=fetch_symbol_bundle(sym)
        if not b.get("ok"): return None
        return build_record(b)
    except Exception as e:
        return {"symbol":sym,"name":sym,"sector":SECTOR_MAP.get(sym,"عام"),"price":np.nan,"score":np.nan,"coverage":0,"data_error":str(e)[:220]}


def scan(symbols, workers=5):
    rows=[]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures={ex.submit(analyze_symbol,s):s for s in symbols}
        for f in as_completed(futures):
            try:
                r=f.result()
                if r is not None: rows.append(r)
            except Exception:
                pass
    df=pd.DataFrame(rows)
    if df.empty:return df
    return df.sort_values(["score","coverage"],ascending=False,na_position="last").reset_index(drop=True)


def fmt_money(x): return "—" if not finite(x) else f"{x:,.2f}"
def fmt_pct(x): return "—" if not finite(x) else f"{x*100:.1f}%"

@st.cache_data(show_spinner=False)
def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")


def main():
    st.title("💰 EGX Financial Intelligence PRO")
    st.caption(f"المحرك المالي فقط — Fundamental + Valuation + 3Y Scenarios | إصدار {APP_VERSION}")
    st.info("المحرك لا يستخدم RSI أو MACD أو أي مؤشرات فنية. التقييم يعتمد على القوائم المالية، التدفقات النقدية، الربحية، المديونية، النمو، والتقييم القطاعي.")

    with st.sidebar:
        st.header("⚙️ إعدادات المحرك")
        workers=st.slider("عدد العمليات المتوازية",2,8,5)
        min_cov=st.slider("أقل تغطية بيانات",30,100,55)
        top_n=st.slider("عدد أفضل الأسهم",10,50,20)
        universe_mode=st.radio("مصدر الكون",["اكتشاف + القائمة الاحتياطية","القائمة الاحتياطية فقط","رموز مخصصة"])
        custom=""
        if universe_mode=="رموز مخصصة":
            custom=st.text_area("اكتب الرموز مفصولة بفاصلة أو سطر",value="COMI,DAPH,MFPC,MICH,HELI,FWRY")
        if st.button("🧹 مسح الكاش وإعادة التحميل",use_container_width=True):
            st.cache_data.clear(); st.rerun()
        st.markdown("---")
        st.markdown("**هوامش الأمان الافتراضية:** 10% / 20% / 30%")
        st.markdown("**التقييم:** 100 نقطة مع عقوبة واضحة لجودة البيانات.")

    if universe_mode=="اكتشاف + القائمة الاحتياطية":
        symbols=discover_universe()
        source_note="اكتشاف عام + قائمة احتياطية"
    elif universe_mode=="القائمة الاحتياطية فقط":
        symbols=list(dict.fromkeys(DEFAULT_EGX_SYMBOLS)); source_note="قائمة احتياطية مدمجة"
    else:
        symbols=list(dict.fromkeys([clean_symbol(x) for x in re.split(r"[,\s]+",custom) if clean_symbol(x)])); source_note="رموز مخصصة"

    st.write(f"**الكون الحالي:** {len(symbols)} رمز — {source_note}")
    tabs=st.tabs(["📊 السوق بالكامل","🏭 القطاعات","🔎 سهم واحد","🧾 البيانات والمنهجية"])

    with tabs[0]:
        if st.button("🚀 ابدأ المسح المالي",type="primary",use_container_width=True):
            with st.spinner(f"جاري تحليل {len(symbols)} سهم ماليًا..."):
                df=scan(symbols,workers)
            st.session_state["egx_fin_df"]=df
        df=st.session_state.get("egx_fin_df",pd.DataFrame())
        if df.empty:
            st.warning("اضغط «ابدأ المسح المالي» لبدء التحليل.")
        else:
            view=df[df["coverage"]>=min_cov].copy()
            st.success(f"تم تحليل {len(df)} سهم — المعروض بعد حد الجودة: {len(view)}")
            c1,c2,c3,c4=st.columns(4)
            c1.metric("عدد الأسهم",len(view)); c2.metric("متوسط الدرجة",f"{view['score'].mean():.1f}" if not view.empty else "—")
            c3.metric("أفضل سهم",view.iloc[0]["symbol"] if not view.empty else "—"); c4.metric("متوسط الارتفاع المحتمل",fmt_pct(view["upside"].median()) if not view.empty else "—")
            cols={"الترتيب":"rank","الرمز":"symbol","اسم الشركة":"name","القطاع":"sector","السعر":"price","القيمة العادلة":"fair_value","شراء ممتاز":"buy_excellent","شراء قوي":"buy_strong","شراء مقبول":"buy_acceptable","هدف 3 سنوات محافظ":"target_3y_cons","هدف 3 سنوات أساسي":"target_3y_base","هدف 3 سنوات متفائل":"target_3y_opt","النمو 3 سنوات":"earnings_growth_3y","ROE":"roe","هامش الربح":"net_margin","الدين/حقوق الملكية":"debt_equity","FCF":"fcf","عائد التوزيع":"dividend_yield","الارتفاع المحتمل":"upside","جودة البيانات":"coverage","الدرجة":"score","التقييم":"rating"}
            view=view.copy(); view["rank"]=np.arange(1,len(view)+1)
            shown=view[list(cols.values())].rename(columns={v:k for k,v in cols.items()})
            st.dataframe(shown,hide_index=True,use_container_width=True,column_config={
                "السعر":st.column_config.NumberColumn(format="%.2f"),"القيمة العادلة":st.column_config.NumberColumn(format="%.2f"),
                "شراء ممتاز":st.column_config.NumberColumn(format="%.2f"),"شراء قوي":st.column_config.NumberColumn(format="%.2f"),"شراء مقبول":st.column_config.NumberColumn(format="%.2f"),
                "هدف 3 سنوات محافظ":st.column_config.NumberColumn(format="%.2f"),"هدف 3 سنوات أساسي":st.column_config.NumberColumn(format="%.2f"),"هدف 3 سنوات متفائل":st.column_config.NumberColumn(format="%.2f"),
                "النمو 3 سنوات":st.column_config.NumberColumn(format="0.0%"),"ROE":st.column_config.NumberColumn(format="0.0%"),"هامش الربح":st.column_config.NumberColumn(format="0.0%"),"عائد التوزيع":st.column_config.NumberColumn(format="0.0%"),"الارتفاع المحتمل":st.column_config.NumberColumn(format="0.0%"),"الدرجة":st.column_config.NumberColumn(format="0.00")})
            st.download_button("⬇️ تحميل الجدول CSV",csv_bytes(shown),"EGX_Financial_Ranking.csv","text/csv",use_container_width=True)

            st.subheader("🏆 أفضل الأسهم")
            top=view.head(top_n)
            st.dataframe(top[["symbol","name","sector","price","fair_value","buy_excellent","buy_strong","target_3y_base","upside","score","coverage"]].rename(columns={"symbol":"الرمز","name":"الشركة","sector":"القطاع","price":"السعر","fair_value":"القيمة العادلة","buy_excellent":"شراء ممتاز","buy_strong":"شراء قوي","target_3y_base":"هدف 3 سنوات","upside":"Upside","score":"الدرجة","coverage":"جودة البيانات"}),hide_index=True,use_container_width=True)

    with tabs[1]:
        df=st.session_state.get("egx_fin_df",pd.DataFrame())
        if df.empty: st.warning("ابدأ المسح أولًا.")
        else:
            s=df[df["coverage"]>=min_cov].groupby("sector").agg(عدد_الأسهم=("symbol","count"),متوسط_الدرجة=("score","mean"),وسيط_الارتفاع=("upside","median"),متوسط_جودة_البيانات=("coverage","mean")).reset_index().sort_values("متوسط_الدرجة",ascending=False)
            st.dataframe(s.rename(columns={"sector":"القطاع"}),hide_index=True,use_container_width=True)
            sec=st.selectbox("اختر قطاعًا",sorted(df["sector"].dropna().unique()))
            ss=df[(df["sector"]==sec)&(df["coverage"]>=min_cov)].sort_values("score",ascending=False).head(30)
            st.dataframe(ss[["symbol","name","price","fair_value","buy_strong","target_3y_base","upside","score","coverage"]].rename(columns={"symbol":"الرمز","name":"الشركة","price":"السعر","fair_value":"القيمة العادلة","buy_strong":"شراء قوي","target_3y_base":"هدف 3 سنوات","upside":"Upside","score":"الدرجة","coverage":"الجودة"}),hide_index=True,use_container_width=True)

    with tabs[2]:
        sym=st.text_input("رمز السهم",value="DAPH").upper().strip()
        if st.button("🔍 تحليل السهم",use_container_width=True):
            r=analyze_symbol(clean_symbol(sym))
            st.session_state["single_fin"]=r
        r=st.session_state.get("single_fin")
        if r:
            st.subheader(f"{r['symbol']} — {r['name']}")
            m=st.columns(5)
            m[0].metric("السعر",fmt_money(r["price"])); m[1].metric("القيمة العادلة",fmt_money(r["fair_value"])); m[2].metric("شراء قوي",fmt_money(r["buy_strong"])); m[3].metric("هدف 3 سنوات",fmt_money(r["target_3y_base"])); m[4].metric("الدرجة",f"{r['score']:.1f}/100" if finite(r["score"]) else "—")
            st.write(f"**القطاع:** {r['sector']} | **التقييم:** {r['rating']} | **تاريخ السعر:** {r['price_date']} | **جودة البيانات:** {r['coverage']}%")
            a,b,c=st.columns(3)
            a.metric("شراء ممتاز",fmt_money(r["buy_excellent"])); b.metric("شراء مقبول",fmt_money(r["buy_acceptable"])); c.metric("Upside",fmt_pct(r["upside"]))
            st.subheader("📌 السيناريوهات لـ 3 سنوات")
            scen=pd.DataFrame([{"السيناريو":"محافظ","الهدف":r["target_3y_cons"]},{"السيناريو":"أساسي","الهدف":r["target_3y_base"]},{"السيناريو":"متفائل","الهدف":r["target_3y_opt"]}])
            st.dataframe(scen,hide_index=True,use_container_width=True)
            st.subheader("📚 البيانات المالية الأساسية")
            fin=pd.DataFrame([
                ["الإيرادات",r["revenue"]],["صافي الربح",r["net_income"]],["EPS",r["eps"]],["حقوق الملكية",r["equity"]],["الدين",r["debt"]],["النقد",r["cash"]],["التدفق التشغيلي",r["ocf"]],["FCF",r["fcf"]],["BVPS",r["bvps"]],["ROE",r["roe"]],["ROA",r["roa"]],["هامش الربح",r["net_margin"]],["نمو الأرباح 3 سنوات",r["earnings_growth_3y"]],["عائد التوزيع",r["dividend_yield"]],["Payout",r["payout"]]
            ],columns=["المؤشر","القيمة"])
            st.dataframe(fin,hide_index=True,use_container_width=True)
            st.subheader("🧮 مكونات التقييم")
            val=pd.DataFrame([["DCF",r["dcf_value"]],["Residual Income",r["residual_value"]],["P/E",r["pe_value"]],["P/B",r["pb_value"]],["Fair Value",r["fair_value"]]],columns=["النموذج","القيمة"])
            st.dataframe(val,hide_index=True,use_container_width=True)
            st.caption("القيمة العادلة مزيج موزون من نماذج مناسبة للقطاع؛ البنوك تستخدم Residual Income وP/B/P/E بدل DCF التقليدي عندما يكون ذلك أنسب.")

    with tabs[3]:
        st.markdown("### ماذا يفعل المحرك؟")
        st.markdown("""
- يجمع السعر والقوائم المالية السنوية من Yahoo Finance عبر yfinance.
- يحسب الإيرادات والأرباح وEPS وحقوق الملكية والدين والنقد والتدفق التشغيلي وFCF وهوامش الربحية وROE/ROA ونسب المديونية.
- يحسب نموًا تاريخيًا تقريبيًا عبر CAGR، ثم يضع افتراضات نمو وحدودًا مرتبطة بالقطاع.
- يقدّر القيمة العادلة بعدة نماذج: DCF للشركات المناسبة، Residual Income للبنوك، وP/E وP/B كمقاربات نسبية.
- ينتج 3 سيناريوهات لثلاث سنوات، مع أسعار شراء بهامش أمان 10%/20%/30%.
- الدرجة من 100 تجمع التقييم والنمو والربحية والقوة المالية وجودة التدفقات والتوزيعات وجودة البيانات.
- لا يتم اختراع البيانات الناقصة؛ نقص البيانات يقلل التغطية والثقة.
        """)
        st.warning("المصادر المجانية قد تكون ناقصة أو متأخرة. لا تعتبر القيمة العادلة أو أهداف 3 سنوات ضمانًا للسعر المستقبلي، وراجع آخر إفصاحات EGX والشركة قبل قرار استثماري.")
        st.markdown("**ملاحظة تقنية:** التطبيق يستخدم `st.cache_data` لتقليل إعادة التحميل، ويدعم تنزيل النتائج CSV. توثيق Streamlit يوصي بـ`st.cache_data` للبيانات القابلة للتسلسل، و`st.download_button` لتنزيل DataFrame كـCSV.")

if __name__ == "__main__":
    main()
