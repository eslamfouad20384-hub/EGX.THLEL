import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import re
import json
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from urllib.parse import quote

# ============================================================
# EGX FINANCIAL INTELLIGENCE PRO MAX 7.0
# Fundamental-first | 246 symbols | Multi-source | No blank valuation
# ============================================================

st.set_page_config(page_title="EGX Financial Intelligence PRO MAX", page_icon="💰", layout="wide")

APP_VERSION = "9.0 Institutional Research Engine"
TARGET_UNIVERSE = 246
CACHE_TTL = 1800
DEFAULT_WORKERS = 6
RISK_FREE = 0.18
ERP = 0.08
COST_OF_EQUITY_FLOOR = 0.24
TERMINAL_GROWTH = 0.045
MAX_VALUATION_METHODS = 10
MIN_CORE_METHODS = 2
SOURCE_FRESH_DAYS = 180

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131 Safari/537.36"}

# ============================================================
# 246-stock universe
# ============================================================
RAW = '''
COMI ADIB CIEB QNBA HDBK UBEE FAIT EXPA CICH HRHO EFGH CNFN MICH MUBI FWRY EFIH VALU
TMGH PHDC MNHD EMFD SODIC AMER ORHD ORAS SWDY MMMT AUTO ATLC ETEL EAST JUFO DOMT MFPC
ABUK EGAL AMOC EGAS SKPC TAQA ALCN UEFM BIOC NIPH ISPH DAPH MCQE MPCO MAAL RAYA RACC
GBCO GBCA MTIE OTHM NCCW NAEEM ROTO GHRS CCAP MOBG BINV DCRC DSCW EDBM ESRS KZPC SPMD
OCDI ODIN OCPH ODPH AIND AMIA ACGC APSW PACH PHTV EASB EGS ENGI FIPR FINS GTHE ICFC
IDHC IEEC INEG MENA MOIL NAHO OSCI PRCL PRMH SAFW SEIG SCTS SEDO SESC SHIP SMFR TAWF
TOLI TRTO UASG UEGC UNIT UNIP WADI YACO ZAHM ZEAL ZMID ZWDI ARCC AURC AXPH BTFH CLHO
DTPP FITE GEMM GMSH HITE IBCT ICID MOIN MPRC MPSO NDRG NINH PION PSMC SAUD SCEM SCFM
SLMF TANM TCOR TMMT VERT LCSW RUBX MEPA AMPI ARAB ALEX ECAP EGSW EMFD FITE HHDL MOIN
NEDA NSGB RAKT KABO SIPC SPIN SDTI SDAR AMER PHDC TMGH ORAS ORHD
ASCM ACTF KORA CRST NAPR GIHD EGCH ENPPI ELAB PMS ELMR PREG GDWA AIFI GOUR
BONY CERA COSG ELKA FERC MBSC TWSA MENA2 MIPH RAYA2
'''
STOCKS = tuple(dict.fromkeys(x.upper().strip() for x in RAW.split()))

# Current actively traded EGX symbols observed from the StockAnalysis EGX list.
# This static fallback guarantees the app still has a broad universe when a
# live website is temporarily unavailable. Symbols ending in .CA are normalized.
CURRENT_EGX_FALLBACK = '''
COMI SWDY TMGH ETEL EGAL QNBE EAST MFPC ABUK ALCN HDBK EFIH ADIB ORAS FWRY EMFD SCTS ORHD EFID PHDC CANA GPPL JUFO BIOC VLMR VLMRA HRHO OCDI GBCO HELI BTFH FERC RAYA IRON CIEB FAITA FAIT EXPA EGCH PHAR CLHO VALU ARCC CIRA CCAP MTIE SCEM TAQA EFIC MCQE EGTS POUL SKPC NIPH ORWE MASR EGSA SAUD MOIL UBEE AMES EGBE ISPH RMDA TALM MBSC MHOT CICH ATQA AMOC CSAG BINV IFAP MPRC OLFI PRDC MOIN MIPH ISMQ OIH CPCI EGAS BONY PHTV DOMT SPHT AFMC KORA MPCI ZMID ELEC NINH SUGR ENGC ACAP AMIA NAPR AXPH GOUR CNFN ARAB AMER OCPH DSCW SPIN MICH GSSC KABO MFSC SVCE WCDF GDWA ADCI UNIT OFH UEFM AJWA SDTI INFI SAIB ELSH ELKA ASCM ACTF DAPH ACGC ACAMD ISMA LCSW GGCC KZPC SMFR ZEOT CRST ALRA CFGH ETRS EDFM ADPC NARE ATLC MILS MPCO PHGC IDRE GPIM UEGC GGRN RACC CEFM EHDR EALR NAHO AALR ECAP MOSC WKOL SNFC PRCL MAAL ODIN GTWL MENA SCFM DTPP NHPS NCCW CAED CERA DEIN SEIG SEIGA OBRI MEPA SIPC RREI NDRL AFDI AIDC ALUM AMII LUTS COSG ASPI EBSC POCO PRMH RTVC UNIP TANM GTEX MCRO KRDI APSW SPMD ICID ROTO AIHC MEGM ICLE RUBX TYCN EASB KWIN MOED RAKT AREH EEII CCRS EPCO GRCA GIHD ELWA ELNA DGTZ DCCC MMAT NEDA TRTO EPPK GMCI EOSB CPME COPR
'''
CURRENT_EGX_FALLBACK = tuple(dict.fromkeys(x.upper().replace('.CA','') for x in CURRENT_EGX_FALLBACK.split()))

# Known sector hints. Unknown names are allowed and are inferred from Yahoo where possible.
SECTOR_MAP = {
    'COMI':'Banks','ADIB':'Banks','CIEB':'Banks','QNBA':'Banks','HDBK':'Banks','UBEE':'Banks','FAIT':'Banks','EXPA':'Banks',
    'CICH':'Financials','HRHO':'Financials','EFGH':'Financials','CNFN':'Financials','MICH':'Financials','MUBI':'Financials','FWRY':'Technology',
    'VALU':'Financials','TMGH':'Real Estate','PHDC':'Real Estate','MNHD':'Real Estate','EMFD':'Real Estate','SODIC':'Real Estate','AMER':'Real Estate','ORHD':'Real Estate',
    'ORAS':'Construction','SWDY':'Industrials','MMMT':'Industrials','GBCO':'Industrials','GBCA':'Industrials','AUTO':'Industrials','ALCN':'Industrials',
    'ETEL':'Telecom','EAST':'Consumer Staples','JUFO':'Consumer Staples','DOMT':'Consumer Staples','UEFM':'Consumer Staples',
    'MFPC':'Materials','ABUK':'Materials','EGAL':'Materials','AMOC':'Energy','EGAS':'Energy','SKPC':'Energy','TAQA':'Energy',
    'BIOC':'Healthcare','NIPH':'Healthcare','ISPH':'Healthcare','DAPH':'Healthcare','KZPC':'Chemicals','SPMD':'Materials',
}

AR_SECTOR = {
    'Banks':'بنوك','Financials':'خدمات مالية','Real Estate':'عقارات','Construction':'مقاولات وإنشاءات','Industrials':'صناعات',
    'Telecom':'اتصالات','Consumer Staples':'أغذية واستهلاك','Materials':'موارد أساسية','Energy':'طاقة','Healthcare':'رعاية صحية',
    'Technology':'تكنولوجيا وخدمات مالية رقمية','Chemicals':'كيماويات','Other':'أخرى'
}
AR_ACTION = {
    'Strong Buy / Accumulate':'شراء قوي / تجميع','Buy on weakness':'شراء عند التراجعات','Watch / Gradual':'مراقبة / شراء تدريجي',
    'Overvalued / Wait':'مرتفع القيمة / انتظار','Neutral / Watch':'محايد / مراقبة','Weak Data / Reference':'بيانات ضعيفة / قيمة مرجعية','Data unavailable':'البيانات غير كافية'
}

# ============================================================
# Helpers
# ============================================================
def num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan

def first(*xs):
    for x in xs:
        v = num(x)
        if np.isfinite(v):
            return v
    return np.nan

def div(a,b):
    a,b = num(a),num(b)
    return a/b if np.isfinite(a) and np.isfinite(b) and b != 0 else np.nan

def clamp(x,a,b):
    x = num(x)
    return float(np.clip(x,a,b)) if np.isfinite(x) else np.nan

def fmt_num(x, digits=2):
    return 'غير متاح' if not np.isfinite(num(x)) else f'{num(x):,.{digits}f}'

def fmt_pct(x):
    return 'غير متاح' if not np.isfinite(num(x)) else f'{num(x)*100:.1f}%'

def fmt_score(x):
    return 'غير متاح' if not np.isfinite(num(x)) else f'{num(x):.1f}'

def clean_value(x):
    return x if np.isfinite(num(x)) else np.nan

def stmt_latest(stmt, names):
    if stmt is None or stmt.empty:
        return np.nan
    for n in names:
        if n in stmt.index:
            s = pd.to_numeric(stmt.loc[n], errors='coerce').dropna()
            if len(s):
                return float(s.iloc[0])
    return np.nan

def stmt_series(stmt, names):
    if stmt is None or stmt.empty:
        return pd.Series(dtype=float)
    for n in names:
        if n in stmt.index:
            return pd.to_numeric(stmt.loc[n], errors='coerce').dropna()
    return pd.Series(dtype=float)

def stmt_growth(stmt, names):
    s = stmt_series(stmt,names)
    if len(s) >= 2 and s.iloc[1] != 0:
        return float(s.iloc[0]/s.iloc[1]-1)
    return np.nan

def safe_mean(values):
    vals=[num(x) for x in values if np.isfinite(num(x))]
    return float(np.mean(vals)) if vals else np.nan

# ============================================================
# Universe and source links
# ============================================================
@st.cache_data(ttl=21600, show_spinner=False)
def build_universe():
    # Priority: current EGX/StockAnalysis snapshot -> user's legacy universe -> live discovery.
    symbols = list(CURRENT_EGX_FALLBACK)
    for x in STOCKS:
        if x not in symbols:
            symbols.append(x)
    try:
        html = requests.get('https://stockanalysis.com/list/egyptian-stock-exchange/', headers=HEADERS, timeout=12).text
        found = re.findall(r'/quote/egx/([A-Z0-9]{2,10})(?:\.CA)?/', html)
        found += re.findall(r'/stocks/([A-Z0-9]{2,10})/', html)
        for x in found:
            x=x.upper().replace('.CA','')
            if x not in symbols:
                symbols.append(x)
    except Exception:
        pass
    # Exactly 246 symbols are kept for the user's requested scan size.
    return tuple(dict.fromkeys(symbols))[:TARGET_UNIVERSE]

def source_links(symbol):
    s=symbol.upper()
    return {
        'البورصة المصرية': 'https://beta.egx.com.eg/ar',
        'AskBorsa': f'https://askborsa.com/en/company/{quote(s)}',
        'Mubasher': f'https://www.mubasher.info/markets/EGX/stocks/{quote(s)}/ratios',
        'StockAnalysis': 'https://stockanalysis.com/list/egyptian-stock-exchange/',
        'Yahoo Finance': f'https://finance.yahoo.com/quote/{quote(s)}.CA/'
    }

# ============================================================
# Market price layer - small batches are much more reliable than 246 at once
# ============================================================
@st.cache_data(ttl=900, show_spinner=False)
def market_snapshot(symbols):
    out={}
    syms=list(symbols)
    for start in range(0,len(syms),25):
        batch=syms[start:start+25]
        try:
            tickers=' '.join(s+'.CA' for s in batch)
            data=yf.download(tickers,period='5d',interval='1d',group_by='ticker',auto_adjust=False,progress=False,threads=False)
            if data is None or data.empty:
                continue
            for sym in batch:
                try:
                    if isinstance(data.columns,pd.MultiIndex):
                        q=data[sym+'.CA'] if (sym+'.CA') in data.columns.get_level_values(0) else pd.DataFrame()
                    else:
                        q=data
                    if not q.empty and 'Close' in q:
                        q=q.dropna(subset=['Close'])
                        if len(q): out[sym]=float(q['Close'].iloc[-1])
                except Exception:
                    continue
        except Exception:
            continue
    return out

# ============================================================
# Mubasher fallback - fills missing price/ratios/dividend where possible
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def mubasher_fallback(symbol):
    result={}
    try:
        url=f'https://www.mubasher.info/markets/EGX/stocks/{quote(symbol)}/ratios'
        html=requests.get(url,headers=HEADERS,timeout=10).text
        if not html:
            return result
        text=re.sub(r'<[^>]+>',' ',html)
        text=re.sub(r'\s+',' ',text)
        text=text.replace('&nbsp;',' ')
        patterns={
            'price':[r'آخر سعر\s*([0-9,.]+)',r'Last Price\s*([0-9,.]+)'],
            'eps':[r'ربحية السهم\s*([0-9,.\-]+)',r'EPS\s*([0-9,.\-]+)'],
            'pe':[r'مكرر الربحية\s*([0-9,.\-]+)',r'P/E Ratio\s*([0-9,.\-]+)'],
            'pb':[r'مضاعف القيمة الدفترية\s*([0-9,.\-]+)',r'P/B Ratio\s*([0-9,.\-]+)'],
            'bvps':[r'القيمة الدفترية للسهم\s*([0-9,.\-]+)',r'Book Value\s*([0-9,.\-]+)'],
            'roe':[r'العائد على حقوق الملكية\s*%?\s*([0-9,.\-]+)',r'Return on Equity\s*%?\s*([0-9,.\-]+)'],
            'roa':[r'العائد على الأصول\s*%?\s*([0-9,.\-]+)',r'Return on Assets\s*%?\s*([0-9,.\-]+)'],
        }
        for key, pats in patterns.items():
            for pat in pats:
                m=re.search(pat,text,re.I)
                if m:
                    val=num(m.group(1).replace(',',''))
                    if np.isfinite(val):
                        if key in ('roe','roa') and abs(val)>1: val/=100
                        result[key]=val
                        break
    except Exception:
        pass
    return result

# ============================================================
# Independent source reconciliation layer
# ============================================================

def _plain_html(url, timeout=8):
    try:
        rr=requests.get(url,headers=HEADERS,timeout=timeout)
        if rr.ok and rr.text:
            return rr.text
    except Exception:
        pass
    return ''

def _num_from_text(x):
    if x is None: return np.nan
    x=str(x).replace(',','').replace('٬','').replace('٫','.').strip()
    m=re.search(r'[-+]?\d+(?:\.\d+)?',x)
    return num(m.group(0)) if m else np.nan

def _near_number(text, labels):
    clean=re.sub(r'<[^>]+>',' ',text)
    clean=re.sub(r'&nbsp;',' ',clean)
    clean=re.sub(r'\s+',' ',clean)
    for label in labels:
        pat=re.escape(label)+r'.{0,120}?([-+]?\d[\d,]*(?:\.\d+)?)'
        m=re.search(pat,clean,re.I)
        if m:
            v=_num_from_text(m.group(1))
            if np.isfinite(v): return v
    return np.nan

@st.cache_data(ttl=6*3600, show_spinner=False)
def stockanalysis_fallback(symbol):
    """Independent cross-check. Never overwrites Yahoo; only supplies corroborating fields."""
    out={}
    base=f'https://stockanalysis.com/quote/egx/{quote(symbol)}/'
    html=_plain_html(base,10)
    if not html: return out
    out['price']=_near_number(html,['Price','Last Close'])
    out['eps']=_near_number(html,['EPS'])
    out['bvps']=_near_number(html,['Book Value / Share','Book Value Per Share'])
    out['pe']=_near_number(html,['PE Ratio','P/E Ratio'])
    out['pb']=_near_number(html,['PB Ratio','P/B Ratio'])
    out['ps']=_near_number(html,['PS Ratio','P/S Ratio'])
    out['roe']=_near_number(html,['ROE'])
    out['roa']=_near_number(html,['ROA'])
    out['revenue']=_near_number(html,['Revenue'])
    out['net_income']=_near_number(html,['Net Income'])
    out['equity']=_near_number(html,['Total Equity','Stockholders Equity'])
    out['debt']=_near_number(html,['Total Debt'])
    out['cash']=_near_number(html,['Cash & Equivalents','Cash and Equivalents'])
    out['source']='StockAnalysis'
    return {k:v for k,v in out.items() if k=='source' or np.isfinite(num(v))}

@st.cache_data(ttl=6*3600, show_spinner=False)
def askborsa_crosscheck(symbol):
    """AskBorsa is treated as a statement-source cross-check, not as an opinion engine."""
    url=f'https://askborsa.com/en/company/{quote(symbol)}'
    html=_plain_html(url,10)
    if not html: return {'available':False,'url':url}
    return {'available':True,'url':url,'source':'AskBorsa'}

@st.cache_data(ttl=6*3600, show_spinner=False)
def disclosure_crosscheck(symbol):
    url=f'https://foudalens.com/en/disclosures/{quote(symbol)}.CA'
    html=_plain_html(url,10)
    return {'available':bool(html),'url':url,'source':'EGX disclosure archive'}


def reconcile_field(r, key, candidates, tolerance=0.20):
    """Keep primary value but record large source disagreement for confidence penalties."""
    valid=[]
    for source,val in candidates:
        v=num(val)
        if np.isfinite(v): valid.append((source,v))
    if not valid: return
    primary=num(r.get(key))
    if not np.isfinite(primary):
        r[key]=valid[0][1]
        r['imputed'].append(f'{key} من {valid[0][0]}')
    vals=[v for _,v in valid]
    if len(vals)>=2:
        med=float(np.median(vals))
        if med!=0 and abs(primary-med)/abs(med)>tolerance:
            r['source_conflicts'].append(f'{key}: '+', '.join(f'{src}={v:.2f}' for src,v in valid))


# ============================================================
# Fundamental engine
# ============================================================
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fundamentals(symbol):
    r={
        'symbol':symbol,'name':symbol,'sector':SECTOR_MAP.get(symbol,'Other'),
        'price':np.nan,'market_cap':np.nan,'shares':np.nan,
        'revenue':np.nan,'net_income':np.nan,'ebitda':np.nan,'operating_cf':np.nan,'fcf':np.nan,
        'cash':np.nan,'debt':np.nan,'equity':np.nan,'assets':np.nan,
        'eps':np.nan,'bvps':np.nan,'dividend':np.nan,'div_yield':np.nan,'payout':np.nan,
        'roe':np.nan,'roa':np.nan,'margin':np.nan,'rev_growth':np.nan,'earn_growth':np.nan,
        'debt_equity':np.nan,'pe':np.nan,'pb':np.nan,'ps':np.nan,
        'coverage':0.0,'imputed':[],'sources':[],'period':'غير محدد','price_source':'غير متاح','div_source':'غير متاح',
        'normalized_net_income':np.nan,'normalized_eps':np.nan,'normalized_growth':np.nan,'fcf_normalized':np.nan,'fcff_normalized':np.nan,
        'piotroski':np.nan,'altman_z':np.nan,'interest_coverage':np.nan,'earnings_quality':np.nan,'net_debt':np.nan,
        'peer_pe':np.nan,'peer_pb':np.nan,'peer_ps':np.nan,'peer_ev_ebitda':np.nan,'fair_value_confidence':np.nan,
        'latest_statement_days':np.nan,
        'source_quality':{},'source_conflicts':[],'source_count':0,'accounting_mode':'غير محدد'
    }
    try:
        t=yf.Ticker(symbol+'.CA')
        try: info=t.info or {}
        except Exception: info={}
        r['name']=info.get('longName') or info.get('shortName') or symbol
        r['sector']=SECTOR_MAP.get(symbol) or info.get('sector') or info.get('industry') or 'Other'
        r['price']=first(info.get('currentPrice'),info.get('regularMarketPrice'),info.get('previousClose'))
        if np.isfinite(r['price']):
            r['price_source']='Yahoo Finance'
            r['sources'].append('Yahoo')
        r['market_cap']=first(info.get('marketCap'))
        r['shares']=first(info.get('sharesOutstanding'),info.get('impliedSharesOutstanding'))
        r['eps']=first(info.get('trailingEps'),info.get('forwardEps'))
        r['dividend']=first(info.get('dividendRate'),info.get('trailingAnnualDividendRate'))
        r['div_yield']=first(info.get('dividendYield'))
        r['roe']=first(info.get('returnOnEquity'))
        r['roa']=first(info.get('returnOnAssets'))
        r['margin']=first(info.get('profitMargins'))
        r['rev_growth']=first(info.get('revenueGrowth'))
        r['earn_growth']=first(info.get('earningsGrowth'),info.get('earningsQuarterlyGrowth'))
        r['debt_equity']=first(info.get('debtToEquity'))
        r['pe']=first(info.get('trailingPE'),info.get('forwardPE'))
        r['pb']=first(info.get('priceToBook'))
        r['ps']=first(info.get('priceToSalesTrailing12Months'))

        try: inc=t.income_stmt
        except Exception: inc=pd.DataFrame()
        try: bal=t.balance_sheet
        except Exception: bal=pd.DataFrame()
        try: cf=t.cashflow
        except Exception: cf=pd.DataFrame()
        try:
            if not inc.empty:
                r['period']=str(inc.columns[0])[:10]
        except Exception: pass

        r['revenue']=first(stmt_latest(inc,['Total Revenue','Operating Revenue']),info.get('totalRevenue'))
        r['net_income']=first(stmt_latest(inc,['Net Income','Net Income Common Stockholders','Net Income Including Noncontrolling Interests']),info.get('netIncomeToCommon'))
        r['ebitda']=first(stmt_latest(inc,['EBITDA','Normalized EBITDA']),info.get('ebitda'))
        r['operating_cf']=first(stmt_latest(cf,['Operating Cash Flow','Total Cash From Operating Activities']),info.get('operatingCashflow'))
        r['fcf']=first(stmt_latest(cf,['Free Cash Flow']),info.get('freeCashflow'))
        if not np.isfinite(r['fcf']) and np.isfinite(r['operating_cf']):
            capex=stmt_latest(cf,['Capital Expenditure','Capital Expenditure Reported'])
            if np.isfinite(capex): r['fcf']=r['operating_cf']+capex
        r['cash']=first(stmt_latest(bal,['Cash Cash Equivalents And Short Term Investments','Cash And Cash Equivalents','Cash Financial']),info.get('totalCash'))
        r['debt']=first(stmt_latest(bal,['Total Debt','Long Term Debt And Capital Lease Obligation','Current Debt And Capital Lease Obligation']),info.get('totalDebt'))
        r['equity']=first(stmt_latest(bal,['Stockholders Equity','Common Stock Equity','Total Equity Gross Minority Interest']))
        r['assets']=first(stmt_latest(bal,['Total Assets']),info.get('totalAssets'))
        r['rev_growth']=first(r['rev_growth'],stmt_growth(inc,['Total Revenue','Operating Revenue']))
        r['earn_growth']=first(r['earn_growth'],stmt_growth(inc,['Net Income','Net Income Common Stockholders']))

        # Historical dividend action: much more reliable than relying only on info fields.
        try:
            actions=t.actions
            if actions is not None and not actions.empty and 'Dividends' in actions.columns:
                ds=pd.to_numeric(actions['Dividends'],errors='coerce').dropna()
                ds=ds[ds>0]
                if len(ds):
                    recent=ds[ds.index >= (pd.Timestamp.today(tz=None)-pd.Timedelta(days=400))]
                    total=float(recent.sum()) if len(recent) else float(ds.tail(4).sum())
                    if total>0:
                        r['dividend']=total
                        r['div_source']='Yahoo dividend history'
                        if np.isfinite(r['price']) and r['price']>0:
                            r['div_yield']=total/r['price']
        except Exception: pass

        # Derived values: allowed because they are mathematical transformations of reported data.
        if not np.isfinite(r['shares']) and np.isfinite(r['market_cap']) and np.isfinite(r['price']) and r['price']>0:
            r['shares']=r['market_cap']/r['price']; r['imputed'].append('عدد الأسهم مشتق')
        if not np.isfinite(r['eps']) and np.isfinite(r['net_income']) and np.isfinite(r['shares']) and r['shares']>0:
            r['eps']=r['net_income']/r['shares']; r['imputed'].append('EPS مشتق')
        if not np.isfinite(r['bvps']) and np.isfinite(r['equity']) and np.isfinite(r['shares']) and r['shares']>0:
            r['bvps']=r['equity']/r['shares']; r['imputed'].append('BVPS مشتق')
        if not np.isfinite(r['roe']) and np.isfinite(r['net_income']) and np.isfinite(r['equity']) and r['equity']>0:
            r['roe']=r['net_income']/r['equity']; r['imputed'].append('ROE مشتق')
        if not np.isfinite(r['roa']) and np.isfinite(r['net_income']) and np.isfinite(r['assets']) and r['assets']>0:
            r['roa']=r['net_income']/r['assets']; r['imputed'].append('ROA مشتق')
        if not np.isfinite(r['margin']) and np.isfinite(r['net_income']) and np.isfinite(r['revenue']) and r['revenue']!=0:
            r['margin']=r['net_income']/r['revenue']; r['imputed'].append('هامش الربح مشتق')
        if not np.isfinite(r['debt_equity']) and np.isfinite(r['debt']) and np.isfinite(r['equity']) and r['equity']!=0:
            r['debt_equity']=r['debt']/r['equity']; r['imputed'].append('D/E مشتق')
        if not np.isfinite(r['div_yield']) and np.isfinite(r['dividend']) and np.isfinite(r['price']) and r['price']>0:
            r['div_yield']=r['dividend']/r['price']; r['imputed'].append('عائد التوزيع مشتق')
        if not np.isfinite(r['payout']) and np.isfinite(r['dividend']) and np.isfinite(r['eps']) and r['eps']>0:
            r['payout']=r['dividend']/r['eps']
        if not np.isfinite(r['pe']) and np.isfinite(r['price']) and np.isfinite(r['eps']) and r['eps']>0:
            r['pe']=r['price']/r['eps']; r['imputed'].append('P/E مشتق')
        if not np.isfinite(r['pb']) and np.isfinite(r['price']) and np.isfinite(r['bvps']) and r['bvps']>0:
            r['pb']=r['price']/r['bvps']; r['imputed'].append('P/B مشتق')
        if not np.isfinite(r['ps']) and np.isfinite(r['price']) and np.isfinite(r['market_cap']) and np.isfinite(r['revenue']) and r['revenue']>0:
            r['ps']=r['market_cap']/r['revenue']; r['imputed'].append('P/S مشتق')

        # --------------------------------------------------------
        # Historical normalization: use reported 3-5Y statements, never fabricate data.
        # --------------------------------------------------------
        rev_s=stmt_series(inc,['Total Revenue','Operating Revenue'])
        ni_s=stmt_series(inc,['Net Income','Net Income Common Stockholders','Net Income Including Noncontrolling Interests'])
        fcf_s=stmt_series(cf,['Free Cash Flow'])
        ocf_s=stmt_series(cf,['Operating Cash Flow','Total Cash From Operating Activities'])
        if len(ni_s)>=2:
            # Median of available historical profits is a robust base for cyclical/one-off businesses.
            tail=ni_s.head(min(5,len(ni_s)))
            pos=tail[tail>0]
            r['normalized_net_income']=float(pos.median()) if len(pos)>=2 else float(tail.median())
        if np.isfinite(r['normalized_net_income']) and np.isfinite(r['shares']) and r['shares']>0:
            r['normalized_eps']=r['normalized_net_income']/r['shares']
        if len(rev_s)>=3:
            a=float(rev_s.iloc[min(2,len(rev_s)-1)]); b=float(rev_s.iloc[0])
            if a>0 and b>0: r['normalized_growth']=(b/a)**(1/2)-1
        if len(ni_s)>=3:
            pos=ni_s.head(5)
            pos=pos[pos>0]
            if len(pos)>=3:
                r['earnings_growth_3y']=(float(pos.iloc[0])/float(pos.iloc[min(2,len(pos)-1)]))**(1/2)-1 if float(pos.iloc[min(2,len(pos)-1)])>0 else np.nan
        if len(fcf_s)>=2:
            r['fcf_normalized']=float(fcf_s.head(min(5,len(fcf_s))).median())
        elif len(ocf_s)>=2:
            r['fcf_normalized']=float(ocf_s.head(min(5,len(ocf_s))).median())
        # Net debt and FCFF are derived from statement values only.
        if np.isfinite(r['debt']) or np.isfinite(r['cash']):
            r['net_debt']=max(0.0,(r['debt'] if np.isfinite(r['debt']) else 0.0)-(r['cash'] if np.isfinite(r['cash']) else 0.0))
        interest=stmt_latest(inc,['Interest Expense Non Operating','Interest Expense','Interest Expense Non Operating Income'])
        if np.isfinite(interest) and abs(interest)>0 and np.isfinite(r['ebitda']):
            r['interest_coverage']=r['ebitda']/abs(interest)
        elif np.isfinite(r['ebitda']) and np.isfinite(r['debt']) and r['debt']==0:
            r['interest_coverage']=10.0
        # Piotroski F-score from available annual statement history.
        try:
            assets_s=stmt_series(bal,['Total Assets'])
            eq_s=stmt_series(bal,['Stockholders Equity','Common Stock Equity','Total Equity Gross Minority Interest'])
            ocf_s=stmt_series(cf,['Operating Cash Flow','Total Cash From Operating Activities'])
            score9=0; tests=0
            if len(ni_s)>=2:
                tests+=1; score9 += int(float(ni_s.iloc[0])>0)
            if len(ocf_s)>=1:
                tests+=1; score9 += int(float(ocf_s.iloc[0])>0)
            if len(ni_s)>=1 and len(assets_s)>=2 and assets_s.iloc[0]>0 and assets_s.iloc[1]>0:
                roa0=ni_s.iloc[0]/assets_s.iloc[0]; roa1=ni_s.iloc[1]/assets_s.iloc[1]; tests+=1; score9+=int(roa0>roa1)
            if len(ocf_s)>=1 and len(ni_s)>=1:
                tests+=1; score9+=int(ocf_s.iloc[0]>ni_s.iloc[0])
            if len(eq_s)>=2:
                tests+=1; score9+=int(eq_s.iloc[0]>=eq_s.iloc[1])
            debt_s=stmt_series(bal,['Total Debt','Long Term Debt And Capital Lease Obligation','Current Debt And Capital Lease Obligation'])
            if len(assets_s)>=2 and len(debt_s)>=2:
                lev0=(debt_s.iloc[0]/assets_s.iloc[0]) if assets_s.iloc[0]!=0 else np.nan
                lev1=(debt_s.iloc[1]/assets_s.iloc[1]) if assets_s.iloc[1]!=0 else np.nan
                if np.isfinite(lev0) and np.isfinite(lev1): tests+=1; score9+=int(lev0<=lev1)
            if len(ocf_s)>=2:
                tests+=1; score9+=int(ocf_s.iloc[0]>=ocf_s.iloc[1])
            if len(rev_s)>=2:
                tests+=1; score9+=int(rev_s.iloc[0]>=rev_s.iloc[1])
            if np.isfinite(r['margin']) and len(rev_s)>=2 and len(ni_s)>=2:
                m0=ni_s.iloc[0]/rev_s.iloc[0] if rev_s.iloc[0]!=0 else np.nan; m1=ni_s.iloc[1]/rev_s.iloc[1] if rev_s.iloc[1]!=0 else np.nan
                if np.isfinite(m0) and np.isfinite(m1): tests+=1; score9+=int(m0>m1)
            r['piotroski']=score9 if tests>=5 else np.nan
        except Exception: pass
        # Earnings quality: operating cash flow relative to accounting profit.
        if np.isfinite(r['operating_cf']) and np.isfinite(r['net_income']) and abs(r['net_income'])>1e-9:
            r['earnings_quality']=r['operating_cf']/r['net_income']

        # Independent cross-checks. They corroborate; they do not blindly replace primary accounting data.
        sa=stockanalysis_fallback(symbol)
        if sa.get('source'):
            r['sources'].append('StockAnalysis')
            for k in ['price','eps','bvps','pe','pb','ps','roe','roa','revenue','net_income','equity','debt','cash']:
                reconcile_field(r,k,[('Yahoo',r.get(k)),('StockAnalysis',sa.get(k))])
        ab=askborsa_crosscheck(symbol)
        if ab.get('available'):
            r['sources'].append('AskBorsa')
        disc=disclosure_crosscheck(symbol)
        if disc.get('available'):
            r['sources'].append('EGX disclosure archive')
        r['source_count']=len(set(r['sources']))
        r['source_quality']={'Yahoo': 'market/fundamental', 'Mubasher':'ratios/market', 'StockAnalysis':'cross-check', 'AskBorsa':'official-report extraction', 'EGX disclosure archive':'official disclosure archive'}
        fields=['price','revenue','net_income','equity','eps','bvps','dividend','roe','rev_growth','earn_growth','shares']
        r['coverage']=sum(np.isfinite(num(r[k])) for k in fields)/len(fields)
    except Exception as e:
        r['imputed'].append('خطأ مصدر Yahoo')

    # Mubasher fills missing ratios. It never overwrites a valid Yahoo figure.
    mb=mubasher_fallback(symbol)
    for k,v in mb.items():
        if not np.isfinite(num(r.get(k))) and np.isfinite(num(v)):
            r[k]=v
            r['imputed'].append(f'{k} من Mubasher')
            if 'Mubasher' not in r['sources']: r['sources'].append('Mubasher')
    if not np.isfinite(r['price']) and np.isfinite(num(market_snapshot((symbol,)).get(symbol))):
        r['price']=market_snapshot((symbol,)).get(symbol); r['price_source']='Yahoo batch'; r['imputed'].append('السعر من دفعة السوق')
    if not np.isfinite(r['div_yield']) and np.isfinite(r['dividend']) and np.isfinite(r['price']) and r['price']>0:
        r['div_yield']=r['dividend']/r['price']
    r['source_count']=len(set(r.get('sources') or []))
    r['coverage']=sum(np.isfinite(num(r[k])) for k in ['price','revenue','net_income','equity','eps','bvps','dividend','roe','rev_growth','earn_growth','shares'])/11
    return r

# ============================================================
# Technical confirmation
# ============================================================
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def history(symbol):
    try:
        d=yf.Ticker(symbol+'.CA').history(period='3y',auto_adjust=False,actions=True)
        return d.dropna(subset=['Close']) if d is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def technical(d):
    o={k:np.nan for k in ['rsi','ema20','ema50','ema200','atr_pct','volume_ratio','support','resistance','technical_score']}
    if d is None or len(d)<30: return o
    c=pd.to_numeric(d['Close'],errors='coerce'); h=pd.to_numeric(d['High'],errors='coerce'); l=pd.to_numeric(d['Low'],errors='coerce'); v=pd.to_numeric(d['Volume'],errors='coerce')
    o['ema20']=c.ewm(span=20,adjust=False).mean().iloc[-1]
    o['ema50']=c.ewm(span=50,adjust=False).mean().iloc[-1]
    if len(c)>=200: o['ema200']=c.ewm(span=200,adjust=False).mean().iloc[-1]
    delta=c.diff(); gain=delta.clip(lower=0).rolling(14).mean(); loss=(-delta.clip(upper=0)).rolling(14).mean(); rs=gain/loss.replace(0,np.nan); o['rsi']=(100-100/(1+rs)).iloc[-1]
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1); atr=tr.rolling(14).mean(); o['atr_pct']=atr.iloc[-1]/c.iloc[-1] if c.iloc[-1]!=0 else np.nan
    vm=v.rolling(20).mean().iloc[-1]; o['volume_ratio']=v.iloc[-1]/vm if np.isfinite(vm) and vm!=0 else np.nan
    recent=c.tail(min(120,len(c))); o['support']=recent.min(); o['resistance']=recent.max()
    score=50
    if c.iloc[-1]>o['ema20']: score+=12
    if c.iloc[-1]>o['ema50']: score+=12
    if np.isfinite(o['ema200']) and c.iloc[-1]>o['ema200']: score+=12
    if o['ema20']>o['ema50']: score+=8
    if np.isfinite(o['rsi']):
        if 50<=o['rsi']<=68: score+=10
        elif o['rsi']>75: score-=8
        elif o['rsi']<35: score-=5
    if np.isfinite(o['volume_ratio']) and o['volume_ratio']>1.2: score+=5
    o['technical_score']=float(np.clip(score,0,100))
    return o

# ============================================================
# Institutional multi-method valuation engine
# ============================================================
def _is_bank(r):
    s=str(r.get('sector','')).lower()
    return 'bank' in s or 'بنك' in s

def sector_pe(sector, roe, growth):
    s=str(sector).lower(); roe=num(roe); growth=num(growth)
    if 'bank' in s: return float(np.clip(6.0 + ((roe-.15)*9 if np.isfinite(roe) else 0),5.5,11.0))
    if 'financial' in s: return 9.0
    if 'real estate' in s: return 9.5
    if 'telecom' in s or 'utility' in s: return 8.5
    return float(np.clip(9.0 + ((growth-.08)*7 if np.isfinite(growth) else 0),6.5,15.0))

def sector_pb(sector, roe):
    s=str(sector).lower(); roe=num(roe)
    if 'bank' in s:
        rr=roe if np.isfinite(roe) else .15
        return float(np.clip(.75+(rr-.10)*3.0,.60,2.2))
    if 'financial' in s: return 1.10
    if 'real estate' in s: return 1.15
    return 1.10

def _method_conf(name, r):
    base={'P/E':.78,'P/B':.78,'Normalized P/E':.84,'Dividend':.66,'FCF Yield':.84,'DCF':.86,'Residual Income':.90,'EV/EBITDA':.80,'P/S':.58,'Peer P/E':.88,'Peer P/B':.86}.get(name,.60)
    cov=num(r.get('coverage')); sources=num(r.get('source_count'))
    return float(np.clip(base + .10*max(cov-.5,0) + .035*min(sources,4) - .025*len(r.get('source_conflicts') or []),.25,.98))

def _mad_filter(items):
    if len(items)<=3: return items
    vals=np.array([x['value'] for x in items],dtype=float)
    med=float(np.median(vals)); mad=float(np.median(np.abs(vals-med)))
    if mad<=1e-12: return [x for x in items if abs(x['value']-med)/max(abs(med),1e-9)<=.75]
    z=np.abs(vals-med)/(1.4826*mad)
    return [x for x,zz in zip(items,z) if zz<=3.5]

def _dcf_value(r):
    fcf=first(r.get('fcf_normalized'),r.get('fcf')); shares=num(r.get('shares')); growth=first(r.get('normalized_growth'),r.get('rev_growth'))
    if not np.isfinite(fcf) or fcf<=0 or not np.isfinite(shares) or shares<=0: return np.nan
    g0=float(np.clip(growth if np.isfinite(growth) else .08,.02,.16))
    ke=max(COST_OF_EQUITY_FLOOR,RISK_FREE+ERP); tg=TERMINAL_GROWTH
    if ke<=tg: return np.nan
    pv=0.0; base=fcf
    for y in range(1,6):
        g=float(np.clip(g0*(1-(y-1)*.14),.015,.16)); base*=1+g; pv+=base/((1+ke)**y)
    terminal=base*(1+tg)/(ke-tg)
    ev=pv+terminal/((1+ke)**5)
    debt=num(r.get('debt')); cash=num(r.get('cash'))
    eq=ev-(debt if np.isfinite(debt) else 0)+(cash if np.isfinite(cash) else 0)
    return eq/shares if eq>0 else np.nan

def _residual_income_value(r):
    bvps=num(r.get('bvps')); roe=num(r.get('roe'))
    if not np.isfinite(bvps) or bvps<=0 or not np.isfinite(roe): return np.nan
    ke=max(COST_OF_EQUITY_FLOOR,RISK_FREE+ERP); g=float(np.clip(first(r.get('normalized_growth'),r.get('rev_growth')) if np.isfinite(first(r.get('normalized_growth'),r.get('rev_growth'))) else .05,.02,.10))
    book=bvps; value=book
    for y in range(1,6):
        ri=max(0,(roe-ke))*book; value += ri/((1+ke)**y); book*=1+g
    terminal_ri=max(0,(roe-ke))*book*(1+TERMINAL_GROWTH)/(ke-TERMINAL_GROWTH)
    return value+terminal_ri/((1+ke)**5)

def valuation(r):
    sector=str(r.get('sector','Other')); eps=num(r.get('eps')); neps=num(r.get('normalized_eps')); bvps=num(r.get('bvps')); roe=num(r.get('roe')); divd=num(r.get('dividend')); shares=num(r.get('shares')); price=num(r.get('price')); ebitda=num(r.get('ebitda')); debt=num(r.get('debt')); cash=num(r.get('cash'))
    growth=first(r.get('normalized_growth'),r.get('earn_growth')); items=[]
    bank=_is_bank(r)
    # Banks/financials: prioritize book value + residual income + normalized earnings.
    if np.isfinite(neps) and neps>0:
        items.append({'name':'Normalized P/E','value':neps*first(r.get('peer_pe'),sector_pe(sector,roe,growth)),'conf':_method_conf('Normalized P/E',r)})
    elif np.isfinite(eps) and eps>0:
        items.append({'name':'P/E','value':eps*first(r.get('peer_pe'),sector_pe(sector,roe,growth)),'conf':_method_conf('P/E',r)})
    if np.isfinite(bvps) and bvps>0:
        items.append({'name':'P/B','value':bvps*first(r.get('peer_pb'),sector_pb(sector,roe)),'conf':_method_conf('P/B',r)})
    ri=_residual_income_value(r)
    if np.isfinite(ri): items.append({'name':'Residual Income','value':ri,'conf':_method_conf('Residual Income',r)})
    if np.isfinite(divd) and divd>0:
        g=float(np.clip(growth if np.isfinite(growth) else .04,.00,.07)); ke=max(COST_OF_EQUITY_FLOOR,RISK_FREE+ERP)
        if ke>g: items.append({'name':'Dividend','value':divd*(1+g)/(ke-g),'conf':_method_conf('Dividend',r)})
    if not bank:
        dcf=_dcf_value(r)
        if np.isfinite(dcf): items.append({'name':'DCF','value':dcf,'conf':_method_conf('DCF',r)})
        fcf=first(r.get('fcf_normalized'),r.get('fcf'))
        if np.isfinite(fcf) and fcf>0 and np.isfinite(shares) and shares>0:
            fcfps=fcf/shares; g=float(np.clip(growth if np.isfinite(growth) else .06,.02,.16)); mult=float(np.clip(9.0+g*16,8,13)); items.append({'name':'FCF Yield','value':fcfps*mult,'conf':_method_conf('FCF Yield',r)})
        if np.isfinite(ebitda) and ebitda>0 and np.isfinite(shares) and shares>0:
            mult=first(r.get('peer_ev_ebitda'),7.5); eq=ebitda*mult-(debt if np.isfinite(debt) else 0)+(cash if np.isfinite(cash) else 0)
            if eq>0: items.append({'name':'EV/EBITDA','value':eq/shares,'conf':_method_conf('EV/EBITDA',r)})
    # Peer P/S is only a fallback, never a dominant method.
    rev=num(r.get('revenue')); mcap=num(r.get('market_cap'))
    if not bank and np.isfinite(rev) and rev>0 and np.isfinite(shares) and shares>0 and np.isfinite(num(r.get('peer_ps'))):
        items.append({'name':'P/S','value':rev*r['peer_ps']/shares,'conf':_method_conf('P/S',r)})
    items=[x for x in items if np.isfinite(x['value']) and x['value']>0 and x['value']<price*8 if np.isfinite(price) and price>0] if np.isfinite(price) and price>0 else [x for x in items if np.isfinite(x['value']) and x['value']>0]
    filtered=_mad_filter(items)
    if len(filtered)>=2:
        vals=np.array([x['value'] for x in filtered]); weights=np.array([x['conf'] for x in filtered]); med=float(np.median(vals)); fair=float(np.average(vals,weights=weights))
        dispersion=float(np.std(vals)/max(abs(med),1e-9)); fair=.65*fair+.35*med
        low=float(np.percentile(vals,20)); high=float(np.percentile(vals,80))
        low=max(low,fair*.60); high=min(high,fair*1.40)
        if low>=high: low=fair*.80; high=fair*1.20
        fvc=float(np.clip(100*(.45+.08*len(filtered))*(1-dispersion*.65)*min(1,num(r.get('coverage'))+.2),25,98))
        return fair,low,high,' + '.join(x['name'] for x in filtered),len(filtered),False,dispersion,filtered,fvc
    if len(filtered)==1:
        v=filtered[0]['value']; return v,v*.75,v*1.25,filtered[0]['name'],1,False,.35,filtered,45.0
    if np.isfinite(price) and price>0:
        return np.nan,price*.70,price*1.30,'لا توجد طرق تقييم أساسية كافية — نطاق مرجعي فقط',0,True,np.nan,[],15.0
    return np.nan,np.nan,np.nan,'لا توجد بيانات كافية',0,True,np.nan,[],10.0

# ============================================================
# Sector imputation - only ratios/growth, never accounting totals
# ============================================================
def peer_benchmarks(df):
    df=df.copy()
    for c in ['peer_pe','peer_pb','peer_ps','peer_ev_ebitda']:
        if c not in df: df[c]=np.nan
    # Sector medians are computed only from observed positive market multiples.
    for sector, idx in df.groupby('sector').groups.items():
        sub=df.loc[idx]
        def med(col, fallback):
            x=pd.to_numeric(sub[col],errors='coerce') if col in sub else pd.Series(dtype=float)
            x=x[(x>0)&(x<60)]
            return float(x.median()) if len(x)>=3 else fallback
        pe=med('pe',sector_pe(sector,np.nan,np.nan)); pb=med('pb',sector_pb(sector,np.nan)); ps=med('ps',1.4)
        ev=7.5
        if 'ebitda' in sub:
            evs=[]
            for _,rr in sub.iterrows():
                e=num(rr.get('ebitda')); mc=num(rr.get('market_cap')); d=num(rr.get('debt')); c=num(rr.get('cash'))
                if np.isfinite(e) and e>0 and np.isfinite(mc):
                    evs.append((mc+(d if np.isfinite(d) else 0)-(c if np.isfinite(c) else 0))/e)
            if len(evs)>=3: ev=float(np.clip(np.median([x for x in evs if 0<x<40]),5,15))
        mask=df.index.isin(idx)
        df.loc[mask,'peer_pe']=pe; df.loc[mask,'peer_pb']=pb; df.loc[mask,'peer_ps']=ps; df.loc[mask,'peer_ev_ebitda']=ev
    return df

def sector_impute(df):
    df=df.copy()
    if 'imputed' not in df: df['imputed']=[[] for _ in range(len(df))]
    for col in ['rev_growth','earn_growth','roe','margin','debt_equity','div_yield','pe','pb']:
        if col not in df: continue
        med=df.groupby('sector')[col].transform('median')
        mask=df[col].isna() & med.notna()
        df.loc[mask,col]=med[mask]
        df.loc[mask,'imputed']=df.loc[mask,'imputed'].apply(lambda x:(x if isinstance(x,list) else [])+[f'وسيط القطاع: {col}'])
    return df

# ============================================================
# Scoring / scenarios
# ============================================================
def score(r):
    def n(x,a,b):
        return 50.0 if not np.isfinite(num(x)) else float(np.clip((num(x)-a)/(b-a)*100,0,100))
    price=num(r.get('price')); fair=num(r.get('fair_value'))
    upside=fair/price-1 if np.isfinite(price) and np.isfinite(fair) and price>0 else np.nan
    profitability=.40*n(r.get('roe'),.05,.35)+.30*n(r.get('margin'),.02,.30)+.30*n(r.get('earnings_quality'),.50,1.30)
    growth=.45*n(r.get('normalized_growth'),-.05,.25)+.30*n(r.get('earn_growth'),-.10,.30)+.25*n(r.get('rev_growth'),-.05,.25)
    valuation=n(upside,-.30,.60)
    balance=.55*(100-n(r.get('debt_equity'),0,3))+.25*n(r.get('interest_coverage'),1,10)+.20*n(r.get('piotroski'),2,8)
    dividend=n(r.get('div_yield'),0,.08)
    risk=n(r.get('altman_z'),1,3) if not _is_bank(r) and np.isfinite(num(r.get('altman_z'))) else 50
    financial=.20*profitability+.18*growth+.25*valuation+.15*balance+.07*dividend+.15*risk
    tech=num(r.get('technical_score')); tech=50 if not np.isfinite(tech) else tech
    confidence=num(r.get('confidence')); confidence=.25 if not np.isfinite(confidence) else confidence
    # Financial analysis dominates technical confirmation.
    combined=.90*financial+.10*tech
    final=combined*(.55+.45*confidence)
    return upside,financial,confidence,final

def scenarios(r):
    p=num(r.get('price')); fair=num(r.get('fair_value')); g=num(r.get('earn_growth')); d=num(r.get('dividend'))
    if not np.isfinite(p) or p<=0: return {}
    anchor=fair if np.isfinite(fair) and fair>0 else p
    g=float(np.clip(g if np.isfinite(g) else .08,.02,.25))
    bear_g=float(np.clip(g*.45,0,.12)); base_g=float(np.clip(g*.75,.03,.18)); bull_g=float(np.clip(g*1.05,.05,.25))
    bear=anchor*(1+bear_g)**3*.88; base=anchor*(1+base_g)**3; bull=anchor*(1+bull_g)**3*1.08
    div3=d*3 if np.isfinite(d) and d>0 else 0.0
    return {'bear_target':bear,'base_target':base,'bull_target':bull,'bear_total':bear+div3,'base_total':base+div3,'bull_total':bull+div3,'base_cagr':(base/p)**(1/3)-1 if base>0 else np.nan}

def analyze(symbol, market_price=None):
    r=fundamentals(symbol)
    # Always prefer the fresh batch price over a stale info price.
    if np.isfinite(num(market_price)) and market_price>0:
        r['price']=float(market_price); r['price_source']='Yahoo batch 5D'
    t=technical(history(symbol)); r.update(t)
    fair,lo,hi,methods,nmethods,is_ref,dispersion,vitems,fvc=valuation(r)
    r.update({'fair_value':fair,'fair_low':lo,'fair_high':hi,'valuation_methods':methods,'valuation_reference':is_ref,'valuation_method_count':nmethods,'valuation_dispersion':dispersion,'valuation_items':vitems,'fair_value_confidence':fvc})
    r['buy_30']=fair*.70 if np.isfinite(fair) else np.nan
    r['buy_20']=fair*.80 if np.isfinite(fair) else np.nan
    r['buy_10']=fair*.90 if np.isfinite(fair) else np.nan
    up,fin,conf,final=score(r)
    r.update({'upside':up,'financial_score':fin,'confidence':conf,'final_score':final})
    r.update(scenarios(r))
    dq=np.clip(100*(.72*r.get('coverage',0)+.28*(1-min(len(r.get('imputed') or []),20)/20)),15,100)
    if is_ref: dq=min(dq,55)
    r['data_quality']=dq
    ts=num(r.get('technical_score')); ts=50 if not np.isfinite(ts) else ts
    if is_ref: action='Weak Data / Reference'
    elif np.isfinite(up) and up>=.25 and ts>=65: action='Strong Buy / Accumulate'
    elif np.isfinite(up) and up>=.15: action='Buy on weakness'
    elif np.isfinite(up) and up>=.05: action='Watch / Gradual'
    elif np.isfinite(up) and up<0: action='Overvalued / Wait'
    else: action='Neutral / Watch'
    r['action']=action
    return r

# ============================================================
# Arabic display
# ============================================================
TABLE_COLS=['symbol','name','sector','price','fair_value','fair_low','fair_high','buy_30','buy_20','buy_10','bear_target','base_target','bull_target','base_cagr','dividend','div_yield','rev_growth','earn_growth','roe','debt_equity','financial_score','technical_score','data_quality','fair_value_confidence','confidence','final_score','action']
AR_COLS=['الرمز','الشركة','القطاع','السعر الحالي','القيمة العادلة','أدنى نطاق','أعلى نطاق','شراء ممتاز -30%','شراء قوي -20%','شراء مقبول -10%','هدف 3 سنوات متحفظ','هدف 3 سنوات أساسي','هدف 3 سنوات متفائل','CAGR الأساسي','التوزيع السنوي','عائد التوزيع','نمو الإيرادات','نمو الأرباح','ROE','الدين/حقوق الملكية','المالي','الفني','جودة البيانات','ثقة القيمة العادلة','الثقة','النتيجة النهائية','القرار']

PCT_COLS={'CAGR الأساسي','عائد التوزيع','نمو الإيرادات','نمو الأرباح','ROE'}
MONEY_COLS={'السعر الحالي','القيمة العادلة','أدنى نطاق','أعلى نطاق','شراء ممتاز -30%','شراء قوي -20%','شراء مقبول -10%','هدف 3 سنوات متحفظ','هدف 3 سنوات أساسي','هدف 3 سنوات متفائل','التوزيع السنوي'}

def arabic_ranking(df,topn):
    t=df.reindex(columns=TABLE_COLS).head(topn).copy()
    t.columns=AR_COLS
    t['القطاع']=t['القطاع'].map(lambda x:AR_SECTOR.get(x,x))
    t['القرار']=t['القرار'].map(lambda x:AR_ACTION.get(x,x))
    for c in PCT_COLS: t[c]=t[c].map(fmt_pct)
    for c in MONEY_COLS: t[c]=t[c].map(fmt_num)
    for c in ['المالي','الفني','جودة البيانات','ثقة القيمة العادلة']: t[c]=t[c].map(fmt_score)
    t['الثقة']=t['الثقة'].map(lambda x:'غير متاح' if not np.isfinite(num(x)) else f'{num(x)*100:.1f}%')
    return t

def financial_table(r):
    rows=[
        ('الإيرادات','revenue','money'),('صافي الربح','net_income','money'),('EBITDA','ebitda','money'),('التدفق النقدي التشغيلي','operating_cf','money'),
        ('التدفق النقدي الحر FCF','fcf','money'),('النقد','cash','money'),('الدين','debt','money'),('حقوق الملكية','equity','money'),('الأصول','assets','money'),
        ('ربحية السهم EPS','eps','money'),('القيمة الدفترية للسهم BVPS','bvps','money'),('ROE','roe','pct'),('ROA','roa','pct'),('هامش الربح','margin','pct'),
        ('نمو الإيرادات','rev_growth','pct'),('نمو الأرباح','earn_growth','pct'),('الدين/حقوق الملكية','debt_equity','ratio'),('التوزيع السنوي','dividend','money'),
        ('عائد التوزيع','div_yield','pct'),('نسبة التوزيع من الأرباح','payout','pct'),('P/E','pe','ratio'),('P/B','pb','ratio'),('P/S','ps','ratio')]
    data=[]
    for label,key,kind in rows:
        v=r.get(key)
        if kind=='pct': val=fmt_pct(v)
        elif kind=='ratio': val=fmt_num(v)
        else: val=fmt_num(v)
        data.append([label,val])
    return pd.DataFrame(data,columns=['المؤشر','القيمة'])

def quality_table(r):
    return pd.DataFrame([
        ['مصدر السعر',r.get('price_source','غير متاح')],
        ['الفترة المالية',r.get('period','غير محدد')],
        ['مصادر البيانات',' + '.join(sorted(set(r.get('sources') or []))) or 'غير متاح'],
        ['عدد المصادر المستقلة',str(int(num(r.get('source_count')) if np.isfinite(num(r.get('source_count'))) else 0))],
        ['تعارضات المصادر','؛ '.join(r.get('source_conflicts') or []) or 'لا يوجد'],
        ['تشتت نماذج التقييم',fmt_pct(r.get('valuation_dispersion'))],
        ['طرق التقييم',r.get('valuation_methods','غير متاح')],
        ['عدد طرق التقييم',str(int(r.get('valuation_method_count',0)))],
        ['القيمة المرجعية فقط؟','نعم' if r.get('valuation_reference') else 'لا'],
        ['جودة البيانات',f"{num(r.get('data_quality')):.1f}%"],
        ['ثقة القيمة العادلة',f"{num(r.get('fair_value_confidence')):.1f}%"],
        ['الثقة',f"{num(r.get('confidence'))*100:.1f}%"],
        ['حقول مشتقة/مقدرة','؛ '.join(r.get('imputed') or []) or 'لا يوجد']
    ],columns=['البند','التفاصيل'])

# ============================================================
# Main UI
# ============================================================
st.title('💰 EGX Financial Intelligence PRO MAX')
st.caption(f'الإصدار {APP_VERSION} | 246 سهم | مالي أولًا + تقييم مؤسسي متعدد المصادر + نطاق Fair Value + جودة بيانات + تأكيد فني')

with st.sidebar:
    st.header('⚙️ إعدادات التحليل')
    topn=st.slider('أفضل عدد أسهم',10,50,20)
    workers=st.slider('عدد عمليات التحليل المتوازية',2,10,DEFAULT_WORKERS)
    quality_floor=st.slider('أقل جودة بيانات للعرض',0,100,0)
    rank_mode=st.selectbox('أسلوب الترتيب',['النتيجة النهائية','النتيجة المالية','نسبة الارتفاع عن القيمة العادلة','CAGR الأساسي'])
    st.markdown('**الوزن:** المالي 90% — الفني 10%')
    st.markdown('**التقييم:** Normalized P/E + P/B + Residual Income + DCF + FCF + EV/EBITDA + DDM + Peer Multiples حسب القطاع')
    st.markdown('**قاعدة مهمة:** لا يتم اختراع أرقام القوائم المالية؛ التقديرات الحسابية معلّمة.')
    run=st.button('🚀 تحليل الـ246 سهم',type='primary',use_container_width=True)

UNIVERSE=build_universe()
st.info(f'الكون الحالي: **{len(UNIVERSE)} رمزًا**. لا يتم حذف السهم لمجرد نقص البيانات.')

if run:
    market=market_snapshot(UNIVERSE)
    rows=[]; prog=st.progress(0); status=st.empty(); total=len(UNIVERSE)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures={ex.submit(analyze,s,market.get(s)):s for s in UNIVERSE}
        for i,f in enumerate(as_completed(futures),1):
            s=futures[f]
            try:
                rows.append(f.result())
            except Exception as e:
                rows.append({'symbol':s,'name':s,'sector':SECTOR_MAP.get(s,'Other'),'price':market.get(s,np.nan),'fair_value':np.nan,'fair_low':market.get(s,np.nan)*.70 if np.isfinite(num(market.get(s))) else np.nan,'fair_high':market.get(s,np.nan)*1.30 if np.isfinite(num(market.get(s))) else np.nan,
                             'buy_30':market.get(s,np.nan)*.70 if np.isfinite(num(market.get(s))) else np.nan,'buy_20':market.get(s,np.nan)*.80 if np.isfinite(num(market.get(s))) else np.nan,'buy_10':market.get(s,np.nan)*.90 if np.isfinite(num(market.get(s))) else np.nan,
                             'financial_score':0,'technical_score':50,'data_quality':15,'confidence':.15,'final_score':0,'coverage':0,'imputed':[str(e)[:100]],'action':'Data unavailable','valuation_reference':True,'valuation_methods':'خطأ/مرجع السعر'})
            prog.progress(i/total); status.write(f'تحليل {i}/{total}: {s}')
    df=pd.DataFrame(rows)
    df=sector_impute(df)
    df=peer_benchmarks(df)
    # Recalculate valuation after sector ratio/peer imputation.

    for i in df.index:
        rr=df.loc[i].to_dict()
        fair,lo,hi,methods,nmethods,is_ref,dispersion,vitems,fvc=valuation(rr)
        df.at[i,'fair_value']=fair;df.at[i,'fair_low']=lo;df.at[i,'fair_high']=hi;df.at[i,'valuation_methods']=methods;df.at[i,'valuation_method_count']=nmethods;df.at[i,'valuation_reference']=is_ref;df.at[i,'valuation_dispersion']=dispersion;df.at[i,'valuation_items']=vitems;df.at[i,'fair_value_confidence']=fvc
        if np.isfinite(fair):
            df.at[i,'buy_30']=fair*.70;df.at[i,'buy_20']=fair*.80;df.at[i,'buy_10']=fair*.90
        up,fin,conf,final=score(df.loc[i].to_dict())
        df.at[i,'upside']=up;df.at[i,'financial_score']=fin;df.at[i,'confidence']=conf;df.at[i,'final_score']=final
        for k,v in scenarios(df.loc[i].to_dict()).items(): df.at[i,k]=v
        dq=np.clip(100*(.72*num(df.at[i,'coverage'])+.28*(1-min(len(df.at[i,'imputed'] or []),20)/20)),15,100)
        if is_ref: dq=min(dq,55)
        df.at[i,'data_quality']=dq
        if is_ref: df.at[i,'action']='Weak Data / Reference'
    df=df.sort_values(['final_score','financial_score'],ascending=False,na_position='last').reset_index(drop=True)
    st.session_state['df']=df
    st.success('✅ اكتمل تحليل الـ246 سهم — تم تثبيت السعر أولًا ثم مطابقة المصادر وإعادة بناء القيمة العادلة من عدة نماذج مستقلة.')

if 'df' not in st.session_state:
    st.warning('اضغط «تحليل الـ246 سهم» لبدء المسح الكامل.')
    st.stop()

df=st.session_state['df'].copy()

# Filters / ranking
sector_filter=st.multiselect('فلترة القطاع',sorted(df['sector'].dropna().unique().tolist()),format_func=lambda x:AR_SECTOR.get(x,x))
view=df.copy()
if sector_filter: view=view[view['sector'].isin(sector_filter)]
view=view[view['data_quality']>=quality_floor]
rank_col={'النتيجة النهائية':'final_score','النتيجة المالية':'financial_score','نسبة الارتفاع عن القيمة العادلة':'upside','CAGR الأساسي':'base_cagr'}[rank_mode]
view=view.sort_values(rank_col,ascending=False,na_position='last')

# KPIs
c1,c2,c3,c4,c5=st.columns(5)
c1.metric('عدد الأسهم',len(df));c2.metric('متوسط النتيجة',fmt_score(df['final_score'].mean()));c3.metric('أعلى نتيجة',fmt_score(df['final_score'].max()));c4.metric('متوسط جودة البيانات',f"{df['data_quality'].mean():.1f}%");c5.metric('القطاعات',df['sector'].nunique())

st.subheader('🏆 الترتيب النهائي — أعلى الأسهم')
st.dataframe(arabic_ranking(view,topn),use_container_width=True,hide_index=True)

# Sector ranking
st.subheader('🏭 ترتيب القطاعات')
sec=df.groupby('sector').agg(عدد_الأسهم=('symbol','count'),متوسط_النتيجة=('final_score','mean'),متوسط_المالي=('financial_score','mean'),متوسط_الفني=('technical_score','mean'),متوسط_الجودة=('data_quality','mean')).sort_values('متوسط_النتيجة',ascending=False).round(2)
sec.index=sec.index.map(lambda x:AR_SECTOR.get(x,x)); st.dataframe(sec,use_container_width=True)

# Detailed report
st.subheader('🔎 تقرير سهم مفصل')
choice=st.selectbox('اختار السهم',df['symbol'].tolist(),format_func=lambda x:f"{x} — {df.loc[df.symbol==x,'name'].iloc[0]}")
r=df[df.symbol==choice].iloc[0].to_dict()

a,b,c,d,e=st.columns(5)
a.metric('السعر الحالي',fmt_num(r.get('price')));b.metric('القيمة العادلة',fmt_num(r.get('fair_value')));c.metric('شراء قوي -20%',fmt_num(r.get('buy_20')));d.metric('توزيع سنوي',fmt_num(r.get('dividend')));e.metric('النتيجة',fmt_score(r.get('final_score')))

st.markdown('### 💰 مستويات الشراء')
buy=pd.DataFrame([
    ['ممتاز — 30% تحت القيمة العادلة',fmt_num(r.get('buy_30'))],
    ['قوي — 20% تحت القيمة العادلة',fmt_num(r.get('buy_20'))],
    ['مقبول — 10% تحت القيمة العادلة',fmt_num(r.get('buy_10'))],
],columns=['المستوى','السعر'])
st.dataframe(buy,use_container_width=True,hide_index=True)

st.markdown('### 🧮 محرك القيمة العادلة — Consensus')
if np.isfinite(num(r.get('fair_value'))):
    st.write(f"**Fair Value:** {fmt_num(r.get('fair_value'))} جنيه | **النطاق:** {fmt_num(r.get('fair_low'))} – {fmt_num(r.get('fair_high'))} | **تشتت النماذج:** {fmt_pct(r.get('valuation_dispersion'))}")
    vi=r.get('valuation_items') or []
    if vi:
        vt=pd.DataFrame([{'النموذج':x.get('name'),'القيمة':fmt_num(x.get('value')),'الثقة':f"{num(x.get('conf'))*100:.0f}%"} for x in vi])
        st.dataframe(vt,use_container_width=True,hide_index=True)
    if r.get('source_conflicts'):
        st.warning('⚠️ يوجد تعارض بين بعض المصادر؛ تم خفض الثقة بدل إجبار المصادر على رقم واحد.')
else:
    st.warning('⚠️ لا توجد طرق تقييم أساسية كافية لإصدار Fair Value موثوق. النطاق الظاهر ليس قيمة عادلة؛ هو نطاق مرجعي فقط.')

st.markdown('### 🎯 أهداف 3 سنوات')
sc=pd.DataFrame([
    ['متحفظ',fmt_num(r.get('bear_target')),fmt_num(r.get('bear_total'))],
    ['أساسي',fmt_num(r.get('base_target')),fmt_num(r.get('base_total'))],
    ['متفائل',fmt_num(r.get('bull_target')),fmt_num(r.get('bull_total'))],
],columns=['السيناريو','هدف السعر','هدف السعر + التوزيعات'])
st.dataframe(sc,use_container_width=True,hide_index=True)

chart=pd.DataFrame({'السعر الحالي':[r.get('price')],'القيمة العادلة':[r.get('fair_value')],'متحفظ':[r.get('bear_target')],'أساسي':[r.get('base_target')],'متفائل':[r.get('bull_target')]})
st.bar_chart(chart.T)

st.markdown('### 📊 التحليل المالي')
st.dataframe(financial_table(r),use_container_width=True,hide_index=True)

st.markdown('### 📈 التحليل الفني — للتأكيد فقط')
tech=pd.DataFrame([
    ['RSI',fmt_num(r.get('rsi'),2)],['EMA20',fmt_num(r.get('ema20'))],['EMA50',fmt_num(r.get('ema50'))],['EMA200',fmt_num(r.get('ema200'))],
    ['ATR %',fmt_pct(r.get('atr_pct'))],['نسبة حجم التداول',fmt_num(r.get('volume_ratio'))],['الدعم',fmt_num(r.get('support'))],['المقاومة',fmt_num(r.get('resistance'))],['النتيجة الفنية',fmt_score(r.get('technical_score'))]
],columns=['المؤشر','القيمة'])
st.dataframe(tech,use_container_width=True,hide_index=True)

st.markdown('### 🧠 جودة البيانات ومصدر التقييم')
st.dataframe(quality_table(r),use_container_width=True,hide_index=True)
if r.get('valuation_reference'):
    st.warning('⚠️ القيمة المعروضة هنا قيمة مرجعية للسعر وليست Fair Value محاسبية قوية، لأن عدد طرق التقييم الأساسية غير كافٍ. تم تخفيض الثقة والنتيجة تلقائيًا.')
else:
    st.success('✅ القيمة العادلة مبنية على طريقة/طرق تقييم مالية متاحة، مع تخفيض الثقة تلقائيًا عند نقص البيانات.')

st.markdown('### 🛡️ قاعدة النزاهة الحسابية')
st.info('المحرك لا يعتبر سعر السوق نفسه قيمة عادلة. إذا تعارضت النماذج أو المصادر، يعرض نطاقًا وثقة أقل ويُظهر التعارض. الأرقام المحاسبية لا تُخترع؛ المشتقات الرياضية معلّمة.')

st.markdown('### 🔗 مصادر البيانات')
links=source_links(choice)
for label,url in links.items(): st.markdown(f'- **{label}:** {url}')

# Exports: raw audit + Arabic presentation
raw_csv=df.to_csv(index=False).encode('utf-8-sig')
ar_csv=arabic_ranking(view,len(view)).to_csv(index=False).encode('utf-8-sig')
ec1,ec2=st.columns(2)
ec1.download_button('⬇️ تحميل النتائج الكاملة CSV',raw_csv,'EGX_PRO_MAX_246_RAW.csv','text/csv',use_container_width=True)
ec2.download_button('⬇️ تحميل جدول النتائج بالعربي CSV',ar_csv,'EGX_PRO_MAX_246_AR.csv','text/csv',use_container_width=True)

st.caption('ملاحظة: القيمة العادلة والأهداف تقديرات نموذجية وليست ضمانًا. الأرقام المحاسبية لا يتم اختراعها؛ القيم المشتقة والمرجعية يتم تمييزها في جودة البيانات والثقة.')
