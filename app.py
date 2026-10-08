import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from urllib.parse import quote

# ============================================================
# EGX FINANCIAL INTELLIGENCE PRO MAX 7.0
# Fundamental-first | 246 symbols | Multi-source | No blank valuation
# ============================================================

st.set_page_config(page_title="EGX Financial Intelligence PRO MAX", page_icon="💰", layout="wide")

APP_VERSION = "7.0"
TARGET_UNIVERSE = 246
CACHE_TTL = 1800
DEFAULT_WORKERS = 6
RISK_FREE = 0.18
ERP = 0.08

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
        'coverage':0.0,'imputed':[],'sources':[],'period':'غير محدد','price_source':'غير متاح','div_source':'غير متاح'
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
# Sector-aware valuation
# ============================================================
def sector_pe(sector, roe, growth):
    s=str(sector).lower(); roe=num(roe); growth=num(growth)
    if 'bank' in s: return float(np.clip(6.0+(roe-0.18)*8 if np.isfinite(roe) else 6.5,5.0,10.0))
    if 'financial' in s: return 8.0
    if 'real estate' in s: return 9.0
    if 'telecom' in s or 'utility' in s: return 8.5
    return float(np.clip(9.0+(growth-0.08)*10 if np.isfinite(growth) else 9.5,6.0,15.0))

def valuation(r):
    sector=str(r.get('sector','Other'))
    eps=num(r.get('eps')); bvps=num(r.get('bvps')); roe=num(r.get('roe')); divd=num(r.get('dividend')); fcf=num(r.get('fcf')); shares=num(r.get('shares')); growth=num(r.get('earn_growth')); price=num(r.get('price'))
    candidates=[]; methods=[]; weights=[]
    if np.isfinite(eps) and eps>0:
        pe=sector_pe(sector,roe,growth); candidates.append(eps*pe); methods.append('P/E'); weights.append(1.0)
    if np.isfinite(bvps) and bvps>0:
        s=sector.lower()
        if 'bank' in s:
            rr=roe if np.isfinite(roe) else .18; target_pb=float(np.clip(.70+(rr-.12)*3.5,.60,2.00))
        elif 'financial' in s: target_pb=1.0
        elif 'real estate' in s: target_pb=1.15
        else: target_pb=1.10
        candidates.append(bvps*target_pb); methods.append('P/B'); weights.append(.9)
    if np.isfinite(divd) and divd>0:
        g=float(np.clip(growth if np.isfinite(growth) else .06,0,.10)); ke=RISK_FREE+ERP
        if ke>g:
            candidates.append(divd*(1+g)/(ke-g)); methods.append('Dividend'); weights.append(.7)
    if np.isfinite(fcf) and np.isfinite(shares) and shares>0 and fcf>0:
        fcfps=fcf/shares; mult=float(np.clip(10+(growth if np.isfinite(growth) else .08)*15,8,14))
        candidates.append(fcfps*mult); methods.append('FCF'); weights.append(.8)

    candidates=[num(v) for v in candidates if np.isfinite(num(v)) and num(v)>0]
    if candidates:
        # Median is robust against one bad method; weighted blend when >=2 methods agree.
        fair=float(np.median(candidates)) if len(candidates)>=3 else float(np.average(candidates,weights=weights[:len(candidates)]))
        dispersion=float(np.std(candidates)/fair) if len(candidates)>1 and fair>0 else .20
        dispersion=float(np.clip(dispersion,.08,.30))
        low=fair*(1-dispersion); high=fair*(1+dispersion)
        return fair,low,high,' + '.join(methods),len(candidates),False

    # Reference valuation: not a fabricated accounting fair value. It is deliberately marked weak.
    if np.isfinite(price) and price>0:
        return price,price*.75,price*1.25,'قيمة مرجعية للسعر فقط',0,True
    return np.nan,np.nan,np.nan,'لا توجد بيانات كافية',0,True

# ============================================================
# Sector imputation - only ratios/growth, never accounting totals
# ============================================================
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
    profitability=.60*n(r.get('roe'),.05,.35)+.40*n(r.get('margin'),.02,.30)
    growth=.55*n(r.get('earn_growth'),-.10,.30)+.45*n(r.get('rev_growth'),-.05,.25)
    valuation=n(upside,-.30,.50)
    balance=100-n(r.get('debt_equity'),0,3)
    dividend=n(r.get('div_yield'),0,.08)
    financial=.28*profitability+.22*growth+.28*valuation+.14*balance+.08*dividend
    tech=num(r.get('technical_score')); tech=50 if not np.isfinite(tech) else tech
    combined=.78*financial+.22*tech
    weak=bool(r.get('valuation_reference',False))
    confidence=np.clip(.30+.70*num(r.get('coverage'))-.025*len(r.get('imputed') or []),.15,1.0)
    if weak: confidence=min(confidence,.45)
    final=combined*(.72+.28*confidence)
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
    fair,lo,hi,methods,nmethods,is_ref=valuation(r)
    r.update({'fair_value':fair,'fair_low':lo,'fair_high':hi,'valuation_methods':methods,'valuation_reference':is_ref,'valuation_method_count':nmethods})
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
TABLE_COLS=['symbol','name','sector','price','fair_value','buy_30','buy_20','buy_10','bear_target','base_target','bull_target','base_cagr','dividend','div_yield','rev_growth','earn_growth','roe','debt_equity','financial_score','technical_score','data_quality','confidence','final_score','action']
AR_COLS=['الرمز','الشركة','القطاع','السعر الحالي','القيمة العادلة','شراء ممتاز -30%','شراء قوي -20%','شراء مقبول -10%','هدف 3 سنوات متحفظ','هدف 3 سنوات أساسي','هدف 3 سنوات متفائل','CAGR الأساسي','التوزيع السنوي','عائد التوزيع','نمو الإيرادات','نمو الأرباح','ROE','الدين/حقوق الملكية','المالي','الفني','جودة البيانات','الثقة','النتيجة النهائية','القرار']

PCT_COLS={'CAGR الأساسي','عائد التوزيع','نمو الإيرادات','نمو الأرباح','ROE'}
MONEY_COLS={'السعر الحالي','القيمة العادلة','شراء ممتاز -30%','شراء قوي -20%','شراء مقبول -10%','هدف 3 سنوات متحفظ','هدف 3 سنوات أساسي','هدف 3 سنوات متفائل','التوزيع السنوي'}

def arabic_ranking(df,topn):
    t=df.reindex(columns=TABLE_COLS).head(topn).copy()
    t.columns=AR_COLS
    t['القطاع']=t['القطاع'].map(lambda x:AR_SECTOR.get(x,x))
    t['القرار']=t['القرار'].map(lambda x:AR_ACTION.get(x,x))
    for c in PCT_COLS: t[c]=t[c].map(fmt_pct)
    for c in MONEY_COLS: t[c]=t[c].map(fmt_num)
    for c in ['المالي','الفني','جودة البيانات']: t[c]=t[c].map(fmt_score)
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
        ['مصادر البيانات',' + '.join(r.get('sources') or []) or 'Yahoo / مشتق'],
        ['طرق التقييم',r.get('valuation_methods','غير متاح')],
        ['عدد طرق التقييم',str(int(r.get('valuation_method_count',0)))],
        ['القيمة المرجعية فقط؟','نعم' if r.get('valuation_reference') else 'لا'],
        ['جودة البيانات',f"{num(r.get('data_quality')):.1f}%"],
        ['الثقة',f"{num(r.get('confidence'))*100:.1f}%"],
        ['حقول مشتقة/مقدرة','؛ '.join(r.get('imputed') or []) or 'لا يوجد']
    ],columns=['البند','التفاصيل'])

# ============================================================
# Main UI
# ============================================================
st.title('💰 EGX Financial Intelligence PRO MAX')
st.caption(f'الإصدار {APP_VERSION} | 246 سهم | مالي أولًا + تقييم + جودة بيانات + تأكيد فني')

with st.sidebar:
    st.header('⚙️ إعدادات التحليل')
    topn=st.slider('أفضل عدد أسهم',10,50,20)
    workers=st.slider('عدد عمليات التحليل المتوازية',2,10,DEFAULT_WORKERS)
    quality_floor=st.slider('أقل جودة بيانات للعرض',0,100,0)
    rank_mode=st.selectbox('أسلوب الترتيب',['النتيجة النهائية','النتيجة المالية','نسبة الارتفاع عن القيمة العادلة','CAGR الأساسي'])
    st.markdown('**الوزن:** المالي 78% — الفني 22%')
    st.markdown('**التقييم:** P/E + P/B + توزيعات + FCF حسب توافر البيانات')
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
                rows.append({'symbol':s,'name':s,'sector':SECTOR_MAP.get(s,'Other'),'price':market.get(s,np.nan),'fair_value':market.get(s,np.nan),'fair_low':np.nan,'fair_high':np.nan,
                             'buy_30':market.get(s,np.nan)*.70 if np.isfinite(num(market.get(s))) else np.nan,'buy_20':market.get(s,np.nan)*.80 if np.isfinite(num(market.get(s))) else np.nan,'buy_10':market.get(s,np.nan)*.90 if np.isfinite(num(market.get(s))) else np.nan,
                             'financial_score':0,'technical_score':50,'data_quality':15,'confidence':.15,'final_score':0,'coverage':0,'imputed':[str(e)[:100]],'action':'Data unavailable','valuation_reference':True,'valuation_methods':'خطأ/مرجع السعر'})
            prog.progress(i/total); status.write(f'تحليل {i}/{total}: {s}')
    df=pd.DataFrame(rows)
    df=sector_impute(df)
    # Recalculate valuation after sector ratio imputation; then guarantee buy levels are populated when a price exists.
    for i in df.index:
        rr=df.loc[i].to_dict()
        fair,lo,hi,methods,nmethods,is_ref=valuation(rr)
        df.at[i,'fair_value']=fair;df.at[i,'fair_low']=lo;df.at[i,'fair_high']=hi;df.at[i,'valuation_methods']=methods;df.at[i,'valuation_method_count']=nmethods;df.at[i,'valuation_reference']=is_ref
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
    st.success('✅ اكتمل تحليل الـ246 سهم — وتمت إعادة حساب القيمة العادلة ومستويات الشراء بعد تثبيت السعر.')

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
