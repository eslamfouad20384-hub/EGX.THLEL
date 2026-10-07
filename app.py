import os
import io
import math
import time
import warnings
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import requests

warnings.filterwarnings('ignore')

# =========================================================
# EGX STOCK INTELLIGENCE PRO MAX v2.2
# Hardened engine: pandas-ME safe + robust indicators + point-in-time backtest
# + train-aware WFO + trade-level Monte Carlo + sector-aware valuation + data quality
# Financial + Technical Multi-Horizon + Valuation + Backtest
# Risk + Stability + Monte Carlo + Data Quality + Fallback Data
# =========================================================

st.set_page_config(page_title='EGX Stock Intelligence PRO MAX', page_icon='📈', layout='wide')

# -----------------------------
# Universe: 248 raw -> 246 unique
# -----------------------------
RAW = '''COMI MFPC PHDC ORAS HDBK EFIH AMES INEG BTFH BIOC CLHO MBSC MTIE EGTS EGSA MHOT EGBE IFAP PRDC MIPH MPCI MOIN ISMQ AXPH PHTV CPCI NINH SPIN ENGC CNFN SVCE KABO OFH GSSC WCDF MFSC SAIB ACGC UEFM KZPC ADCI INFI ASCM VALU ZEOT SMFR ETRS CIRA QNBE EDFM MILS GBCO ACTF SCTS HRHO TMGH FWRY SWDY ETEL AMOC HELI EAST EFID JUFO ABUK ESRS EMFD CCAP ACAP CICH OCDI ORHD MASR AIHC ADIB SAUD CIEB FAIT AFDI CANA EXPA ARCC AJWA MICH SUGR POUL DOMT ISMA UEGC FERC UBEE FAITA MNHD SUCE SMPP ALEX CRST DCRC DIFC MAAL GGRN GGCC IEEC NDRL EFIC GPIM RTVC RUBX PRMH UNIP TWSA ICLE MEGM EASB APSW MOED KWIN KORA RMDA OIH NAPR BONY SPHT SDTI GTWL CFGH NAHO ACAMD NARE CEFM ASPI SCFM CERA DEIN MBEG SIPC NHPS ROTO TYCN RAKT EEII CCRS AREH EPCO FCMD GRCA GIHD ELWA MMAT NEDA EPPK GMCI CPME VLMR GPPL ADPC ADRI AIDC OBRI RREI RKAZ SEIG SNFC TANM UPMS UTOP VERT WKOL LUTS AIFI AMIA AMII ACRO DGTZ DTPP EALR EBSC EGREF EHDR ELNA EOSB FIRE FNAR FTNS GOUR ICID IDRE KASABF KRDI LCSW MOSC TAQA OLFI SKPC AMER TALM ALUM ORWE SPMD ZMID MENA DAPH RAYA EGAL ECAP MPRC AFMC NCCW SCEM ARAB GDWA ELEC IRON ATQA EGCH ALCN MPCO ELSH MEPA ODIN EGAS RACC PRCL BINV EDBM MCQE MOIL NIPH ISPH DSCW AALR UNIT PHAR TRTO CAED CSAG ICFC ELKA PHGC NCGC MCRO ATLC COSG AMPI COPR OCPH RUBX LCSW'''.split()
STOCKS = tuple(dict.fromkeys(x + '.CA' for x in RAW))

BANKS = {'COMI','HDBK','EGBE','SAIB','QNBE','CICH','SAUD','CIEB','ADIB','ABUK','UBEE'}
FINANCIALS = BANKS | {'EFIH','BTFH','VALU','OFH','CNFN','MCQE','ADCI','ACAP','FAIT','AFDI','FAITA','AIFI','AMIA','AMII','ATLC','BINV','DIFC','EFIC'}

# -----------------------------
# Parameters
# -----------------------------
TTL = 900
RFR = 0.18
ERP = 0.08
TERMINAL_G = 0.05
MAX_POSITION_PCT = 0.20
RISK_PER_TRADE_PCT = 0.02
STOP_ATR_MULT = 1.8
TP_R_MULT = 2.5
MIN_TRADE_PRICE = 0.01
COMMISSION = 0.0015
SLIPPAGE = 0.001
TOTAL_COST = COMMISSION + SLIPPAGE
INITIAL_CAPITAL = 100000.0
MC_RUNS_DEFAULT = 500

# -----------------------------
# Generic helpers
# -----------------------------
def num(x, default=np.nan):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return default
        return float(x)
    except Exception:
        return default

def finite(x):
    try:
        return np.isfinite(float(x))
    except Exception:
        return False

def clamp(x, lo, hi):
    if not finite(x): return lo
    return max(lo, min(hi, float(x)))

def fmt(x, digits=2):
    return '—' if not finite(x) else f'{x:,.{digits}f}'

def pct(x, digits=1):
    return '—' if not finite(x) else f'{x*100:.{digits}f}%'

def latest(s):
    s = pd.Series(s).dropna()
    return num(s.iloc[-1]) if len(s) else np.nan

def safe_div(a,b):
    a,b=num(a),num(b)
    return np.nan if not finite(a) or not finite(b) or abs(b)<1e-12 else a/b

def growth(new, old):
    if not finite(new) or not finite(old) or old == 0: return np.nan
    return new/old - 1

def cagr(start, end, years):
    if not finite(start) or not finite(end) or start <= 0 or end <= 0 or years <= 0: return np.nan
    return (end/start)**(1/years)-1

def clean_ohlcv(df):
    if df is None or len(df)==0: return pd.DataFrame()
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    rename = {str(c).title(): c for c in df.columns}
    for needed in ['Open','High','Low','Close','Volume']:
        if needed not in df.columns:
            for c in df.columns:
                if str(c).lower() == needed.lower(): df[needed]=df[c]
    keep=[c for c in ['Open','High','Low','Close','Volume'] if c in df.columns]
    df=df[keep].copy()
    for c in keep: df[c]=pd.to_numeric(df[c], errors='coerce')
    df=df.dropna(subset=['Close'])
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index=pd.to_datetime(df.index, errors='coerce')
    df=df[~df.index.isna()].sort_index()
    return df

# -----------------------------
# Data sources
# -----------------------------
@st.cache_data(ttl=TTL, show_spinner=False)
def yahoo_history(symbol, period='2y', interval='1d'):
    try:
        df=yf.download(symbol, period=period, interval=interval, auto_adjust=False, actions=False, progress=False, threads=False)
        return clean_ohlcv(df)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=TTL, show_spinner=False)
def stooq_history(symbol, years=2):
    # Stooq accepts EGX symbols in many cases as .CA; request is best-effort.
    try:
        sym=symbol.lower()
        end=datetime.utcnow().date()
        start=end-timedelta(days=365*years+30)
        url=f'https://stooq.com/q/d/l/?s={sym}&d1={start:%Y%m%d}&d2={end:%Y%m%d}&i=d'
        r=requests.get(url, timeout=12, headers={'User-Agent':'Mozilla/5.0'})
        if r.ok and 'Date' in r.text:
            df=pd.read_csv(io.StringIO(r.text))
            df['Date']=pd.to_datetime(df['Date'], errors='coerce')
            df=df.set_index('Date').rename(columns={'Close':'Close','Open':'Open','High':'High','Low':'Low','Volume':'Volume'})
            return clean_ohlcv(df)
    except Exception:
        pass
    return pd.DataFrame()

@st.cache_data(ttl=TTL, show_spinner=False)
def alpha_history(symbol, years=2):
    key=os.getenv('ALPHAVANTAGE_API_KEY','').strip()
    if not key: return pd.DataFrame()
    try:
        url='https://www.alphavantage.co/query'
        params={'function':'TIME_SERIES_DAILY_ADJUSTED','symbol':symbol,'outputsize':'full','apikey':key}
        r=requests.get(url, params=params, timeout=15)
        data=r.json()
        ts=data.get('Time Series (Daily)',{})
        if not ts: return pd.DataFrame()
        rows=[]
        for d,v in ts.items():
            rows.append([pd.to_datetime(d), num(v.get('1. open')),num(v.get('2. high')),num(v.get('3. low')),num(v.get('4. close')),num(v.get('6. volume'))])
        df=pd.DataFrame(rows,columns=['Date','Open','High','Low','Close','Volume']).set_index('Date').sort_index()
        cutoff=pd.Timestamp.today()-pd.Timedelta(days=365*years+30)
        return clean_ohlcv(df[df.index>=cutoff])
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=TTL, show_spinner=False)
def get_history(symbol, period='2y'):
    df=yahoo_history(symbol, period=period, interval='1d')
    source='Yahoo'
    if len(df)<60:
        df=stooq_history(symbol, 3 if period in ('2y','3y') else 5)
        source='Stooq' if len(df) else source
    if len(df)<60:
        df=alpha_history(symbol, 3)
        source='Alpha Vantage' if len(df) else source
    return df, source

@st.cache_data(ttl=TTL, show_spinner=False)
def ticker_info(symbol):
    try:
        return yf.Ticker(symbol).info or {}
    except Exception:
        return {}

@st.cache_data(ttl=TTL, show_spinner=False)
def financials(symbol):
    try:
        t=yf.Ticker(symbol)
        inc=t.income_stmt
        bal=t.balance_sheet
        cf=t.cashflow
        return inc if isinstance(inc,pd.DataFrame) else pd.DataFrame(), bal if isinstance(bal,pd.DataFrame) else pd.DataFrame(), cf if isinstance(cf,pd.DataFrame) else pd.DataFrame()
    except Exception:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()

def statement_value(df, names):
    if df is None or df.empty: return np.nan
    for name in names:
        matches=[idx for idx in df.index if str(idx).lower()==name.lower()]
        if not matches:
            matches=[idx for idx in df.index if name.lower() in str(idx).lower()]
        if matches:
            try:
                s=pd.to_numeric(df.loc[matches[0]], errors='coerce').dropna()
                if len(s): return num(s.iloc[0])
            except Exception: pass
    return np.nan

def statement_series(df, names):
    if df is None or df.empty: return pd.Series(dtype=float)
    for name in names:
        matches=[idx for idx in df.index if name.lower() in str(idx).lower()]
        if matches:
            s=pd.to_numeric(df.loc[matches[0]], errors='coerce').dropna()
            return s.sort_index()
    return pd.Series(dtype=float)

# -----------------------------
# Technical indicators
# -----------------------------
def ema(s,n): return pd.Series(s).ewm(span=n, adjust=False, min_periods=n).mean()

def rsi(s,n=14):
    s=pd.Series(s); d=s.diff(); up=d.clip(lower=0); dn=-d.clip(upper=0)
    au=up.ewm(alpha=1/n, adjust=False, min_periods=n).mean(); ad=dn.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs=au/ad.replace(0,np.nan)
    return 100-(100/(1+rs))

def atr(df,n=14):
    h,l,c=df['High'],df['Low'],df['Close']; pc=c.shift(1)
    tr=pd.concat([(h-l),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False,min_periods=n).mean()

def adx(df,n=14):
    h,l,c=df['High'],df['Low'],df['Close']; up=h.diff(); dn=-l.diff()
    plus=np.where((up>dn)&(up>0),up,0.0); minus=np.where((dn>up)&(dn>0),dn,0.0)
    tr=pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atrv=tr.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    pdi=100*pd.Series(plus,index=df.index).ewm(alpha=1/n,adjust=False,min_periods=n).mean()/atrv
    mdi=100*pd.Series(minus,index=df.index).ewm(alpha=1/n,adjust=False,min_periods=n).mean()/atrv
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    return dx.ewm(alpha=1/n,adjust=False,min_periods=n).mean()

def macd(s):
    m=ema(s,12)-ema(s,26); sig=m.ewm(span=9,adjust=False,min_periods=9).mean(); return m,sig

def obv(df):
    d=np.sign(df['Close'].diff()).fillna(0); return (d*df['Volume'].fillna(0)).cumsum()

def mfi(df,n=14):
    tp=(df['High']+df['Low']+df['Close'])/3; mf=tp*df['Volume'].fillna(0)
    pos=mf.where(tp.diff()>0,0).rolling(n).sum(); neg=mf.where(tp.diff()<0,0).rolling(n).sum().abs()
    return 100-(100/(1+pos/neg.replace(0,np.nan)))

def stoch_rsi(s,n=14):
    r=rsi(s,n); lo=r.rolling(n).min(); hi=r.rolling(n).max(); return (r-lo)/(hi-lo).replace(0,np.nan)*100

def vwap(df,n=20):
    tp=(df['High']+df['Low']+df['Close'])/3; vol=df['Volume'].fillna(0)
    return (tp*vol).rolling(n).sum()/vol.rolling(n).sum().replace(0,np.nan)

def roc(s,n=20): return pd.Series(s).pct_change(n)

def resample_ohlcv(df, rule):
    """Safe OHLCV resampling for modern pandas.
    Monthly MUST use ME (month-end); legacy M is intentionally never used.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=['Open','High','Low','Close','Volume'])
    x=clean_ohlcv(df)
    if x.empty:
        return x
    normalized = {'M':'ME', 'BM':'BME', 'Q':'QE'}
    rule = normalized.get(str(rule), str(rule))
    agg={'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}
    out=x.resample(rule).agg(agg)
    return out.dropna(subset=['Close'])

def add_indicators(df):
    x=clean_ohlcv(df)
    if x.empty:
        return x
    x['EMA20']=ema(x.Close,20); x['EMA50']=ema(x.Close,50); x['EMA200']=ema(x.Close,200)
    x['RSI']=rsi(x.Close); x['MACD'],x['MACDSignal']=macd(x.Close)
    x['ATR']=atr(x); x['ADX']=adx(x); x['OBV']=obv(x); x['MFI']=mfi(x); x['StochRSI']=stoch_rsi(x.Close); x['VWAP']=vwap(x)
    vol_ma=x.Volume.rolling(20,min_periods=5).mean()
    x['VolumeRatio']=(x.Volume/vol_ma.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan)
    x['VolumeRatio']=x['VolumeRatio'].fillna(1.0)
    x['ROC20']=roc(x.Close,20)
    x['Support20']=x.Low.rolling(20,min_periods=5).min(); x['Support60']=x.Low.rolling(60,min_periods=10).min()
    x['Resistance20']=x.High.rolling(20,min_periods=5).max(); x['Resistance60']=x.High.rolling(60,min_periods=10).max()
    x=x.replace([np.inf,-np.inf],np.nan)
    return x

# -----------------------------
# Technical scoring / Fibonacci / confirmation
# -----------------------------
def horizon_score(df):
    if df is None or len(df)<40:
        return {'score':np.nan,'trend':'غير كافٍ','rsi':np.nan,'adx':np.nan,'support':np.nan,'resistance':np.nan,'pullback':np.nan,'fib50':np.nan,'fib618':np.nan,'confirmation':0,'atr':np.nan,'volume_ratio':np.nan,'mfi':np.nan,'stochrsi':np.nan,'roc20':np.nan}
    x=add_indicators(df); r=x.iloc[-1]; close=num(r.get('Close'))
    score=0
    score += 10 if finite(r.EMA20) and close>r.EMA20 else 0
    score += 8 if finite(r.EMA50) and close>r.EMA50 else 0
    score += 8 if finite(r.EMA200) and close>r.EMA200 else 0
    score += 4 if finite(r.RSI) and 50<r.RSI<75 else (2 if finite(r.RSI) and r.RSI>=40 else 0)
    score += 4 if finite(r.MACD) and finite(r.MACDSignal) and r.MACD>r.MACDSignal else 0
    score += 4 if finite(r.ADX) and r.ADX>=20 else 0
    score += 3 if finite(r.VolumeRatio) and r.VolumeRatio>=1.0 else 0
    score += 3 if finite(r.OBV) and len(x)>5 and finite(x.OBV.iloc[-5]) and x.OBV.iloc[-1]>x.OBV.iloc[-5] else 0
    score += 3 if finite(r.MFI) and r.MFI>50 else 0
    score += 3 if finite(r.VWAP) and close>r.VWAP else 0
    score=clamp(score,0,50)
    trend='صاعد قوي' if score>=39 else 'صاعد' if score>=30 else 'عرضي' if score>=21 else 'هابط'
    look=x.tail(min(120,len(x))); hi=num(look.High.max()); lo=num(look.Low.min()); rng=hi-lo if finite(hi) and finite(lo) else np.nan
    fib50=hi-rng*.50 if finite(rng) else np.nan; fib618=hi-rng*.618 if finite(rng) else np.nan
    supports=[num(r.Support20),num(r.Support60)]; resistances=[num(r.Resistance20),num(r.Resistance60)]
    support=max([z for z in supports if finite(z)],default=np.nan); resistance=max([z for z in resistances if finite(z)],default=np.nan)
    pullback=safe_div(close-support,support) if finite(support) else np.nan
    confirm=0
    if finite(support) and support*0.94<=close<=support*1.06: confirm+=1
    if finite(r.EMA20) and r.EMA20*.94<=close<=r.EMA20*1.06: confirm+=1
    if finite(fib618) and finite(fib50) and fib618*.97<=close<=fib50*1.03: confirm+=1
    if finite(r.VolumeRatio) and r.VolumeRatio>=1: confirm+=1
    if finite(r.RSI) and r.RSI>=45: confirm+=1
    return {'score':score,'trend':trend,'rsi':num(r.RSI),'adx':num(r.ADX),'support':support,'resistance':resistance,'pullback':pullback,'fib50':fib50,'fib618':fib618,'confirmation':confirm,'atr':num(r.ATR),'volume_ratio':num(r.VolumeRatio),'mfi':num(r.MFI),'stochrsi':num(r.StochRSI),'roc20':num(r.ROC20)}

def multi_horizon(df):
    daily=horizon_score(df)
    weekly=horizon_score(resample_ohlcv(df,'W-FRI'))
    monthly=horizon_score(resample_ohlcv(df,'ME'))
    vals=[v['score'] for v in [daily,weekly,monthly] if finite(v.get('score'))]
    score=float(np.mean(vals)) if vals else np.nan
    alignment=sum(1 for v in [daily,weekly,monthly] if v.get('trend') in ('صاعد','صاعد قوي'))
    return daily,weekly,monthly,score,alignment

# -----------------------------
# Financial extraction
# -----------------------------
def financial_metrics(symbol, price, info):
    inc,bal,cf=financials(symbol)
    revenue=statement_value(inc,['Total Revenue','Operating Revenue','Revenue'])
    net=statement_value(inc,['Net Income','Net Income Common Stockholders','Net Income Applicable To Common Shares'])
    ebitda=statement_value(inc,['EBITDA','Normalized EBITDA'])
    op_income=statement_value(inc,['Operating Income'])
    assets=statement_value(bal,['Total Assets'])
    equity=statement_value(bal,['Stockholders Equity','Total Stockholder Equity','Common Stock Equity'])
    debt=statement_value(bal,['Total Debt','Long Term Debt'])
    cash=statement_value(bal,['Cash And Cash Equivalents','Cash Cash Equivalents And Short Term Investments','Cash Financial'])
    ocf=statement_value(cf,['Operating Cash Flow','Total Cash From Operating Activities'])
    capex=statement_value(cf,['Capital Expenditure','Capital Expenditures'])
    fcf=statement_value(cf,['Free Cash Flow'])
    if not finite(fcf) and finite(ocf): fcf=ocf+(capex if finite(capex) else 0)
    shares=num(info.get('sharesOutstanding'))
    eps=num(info.get('trailingEps'))
    book_ps=num(info.get('bookValue'))
    if not finite(shares) and finite(equity) and finite(book_ps) and book_ps>0: shares=equity/book_ps
    if not finite(eps) and finite(net) and finite(shares) and shares>0: eps=net/shares
    if not finite(book_ps) and finite(equity) and finite(shares) and shares>0: book_ps=equity/shares
    revs=statement_series(inc,['Total Revenue','Operating Revenue','Revenue'])
    nets=statement_series(inc,['Net Income','Net Income Common Stockholders'])
    rev_growth=growth(revenue, revs.iloc[-2]) if len(revs)>=2 else num(info.get('revenueGrowth'))
    profit_growth=growth(net, nets.iloc[-2]) if len(nets)>=2 else num(info.get('earningsGrowth'))
    rev_cagr=cagr(revs.iloc[0],revs.iloc[-1],max(1,len(revs)-1)) if len(revs)>=2 else np.nan
    profit_cagr=cagr(nets.iloc[0],nets.iloc[-1],max(1,len(nets)-1)) if len(nets)>=2 else np.nan
    roe=safe_div(net,equity); roa=safe_div(net,assets); margin=safe_div(net,revenue)
    de=safe_div(debt,equity); current_ratio=num(info.get('currentRatio')); pe=safe_div(price,eps); pb=safe_div(price,book_ps)
    dy=num(info.get('dividendYield'))
    if finite(dy) and dy>1: dy=dy/100
    return {'revenue':revenue,'net_income':net,'ebitda':ebitda,'operating_income':op_income,'assets':assets,'equity':equity,'debt':debt,'cash':cash,'ocf':ocf,'capex':capex,'fcf':fcf,'shares':shares,'eps':eps,'book_ps':book_ps,'revenue_growth':rev_growth,'profit_growth':profit_growth,'revenue_cagr':rev_cagr,'profit_cagr':profit_cagr,'roe':roe,'roa':roa,'margin':margin,'de':de,'current_ratio':current_ratio,'pe':pe,'pb':pb,'dividend_yield':dy,'coverage':0.0,'income_df':inc,'balance_df':bal,'cashflow_df':cf}

def data_quality(m, df, sector='general'):
    """Sector-aware quality: do not punish banks/financials for irrelevant FCF fields."""
    checks=[
        finite(m.get('revenue')), finite(m.get('net_income')), finite(m.get('equity')),
        finite(m.get('assets')), finite(m.get('eps')), finite(m.get('book_ps')), finite(m.get('roe')),
        len(df)>=200
    ]
    if sector not in ('bank','financial'):
        checks.append(finite(m.get('fcf')))
    if finite(m.get('de')):
        checks.append(True)
    return round(sum(checks)/len(checks)*100,2)

# -----------------------------
# Valuation engines
# -----------------------------
def dcf_value(fcf, growth_rate, discount=RFR+ERP, terminal_g=TERMINAL_G, shares=np.nan):
    if not finite(fcf) or fcf<=0 or not finite(shares) or shares<=0: return np.nan
    g=clamp(growth_rate if finite(growth_rate) else .06,-.05,.20)
    r=max(discount,.12); tg=min(terminal_g,r-.03)
    pv=0
    for y in range(1,6):
        cf=fcf*((1+g)**y); pv += cf/((1+r)**y)
    terminal=fcf*((1+g)**5)*(1+tg)/(r-tg)
    pv += terminal/((1+r)**5)
    return pv/shares

def valuation_engine(symbol, price, m):
    ticker=symbol.replace('.CA',''); bank=ticker in BANKS; financial=ticker in FINANCIALS
    sector='bank' if bank else 'financial' if financial else 'general'
    models={}; weights={}
    growth_values=[m.get('revenue_cagr'),m.get('profit_cagr'),m.get('revenue_growth'),m.get('profit_growth')]
    g=float(np.nanmedian([x for x in growth_values if finite(x)])) if any(finite(x) for x in growth_values) else .06
    g=clamp(g,-.05,.18)
    roe=num(m.get('roe')); margin=num(m.get('margin'))
    if finite(m.get('eps')) and m['eps']>0:
        if bank: pe_target=8.0 + clamp((roe-.12)*12 if finite(roe) else 0,-1,4)
        elif financial: pe_target=9.0 + clamp((roe-.12)*10 if finite(roe) else 0,-1,4)
        else: pe_target=9.0 + clamp(g*100*.15,0,6)
        models['P/E']=m['eps']*pe_target; weights['P/E']=0.35 if bank else 0.30
    if finite(m.get('book_ps')) and m['book_ps']>0:
        if bank: pb_target=0.85 + clamp((roe-.08)*2.5 if finite(roe) else 0,-.15,.60)
        elif financial: pb_target=1.15 + clamp((roe-.10)*2.0 if finite(roe) else 0,-.10,.50)
        else: pb_target=1.20 + clamp((roe-.10)*1.5 if finite(roe) else 0,-.10,.50)
        models['P/B']=m['book_ps']*pb_target; weights['P/B']=0.40 if bank else 0.25
    if not financial:
        dcf=dcf_value(m.get('fcf'),g,shares=m.get('shares'))
        if finite(dcf) and dcf>0:
            models['DCF/FCF']=dcf; weights['DCF/FCF']=0.45 if finite(m.get('fcf')) else 0.0
    vals=[(k,v,weights.get(k,0)) for k,v in models.items() if finite(v) and v>0]
    if vals:
        total_w=sum(w for _,_,w in vals)
        if total_w>0: fair=sum(v*w for _,v,w in vals)/total_w
        else: fair=float(np.median([v for _,v,_ in vals]))
    else: fair=np.nan
    # Quality adjustment is deliberately small; valuation models remain dominant.
    adj=1.0
    if finite(roe): adj += clamp((roe-.10)*.10,-.03,.05)
    if finite(margin) and margin<0: adj-=.03
    if finite(m.get('de')) and m['de']>2: adj-=.04
    fair=fair*clamp(adj,.90,1.08) if finite(fair) else np.nan
    conservative=fair*.78 if finite(fair) else np.nan
    optimistic=fair*1.25 if finite(fair) else np.nan
    safe=fair*(1-DEFAULT_MARGIN_OF_SAFETY) if finite(fair) else np.nan
    excellent=fair*.70 if finite(fair) else np.nan
    target=np.nan
    if finite(fair): target=fair*((1+clamp(g,-.03,.18))**3)
    return {'fair':fair,'conservative':conservative,'optimistic':optimistic,'safe_buy':safe,'excellent_buy':excellent,'target3':target,'growth_assumption':g,'models':models,'weights':weights,'sector_type':sector}

# -----------------------------
# Quality / risk / forensic scores
# -----------------------------
def fundamental_score(m, quality):
    s=0
    s += clamp((m.get('roe',0) if finite(m.get('roe')) else 0)*45,0,15)
    s += clamp((m.get('margin',0) if finite(m.get('margin')) else 0)*35,0,8)
    s += clamp((m.get('profit_growth',0) if finite(m.get('profit_growth')) else 0)*20,0,8)
    s += 5 if finite(m.get('current_ratio')) and m['current_ratio']>=1 else 0
    s += 5 if finite(m.get('de')) and m['de']<1 else 0
    s += 4 if finite(m.get('fcf')) and m['fcf']>0 else 0
    s += 5 if finite(m.get('revenue_growth')) and m['revenue_growth']>0 else 0
    s += clamp(quality/100*5,0,5)
    return clamp(s,0,50)

def valuation_score(price,v):
    fair=v.get('fair')
    if not finite(price) or not finite(fair) or fair<=0: return 4
    upside=fair/price-1
    return clamp(7.5+upside*18,0,15)

def technical_score(d,w,mo,alignment):
    vals=[x['score'] for x in [d,w,mo] if finite(x['score'])]
    if not vals: return 0
    base=np.mean(vals)/50*28
    base += alignment/3*4
    base += clamp(d.get('confirmation',0)/5*3,0,3)
    return clamp(base,0,35)

def liquidity_score(df):
    if len(df)<20: return 0
    avg=float(df.Volume.tail(20).mean()) if 'Volume' in df else 0
    vr=num(df.Volume.iloc[-1]/avg) if avg>0 else np.nan
    return clamp(3 + (2 if finite(vr) and vr>=1 else 0) + (1 if avg>50000 else 0),0,6)

def forensic_score(m):
    # A conservative proxy when all detailed forensic fields are unavailable from Yahoo.
    s=50
    if finite(m.get('de')) and m['de']>2: s-=18
    if finite(m.get('profit_growth')) and m['profit_growth']>0.5 and (not finite(m.get('revenue_growth')) or m['revenue_growth']<0): s-=15
    if finite(m.get('roe')) and m['roe']<0: s-=20
    if finite(m.get('fcf')) and m['fcf']<0 and finite(m.get('net_income')) and m['net_income']>0: s-=12
    return clamp(s,0,100)

# -----------------------------
# Backtest: pullback/support confirmation strategy
# -----------------------------
def _empty_bt(reason='بيانات غير كافية'):
    return {'trades':0,'win_rate':np.nan,'return':np.nan,'max_dd':np.nan,'profit_factor':np.nan,'sharpe':np.nan,'sortino':np.nan,'calmar':np.nan,'expectancy':np.nan,'equity':pd.Series(dtype=float),'trade_returns':[],'trade_log':[],'signals':0,'valid_entries':0,'reason':reason}

def _strategy_params(df):
    """Small deterministic parameter search used by WFO training."""
    candidates=[]
    x=add_indicators(df)
    for atr_mult in (1.5,1.8,2.1):
        for tp_mult in (2.0,2.5,3.0):
            for near in (0.05,0.06,0.08):
                candidates.append((atr_mult,tp_mult,near))
    best=(STOP_ATR_MULT,TP_R_MULT,.06); best_score=-1e9
    # Fast proxy on training data: count valid signals and reward trend/momentum quality.
    for atr_mult,tp_mult,near in candidates:
        sub=x.dropna(subset=['EMA20','EMA50','RSI','MACD','MACDSignal','ATR','Support20'])
        if len(sub)<80: continue
        price=sub.Close
        cond=(price>sub.EMA20)&(sub.EMA20>sub.EMA50)&sub.RSI.between(45,72)&(sub.MACD>sub.MACDSignal)&(sub.VolumeRatio>=.8)&(price<=sub.Support20*(1+near))
        n=int(cond.sum())
        if n<3: continue
        # Conservative proxy favors enough signals without overfitting to a tiny count.
        score=n*0.5 + (tp_mult/atr_mult)*2 + min(n,40)*0.05
        if score>best_score: best_score=score; best=(atr_mult,tp_mult,near)
    return best

def run_backtest(df, capital=INITIAL_CAPITAL, warmup=220, params=None):
    if df is None or len(df)<max(warmup+30,260): return _empty_bt()
    x=add_indicators(df.copy())
    x=x.iloc[warmup:].copy() if len(x)>warmup else x
    x=x.dropna(subset=['EMA20','EMA50','RSI','MACD','MACDSignal','ATR','Support20'])
    if len(x)<60: return _empty_bt()
    atr_mult,tp_mult,near=(params or (STOP_ATR_MULT,TP_R_MULT,.06))
    cash=float(capital); shares=0; entry=stop=tp=np.nan; entry_date=None; equity=[]; trades=[]; signals=valid=0
    for idx,r in x.iterrows():
        price=num(r.Close)
        if not finite(price) or price<=MIN_TRADE_PRICE: continue
        # Exit first using the bar close. Costs are applied exactly once per side.
        if shares>0:
            exit_reason=None
            if price<=stop: exit_reason='Stop'
            elif price>=tp: exit_reason='TP'
            if exit_reason:
                exitp=price*(1-TOTAL_COST)
                proceeds=exitp*shares
                pnl=proceeds-entry*shares
                invested=entry*shares
                trade_ret=pnl/invested if invested>0 else np.nan
                cash+=proceeds
                trades.append(pnl); 
                if finite(trade_ret):
                    trades[-1]=pnl
                valid_log={'entry_date':entry_date,'exit_date':idx,'entry':entry,'exit':exitp,'shares':shares,'pnl':pnl,'return':trade_ret,'reason':exit_reason}
                valid_log['r_multiple']=pnl/(max(entry-stop,1e-12)*shares) if shares else np.nan
                # store as separate log below
                if 'trade_log' not in locals(): trade_log=[]
                trade_log.append(valid_log)
                shares=0; entry=stop=tp=np.nan; entry_date=None
        if shares==0:
            support=num(r.Support20); atrv=num(r.ATR)
            signals += 1 if (finite(support) and finite(atrv)) else 0
            trend=finite(r.EMA20) and finite(r.EMA50) and price>r.EMA20>r.EMA50
            momentum=finite(r.RSI) and 45<=r.RSI<=72 and finite(r.MACD) and finite(r.MACDSignal) and r.MACD>r.MACDSignal
            liquidity=finite(r.VolumeRatio) and r.VolumeRatio>=.8
            near_support=finite(support) and price<=support*(1+near)
            confirm=finite(r.RSI) and r.RSI>=45 and (not finite(r.ADX) or r.ADX>=15)
            if trend and momentum and liquidity and near_support and confirm and finite(atrv) and atrv>0:
                risk_per_share=max(atrv*atr_mult,price*.035)
                risk_budget=cash*RISK_PER_TRADE_PCT
                qty_by_risk=math.floor(risk_budget/risk_per_share) if risk_per_share>0 else 0
                qty_by_alloc=math.floor((cash*MAX_POSITION_PCT)/price)
                qty=max(0,min(qty_by_risk,qty_by_alloc))
                if qty>0:
                    valid+=1
                    entry=price*(1+TOTAL_COST); invested=entry*qty
                    if invested<=cash:
                        shares=qty; cash-=invested; entry_date=idx; stop=entry-risk_per_share; tp=entry+risk_per_share*tp_mult
        equity.append((idx,cash+shares*price))
    if shares>0:
        price=num(x.Close.iloc[-1]); exitp=price*(1-TOTAL_COST); proceeds=exitp*shares; pnl=proceeds-entry*shares; invested=entry*shares
        tr=pnl/invested if invested>0 else np.nan
        cash+=proceeds; trades.append(pnl)
        if 'trade_log' not in locals(): trade_log=[]
        trade_log.append({'entry_date':entry_date,'exit_date':x.index[-1],'entry':entry,'exit':exitp,'shares':shares,'pnl':pnl,'return':tr,'reason':'End','r_multiple':pnl/(max(entry-stop,1e-12)*shares)})
        shares=0
    eq=pd.Series(dict(equity)).sort_index()
    if len(eq):
        rets=eq.pct_change().replace([np.inf,-np.inf],np.nan).dropna(); peak=eq.cummax(); dd=eq/peak-1
        maxdd=abs(dd.min()) if len(dd) else np.nan; ret=eq.iloc[-1]/capital-1
        sharpe=(rets.mean()/rets.std()*math.sqrt(252)) if len(rets)>10 and rets.std()>0 else np.nan
        neg=rets[rets<0]; sortino=(rets.mean()/neg.std()*math.sqrt(252)) if len(neg)>5 and neg.std()>0 else np.nan
        calmar=ret/maxdd if finite(maxdd) and maxdd>0 else np.nan
    else: ret=maxdd=sharpe=sortino=calmar=np.nan
    wins=[p for p in trades if p>0]; losses=[p for p in trades if p<0]
    pf=sum(wins)/abs(sum(losses)) if losses else (np.inf if wins else np.nan)
    wr=len(wins)/len(trades) if trades else np.nan
    exp=float(np.mean(trades)) if trades else np.nan
    trade_returns=[num(t.get('return')) for t in locals().get('trade_log',[]) if finite(t.get('return'))]
    return {'trades':len(trades),'win_rate':wr,'return':ret,'max_dd':maxdd,'profit_factor':pf,'sharpe':sharpe,'sortino':sortino,'calmar':calmar,'expectancy':exp,'equity':eq,'trade_returns':trade_returns,'trade_log':locals().get('trade_log',[]),'signals':signals,'valid_entries':valid,'reason':'ok'}

def walk_forward(df):
    if df is None or len(df)<650: return {'wfo_score':np.nan,'oos_return':np.nan,'windows':0,'positive_windows':0,'details':[]}
    n=len(df); train=max(320,int(n*.55)); test=max(80,int(n*.15)); start=0; results=[]
    while start+train+test<=n and len(results)<5:
        train_df=df.iloc[start:start+train]; test_df=df.iloc[start+train:start+train+test]
        params=_strategy_params(train_df)
        # Warm-up comes from the past only; all scored trades belong to the OOS window.
        warmup_tail=train_df.tail(220)
        oos_frame=pd.concat([warmup_tail,test_df]).loc[~pd.concat([warmup_tail,test_df]).index.duplicated(keep='last')]
        bt=run_backtest(oos_frame,warmup=min(220,len(warmup_tail)),params=params)
        if finite(bt.get('return')):
            results.append({'return':bt['return'],'win_rate':bt.get('win_rate'),'trades':bt.get('trades',0),'params':params})
        start+=test
    if not results: return {'wfo_score':np.nan,'oos_return':np.nan,'windows':0,'positive_windows':0,'details':[]}
    rets=[r['return'] for r in results]; positive=sum(r>0 for r in rets)
    avg=float(np.mean(rets)); median=float(np.median(rets)); consistency=positive/len(rets)
    score=clamp(consistency*60 + clamp(avg,-.5,1)*25 + clamp(median,-.5,1)*15,0,100)
    return {'wfo_score':score,'oos_return':avg,'windows':len(results),'positive_windows':positive,'details':results}

def monte_carlo(bt, runs=MC_RUNS_DEFAULT, seed=42):
    trade_returns=[x for x in bt.get('trade_returns',[]) if finite(x) and x>-0.99]
    if len(trade_returns)<5: return {'median':np.nan,'p10':np.nan,'p90':np.nan,'prob_profit':np.nan,'p5_drawdown':np.nan,'ruin_prob':np.nan}
    rng=np.random.default_rng(seed); n=len(trade_returns); finals=[]; maxdds=[]; ruin=0
    arr=np.asarray(trade_returns,float)
    for _ in range(max(100,int(runs))):
        sample=rng.choice(arr,size=n,replace=True)
        wealth=np.cumprod(1+sample); peak=np.maximum.accumulate(wealth); dd=1-wealth/peak
        finals.append(wealth[-1]-1); maxdds.append(dd.max() if len(dd) else 0)
        if wealth[-1] < .70: ruin+=1
    a=np.asarray(finals); d=np.asarray(maxdds)
    return {'median':float(np.median(a)),'p10':float(np.quantile(a,.10)),'p90':float(np.quantile(a,.90)),'prob_profit':float(np.mean(a>0)),'p5_drawdown':float(np.quantile(d,.95)),'ruin_prob':float(ruin/len(a))}

def stability_score(bt,wfo,mc):
    s=0
    if finite(bt.get('win_rate')): s+=clamp(bt['win_rate']*25,0,25)
    if finite(bt.get('profit_factor')): s+=clamp((min(bt['profit_factor'],5)-1)*5,0,20)
    if finite(bt.get('max_dd')): s+=clamp((.25-bt['max_dd'])/.25*20,0,20)
    if finite(wfo.get('wfo_score')): s+=wfo['wfo_score']*.20
    if finite(mc.get('prob_profit')): s+=mc['prob_profit']*15
    return clamp(s,0,100)

# -----------------------------
# Full stock analysis
# -----------------------------
def analyze(symbol, mc_runs=MC_RUNS_DEFAULT):
    ticker=symbol.replace('.CA','').upper()
    df,source=get_history(symbol,'3y')
    if len(df)<60: raise ValueError('بيانات سعرية غير كافية')
    info=ticker_info(symbol)
    close=num(df.Close.iloc[-1]); info_price=num(info.get('currentPrice'))
    price=close if finite(close) else info_price
    if not finite(price) or price<=0: raise ValueError('السعر الحالي غير متاح')
    m=financial_metrics(symbol,price,info)
    sector='bank' if ticker in BANKS else 'financial' if ticker in FINANCIALS else 'general'
    q=data_quality(m,df,sector); m['coverage']=q
    d,w,mo,multi,alignment=multi_horizon(df)
    val=valuation_engine(symbol,price,m)
    bt=run_backtest(df)
    wfo=walk_forward(df)
    mc=monte_carlo(bt,mc_runs)
    stab=stability_score(bt,wfo,mc)
    tech=technical_score(d,w,mo,alignment); fund=fundamental_score(m,q); vals=valuation_score(price,val)
    liq=liquidity_score(df); forensic=forensic_score(m)
    bt_score=0
    if finite(bt.get('return')): bt_score+=clamp((bt['return']+0.20)*10/1.20,0,10)
    if finite(bt.get('sharpe')): bt_score+=clamp(bt['sharpe']*2.5,0,5)
    # Keep the existing 100-point philosophy, but add liquidity/forensic as guardrails rather than hidden unused metrics.
    base=tech/35*28 + fund/50*27 + vals + bt_score + stab/100*5 + q/100*5
    guard=(liq/6*2 if finite(liq) else 0) + (forensic/100*3 if finite(forensic) else 0)
    final=clamp(base+guard,0,100)
    risk='منخفض' if final>=80 and q>=75 and (not finite(bt.get('max_dd')) or bt.get('max_dd')<.25) else 'متوسط' if final>=65 else 'مرتفع'
    fair=val.get('fair'); upside=safe_div(fair,price)-1 if finite(fair) else np.nan
    target=val.get('target3'); cagr3=cagr(price,target,3) if finite(target) and price>0 else np.nan
    target_status='هدف صاعد' if finite(target) and target>price*1.03 else 'قريب من السعر' if finite(target) and target>=price*.97 else 'هدف هبوطي/إعادة تقييم'
    return {'symbol':ticker,'symbol_full':symbol,'price':price,'last_date':df.index[-1].date(),'source':source,'df':df,'info':info,'m':m,'quality':q,'daily':d,'weekly':w,'monthly':mo,'multi':multi,'alignment':alignment,'valuation':val,'backtest':bt,'wfo':wfo,'mc':mc,'stability':stab,'forensic':forensic,'liquidity':liq,'final':final,'risk':risk,'fair_upside':upside,'target3':target,'target3_cagr':cagr3,'target_status':target_status}

# -----------------------------
# Cached bulk scan
# -----------------------------
@st.cache_data(ttl=TTL, show_spinner=False)
def scan_all(symbols, mc_runs=300, workers=8):
    rows=[]; failures=[]
    with ThreadPoolExecutor(max_workers=int(clamp(workers,2,16))) as ex:
        futures={ex.submit(analyze,s,mc_runs):s for s in symbols}
        for f in as_completed(futures):
            s=futures[f]
            try:
                a=f.result(); v=a['valuation']; bt=a['backtest']; wfo=a['wfo']; mc=a['mc']
                rows.append({'السهم':a['symbol'],'السعر':a['price'],'القيمة العادلة':v['fair'],'شراء آمن':v['safe_buy'],'شراء ممتاز':v['excellent_buy'],'هدف 3 سنوات':a['target3'],'حالة الهدف':a['target_status'],'العائد المتوقع 3س':a['target3_cagr'],'فني/35':a['daily']['score'],'توافق الفترات':a['alignment'],'Backtest Trades':bt['trades'],'Signals':bt.get('signals',0),'Valid Entries':bt.get('valid_entries',0),'Win Rate':bt['win_rate'],'Max DD':bt['max_dd'],'PF':bt['profit_factor'],'Sharpe':bt['sharpe'],'Sortino':bt['sortino'],'WFO':wfo['wfo_score'],'OOS':wfo['oos_return'],'MC P(+)':mc['prob_profit'],'MC P95 DD':mc.get('p5_drawdown'),'Stability':a['stability'],'جودة البيانات':a['quality'],'Forensic':a['forensic'],'Liquidity':a['liquidity'],'الدرجة النهائية':a['final'],'المخاطر':a['risk'],'الاتجاه':a['daily']['trend'],'المصدر':a['source'],'آخر شمعة':str(a['last_date'])})
            except Exception as e: failures.append((s.replace('.CA',''),str(e)[:180]))
    df=pd.DataFrame(rows)
    if not df.empty: df=df.sort_values(['الدرجة النهائية','جودة البيانات'],ascending=[False,False]).reset_index(drop=True)
    return df,failures

# -----------------------------
# UI
# -----------------------------
st.title('📈 EGX Stock Intelligence PRO MAX')
st.caption('محرك مؤسسي موحد: مالي + تقييم + فني متعدد الفترات + Backtest + WFO/OOS + Monte Carlo + جودة البيانات + مصادر احتياطية')

with st.sidebar:
    st.header('⚙️ الإعدادات')
    workers=st.slider('عدد العمال',4,16,8)
    top_n=st.slider('أعلى نتائج',5,30,15)
    min_quality=st.slider('أقل جودة بيانات %',0,100,50)
    mc_runs=st.slider('Monte Carlo',100,2000,MC_RUNS_DEFAULT,100)
    st.info(f'الكون: {len(STOCKS)} سهم فريد')
    st.write('المصدر الأساسي: Yahoo Finance')
    st.write('Fallback: Stooq ثم Alpha Vantage عند توفر المفتاح')
    if st.button('🧹 مسح الكاش'):
        st.cache_data.clear(); st.rerun()

c1,c2,c3,c4=st.columns(4)
c1.metric('Universe',len(STOCKS))
c2.metric('العمولة/جانب',f'{COMMISSION*100:.2f}%')
c3.metric('Slippage/جانب',f'{SLIPPAGE*100:.2f}%')
c4.metric('رأس المال الافتراضي',f'{INITIAL_CAPITAL:,.0f} EGP')

st.divider()

if 'scan_df' not in st.session_state: st.session_state.scan_df=pd.DataFrame()
if 'failures' not in st.session_state: st.session_state.failures=[]

col1,col2=st.columns([1,3])
with col1:
    run=st.button(f'🚀 افحص الـ{len(STOCKS)} سهم الآن',type='primary',use_container_width=True)
with col2:
    st.write('التحليل الكامل قد يستغرق وقتًا لأن كل سهم يمر عبر السعر + المؤشرات + القوائم المالية + الباكتيست.')

if run:
    with st.spinner('جاري تشغيل المحرك المؤسسي...'):
        # Rebuild cached function with requested worker count by temporarily using a local executor is unnecessary;
        # global cached scan uses 8 for deterministic performance.
        sdf,fail=scan_all(STOCKS,mc_runs,workers)
        st.session_state.scan_df=sdf; st.session_state.failures=fail

sdf=st.session_state.scan_df
if not sdf.empty:
    st.subheader('🏆 الترتيب النهائي')
    filtered=sdf[sdf['جودة البيانات']>=min_quality].head(top_n).copy()
    display_cols=['السهم','السعر','القيمة العادلة','شراء آمن','شراء ممتاز','هدف 3 سنوات','العائد المتوقع 3س','حالة الهدف','Win Rate','Max DD','PF','Sharpe','WFO','OOS','MC P(+)','Stability','جودة البيانات','الدرجة النهائية','المخاطر','الاتجاه','آخر شمعة']
    st.dataframe(filtered[display_cols],use_container_width=True,hide_index=True)
    st.download_button('⬇️ تحميل النتائج CSV',sdf.to_csv(index=False).encode('utf-8-sig'),'egx_pro_max_results.csv','text/csv')
    st.subheader('📊 أقوى الأسهم حسب الدرجة')
    chart=filtered.set_index('السهم')['الدرجة النهائية']
    st.bar_chart(chart)
    if st.session_state.failures:
        with st.expander(f'⚠️ الأسهم التي فشل تحليلها ({len(st.session_state.failures)})'):
            st.dataframe(pd.DataFrame(st.session_state.failures,columns=['السهم','السبب']),use_container_width=True,hide_index=True)
else:
    st.info('اضغط «افحص الـ246 سهم الآن» لبدء المسح الكامل.')

st.divider()
st.subheader('🔎 تقرير سهم تفصيلي')
selected=st.selectbox('اختر السهم', [s.replace('.CA','') for s in STOCKS])
if st.button('📋 تشغيل التقرير التفصيلي',use_container_width=True):
    with st.spinner(f'تحليل {selected}...'):
        try:
            a=analyze(selected+'.CA',mc_runs)
            st.session_state.detail=a
        except Exception as e:
            st.error(f'تعذر التحليل: {e}')

if 'detail' in st.session_state:
    a=st.session_state.detail; m=a['m']; v=a['valuation']; bt=a['backtest']; wfo=a['wfo']; mc=a['mc']
    st.markdown(f"### {a['symbol']} — التقرير المؤسسي")
    st.caption(f"السعر المستخدم: {fmt(a['price'])} EGP | آخر شمعة: {a['last_date']} | المصدر: {a['source']} | حالة الهدف: {a['target_status']}")
    k=st.columns(6)
    k[0].metric('السعر',fmt(a['price']))
    k[1].metric('القيمة العادلة',fmt(v['fair']))
    k[2].metric('شراء آمن',fmt(v['safe_buy']))
    k[3].metric('شراء ممتاز',fmt(v['excellent_buy']))
    k[4].metric('هدف 3 سنوات',fmt(a['target3']))
    k[5].metric('الدرجة /100',fmt(a['final'],1))

    s1,s2,s3,s4=st.columns(4)
    s1.metric('المخاطر',a['risk']); s2.metric('جودة البيانات',pct(a['quality']/100)); s3.metric('توافق الفترات',f"{a['alignment']}/3"); s4.metric('Stability',fmt(a['stability'],1))

    tabs=st.tabs(['💰 التقييم','🏦 المالي','📈 الفني','🧪 Backtest','🎲 Monte Carlo','🛡️ المخاطر وجودة البيانات'])
    with tabs[0]:
        st.markdown('#### القيمة العادلة والسيناريوهات')
        st.dataframe(pd.DataFrame([
            ['محافظ',v['conservative'],safe_div(v['conservative'],a['price'])-1 if finite(v['conservative']) else np.nan],
            ['أساسي / Fair Value',v['fair'],safe_div(v['fair'],a['price'])-1 if finite(v['fair']) else np.nan],
            ['متفائل',v['optimistic'],safe_div(v['optimistic'],a['price'])-1 if finite(v['optimistic']) else np.nan],
            ['شراء آمن',v['safe_buy'],safe_div(v['safe_buy'],a['price'])-1 if finite(v['safe_buy']) else np.nan],
            ['شراء ممتاز',v['excellent_buy'],safe_div(v['excellent_buy'],a['price'])-1 if finite(v['excellent_buy']) else np.nan],
            ['هدف 3 سنوات',a['target3'],a['target3_cagr']],
            ['حالة الهدف',a['target_status'],np.nan],
        ],columns=['السيناريو','السعر','العائد/الـCAGR']),use_container_width=True,hide_index=True)
        st.write('نماذج التقييم:', ', '.join([f"{k}: {fmt(x)}" for k,x in v['models'].items()]) or 'لا توجد نماذج كافية')
        st.caption(f"افتراض النمو: {pct(v['growth_assumption'])} | معدل الخصم: {pct(RFR+ERP)} | النمو النهائي: {pct(TERMINAL_G)} | نوع التقييم: {v['sector_type']} | الأوزان: {v.get('weights',{})}")

    with tabs[1]:
        fm={k:v for k,v in m.items() if k not in ['income_df','balance_df','cashflow_df']}
        labels={'revenue':'الإيرادات','net_income':'صافي الربح','ebitda':'EBITDA','assets':'الأصول','equity':'حقوق الملكية','debt':'الدين','cash':'النقدية','fcf':'التدفق النقدي الحر','eps':'EPS','book_ps':'القيمة الدفترية/سهم','revenue_growth':'نمو الإيرادات','profit_growth':'نمو الأرباح','revenue_cagr':'CAGR الإيرادات','profit_cagr':'CAGR الأرباح','roe':'ROE','roa':'ROA','margin':'هامش صافي الربح','de':'Debt/Equity','current_ratio':'Current Ratio','pe':'P/E','pb':'P/B','dividend_yield':'عائد التوزيع'}
        rows=[]
        for key,label in labels.items():
            x=fm.get(key)
            if key in ['revenue_growth','profit_growth','revenue_cagr','profit_cagr','roe','roa','margin','dividend_yield']: valtxt=pct(x)
            else: valtxt=fmt(x)
            rows.append([label,valtxt])
        st.dataframe(pd.DataFrame(rows,columns=['المؤشر','القيمة']),use_container_width=True,hide_index=True)
        st.metric('Forensic /100',fmt(a['forensic'],1))

    with tabs[2]:
        st.line_chart(a['df']['Close'])
        rows=[]
        for name,h in [('Daily',a['daily']),('Weekly',a['weekly']),('Monthly',a['monthly'])]:
            rows.append([name,h['trend'],h['score'],h['rsi'],h['adx'],h['support'],h['resistance'],h['confirmation'],h['volume_ratio'],h['mfi'],h['stochrsi']])
        st.dataframe(pd.DataFrame(rows,columns=['الفترة','الاتجاه','Score','RSI','ADX','Support','Resistance','Confirmation','Volume Ratio','MFI','StochRSI']),use_container_width=True,hide_index=True)
        st.info(f"Pullback الحالي: {pct(a['daily']['pullback'])} | Fibonacci 50%: {fmt(a['daily']['fib50'])} | Fibonacci 61.8%: {fmt(a['daily']['fib618'])} | ATR: {fmt(a['daily']['atr'])}")
        st.success('الفلتر الفني: اتجاه صاعد + دعم/EMA/Fibonacci + تحسن ضغط البيع + تأكيد سعري + Pullback قريب من الدعم. VolumeRatio آمن حتى عند غياب/صفر حجم.')

    with tabs[3]:
        bdf=pd.DataFrame([{
            'Trades':bt['trades'],'Win Rate':bt['win_rate'],'Return':bt['return'],'Max DD':bt['max_dd'],'Profit Factor':bt['profit_factor'],'Sharpe':bt['sharpe'],'Sortino':bt['sortino'],'Calmar':bt['calmar'],'Expectancy EGP':bt['expectancy'],'Signals':bt.get('signals',0),'Valid Entries':bt.get('valid_entries',0),'WFO Score':wfo['wfo_score'],'OOS Return':wfo['oos_return']
        }])
        st.dataframe(bdf,use_container_width=True,hide_index=True)
        if len(bt['equity']): st.line_chart(bt['equity'])
        st.caption('الاختبار يستخدم عمولة + انزلاق على جانبي الصفقة، مخاطرة 2% من رأس المال لكل صفقة، حد أقصى للمركز 20%، وقف ATR، وهدف R متعدد؛ والـWFO يختار المعلمات من بيانات التدريب فقط.')

    with tabs[4]:
        c=st.columns(4)
        c[0].metric('Median',pct(mc['median'])); c[1].metric('P10',pct(mc['p10'])); c[2].metric('P90',pct(mc['p90'])); c[3].metric('احتمال الربح',pct(mc['prob_profit']))
        st.caption(f"عدد المحاكاة: {mc_runs}. تعتمد المحاكاة على نتائج الصفقات الفعلية للـBacktest، وليست وعدًا بعائد مستقبلي. P95 DD: {pct(mc.get('p5_drawdown'))} | احتمال هبوط رأس المال >30%: {pct(mc.get('ruin_prob'))}")

    with tabs[5]:
        risk_rows=[['جودة البيانات',a['quality']],['Forensic',a['forensic']],['Stability',a['stability']],['Max Drawdown',bt['max_dd']],['WFO',wfo['wfo_score']],['MC Probability',mc['prob_profit']]]
        st.dataframe(pd.DataFrame(risk_rows,columns=['البند','القيمة']),use_container_width=True,hide_index=True)
        st.warning('هذا محرك تحليلي وليس توصية مضمونة للشراء أو البيع. البيانات المجانية قد تتأخر أو تحتوي على نواقص، لذلك يجب مراجعة القوائم المالية وإفصاحات البورصة قبل القرار.')

st.divider()
st.caption('EGX Stock Intelligence PRO MAX v2.2 — نسخة مطوّرة. الأسعار والقوائم تعتمد على مصادر مجانية متاحة وقد تتعرض لمشاكل Yahoo/Crumb؛ المحرك يستخدم أحدث شمعة متاحة بدل الاعتماد الأعمى على currentPrice.')
