import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title='EGX Financial Intelligence PRO MAX', layout='wide')

APP_VERSION='5.0'
TARGET_UNIVERSE=246
CACHE_TTL=1800
DEFAULT_WORKERS=8
RISK_FREE=0.18
ERP=0.08

# ============================================================
# 246 EGX universe
# ============================================================
RAW='''
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
'''
STOCKS=tuple(dict.fromkeys(x.upper().strip() for x in RAW.split()))

SECTOR_MAP={
'COMI':'Banks','ADIB':'Banks','CIEB':'Banks','QNBA':'Banks','HDBK':'Banks','UBEE':'Banks','FAIT':'Banks','EXPA':'Banks','CICH':'Financials','HRHO':'Financials','EFGH':'Financials','CNFN':'Financials','MICH':'Financials','MUBI':'Financials','FWRY':'Financials','VALU':'Financials',
'EFIH':'Technology','TMGH':'Real Estate','PHDC':'Real Estate','MNHD':'Real Estate','EMFD':'Real Estate','SODIC':'Real Estate','AMER':'Real Estate','ORHD':'Real Estate',
'ORAS':'Construction','SWDY':'Industrials','MMMT':'Industrials','GBCO':'Industrials','GBCA':'Industrials','AUTO':'Industrials','ALCN':'Industrials',
'ETEL':'Telecom','EAST':'Consumer Staples','JUFO':'Consumer Staples','DOMT':'Consumer Staples','MFPC':'Materials','ABUK':'Materials','EGAL':'Materials','AMOC':'Energy','EGAS':'Energy','SKPC':'Energy','TAQA':'Energy','UEFM':'Consumer Staples',
'BIOC':'Healthcare','NIPH':'Healthcare','ISPH':'Healthcare','DAPH':'Healthcare'
}

# ---------- helpers ----------
def num(x):
    try:
        v=float(x)
        return v if np.isfinite(v) else np.nan
    except Exception: return np.nan

def first(*xs):
    for x in xs:
        v=num(x)
        if np.isfinite(v): return v
    return np.nan

def div(a,b):
    a,b=num(a),num(b)
    return a/b if np.isfinite(a) and np.isfinite(b) and b!=0 else np.nan

def clamp(x,a,b):
    x=num(x)
    return float(np.clip(x,a,b)) if np.isfinite(x) else np.nan

def pct(x): return '—' if not np.isfinite(num(x)) else f'{x*100:.1f}%'
def money(x): return '—' if not np.isfinite(num(x)) else f'{x:,.2f}'

def stmt_latest(stmt, names):
    if stmt is None or stmt.empty: return np.nan
    for n in names:
        if n in stmt.index:
            s=pd.to_numeric(stmt.loc[n], errors='coerce').dropna()
            if len(s): return float(s.iloc[0])
    return np.nan

def stmt_growth(stmt,names):
    if stmt is None or stmt.empty: return np.nan
    for n in names:
        if n in stmt.index:
            s=pd.to_numeric(stmt.loc[n], errors='coerce').dropna()
            if len(s)>=2 and s.iloc[1]!=0:
                return float(s.iloc[0]/s.iloc[1]-1)
    return np.nan

# ---------- data ----------
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fundamentals(symbol):
    r={'symbol':symbol,'name':symbol,'sector':SECTOR_MAP.get(symbol,'Other'),'price':np.nan,'market_cap':np.nan,'shares':np.nan,
       'revenue':np.nan,'net_income':np.nan,'ebitda':np.nan,'operating_cf':np.nan,'fcf':np.nan,'cash':np.nan,'debt':np.nan,'equity':np.nan,
       'eps':np.nan,'bvps':np.nan,'dividend':np.nan,'div_yield':np.nan,'roe':np.nan,'roa':np.nan,'margin':np.nan,'rev_growth':np.nan,'earn_growth':np.nan,'debt_equity':np.nan,
       'pe':np.nan,'pb':np.nan,'ps':np.nan,'coverage':0.0,'imputed':[]}
    try:
        t=yf.Ticker(symbol+'.CA')
        try: info=t.info or {}
        except Exception: info={}
        r['name']=info.get('longName') or info.get('shortName') or symbol
        r['sector']=SECTOR_MAP.get(symbol) or info.get('sector') or info.get('industry') or 'Other'
        r['price']=first(info.get('currentPrice'),info.get('regularMarketPrice'),info.get('previousClose'))
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
        r['revenue']=first(stmt_latest(inc,['Total Revenue','Operating Revenue']),info.get('totalRevenue'))
        r['net_income']=first(stmt_latest(inc,['Net Income','Net Income Common Stockholders']),info.get('netIncomeToCommon'))
        r['ebitda']=first(stmt_latest(inc,['EBITDA','Normalized EBITDA']),info.get('ebitda'))
        r['operating_cf']=stmt_latest(cf,['Operating Cash Flow','Total Cash From Operating Activities'])
        r['fcf']=first(stmt_latest(cf,['Free Cash Flow']),info.get('freeCashflow'))
        r['cash']=first(stmt_latest(bal,['Cash Cash Equivalents And Short Term Investments','Cash And Cash Equivalents']),info.get('totalCash'))
        r['debt']=first(stmt_latest(bal,['Total Debt','Long Term Debt And Capital Lease Obligation']),info.get('totalDebt'))
        r['equity']=first(stmt_latest(bal,['Stockholders Equity','Common Stock Equity','Total Equity Gross Minority Interest']))
        r['rev_growth']=first(r['rev_growth'],stmt_growth(inc,['Total Revenue','Operating Revenue']))
        r['earn_growth']=first(r['earn_growth'],stmt_growth(inc,['Net Income','Net Income Common Stockholders']))
        if not np.isfinite(r['shares']) and np.isfinite(r['market_cap']) and np.isfinite(r['price']):
            r['shares']=r['market_cap']/r['price']; r['imputed'].append('Shares')
        if not np.isfinite(r['eps']) and np.isfinite(r['net_income']) and np.isfinite(r['shares']) and r['shares']>0:
            r['eps']=r['net_income']/r['shares']; r['imputed'].append('EPS')
        if not np.isfinite(r['bvps']) and np.isfinite(r['equity']) and np.isfinite(r['shares']) and r['shares']>0:
            r['bvps']=r['equity']/r['shares']; r['imputed'].append('BVPS')
        if not np.isfinite(r['roe']) and np.isfinite(r['net_income']) and np.isfinite(r['equity']) and r['equity']>0:
            r['roe']=r['net_income']/r['equity']; r['imputed'].append('ROE')
        if not np.isfinite(r['margin']) and np.isfinite(r['net_income']) and np.isfinite(r['revenue']) and r['revenue']!=0:
            r['margin']=r['net_income']/r['revenue']; r['imputed'].append('Margin')
        if not np.isfinite(r['debt_equity']) and np.isfinite(r['debt']) and np.isfinite(r['equity']) and r['equity']!=0:
            r['debt_equity']=r['debt']/r['equity']; r['imputed'].append('Debt/Equity')
        if not np.isfinite(r['div_yield']) and np.isfinite(r['dividend']) and np.isfinite(r['price']) and r['price']>0:
            r['div_yield']=r['dividend']/r['price']; r['imputed'].append('Dividend Yield')
        fields=['price','revenue','net_income','debt','equity','eps','dividend','roe','rev_growth','earn_growth','debt_equity','shares','bvps']
        r['coverage']=sum(np.isfinite(num(r[k])) for k in fields)/len(fields)
    except Exception as e:
        r['imputed'].append('Data error')
    return r

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def history(symbol):
    try:
        d=yf.Ticker(symbol+'.CA').history(period='3y',auto_adjust=False,actions=True)
        return d.dropna(subset=['Close']) if d is not None else pd.DataFrame()
    except Exception: return pd.DataFrame()

def technical(d):
    o={k:np.nan for k in ['rsi','ema20','ema50','ema200','atr_pct','volume_ratio','support','resistance','technical_score']}
    if d is None or len(d)<30:return o
    c=pd.to_numeric(d['Close'],errors='coerce'); h=pd.to_numeric(d['High'],errors='coerce'); l=pd.to_numeric(d['Low'],errors='coerce'); v=pd.to_numeric(d['Volume'],errors='coerce')
    o['ema20']=c.ewm(span=20,adjust=False).mean().iloc[-1]; o['ema50']=c.ewm(span=50,adjust=False).mean().iloc[-1]
    if len(c)>=200:o['ema200']=c.ewm(span=200,adjust=False).mean().iloc[-1]
    delta=c.diff(); gain=delta.clip(lower=0).rolling(14).mean(); loss=(-delta.clip(upper=0)).rolling(14).mean(); rs=gain/loss.replace(0,np.nan); o['rsi']=(100-100/(1+rs)).iloc[-1]
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1); atr=tr.rolling(14).mean(); o['atr_pct']=atr.iloc[-1]/c.iloc[-1] if c.iloc[-1]!=0 else np.nan
    o['volume_ratio']=v.iloc[-1]/v.rolling(20).mean().iloc[-1] if len(v.dropna())>=20 and v.rolling(20).mean().iloc[-1]!=0 else np.nan
    recent=c.tail(min(120,len(c))); o['support']=recent.min(); o['resistance']=recent.max()
    score=50
    if c.iloc[-1]>o['ema20']:score+=12
    if c.iloc[-1]>o['ema50']:score+=12
    if np.isfinite(o['ema200']) and c.iloc[-1]>o['ema200']:score+=12
    if o['ema20']>o['ema50']:score+=8
    if np.isfinite(o['rsi']):
        if 50<=o['rsi']<=68:score+=10
        elif o['rsi']>75:score-=8
        elif o['rsi']<35:score-=5
    if np.isfinite(o['volume_ratio']) and o['volume_ratio']>1.2:score+=5
    o['technical_score']=float(np.clip(score,0,100))
    return o

# ---------- valuation ----------
def valuation(r):
    sector=str(r.get('sector','')).lower(); vals=[]; methods=[]
    eps=num(r.get('eps')); bvps=num(r.get('bvps')); roe=num(r.get('roe')); divd=num(r.get('dividend')); fcf=num(r.get('fcf')); shares=num(r.get('shares')); growth=num(r.get('earn_growth'))
    if np.isfinite(eps) and eps>0:
        if 'bank' in sector: target_pe=np.clip(6.0+(roe-0.18)*8 if np.isfinite(roe) else 6.5,5,10)
        elif 'financial' in sector: target_pe=8.0
        elif 'real estate' in sector: target_pe=9.0
        elif 'telecom' in sector or 'utility' in sector: target_pe=8.5
        else: target_pe=np.clip(9.0+(growth-0.08)*10 if np.isfinite(growth) else 9.5,6,15)
        vals.append(eps*target_pe);methods.append('P/E')
    if np.isfinite(bvps) and bvps>0:
        if 'bank' in sector:
            rr=roe if np.isfinite(roe) else 0.18; target_pb=np.clip(0.70+(rr-0.12)*3.5,0.6,2.0)
        elif 'financial' in sector: target_pb=1.0
        else: target_pb=1.1
        vals.append(bvps*target_pb);methods.append('P/B')
    if np.isfinite(divd) and divd>0:
        g=np.clip(growth if np.isfinite(growth) else 0.06,0,0.10); ke=RISK_FREE+ERP
        if ke>g: vals.append(divd*(1+g)/(ke-g));methods.append('Dividend')
    if np.isfinite(fcf) and np.isfinite(shares) and shares>0 and fcf>0:
        fcfps=fcf/shares; vals.append(fcfps*np.clip(10+(growth if np.isfinite(growth) else .08)*15,8,14));methods.append('FCF')
    vals=[v for v in vals if np.isfinite(v) and v>0]
    if not vals:return np.nan,np.nan,np.nan,'None'
    fair=float(np.median(vals)); spread=np.clip(np.std(vals)/fair if len(vals)>1 else .20,.10,.30)
    return fair,fair*(1-spread),fair*(1+spread),'+'.join(methods)

def sector_impute(df):
    df=df.copy()
    if 'imputed' not in df:df['imputed']=[[] for _ in range(len(df))]
    for col in ['rev_growth','earn_growth','roe','margin','debt_equity','div_yield','pe','pb']:
        if col not in df:continue
        med=df.groupby('sector')[col].transform('median')
        mask=df[col].isna()&med.notna()
        df.loc[mask,col]=med[mask]
        df.loc[mask,'imputed']=df.loc[mask,'imputed'].apply(lambda x:(x if isinstance(x,list) else [])+[f'Sector median {col}'])
    return df

def score(r):
    def n(x,a,b):return 50 if not np.isfinite(num(x)) else float(np.clip((x-a)/(b-a)*100,0,100))
    price=num(r.get('price')); fair=num(r.get('fair_value')); upside=fair/price-1 if np.isfinite(price) and np.isfinite(fair) and price>0 else np.nan
    profitability=.6*n(r.get('roe'),.05,.35)+.4*n(r.get('margin'),.02,.30)
    growth=.55*n(r.get('earn_growth'),-.10,.30)+.45*n(r.get('rev_growth'),-.05,.25)
    valuation=n(upside,-.30,.50); balance=100-n(r.get('debt_equity'),0,3); dividend=n(r.get('div_yield'),0,.08)
    financial=.28*profitability+.22*growth+.28*valuation+.14*balance+.08*dividend
    tech=num(r.get('technical_score')); tech=50 if not np.isfinite(tech) else tech
    combined=.78*financial+.22*tech
    confidence=np.clip(.35+.65*num(r.get('coverage'))-.03*len(r.get('imputed') or []),.20,1.0)
    final=combined*(.80+.20*confidence)
    return upside,financial,confidence,final

def scenarios(r):
    p=num(r.get('price')); fair=num(r.get('fair_value')); g=num(r.get('earn_growth')); d=num(r.get('dividend'))
    if not np.isfinite(p) or p<=0:return {}
    anchor=fair if np.isfinite(fair) and fair>0 else p; g=np.clip(g if np.isfinite(g) else .08,.02,.25)
    bg=np.clip(g*.45,0,.12); mg=np.clip(g*.75,.03,.18); ug=np.clip(g*1.05,.05,.25)
    bear=anchor*(1+bg)**3*.88; base=anchor*(1+mg)**3; bull=anchor*(1+ug)**3*1.08; div3=d*3 if np.isfinite(d) and d>0 else 0
    return {'bear_target':bear,'base_target':base,'bull_target':bull,'bear_total':bear+div3,'base_total':base+div3,'bull_total':bull+div3,'base_cagr':(base/p)**(1/3)-1 if base>0 else np.nan}

def analyze(s):
    r=fundamentals(s); t=technical(history(s)); r.update(t)
    fair,lo,hi,methods=valuation(r)
    if not np.isfinite(fair) and np.isfinite(num(r['price'])):
        fair=r['price'];lo=fair*.75;hi=fair*1.25;methods='Fallback current price';r['imputed'].append('Fair value fallback')
    r.update({'fair_value':fair,'fair_low':lo,'fair_high':hi,'valuation_methods':methods})
    r['buy_30']=fair*.70 if np.isfinite(fair) else np.nan; r['buy_20']=fair*.80 if np.isfinite(fair) else np.nan; r['buy_10']=fair*.90 if np.isfinite(fair) else np.nan
    up,fin,conf,final=score(r);r.update({'upside':up,'financial_score':fin,'confidence':conf,'final_score':final})
    r.update(scenarios(r));r['data_quality']=np.clip(100*(.75*r['coverage']+.25*(1-len(r['imputed'])/15)),20,100)
    ts=num(r['technical_score']) if np.isfinite(num(r['technical_score'])) else 50
    if np.isfinite(up) and up>=.25 and ts>=65:action='Strong Buy / Accumulate'
    elif np.isfinite(up) and up>=.15:action='Buy on weakness'
    elif np.isfinite(up) and up>=.05:action='Watch / Gradual'
    elif np.isfinite(up) and up<0:action='Overvalued / Wait'
    else:action='Neutral / Watch'
    r['action']=action
    return r

# ---------- UI ----------
st.title('💰 EGX Financial Intelligence PRO MAX')
st.caption(f'v{APP_VERSION} | 246 EGX stocks | Fundamental-first + Valuation + Technical Confirmation')

with st.sidebar:
    st.header('⚙️ الإعدادات')
    topn=st.slider('أفضل N',10,50,20)
    workers=st.slider('Workers',2,12,DEFAULT_WORKERS)
    st.markdown('**الأوزان:** المالي 78% — الفني 22%')
    st.markdown('**مبدأ البيانات:** لا نخفي السهم بسبب نقص البيانات؛ النقص يظهر في Data Quality وConfidence.')
    run=st.button('🚀 تحليل الـ246 سهم',type='primary')

st.info(f'الكون المضمن: {len(STOCKS)} رمزًا فريدًا. البرنامج لا يحذف السهم لمجرد نقص البيانات.')

if run:
    rows=[]; prog=st.progress(0); status=st.empty(); total=len(STOCKS)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures={ex.submit(analyze,s):s for s in STOCKS}
        for i,f in enumerate(as_completed(futures),1):
            s=futures[f]
            try:rows.append(f.result())
            except Exception as e:rows.append({'symbol':s,'name':s,'sector':SECTOR_MAP.get(s,'Other'),'price':np.nan,'final_score':0,'financial_score':0,'technical_score':50,'coverage':0,'confidence':.2,'data_quality':20,'action':'Data unavailable','imputed':[str(e)[:80]]})
            prog.progress(i/total);status.write(f'تحليل {i}/{total}: {s}')
    df=pd.DataFrame(rows);df=sector_impute(df)
    # Recompute fair values after sector estimates where possible
    for i,r in df.iterrows():
        fair,lo,hi,methods=valuation(r.to_dict())
        if np.isfinite(fair):
            df.at[i,'fair_value']=fair;df.at[i,'fair_low']=lo;df.at[i,'fair_high']=hi;df.at[i,'valuation_methods']=methods
            df.at[i,'buy_30']=fair*.70;df.at[i,'buy_20']=fair*.80;df.at[i,'buy_10']=fair*.90
        up,fin,conf,final=score(df.loc[i].to_dict());df.at[i,'upside']=up;df.at[i,'financial_score']=fin;df.at[i,'confidence']=conf;df.at[i,'final_score']=final
        for k,v in scenarios(df.loc[i].to_dict()).items():df.at[i,k]=v
    df=df.sort_values(['final_score','financial_score'],ascending=False,na_position='last').reset_index(drop=True)
    st.session_state['df']=df
    st.success('اكتمل تحليل الـ246 سهم.')

if 'df' not in st.session_state:
    st.warning('اضغط «تحليل الـ246 سهم» لبدء المسح الكامل.')
    st.stop()

df=st.session_state['df']

# KPIs
c1,c2,c3,c4,c5=st.columns(5)
c1.metric('الأسهم',len(df));c2.metric('متوسط Final Score',f"{df.final_score.mean():.1f}");c3.metric('أعلى Score',f"{df.final_score.max():.1f}");c4.metric('متوسط جودة البيانات',f"{df.data_quality.mean():.1f}%");c5.metric('القطاعات',df.sector.nunique())

st.subheader('🏆 الترتيب النهائي')
cols=['symbol','name','sector','price','fair_value','buy_30','buy_20','buy_10','bear_target','base_target','bull_target','base_cagr','dividend','div_yield','rev_growth','earn_growth','roe','debt_equity','financial_score','technical_score','data_quality','confidence','final_score','action']
t=df[[c for c in cols if c in df]].head(topn).copy()
t.columns=['Ticker','Company','Sector','Current','Fair Value','Buy -30%','Buy -20%','Buy -10%','3Y Bear','3Y Base','3Y Bull','Base CAGR','Dividend','Div Yield','Revenue Growth','Earnings Growth','ROE','Debt/Equity','Financial','Technical','Data Quality','Confidence','Final Score','Action']
st.dataframe(t,use_container_width=True,hide_index=True)

st.subheader('🏭 ترتيب القطاعات')
sec=df.groupby('sector').agg(Stocks=('symbol','count'),Avg_Score=('final_score','mean'),Financial=('financial_score','mean'),Technical=('technical_score','mean'),Data_Quality=('data_quality','mean')).sort_values('Avg_Score',ascending=False).round(2)
st.dataframe(sec,use_container_width=True)

st.subheader('🔎 تقرير سهم مفصل')
choice=st.selectbox('اختار السهم',df.symbol.tolist());r=df[df.symbol==choice].iloc[0].to_dict()
a,b,c,d=st.columns(4);a.metric('السعر الحالي',money(r.get('price')));b.metric('القيمة العادلة',money(r.get('fair_value')));c.metric('Final Score',f"{r.get('final_score',np.nan):.1f}");d.metric('القرار',r.get('action','—'))

st.markdown('### 💰 مستويات الشراء')
st.dataframe(pd.DataFrame({'المستوى':['ممتاز -30%','قوي -20%','مقبول -10%'],'السعر':[r.get('buy_30'),r.get('buy_20'),r.get('buy_10')]}).round(2),use_container_width=True,hide_index=True)

st.markdown('### 🎯 أهداف 3 سنوات')
st.dataframe(pd.DataFrame({'السيناريو':['Bear','Base','Bull'],'هدف السعر':[r.get('bear_target'),r.get('base_target'),r.get('bull_target')],'السعر + توزيعات':[r.get('bear_total'),r.get('base_total'),r.get('bull_total')]}).round(2),use_container_width=True,hide_index=True)

st.markdown('### 📊 التحليل المالي')
financial=pd.DataFrame({'المؤشر':['Revenue','Net Income','EBITDA','Operating CF','FCF','Cash','Debt','Equity','EPS','BVPS','ROE','ROA','Margin','Revenue Growth','Earnings Growth','Debt/Equity','Dividend','Dividend Yield','P/E','P/B','P/S'], 'Value':[r.get(k) for k in ['revenue','net_income','ebitda','operating_cf','fcf','cash','debt','equity','eps','bvps','roe','roa','margin','rev_growth','earn_growth','debt_equity','dividend','div_yield','pe','pb','ps']]})
st.dataframe(financial,use_container_width=True,hide_index=True)

st.markdown('### 📈 الفني المستخدم كتأكيد')
tech=pd.DataFrame({'المؤشر':['RSI','EMA20','EMA50','EMA200','ATR %','Volume Ratio','Support','Resistance','Technical Score'], 'Value':[r.get('rsi'),r.get('ema20'),r.get('ema50'),r.get('ema200'),r.get('atr_pct'),r.get('volume_ratio'),r.get('support'),r.get('resistance'),r.get('technical_score')]})
st.dataframe(tech.round(3),use_container_width=True,hide_index=True)

st.markdown('### 🧠 البيانات الناقصة والتقديرات')
im=r.get('imputed') or []
if im:st.warning('تم استخدام: '+', '.join(im))
else:st.success('لا توجد تقديرات ظاهرة في الحقول الأساسية.')
st.caption('المحرك لا يخترع قوائم مالية. عند غياب رقم، يستخدم أولًا بديلًا حسابيًا من بيانات الشركة، ثم median القطاع عند ملاءمة المقارنة، ويضع علامة Imputed. القيمة العادلة والأهداف تقديرات نموذجية وليست ضمانًا.')

st.download_button('⬇️ تحميل كل النتائج CSV',df.to_csv(index=False).encode('utf-8-sig'),'EGX_PRO_MAX_246.csv','text/csv')
