import re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title='EGX Financial Intelligence PRO MAX', page_icon='📊', layout='wide', initial_sidebar_state='collapsed')

# ============================================================
# 246 EGX UNIVERSE
# 229 actively traded symbols + 17 newer/less-covered EGX listings.
# The universe is NEVER reduced because financial data is missing.
# ============================================================
BASE_229 = '''COMI ETEL SWDY TMGH EGAL MFPC QNBE HDBK ABUK ALCN ORAS EFIH EAST ADIB EMFD FWRY SCTS CANA ORHD GPPL VLMRA VLMR EFID HRHO JUFO PHDC OCDI GBCO FAIT FAITA FERC CIEB HELI BTFH IRON EXPA RAYA BIOC ARCC EGCH CCAP VALU CIRA TAQA SCEM CLHO MBSC PHAR MCQE AMOC SKPC MTIE ORWE POUL UBEE EGSA EGTS MOIL MASR SAUD EFIC TALM NIPH ATQA EGBE KORA MHOT BINV CICH ISPH AMES CSAG RMDA NAPR OIH AMIA MIPH IFAP MOIN MPRC CPCI OLFI PHTV BONY EGAS ISMQ SUGR MPCI ZMID AXPH PRDC DOMT ARAB GOUR SPHT ELEC ACAP SPIN NINH ENGC OCPH MCRO CNFN MICH KABO SVCE AFMC GSSC WCDF SDTI DSCW AMER SAIB MFSC OFH UNIT ACGC AJWA KZPC UEFM GDWA CRST ADCI GPIM ACTF ELKA ASCM LCSW ELSH ICFC CFGH NAHO INFI ALRA ATLC ZEOT DAPH ACAMD EDFM ETRS MPCO ISMA GTWL MAAL SMFR NARE CEFM MILS PHGC SNFC IDRE ICID RACC GGRN EALR NCCW ADPC DTPP EHDR AALR KRDI MOSC WKOL MENA MBEG ECAP GGCC SIPC ODIN CERA SCFM DEIN CAED ASPI AIHC PRCL NDRL LUTS NHPS SEIGA SEIG OBRI IEEC MEPA UEGC RUBX AIDC COSG POCO ALUM TWSA GTEX AFDI RREI TANM EBSC RTVC AMII UNIP APSW ICLE MEGM PRMH EASB RAKT ROTO MOED TYCN SPMD KWIN EEII AREH CCRS FCMD EPCO GRCA TRTO GIHD ELWA ELNA MMAT DGTZ DCCC NEDA EPPK GMCI EOSB CPME COPR'''.split()
NEW_17 = '''NFCI SIEG ENPI EGOTH ALXD ANCC PMSC EFAC CID INEG MMHC MLIC MITR TOUR FTNS HBCO HDST'''.split()
EGX_246 = list(dict.fromkeys(BASE_229 + NEW_17))
# Guardrail: the app must contain exactly 246 unique universe entries.
if len(EGX_246) != 246:
    raise RuntimeError(f'Universe error: expected 246 unique symbols, found {len(EGX_246)}')

BANKS = set('COMI QNBE HDBK ADIB CANA FAIT FAITA CIEB EXPA UBEE SAUD EGBE SAIB'.split())
REAL_ESTATE = set('TMGH EMFD PHDC OCDI ORHD HELI MASR PRDC BONY ZMID ACAP UNIT ELKA ELSH DAPH ACAMD NARE EHDR IDRE ICID MENA UEGC OBRI RREI TANM NHPS GIHD UTOP ALXD MMHC'.split())
HEALTH = set('BIOC PHAR NIPH ISPH MIPH CPCI RMDA MPCI AXPH OCPH MCRO NINH ADCI SIPC MEPA FCMD PHGC'.split())
ENERGY = set('ABUK MFPC EGCH EFIC MICH SKPC AMOC ATQA KZPC TAQA EGAS MOIL NDRL FERC NFCI ENPI EFAC SIEG'.split())
FINANCIAL = set('EFIH FWRY HRHO BTFH RAYA CCAP VALU BINV CICH CNFN OFH ACTF NAHO AMIA MOIN ATLC ASPI AIHC AIDC EBSC PRMH EASB TYCN KWIN EOSB CPME TWSA AFDI DEIN SEIGA SEIG MLIC GRCA'.split())
FOOD = set('EAST EFID JUFO POUL OLFI SUGR DOMT GOUR AFMC WCDF UEFM GSSC AJWA MPCO EDFM ISMA CEFM MILS SNFC ADPC KRDI MOSC SCFM INFI ALRA ZEOT EPCO ELNA NEDA GGRN LUTS'.split())
TECH = set('ETEL EFIH FWRY SCTS EGSA OIH MPRC RACC DGTZ MOED CAED'.split())
INDUSTRIAL = set('SWDY ORAS GBCO IRON ARCC SCEM MBSC MCQE ORWE MTIE ELEC KABO GTWL ACGC SPIN DSCW MICH ASCM LCSW ECAP RUBX AMII UNIP APSW MEGM RAKT EEII IEEC PRCL ALUM GTEX CID INEG HBCO HDST YAYT KNGC'.split())
TRANSPORT = set('ALCN CSAG EGTS ETRS POCO DCCC'.split())
TOURISM = set('GPPL PHTV SPHT SDTI MHOT RTVC ROTO TRTO GIHD ELWA MMAT TOUR FTNS EGOTH MITR'.split())

def sector(t):
    t=t.upper()
    for name,s in [('BANKS',BANKS),('FINANCIAL',FINANCIAL),('REAL_ESTATE',REAL_ESTATE),('HEALTHCARE',HEALTH),('ENERGY_CHEMICALS',ENERGY),('FOOD_CONSUMER',FOOD),('TECH_TMT',TECH),('INDUSTRIAL',INDUSTRIAL),('TRANSPORT',TRANSPORT),('TOURISM',TOURISM)]:
        if t in s: return name
    return 'GENERAL'

def num(x):
    try:
        x=float(x); return x if np.isfinite(x) else np.nan
    except: return np.nan

def first(*xs):
    for x in xs:
        x=num(x)
        if np.isfinite(x): return x
    return np.nan

def clamp(x,a=0,b=100):
    x=num(x); return np.nan if not np.isfinite(x) else float(np.clip(x,a,b))

def growth_score(x):
    x=num(x)
    return np.nan if not np.isfinite(x) else clamp(50+x*1.5)

def yahoo(t): return t + '.CA'

ALIASES={
 'revenue':['Total Revenue','Operating Revenue','Revenue'],
 'net_income':['Net Income','Net Income Common Stockholders','Net Income Including Noncontrolling Interests'],
 'ebit':['EBIT','Operating Income'], 'ebitda':['EBITDA','Normalized EBITDA'],
 'assets':['Total Assets'], 'liabilities':['Total Liabilities Net Minority Interest','Total Liabilities'],
 'equity':['Stockholders Equity','Total Equity Gross Minority Interest','Common Stock Equity'],
 'cash':['Cash Cash Equivalents And Short Term Investments','Cash And Cash Equivalents'],
 'debt':['Total Debt','Long Term Debt And Capital Lease Obligation','Current Debt And Capital Lease Obligation'],
 'ocf':['Operating Cash Flow','Total Cash From Operating Activities'],
 'capex':['Capital Expenditure','Capital Expenditure Reported'], 'fcf':['Free Cash Flow'],
 'div_paid':['Cash Dividends Paid','Common Stock Dividend Paid'],
}

def statement(obj,name):
    try:
        x=getattr(obj,name); return x if isinstance(x,pd.DataFrame) else pd.DataFrame()
    except: return pd.DataFrame()

def row_value(df, aliases):
    if df is None or df.empty: return np.nan
    labels=[str(i).lower() for i in df.index]
    for a in aliases:
        a=a.lower()
        for i,l in enumerate(labels):
            if l==a or a in l:
                s=pd.to_numeric(df.iloc[i],errors='coerce').dropna()
                if len(s): return num(s.iloc[0])
    return np.nan

def growth(df,aliases):
    if df is None or df.empty: return np.nan
    labels=[str(i).lower() for i in df.index]
    for a in aliases:
        a=a.lower()
        for i,l in enumerate(labels):
            if l==a or a in l:
                s=pd.to_numeric(df.iloc[i],errors='coerce').dropna()
                if len(s)>=2:
                    return (s.iloc[0]/s.iloc[1]-1)*100 if s.iloc[1]!=0 else np.nan
    return np.nan

def add_metric(m,key,value,source,confidence,kind='Reported'):
    m[key]={'value':num(value),'source':source,'confidence':float(confidence) if np.isfinite(num(confidence)) else 0,'type':kind}

def getv(m,key): return num(m.get(key,{}).get('value'))

def recover(ticker, backup=None):
    y=yahoo(ticker); obj=yf.Ticker(y)
    try: info=obj.get_info()
    except: info={}
    inc=statement(obj,'income_stmt'); bs=statement(obj,'balance_sheet'); cf=statement(obj,'cashflow')
    m={}
    for k,a in ALIASES.items():
        src='Yahoo Statement'; conf=95
        df=inc if k in {'revenue','net_income','ebit','ebitda'} else bs if k in {'assets','liabilities','equity','cash','debt'} else cf
        v=row_value(df,a)
        if not np.isfinite(v):
            info_map={'revenue':'totalRevenue','net_income':'netIncomeToCommon','ebitda':'ebitda','equity':'bookValue','debt':'totalDebt','cash':'totalCash'}
            if k in info_map: v=info.get(info_map[k],np.nan); src='Yahoo Profile'; conf=80
        add_metric(m,k,v,src,conf)
    price=first(info.get('currentPrice'),info.get('regularMarketPrice'))
    shares=first(info.get('sharesOutstanding'),info.get('impliedSharesOutstanding'))
    add_metric(m,'price',price,'Yahoo Quote',90)
    add_metric(m,'shares',shares,'Yahoo Profile',85)
    # Uploaded backup only fills missing values; it never silently overwrites Yahoo.
    if backup is not None:
        for k in set(ALIASES)|{'price','shares','eps','fcf'}:
            if not np.isfinite(getv(m,k)) and k in backup:
                add_metric(m,k,backup[k],'Uploaded Backup',88,'Reported')
    rev,ni,eq,assets,debt,cash,ocf,capex,sh=getv(m,'revenue'),getv(m,'net_income'),getv(m,'equity'),getv(m,'assets'),getv(m,'debt'),getv(m,'cash'),getv(m,'ocf'),getv(m,'capex'),getv(m,'shares')
    fcf=first(getv(m,'fcf'),ocf+capex if np.isfinite(ocf) and np.isfinite(capex) else np.nan)
    eps=ni/sh if np.isfinite(ni) and np.isfinite(sh) and sh>0 else np.nan
    bvps=eq/sh if np.isfinite(eq) and np.isfinite(sh) and sh>0 else np.nan
    roe=ni/eq*100 if np.isfinite(ni) and np.isfinite(eq) and eq else np.nan
    roa=ni/assets*100 if np.isfinite(ni) and np.isfinite(assets) and assets else np.nan
    margin=ni/rev*100 if np.isfinite(ni) and np.isfinite(rev) and rev else np.nan
    de=debt/eq if np.isfinite(debt) and np.isfinite(eq) and eq else np.nan
    pe=getv(m,'price')/eps if np.isfinite(getv(m,'price')) and np.isfinite(eps) and eps>0 else np.nan
    pb=getv(m,'price')/bvps if np.isfinite(getv(m,'price')) and np.isfinite(bvps) and bvps>0 else np.nan
    rg=growth(inc,ALIASES['revenue']); nig=growth(inc,ALIASES['net_income'])
    for k,v in {'fcf':fcf,'eps':eps,'bvps':bvps,'roe':roe,'roa':roa,'margin':margin,'de':de,'pe':pe,'pb':pb,'revenue_growth':rg,'net_income_growth':nig}.items():
        add_metric(m,k,v,'Derived','82' if np.isfinite(v) else 0,'Derived')
    dy=num(info.get('dividendYield'))
    if np.isfinite(dy) and dy<1: dy*=100
    add_metric(m,'dividend_yield',dy,'Yahoo Profile',75)
    # Piotroski-lite: transparent subset, not falsely presented as a full audited F-score.
    checks=[]; checks += [ni>0] if np.isfinite(ni) else []; checks += [ocf>0] if np.isfinite(ocf) else []
    checks += [roe>0] if np.isfinite(roe) else []; checks += [fcf>0] if np.isfinite(fcf) else []
    checks += [de<1.5] if np.isfinite(de) else []; checks += [rg>0] if np.isfinite(rg) else []; checks += [nig>0] if np.isfinite(nig) else []
    add_metric(m,'quality_score',sum(checks)/len(checks)*100 if checks else np.nan,'Derived Quality',70,'Derived')
    return m,info

def tech(ticker):
    try: h=yf.download(yahoo(ticker),period='2y',interval='1d',auto_adjust=False,progress=False,threads=False)
    except: return {}
    if h is None or h.empty: return {}
    if isinstance(h.columns,pd.MultiIndex): h=h.droplevel(1,axis=1)
    c=pd.to_numeric(h['Close'],errors='coerce'); h=h.assign(Close=c)
    ema20=c.ewm(span=20,adjust=False).mean(); ema50=c.ewm(span=50,adjust=False).mean(); ema200=c.ewm(span=200,adjust=False).mean()
    d=c.diff(); up=d.clip(lower=0); dn=-d.clip(upper=0); rs=up.ewm(alpha=1/14,adjust=False).mean()/dn.ewm(alpha=1/14,adjust=False).mean().replace(0,np.nan); rsi=100-100/(1+rs)
    tr=pd.concat([h['High']-h['Low'],(h['High']-c.shift()).abs(),(h['Low']-c.shift()).abs()],axis=1).max(axis=1); atr=tr.ewm(alpha=1/14,adjust=False).mean()
    plus=h['High'].diff().clip(lower=0).ewm(alpha=1/14,adjust=False).mean(); minus=(-h['Low'].diff()).clip(lower=0).ewm(alpha=1/14,adjust=False).mean(); pdi=100*plus/atr.replace(0,np.nan); mdi=100*minus/atr.replace(0,np.nan); adx=(100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)).ewm(alpha=1/14,adjust=False).mean()
    macd=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean(); sig=macd.ewm(span=9,adjust=False).mean(); vr=h['Volume']/h['Volume'].rolling(20).mean().replace(0,np.nan)
    last=h.iloc[-1]; p=num(last['Close']); trend=sum([p>ema20.iloc[-1],ema20.iloc[-1]>ema50.iloc[-1],p>ema200.iloc[-1] if np.isfinite(ema200.iloc[-1]) else False,macd.iloc[-1]>sig.iloc[-1],adx.iloc[-1]>=20])*20
    mom=(40 if 45<=rsi.iloc[-1]<=72 else 20 if rsi.iloc[-1]>72 else 10)+(35 if macd.iloc[-1]>sig.iloc[-1] else 10)+(25 if vr.iloc[-1]>=1 else 10)
    return {'price':p,'last_date':h.index[-1],'ema20':num(ema20.iloc[-1]),'ema50':num(ema50.iloc[-1]),'ema200':num(ema200.iloc[-1]),'rsi':num(rsi.iloc[-1]),'adx':num(adx.iloc[-1]),'volume_ratio':num(vr.iloc[-1]),'trend':clamp(trend),'momentum':clamp(mom),'support':num(h['Low'].tail(60).quantile(.15)),'resistance':num(h['High'].tail(60).quantile(.85))}

def confidence(m,sec):
    keys=['revenue','net_income','equity','cash','debt','shares','eps','fcf','roe','revenue_growth','net_income_growth','bvps']
    covered=[k for k in keys if np.isfinite(getv(m,k))]
    coverage=len(covered)/len(keys)*100
    conf=np.mean([m[k]['confidence'] for k in covered]) if covered else 0
    critical={'BANKS':['net_income','equity','shares','eps','roe'],'REAL_ESTATE':['net_income','equity','debt'],'GENERAL':['revenue','net_income','equity','fcf']}.get(sec,['revenue','net_income','equity','fcf'])
    cc=sum(np.isfinite(getv(m,k)) for k in critical)/len(critical)*100
    return coverage,clamp(.45*coverage+.35*conf+.20*cc),cc

def valuation(m,sec):
    p,eps,bvps,fcf,sh,g,roe=getv(m,'price'),getv(m,'eps'),getv(m,'bvps'),getv(m,'fcf'),getv(m,'shares'),getv(m,'net_income_growth'),getv(m,'roe')
    vals=[]; ws=[]
    if np.isfinite(eps) and eps>0:
        pe=12 if sec=='BANKS' else clamp(10+(g if np.isfinite(g) else 5)*.25,7,22); vals.append(eps*pe); ws.append(.35)
    if np.isfinite(bvps) and bvps>0:
        pb=1.2 if sec=='BANKS' else clamp(1+(roe/25 if np.isfinite(roe) else 0),.7,2.5); vals.append(bvps*pb); ws.append(.25)
    dcf=np.nan
    if sec!='BANKS' and np.isfinite(fcf) and fcf>0 and np.isfinite(sh) and sh>0:
        fps=fcf/sh; gg=clamp(g if np.isfinite(g) else 7,0,15); wacc=.18; tg=.06
        pv=sum(fps*(1+gg/100)**y/(1+wacc)**y for y in range(1,6)); tv=fps*(1+gg/100)**5*(1+tg)/(wacc-tg); dcf=pv+tv/(1+wacc)**5
        if np.isfinite(dcf) and dcf>0: vals.append(dcf); ws.append(.40)
    fair=float(np.average(vals,weights=ws)) if vals else p
    return {'fair':fair,'safe':fair*.70 if np.isfinite(fair) else np.nan,'strong':fair*.80 if np.isfinite(fair) else np.nan,'target1':fair*1.15 if np.isfinite(fair) else np.nan,'target2':fair*1.35 if np.isfinite(fair) else np.nan,'target3':fair*1.60 if np.isfinite(fair) else np.nan,'upside':(fair/p-1)*100 if np.isfinite(fair) and np.isfinite(p) and p else np.nan,'dcf':dcf}

def analyze(ticker,backup=None):
    m,info=recover(ticker,backup); sec=sector(ticker); t=tech(ticker)
    if np.isfinite(num(t.get('price'))): add_metric(m,'price',t['price'],'Yahoo History',92)
    cov,conf,crit=confidence(m,sec); v=valuation(m,sec)
    growth=growth_score(getv(m,'net_income_growth')); roe=clamp(getv(m,'roe')*4) if np.isfinite(getv(m,'roe')) else 50; qual=getv(m,'quality_score'); qual=50 if not np.isfinite(qual) else qual
    pe=getv(m,'pe'); pb=getv(m,'pb'); val=50
    if np.isfinite(pe) and pe>0: val += clamp((15-pe)*3,-30,30)
    if np.isfinite(pb) and pb>0: val += clamp((2-pb)*12,-20,20)
    val=clamp(val); fund=np.nanmean([x for x in [growth,roe,qual,val] if np.isfinite(x)])
    techscore=np.nanmean([t.get('trend',50),t.get('momentum',50)]) if t else 50
    raw=clamp(fund*.75+techscore*.25); final=raw*(.65+.35*conf/100)
    return {'Ticker':ticker,'Sector':sec,'Price':getv(m,'price'),'Score':final,'Raw Score':raw,'Fundamental Score':fund,'Technical Score':techscore,'Data Coverage %':cov,'Data Confidence %':conf,'Critical Coverage %':crit,'Fair Value':v['fair'],'Safe Buy':v['safe'],'Strong Buy':v['strong'],'Target 1':v['target1'],'Target 2':v['target2'],'Target 3':v['target3'],'Upside %':v['upside'],'DCF':v['dcf'],'Revenue Growth %':getv(m,'revenue_growth'),'Net Income Growth %':getv(m,'net_income_growth'),'ROE %':getv(m,'roe'),'ROA %':getv(m,'roa'),'P/E':getv(m,'pe'),'P/B':getv(m,'pb'),'Debt/Equity':getv(m,'de'),'Dividend Yield %':getv(m,'dividend_yield'),'Quality %':getv(m,'quality_score'),'Last Candle':t.get('last_date'),'RSI':t.get('rsi'),'ADX':t.get('adx'),'Volume Ratio':t.get('volume_ratio'),'Support':t.get('support'),'Resistance':t.get('resistance'),'Recovery':m}

def normalize_backup(df):
    d=df.copy(); d.columns=[re.sub(r'[^a-z0-9_]','_',str(c).lower().strip()) for c in d.columns]
    aliases={'symbol':'ticker','code':'ticker','ticker':'ticker','revenue':'revenue','net_income':'net_income','equity':'equity','cash':'cash','debt':'debt','shares':'shares','price':'price','fcf':'fcf','eps':'eps'}
    d=d.rename(columns={c:aliases.get(c,c) for c in d.columns})
    if 'ticker' not in d: return pd.DataFrame()
    d['ticker']=d['ticker'].astype(str).str.upper().str.replace('.CA','',regex=False).str.strip(); return d

st.title('📊 EGX Financial Intelligence PRO MAX')
st.caption('246 سهم • Financial Recovery • Derived Metrics • Sector-Aware Valuation • Data Confidence')
with st.expander('⚙️ إعدادات',expanded=True):
    a,b,c=st.columns(3)
    min_cov=a.slider('الحد الأدنى لجودة البيانات',0,100,40,5)
    top_n=b.selectbox('أفضل N',[10,20,30,50,100],index=1)
    workers=c.selectbox('Workers',[4,6,8,10],index=2)
backup=st.file_uploader('📥 Backup اختياري CSV / Excel للقوائم المالية',type=['csv','xlsx'])
bdf=pd.DataFrame()
if backup:
    try: bdf=normalize_backup(pd.read_csv(backup) if backup.name.endswith('.csv') else pd.read_excel(backup)); st.success(f'Backup جاهز: {len(bdf)} صف')
    except Exception as e: st.error(str(e))

st.markdown(f'**Universe ثابت: {len(EGX_246)} سهم**')
if st.button('🚀 تشغيل التحليل الكامل للـ246 سهم',type='primary',use_container_width=True):
    bmap={r['ticker']:r.to_dict() for _,r in bdf.iterrows()} if not bdf.empty else {}
    out=[]; bar=st.progress(0); msg=st.empty()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut={ex.submit(analyze,t,bmap.get(t)):t for t in EGX_246}
        for i,f in enumerate(as_completed(fut),1):
            t=fut[f]
            try: out.append(f.result())
            except Exception as e: out.append({'Ticker':t,'Sector':sector(t),'Score':0,'Data Coverage %':0,'Data Confidence %':0,'Error':str(e),'Recovery':{}})
            msg.write(f'تحليل {t} — {i}/246'); bar.progress(i/246)
    result=pd.DataFrame(out).sort_values(['Score','Data Confidence %'],ascending=False)
    st.session_state['result']=result

if 'result' in st.session_state:
    r=st.session_state['result'].copy(); eligible=r[r['Data Coverage %'].fillna(0)>=min_cov].head(top_n)
    st.subheader('🏆 الترتيب النهائي')
    cols=['Ticker','Sector','Score','Data Coverage %','Data Confidence %','Price','Fair Value','Safe Buy','Target 1','Target 2','Target 3','Upside %','Fundamental Score','Technical Score','P/E','P/B','ROE %','Revenue Growth %','Net Income Growth %','Dividend Yield %']
    st.dataframe(eligible[[c for c in cols if c in eligible]],use_container_width=True,hide_index=True)
    csv=r.drop(columns=['Recovery'],errors='ignore').to_csv(index=False).encode('utf-8-sig')
    st.download_button('⬇️ تحميل EGX_PRO_MAX_Ranking.csv',csv,'EGX_PRO_MAX_Ranking.csv','text/csv')
    st.subheader('🔎 Recovery Audit')
    pick=st.selectbox('اختار سهم للتدقيق',r['Ticker'].tolist())
    one=r[r['Ticker']==pick].iloc[0]
    q1,q2,q3,q4=st.columns(4); q1.metric('Score',f"{one.get('Score',0):.1f}"); q2.metric('Coverage',f"{one.get('Data Coverage %',0):.1f}%"); q3.metric('Confidence',f"{one.get('Data Confidence %',0):.1f}%"); q4.metric('Fair Value',f"{one.get('Fair Value',np.nan):.2f}" if np.isfinite(num(one.get('Fair Value'))) else 'N/A')
    rec=one.get('Recovery',{})
    if isinstance(rec,dict):
        audit=[]
        for k,v in rec.items(): audit.append({'Metric':k,'Value':v.get('value'),'Source':v.get('source'),'Type':v.get('type'),'Confidence':v.get('confidence')})
        st.dataframe(pd.DataFrame(audit),use_container_width=True,hide_index=True)
    st.info('Reported = مصدر مباشر. Derived = محسوب من أرقام أصلية. Backup = ملف خارجي أدخله المستخدم. النقص يقلل الثقة ولا يستبعد السهم تلقائياً.')
