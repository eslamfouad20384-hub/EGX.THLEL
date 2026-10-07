import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# EGX FINANCIAL INTELLIGENCE PRO MAX v3.0
# Fundamental / Valuation / Risk / Quality / 3Y Scenarios
# NO technical indicators.
# ============================================================

st.set_page_config(page_title="EGX Financial Intelligence PRO MAX", page_icon="💰", layout="wide")
st.markdown("""
<style>
html,body,[class*="css"]{direction:rtl}
.block-container{max-width:1600px;padding-top:1rem}
[data-testid="stDataFrame"]{direction:rtl}
</style>
""", unsafe_allow_html=True)

APP_VERSION="3.0 PRO MAX"
YAHOO_SUFFIX=".CA"

DEFAULT_EGX_SYMBOLS=list(dict.fromkeys("""ABUK ACGC ADIB AIVC ALCN AMER ARCC ARPI ASCM ATLC AUTO AXPH BINV BIOC BTFH CICH CIRA CIEB COMI COPR COSG CPCI CRST CSAG DAPH DOMT EAST ECAP EFID EFIH EGAS EGCH EGTS ELEC EMFD ENGC ETEL ETRS FAIT FWRY GDWA GBCO HELI HDBK HRHO ICFC IDHC IEEC IFAP IRON ISMA JUFO KABO KZPC LCID MAAL MCQE MFPC MICH MISR MNHD MOIL MPBS MPCO MPCI MTIE NAHO NCCW NIPH NILE OCIC OCPH ODIN ORAS ORHD ORWE PHAR PHDC PRCL PRMH QNBE RACC RAYA RDFI REAC RMDA ROTO SAIB SAUD SCEM SDTI SKPC SMFR SPIN SPMD SUGR SWDY TALM TMGH TORA UASG UNIT UNIP UPMS VERT WCDF WEAS WKOL ZECO ZAHI AFMC ARAB CCRS CLHO CNFN EASB ELSH EXPA FARE GEMA GSSC HITP IDBE INFI LEDA MENA MEPA NEDA OBOU PHTV PION RREI SIPC TAQA TATW TRTO UNBE VTMN WADI ZMID BICC BODA BTMN CITI EGBE MOBG NBEG QNBA UBEE AMIA APPC CERA GISS GOLD ICID ISPH PACH PRDC SCTS SCFM TMMT TOWN""".split()))

SECTOR_MAP={
"COMI":"بنوك","CIEB":"بنوك","ADIB":"بنوك","HDBK":"بنوك","QNBE":"بنوك","SAIB":"بنوك","FAIT":"بنوك","EGBE":"بنوك","UBEE":"بنوك","EXPA":"بنوك","CICH":"بنوك","BTFH":"بنوك","NBEG":"بنوك","QNBA":"بنوك",
"TMGH":"عقارات","HELI":"عقارات","ORHD":"عقارات","MNHD":"عقارات","PHDC":"عقارات","TALA":"عقارات","EMFD":"عقارات","MENA":"عقارات","ARAB":"عقارات",
"DAPH":"رعاية صحية/دواء","NIPH":"رعاية صحية/دواء","PHAR":"رعاية صحية/دواء","AXPH":"رعاية صحية/دواء","IDHC":"رعاية صحية/دواء","RMDA":"رعاية صحية/دواء","MPCI":"رعاية صحية/دواء",
"MFPC":"كيماويات/أسمدة","ABUK":"كيماويات/أسمدة","SKPC":"كيماويات/أسمدة","KZPC":"كيماويات/أسمدة","MICH":"كيماويات/أسمدة","EGCH":"كيماويات/أسمدة",
"DOMT":"أغذية","JUFO":"أغذية","EAST":"أغذية/تبغ","EFID":"أغذية","UASG":"أغذية",
"ETEL":"اتصالات","FWRY":"مدفوعات/تكنولوجيا","EFIH":"مدفوعات/تكنولوجيا","RAYA":"تكنولوجيا/خدمات","MTIE":"تكنولوجيا/توزيع","CIRA":"تعليم/خدمات",
"HRHO":"خدمات مالية","EFG":"خدمات مالية","BINV":"خدمات مالية","MOBG":"خدمات مالية","NILE":"خدمات مالية","INFI":"خدمات مالية",
"EGAS":"طاقة/غاز","TAQA":"طاقة","MOIL":"طاقة","AMOC":"طاقة","GASCO":"طاقة",
"ESRS":"معادن/حديد","IRCC":"معادن/حديد","IRON":"معادن/حديد","SWDY":"صناعة/كابلات","ARCC":"مواد بناء","TORA":"مواد بناء","SPMD":"مواد بناء","SCEM":"مواد بناء","WCDF":"مواد بناء",
"AMER":"خدمات استهلاكية","AUTO":"سيارات","MPCO":"استهلاكي","LCID":"استهلاكي","ORWE":"منسوجات"
}

SECTOR_DEFAULTS={
"بنوك":dict(ke=.20,pe=9,pb=1.0,growth_cap=.15,terminal=.05,model="bank"),
"عقارات":dict(ke=.19,pe=10,pb=.85,growth_cap=.15,terminal=.04,model="realestate"),
"رعاية صحية/دواء":dict(ke=.20,pe=13,pb=1.25,growth_cap=.18,terminal=.05,model="standard"),
"كيماويات/أسمدة":dict(ke=.20,pe=8.5,pb=1,growth_cap=.12,terminal=.04,model="cyclical"),
"أغذية":dict(ke=.19,pe=11,pb=1.15,growth_cap=.12,terminal=.04,model="standard"),
"اتصالات":dict(ke=.18,pe=10,pb=1.1,growth_cap=.10,terminal=.04,model="telecom"),
"مدفوعات/تكنولوجيا":dict(ke=.21,pe=18,pb=2,growth_cap=.25,terminal=.06,model="growth"),
"تكنولوجيا/خدمات":dict(ke=.21,pe=16,pb=1.8,growth_cap=.22,terminal=.06,model="growth"),
"خدمات مالية":dict(ke=.20,pe=11,pb=1.2,growth_cap=.16,terminal=.05,model="financial"),
"طاقة/غاز":dict(ke=.19,pe=8,pb=1,growth_cap=.10,terminal=.04,model="cyclical"),
"طاقة":dict(ke=.19,pe=8,pb=1,growth_cap=.10,terminal=.04,model="cyclical"),
"معادن/حديد":dict(ke=.21,pe=8,pb=.9,growth_cap=.10,terminal=.04,model="cyclical"),
"مواد بناء":dict(ke=.20,pe=9,pb=1,growth_cap=.10,terminal=.04,model="cyclical"),
"صناعة/كابلات":dict(ke=.20,pe=11,pb=1.15,growth_cap=.12,terminal=.04,model="standard"),
"استهلاكي":dict(ke=.20,pe=11,pb=1.1,growth_cap=.12,terminal=.04,model="standard"),
"سيارات":dict(ke=.21,pe=10,pb=1,growth_cap=.10,terminal=.04,model="standard"),
"منسوجات":dict(ke=.21,pe=9,pb=.9,growth_cap=.10,terminal=.04,model="cyclical"),
"تعليم/خدمات":dict(ke=.20,pe=14,pb=1.4,growth_cap=.18,terminal=.05,model="growth"),
"عام":dict(ke=.21,pe=10,pb=1,growth_cap=.12,terminal=.04,model="standard")}

ALIASES={
"revenue":["Total Revenue","Operating Revenue","Revenue"],"net_income":["Net Income","Net Income Common Stockholders","Net Income Including Noncontrolling Interests"],"pretax":["Pretax Income"],"ebit":["EBIT","Operating Income"],"ebitda":["EBITDA","Normalized EBITDA"],"eps":["Diluted EPS","Basic EPS","Diluted EPS from Continuing Operations"],"equity":["Stockholders Equity","Common Stock Equity","Total Equity Gross Minority Interest","Total Equity"],"assets":["Total Assets"],"debt":["Total Debt","Long Term Debt And Capital Lease Obligation","Long Term Debt"],"cash":["Cash Cash Equivalents And Short Term Investments","Cash And Cash Equivalents","Cash Financial"],"ocf":["Operating Cash Flow","Total Cash From Operating Activities"],"capex":["Capital Expenditure","Capital Expenditures"],"interest":["Interest Expense Non Operating","Interest Expense"],"current_assets":["Current Assets"],"current_liabilities":["Current Liabilities"],"shares":["Ordinary Shares Number","Share Issued"],"gross_profit":["Gross Profit"],"operating_income":["Operating Income"],"tax":["Tax Provision"],"dividends":["Cash Dividends Paid","Common Stock Dividend Paid"]}

def finite(x):
    try:return x is not None and np.isfinite(float(x))
    except:return False

def sf(x,default=np.nan):
    try:
        v=float(x); return v if np.isfinite(v) else default
    except:return default

def clean(s):return re.sub(r"[^A-Z0-9]","",str(s).upper().strip().replace(".CA",""))
def ys(s):return clean(s)+YAHOO_SUFFIX

def sector(sym,yfs=""):
    if sym in SECTOR_MAP:return SECTOR_MAP[sym]
    s=str(yfs).lower()
    if "bank" in s:return "بنوك"
    if "real estate" in s or "reit" in s:return "عقارات"
    if any(x in s for x in ["health","drug","biotech"]):return "رعاية صحية/دواء"
    if any(x in s for x in ["technology","software","information"]):return "تكنولوجيا/خدمات"
    if "financial" in s:return "خدمات مالية"
    if any(x in s for x in ["energy","oil","gas"]):return "طاقة"
    if any(x in s for x in ["communication","telecom"]):return "اتصالات"
    if any(x in s for x in ["consumer","food"]):return "أغذية"
    if "chemical" in s:return "كيماويات/أسمدة"
    return "عام"

@st.cache_data(ttl=3600,show_spinner=False)
def discover_universe():
    found=[]
    for url in ["https://stockanalysis.com/list/egyptian-stock-exchange/","https://stockanalysis.com/stocks/egx/"]:
        try:
            r=requests.get(url,headers={"User-Agent":"Mozilla/5.0"},timeout=12)
            if r.ok:found += [x.replace('.CA','') for x in re.findall(r'\b[A-Z]{3,5}\.CA\b',r.text.upper())]
        except:pass
    return list(dict.fromkeys(found+DEFAULT_EGX_SYMBOLS))

@st.cache_data(ttl=1800,show_spinner=False)
def fetch_bundle(sym):
    try:
        t=yf.Ticker(ys(sym)); info={}
        try:info=t.info or {}
        except:pass
        inc=t.income_stmt; bal=t.balance_sheet; cf=t.cashflow
        hist=t.history(period="10d",interval="1d",auto_adjust=False,actions=True)
        price=sf(hist["Close"].dropna().iloc[-1]) if hist is not None and not hist.empty else sf(info.get("currentPrice",info.get("regularMarketPrice")))
        date=str(hist.index[-1].date()) if hist is not None and not hist.empty else ""
        ok=finite(price) or (inc is not None and not inc.empty)
        return dict(symbol=sym,info=info,income=inc,balance=bal,cashflow=cf,history=hist,price=price,price_date=date,ok=ok,error="")
    except Exception as e:return dict(symbol=sym,ok=False,error=str(e)[:250])

def series(df,names):
    if df is None or not isinstance(df,pd.DataFrame) or df.empty:return pd.Series(dtype=float)
    for n in names:
        if n in df.index:
            s=pd.to_numeric(df.loc[n],errors="coerce").dropna()
            if not s.empty:return s.sort_index()
    norm={re.sub(r"[^a-z0-9]","",str(i).lower()):i for i in df.index}
    for n in names:
        k=re.sub(r"[^a-z0-9]","",n.lower())
        if k in norm:
            s=pd.to_numeric(df.loc[norm[k]],errors="coerce").dropna()
            if not s.empty:return s.sort_index()
    return pd.Series(dtype=float)

def last(df,key):
    s=series(df,ALIASES[key]);return sf(s.iloc[-1]) if not s.empty else np.nan

def cagr(s,years=3):
    if s is None or len(s)<2:return np.nan
    s=s.sort_index(); n=min(years,len(s)-1); a=sf(s.iloc[-1-n]);b=sf(s.iloc[-1])
    if not all([finite(a),finite(b)]) or a<=0 or b<=0:return np.nan
    return (b/a)**(1/n)-1

def slope(s):
    if s is None or len(s)<2:return np.nan
    a=pd.to_numeric(s,errors="coerce").dropna().values
    if len(a)<2:return np.nan
    sl=np.polyfit(np.arange(len(a)),a,1)[0]; base=np.mean(np.abs(a))
    return sl/base if base else np.nan

def medianv(v):
    a=[float(x) for x in v if finite(x)]
    return float(np.median(a)) if a else np.nan

def percentile_score(x,lo,hi):
    if not finite(x):return 50.
    return float(np.clip((x-lo)/(hi-lo)*100,0,100))

def dcf(fcf,shares,ke,g,gt,years=5):
    if not all(finite(x) for x in [fcf,shares,ke,g,gt]) or fcf<=0 or shares<=0 or ke<=gt:return np.nan
    g=float(np.clip(g,-.03,.20));gt=float(np.clip(gt,.02,min(.05,ke-.01)))
    pv=sum(fcf*(1+g)**y/(1+ke)**y for y in range(1,years+1))
    tv=fcf*(1+g)**years*(1+gt)/(ke-gt);pv+=tv/(1+ke)**years
    return pv/shares

def residual_income(bvps,eps,ke,g,years=5):
    if not all(finite(x) for x in [bvps,eps,ke,g]) or bvps<=0 or ke<=0:return np.nan
    g=float(np.clip(g,-.03,.18));book=bvps;value=bvps
    for y in range(1,years+1):
        earn=eps*(1+g)**y;book*=1+g;ri=earn-book*ke;value+=ri/(1+ke)**y
    tg=min(g,.05);tv=(eps*(1+g)**years*(1+g)-book*ke)/(ke-max(.02,tg));value+=tv/(1+ke)**years
    return max(value,0)

def ev_ebitda_value(ebitda,debt,cash,shares,multiple):
    if not all(finite(x) for x in [ebitda,debt,cash,shares,multiple]) or ebitda<=0 or shares<=0:return np.nan
    return max((ebitda*multiple-debt+cash)/shares,0)

def normalize_capex(ocf,capex):
    if not all(finite(x) for x in [ocf,capex]):return np.nan
    return ocf-abs(capex)

def piotroski(r, hist):
    inc,bal,cf=hist
    ni=series(inc,ALIASES['net_income']);ocf=series(cf,ALIASES['ocf']);assets=series(bal,ALIASES['assets']);debt=series(bal,ALIASES['debt']);ca=series(bal,ALIASES['current_assets']);cl=series(bal,ALIASES['current_liabilities']);shares=series(bal,ALIASES['shares']);gp=series(inc,ALIASES['gross_profit']);rev=series(inc,ALIASES['revenue'])
    if any(len(x)<2 for x in [ni,ocf,assets]):return np.nan
    score=0
    roa0=ni.iloc[-1]/assets.iloc[-1] if assets.iloc[-1] else np.nan; roa1=ni.iloc[-2]/assets.iloc[-2] if assets.iloc[-2] else np.nan
    score+=int(finite(roa0) and roa0>0);score+=int(finite(ocf.iloc[-1]) and ocf.iloc[-1]>0);score+=int(finite(roa0) and finite(roa1) and roa0>roa1)
    accr=ocf.iloc[-1]-ni.iloc[-1];score+=int(accr>0)
    if len(debt)>=2:score+=int(debt.iloc[-1]<=debt.iloc[-2])
    if len(ca)>=2 and len(cl)>=2:
        cr0=ca.iloc[-1]/cl.iloc[-1] if cl.iloc[-1] else np.nan;cr1=ca.iloc[-2]/cl.iloc[-2] if cl.iloc[-2] else np.nan;score+=int(finite(cr0) and finite(cr1) and cr0>cr1)
    if len(shares)>=2:score+=int(shares.iloc[-1]<=shares.iloc[-2])
    if len(gp)>=2 and len(rev)>=2:
        gm0=gp.iloc[-1]/rev.iloc[-1] if rev.iloc[-1] else np.nan;gm1=gp.iloc[-2]/rev.iloc[-2] if rev.iloc[-2] else np.nan;score+=int(finite(gm0) and finite(gm1) and gm0>gm1)
        at0=rev.iloc[-1]/assets.iloc[-1] if assets.iloc[-1] else np.nan;at1=rev.iloc[-2]/assets.iloc[-2] if assets.iloc[-2] else np.nan;score+=int(finite(at0) and finite(at1) and at0>at1)
    return float(score)

def altman(r):
    if r['sector']=='بنوك':return np.nan
    A=r['assets']; WC=(r['current_assets']-r['current_liabilities']) if finite(r['current_assets']) and finite(r['current_liabilities']) else np.nan
    RE=r['retained_earnings']; EBIT=r['ebit']; MC=r['market_cap']; TL=r['debt']; SALES=r['revenue']
    if not all(finite(x) for x in [A,WC,RE,EBIT,MC,TL,SALES]) or A<=0 or TL<=0:return np.nan
    return 1.2*WC/A+1.4*RE/A+3.3*EBIT/A+.6*MC/TL+1.0*SALES/A

def valuation(r):
    cfg=SECTOR_DEFAULTS.get(r['sector'],SECTOR_DEFAULTS['عام']);g=float(np.clip(r['normalized_growth'],-.03,cfg['growth_cap']));ke=cfg['ke']
    d=dcf(r['fcf_normalized'],r['shares'],ke,g,cfg['terminal']) if cfg['model'] not in ['bank'] else np.nan
    ri=residual_income(r['bvps'],r['eps_normalized'],ke,g) if cfg['model']=='bank' else np.nan
    pe=r['eps_normalized']*cfg['pe'] if finite(r['eps_normalized']) and r['eps_normalized']>0 else np.nan
    pb=r['bvps']*cfg['pb'] if finite(r['bvps']) and r['bvps']>0 else np.nan
    evm=ev_ebitda_value(r['ebitda_normalized'],r['debt'],r['cash'],r['shares'],max(5,cfg['pe']-1))
    fcfv=(r['fcf_normalized']/max(.06,ke))/r['shares'] if finite(r['fcf_normalized']) and r['fcf_normalized']>0 and finite(r['shares']) else np.nan
    vals=[];weights=[]
    model=cfg['model']
    if model=='bank':
        for v,w in [(ri,.45),(pb,.30),(pe,.25)]:
            if finite(v):vals.append(v);weights.append(w)
    elif model in ['realestate','financial']:
        for v,w in [(d,.30),(pb,.25),(pe,.20),(evm,.15),(fcfv,.10)]:
            if finite(v):vals.append(v);weights.append(w)
    else:
        for v,w in [(d,.35),(pe,.20),(evm,.20),(fcfv,.15),(pb,.10)]:
            if finite(v):vals.append(v);weights.append(w)
    if not vals:return (np.nan,d,ri,pe,pb,evm,fcfv,0)
    w=np.array(weights);w/=w.sum();fair=float(np.dot(vals,w)); dispersion=float(np.std(vals)/max(abs(fair),1e-9));confidence=float(np.clip(100- dispersion*100,20,98))
    coverage=float(r.get('coverage',50) if finite(r.get('coverage',50)) else 50)
    confidence*=.65+.35*(coverage/100)
    return fair,d,ri,pe,pb,evm,fcfv,confidence

def growth_blend(rev,ni,eps,fcf):
    vals=[x for x in [rev,ni,eps,fcf] if finite(x) and x>-0.5]
    return float(np.median(vals)) if vals else .08

def quality_scores(r):
    # Continuous, sector-aware sub-scores.
    growth=percentile_score(r['normalized_growth'],-.05,.25)
    roe=percentile_score(r['roe'],-.05,.30)
    margin=percentile_score(r['net_margin'],-.05,.30)
    profitability=.65*roe+.35*margin
    if r['sector']=='بنوك':
        strength=percentile_score(r['roe'],-.02,.30)
    else:
        de=r['debt_equity'];cr=r['current_ratio'];ic=r['interest_coverage']
        ds=100 if not finite(de) else percentile_score(1/(1+max(de,0)),-0.0,1.0)
        cs=percentile_score(cr,.3,3)
        ins=percentile_score(ic,1,15)
        strength=.45*ds+.30*cs+.25*ins
    cashq=percentile_score(r['ocf_ni'],.3,1.5)
    if finite(r['fcf_normalized']) and r['fcf_normalized']>0:cashq=min(100,cashq+15)
    val=percentile_score(r['upside'],-.30,.50)
    div=percentile_score(r['dividend_yield'],0,.10)
    pi=percentile_score(r['piotroski'],2,9) if finite(r['piotroski']) else 50
    return val,growth,profitability,strength,cashq,div,pi

def score_engine(r):
    val,growth,prof,strength,cashq,div,pi=quality_scores(r)
    # Financial-only weighted score; sector-normalized through the input metrics.
    base=.24*val+.18*growth+.18*prof+.15*strength+.10*cashq+.05*div+.05*pi+.05*r['coverage']
    confidence_penalty=0 if r['coverage']>=75 else (75-r['coverage'])*.08
    model_penalty=0 if r['valuation_confidence']>=70 else (70-r['valuation_confidence'])*.08
    return round(float(np.clip(base-confidence_penalty-model_penalty,0,100)),2)

def targets3(r):
    cfg=SECTOR_DEFAULTS.get(r['sector'],SECTOR_DEFAULTS['عام']);g=r['normalized_growth'];
    gbase=float(np.clip(g,cfg['growth_cap']*-0.25,cfg['growth_cap']));gc=float(np.clip(gbase-.04,-.05,cfg['growth_cap']-.01));go=float(np.clip(gbase+.05,-.03,cfg['growth_cap']+.06))
    fair=r['fair_value'];eps=r['eps_normalized'];bv=r['bvps']
    def pe_target(g,m):return eps*(1+g)**3*m if finite(eps) and eps>0 else np.nan
    def pb_target(g,m):return bv*(1+g)**3*m if finite(bv) and bv>0 else np.nan
    if cfg['model']=='bank' or not finite(eps) or eps<=0:
        c,b,o=pb_target(gc,cfg['pb']*.85),pb_target(gbase,cfg['pb']),pb_target(go,cfg['pb']*1.15)
    else:c,b,o=pe_target(gc,cfg['pe']*.85),pe_target(gbase,cfg['pe']),pe_target(go,cfg['pe']*1.15)
    if finite(fair):
        c=medianv([c,fair*(1+gc)**3]);b=medianv([b,fair*(1+gbase)**3]);o=medianv([o,fair*(1+go)**3])
    dy=r['dividend_yield'] if finite(r['dividend_yield']) else 0
    return c,b,o,dy*3,(b/r['price'])**(1/3)-1 if finite(b) and finite(r['price']) and b>0 and r['price']>0 else np.nan

def build(bundle):
    info=bundle.get('info',{}) or {};inc=bundle.get('income');bal=bundle.get('balance');cf=bundle.get('cashflow');sym=bundle['symbol'];price=bundle['price'];sec=sector(sym,info.get('sector',''))
    rev=series(inc,ALIASES['revenue']);ni=series(inc,ALIASES['net_income']);epss=series(inc,ALIASES['eps']);eq=series(bal,ALIASES['equity']);assets=series(bal,ALIASES['assets']);debt=series(bal,ALIASES['debt']);cash=series(bal,ALIASES['cash']);ocf=series(cf,ALIASES['ocf']);cap=series(cf,ALIASES['capex']);curA=series(bal,ALIASES['current_assets']);curL=series(bal,ALIASES['current_liabilities']);ebit=series(inc,ALIASES['ebit']);ebitda=series(inc,ALIASES['ebitda']);gross=series(inc,ALIASES['gross_profit']);
    shares=sf(info.get('sharesOutstanding'))
    if not finite(shares) and len(series(bal,ALIASES['shares'])):shares=sf(series(bal,ALIASES['shares']).iloc[-1])
    if not finite(shares) and finite(info.get('marketCap')) and price>0:shares=sf(info.get('marketCap'))/price
    revenue=sf(rev.iloc[-1]) if len(rev) else sf(info.get('totalRevenue'));net=sf(ni.iloc[-1]) if len(ni) else sf(info.get('netIncomeToCommon'));eps=sf(epss.iloc[-1]) if len(epss) else sf(info.get('trailingEps'));equity=sf(eq.iloc[-1]);asset=sf(assets.iloc[-1]);deb=sf(debt.iloc[-1]) if len(debt) else sf(info.get('totalDebt'));cashv=sf(cash.iloc[-1]) if len(cash) else sf(info.get('totalCash'));ocfv=sf(ocf.iloc[-1]);capv=sf(cap.iloc[-1]);
    fcf=normalize_capex(ocfv,capv);bvps=equity/shares if finite(equity) and finite(shares) and shares>0 else sf(info.get('bookValue'));revps=revenue/shares if finite(revenue) and finite(shares) and shares>0 else np.nan
    roe=net/equity if finite(net) and finite(equity) and equity else sf(info.get('returnOnEquity'));roa=net/asset if finite(net) and finite(asset) and asset else np.nan; margin=net/revenue if finite(net) and finite(revenue) and revenue else np.nan;de=deb/equity if finite(deb) and finite(equity) and equity else np.nan;cr=sf(curA.iloc[-1])/sf(curL.iloc[-1]) if len(curA) and len(curL) and sf(curL.iloc[-1]) else sf(info.get('currentRatio'));ic=sf(ebit.iloc[-1])/abs(sf(series(inc,['Interest Expense Non Operating','Interest Expense']).iloc[-1])) if len(ebit) and len(series(inc,['Interest Expense Non Operating','Interest Expense'])) else np.nan
    eg=cagr(rev);ng=cagr(ni);xg=cagr(epss);fg=cagr(pd.Series([normalize_capex(ocf.iloc[i],cap.iloc[i]) if i<len(cap) else np.nan for i in range(len(ocf))],index=ocf.index));
    normalized_growth=growth_blend(eg,ng,xg,fg);eps_norm=medianv([eps,eps*(1+normalized_growth*.25)]) if finite(eps) else np.nan;fcf_norm=medianv([fcf,fcf*(1+max(min(normalized_growth,.15),-.05))]) if finite(fcf) else np.nan;ebitda_norm=sf(ebitda.iloc[-1]) if len(ebitda) else np.nan
    market_cap=sf(info.get('marketCap'));current_assets=sf(curA.iloc[-1]) if len(curA) else np.nan;current_liabilities=sf(curL.iloc[-1]) if len(curL) else np.nan;retained=sf(series(bal,['Retained Earnings','Retained Earnings Common Stockholders']).iloc[-1]) if len(series(bal,['Retained Earnings','Retained Earnings Common Stockholders'])) else np.nan;ebitv=sf(ebit.iloc[-1]) if len(ebit) else np.nan
    div=sf(info.get('dividendYield'));div=div/100 if finite(div) and div>1 else div;payout=sf(info.get('payoutRatio'));payout=payout/100 if finite(payout) and payout>1 else payout
    r=dict(symbol=sym,name=info.get('longName',info.get('shortName',sym)),sector=sec,price=price,price_date=bundle.get('price_date',''),shares=shares,revenue=revenue,net_income=net,eps=eps,equity=equity,assets=asset,debt=deb,cash=cashv,ocf=ocfv,capex=capv,fcf=fcf,bvps=bvps,revenue_per_share=revps,roe=roe,roa=roa,net_margin=margin,debt_equity=de,current_ratio=cr,interest_coverage=ic,revenue_growth_3y=eg,earnings_growth_3y=ng,eps_growth_3y=xg,fcf_growth_3y=fg,normalized_growth=normalized_growth,eps_normalized=eps_norm,fcf_normalized=fcf_norm,ebitda_normalized=ebitda_norm,current_assets=current_assets,current_liabilities=current_liabilities,retained_earnings=retained,ebit=ebitv,market_cap=market_cap,dividend_yield=div,payout=payout)
    r['piotroski']=piotroski(r,(inc,bal,cf));r['altman_z']=altman(r)
    r['ocf_ni']=ocfv/net if finite(ocfv) and finite(net) and net!=0 else np.nan
    keys=['price','revenue','net_income','eps','equity','debt','cash','ocf','fcf','shares','bvps','roe','revenue_growth_3y','earnings_growth_3y','normalized_growth']
    r['coverage']=round(100*sum(finite(r.get(k)) for k in keys)/len(keys),1)
    fair,d,ri,pe,pb,evm,fcfv,vc=valuation(r);r.update(fair_value=fair,dcf_value=d,residual_value=ri,pe_value=pe,pb_value=pb,ev_ebitda_value=evm,fcf_yield_value=fcfv,valuation_confidence=vc)
    if finite(fair) and fair>0 and finite(price):
        r['upside']=fair/price-1;r['buy_excellent']=fair*.70;r['buy_strong']=fair*.80;r['buy_acceptable']=fair*.90
    else:r.update(upside=np.nan,buy_excellent=np.nan,buy_strong=np.nan,buy_acceptable=np.nan)
    r['target_3y_cons'],r['target_3y_base'],r['target_3y_opt'],r['dividend_3y_yield'],r['return_3y_base']=targets3(r)
    r['score']=score_engine(r);r['rating']='ممتاز جدًا' if r['score']>=88 else 'ممتاز' if r['score']>=80 else 'قوي' if r['score']>=72 else 'جيد' if r['score']>=62 else 'متوسط' if r['score']>=52 else 'ضعيف'
    r['data_quality_label']='عالية' if r['coverage']>=80 else 'جيدة' if r['coverage']>=65 else 'متوسطة' if r['coverage']>=50 else 'منخفضة'
    return r

def analyze(sym):
    b=fetch_bundle(sym)
    if not b.get('ok'):return None
    try:return build(b)
    except Exception as e:return dict(symbol=sym,name=sym,sector=SECTOR_MAP.get(sym,'عام'),price=np.nan,score=np.nan,valuation_confidence=np.nan,coverage=0,data_error=str(e)[:250])

def scan(symbols,workers):
    rows=[]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fs={ex.submit(analyze,s):s for s in symbols}
        for f in as_completed(fs):
            try:
                x=f.result()
                if x:rows.append(x)
            except:pass
    df=pd.DataFrame(rows)
    if df.empty:return pd.DataFrame(columns=['symbol','name','sector','score','valuation_confidence','coverage'])
    for col in ['score','valuation_confidence','coverage']:
        if col not in df.columns: df[col]=np.nan
        df[col]=pd.to_numeric(df[col],errors='coerce')
    return df.sort_values(['score','valuation_confidence','coverage'],ascending=False,na_position='last').reset_index(drop=True)

def money(x):return '—' if not finite(x) else f'{x:,.2f}'
def pct(x):return '—' if not finite(x) else f'{x*100:.1f}%'
@st.cache_data(show_spinner=False)
def csv_bytes(df):return df.to_csv(index=False).encode('utf-8-sig')

def main():
    st.title('💰 EGX Financial Intelligence PRO MAX')
    st.caption(f'Financial-only Engine | Fundamentals + Quality + Valuation + 3Y Scenarios | {APP_VERSION}')
    st.info('محرك مالي فقط: لا RSI ولا MACD ولا مؤشرات فنية. الهدف هو ترتيب الأسهم حسب جودة الأعمال، القيمة العادلة، القوة المالية، جودة الأرباح، النمو، وسعر الشراء.')
    with st.sidebar:
        st.header('⚙️ إعدادات PRO MAX');workers=st.slider('Parallel workers',2,8,5);mincov=st.slider('الحد الأدنى لجودة البيانات',40,100,60);topn=st.slider('أفضل N',10,50,20)
        mode=st.radio('الكون', ['اكتشاف + احتياطي','احتياطي فقط','رموز مخصصة']);custom=''
        if mode=='رموز مخصصة':custom=st.text_area('رموز',value='COMI,DAPH,MFPC,MICH,HELI,FWRY,MAAL')
        if st.button('🧹 مسح الكاش',use_container_width=True):st.cache_data.clear();st.rerun()
        st.markdown('**Buy Zones:** ممتاز 30% / قوي 20% / مقبول 10% تحت Fair Value')
        st.markdown('**Score:** 100 نقطة + عقوبات الثقة والبيانات')
    symbols=discover_universe() if mode=='اكتشاف + احتياطي' else list(DEFAULT_EGX_SYMBOLS) if mode=='احتياطي فقط' else list(dict.fromkeys(clean(x) for x in re.split(r'[,\s]+',custom) if clean(x)))
    st.write(f'**الكون:** {len(symbols)} رمز')
    tabs=st.tabs(['📊 السوق','🏭 القطاعات','🔎 سهم واحد','📈 تاريخ مالي','🧮 منهجية'])
    with tabs[0]:
        if st.button('🚀 ابدأ PRO MAX Scan',type='primary',use_container_width=True):
            with st.spinner(f'جاري تحليل {len(symbols)} سهم ماليًا...'):st.session_state['df']=scan(symbols,workers)
        df=st.session_state.get('df',pd.DataFrame())
        if df.empty:st.warning('ابدأ المسح المالي.');return
        view=df[df.coverage>=mincov].copy();view['rank']=np.arange(1,len(view)+1)
        a,b,c,d=st.columns(4);a.metric('الأسهم المؤهلة',len(view));b.metric('متوسط الدرجة',f"{view.score.mean():.1f}" if len(view) else '—');c.metric('الأفضل',view.iloc[0].symbol if len(view) else '—');d.metric('وسيط Upside',pct(view.upside.median()) if len(view) else '—')
        cols=['rank','symbol','name','sector','price','fair_value','valuation_confidence','buy_excellent','buy_strong','buy_acceptable','target_3y_cons','target_3y_base','target_3y_opt','normalized_growth','roe','piotroski','altman_z','fcf_normalized','dividend_yield','upside','coverage','score','rating']
        names=['الترتيب','الرمز','الشركة','القطاع','السعر','القيمة العادلة','ثقة التقييم','شراء ممتاز','شراء قوي','شراء مقبول','هدف محافظ 3Y','هدف أساسي 3Y','هدف متفائل 3Y','النمو الطبيعي','ROE','Piotroski','Altman Z','FCF طبيعي','التوزيعات','Upside','جودة البيانات','الدرجة','التقييم']
        shown=view[cols].copy();shown.columns=names
        st.dataframe(shown,hide_index=True,use_container_width=True)
        st.download_button('⬇️ CSV كامل',csv_bytes(shown),'EGX_PRO_MAX_Ranking.csv','text/csv',use_container_width=True)
        st.subheader('🏆 أفضل 20 حسب المحرك');st.dataframe(view.head(topn)[['rank','symbol','sector','price','fair_value','buy_strong','target_3y_base','upside','valuation_confidence','score','coverage','rating']],hide_index=True,use_container_width=True)
    with tabs[1]:
        df=st.session_state.get('df',pd.DataFrame())
        if df.empty:st.warning('ابدأ المسح أولًا.')
        else:
            v=df[df.coverage>=mincov];s=v.groupby('sector').agg(عدد=('symbol','count'),متوسط_الدرجة=('score','mean'),وسيط_Upside=('upside','median'),متوسط_الثقة=('valuation_confidence','mean'),جودة_البيانات=('coverage','mean')).reset_index().sort_values('متوسط_الدرجة',ascending=False);st.dataframe(s.rename(columns={'sector':'القطاع'}),hide_index=True,use_container_width=True)
            sec=st.selectbox('القطاع',sorted(v.sector.dropna().unique()));ss=v[v.sector==sec].sort_values('score',ascending=False);st.dataframe(ss[['symbol','name','price','fair_value','buy_strong','target_3y_base','upside','score','coverage']],hide_index=True,use_container_width=True)
    with tabs[2]:
        sym=st.text_input('رمز السهم',value='DAPH').upper()
        if st.button('🔍 تحليل PRO MAX',use_container_width=True):st.session_state['single']=analyze(clean(sym))
        r=st.session_state.get('single')
        if r:
            st.subheader(f"{r['symbol']} — {r['name']}");m=st.columns(6)
            for col,label,val in zip(m,['السعر','Fair Value','شراء قوي','هدف 3Y','الدرجة','ثقة التقييم'],[money(r['price']),money(r['fair_value']),money(r['buy_strong']),money(r['target_3y_base']),f"{r['score']:.1f}/100",f"{r['valuation_confidence']:.1f}%"]):col.metric(label,val)
            st.write(f"**القطاع:** {r['sector']} | **التقييم:** {r['rating']} | **جودة البيانات:** {r['coverage']}% ({r['data_quality_label']}) | **السعر بتاريخ:** {r['price_date']}")
            st.subheader('🎯 مناطق الشراء');st.dataframe(pd.DataFrame([['شراء ممتاز',r['buy_excellent']],['شراء قوي',r['buy_strong']],['شراء مقبول',r['buy_acceptable']],['القيمة العادلة',r['fair_value']]],columns=['المستوى','السعر']),hide_index=True,use_container_width=True)
            st.subheader('📌 سيناريوهات 3 سنوات');st.dataframe(pd.DataFrame([['محافظ',r['target_3y_cons']],['أساسي',r['target_3y_base']],['متفائل',r['target_3y_opt']]],columns=['السيناريو','الهدف']),hide_index=True,use_container_width=True)
            st.subheader('📊 المؤشرات المالية');metrics=[('الإيرادات',r['revenue']),('صافي الربح',r['net_income']),('EPS',r['eps']),('حقوق الملكية',r['equity']),('الدين',r['debt']),('النقد',r['cash']),('OCF',r['ocf']),('FCF',r['fcf']),('FCF طبيعي',r['fcf_normalized']),('BVPS',r['bvps']),('ROE',r['roe']),('ROA',r['roa']),('هامش الربح',r['net_margin']),('D/E',r['debt_equity']),('Interest Coverage',r['interest_coverage']),('نمو الإيرادات 3Y',r['revenue_growth_3y']),('نمو الأرباح 3Y',r['earnings_growth_3y']),('Piotroski',r['piotroski']),('Altman Z',r['altman_z']),('Dividend Yield',r['dividend_yield'])];st.dataframe(pd.DataFrame(metrics,columns=['المؤشر','القيمة']),hide_index=True,use_container_width=True)
            st.subheader('🧮 التقييم متعدد النماذج');st.dataframe(pd.DataFrame([['DCF',r['dcf_value']],['Residual Income',r['residual_value']],['P/E',r['pe_value']],['P/B',r['pb_value']],['EV/EBITDA',r['ev_ebitda_value']],['FCF Yield',r['fcf_yield_value']],['Fair Value',r['fair_value']],['Confidence',r['valuation_confidence']]],columns=['النموذج','القيمة']),hide_index=True,use_container_width=True)
    with tabs[3]:
        sym2=st.text_input('رمز للتاريخ المالي',value='COMI',key='hist_sym');
        if st.button('📈 تحميل التاريخ المالي',use_container_width=True):
            b=fetch_bundle(clean(sym2));st.session_state['hist']=b
        b=st.session_state.get('hist')
        if b and b.get('ok'):
            for title,df,keys in [('Income Statement',b.get('income'),['revenue','net_income','eps','gross_profit','ebit','ebitda']),('Balance Sheet',b.get('balance'),['equity','assets','debt','cash','current_assets','current_liabilities']),('Cash Flow',b.get('cashflow'),['ocf','capex','dividends'])]:
                st.subheader(title); rows=[]
                for k in keys:
                    s=series(df,ALIASES[k]);
                    if len(s):rows.append([k]+[sf(x) for x in s.iloc[-5:]])
                if rows:
                    n=len(rows[0]);cols=['المؤشر']+[f'سنة {i}' for i in range(n-1,0,-1)];st.dataframe(pd.DataFrame(rows,columns=cols),hide_index=True,use_container_width=True)
    with tabs[4]:
        st.markdown('### المحرك المالي PRO MAX')
        st.markdown("""
**1) Historical Financials:** حتى 5 سنوات من القوائم المتاحة.\
**2) Growth Engine:** CAGR للإيرادات والأرباح وEPS وFCF مع Growth Normalization.\
**3) Quality Engine:** Piotroski F-Score + Cash Conversion + جودة الأرباح.\
**4) Financial Strength:** الدين والسيولة وتغطية الفائدة، وAltman Z للشركات غير المالية عندما تتوافر المدخلات.\
**5) Valuation:** DCF + P/E + P/B + EV/EBITDA + FCF Yield + Residual Income للبنوك.\
**6) Fair Value Confidence:** يقيس اتفاق النماذج + جودة البيانات، وليس مجرد رقم واحد.\
**7) Buy Zones:** 30% / 20% / 10% Margin of Safety.\
**8) 3Y Scenarios:** محافظ / أساسي / متفائل باستخدام نمو طبيعي + مضاعفات قطاعية + Fair Value.\
**9) Score /100:** Valuation 24%، Growth 18%، Profitability 18%، Strength 15%، Cash Quality 10%، Dividend 5%، Piotroski 5%، Data Quality 5%.\
**10) Sector-aware:** البنوك لها نموذج مختلف عن الشركات التشغيلية، والقطاعات الدورية لها حدود نمو أكثر تحفظًا.\
""")
        st.warning('⚠️ البيانات المجانية قد تكون ناقصة أو متأخرة. القيمة العادلة والسيناريوهات تقديرات وليست ضمانًا. راجع آخر إفصاحات EGX والشركة قبل قرار استثماري.')

if __name__=='__main__':main()
