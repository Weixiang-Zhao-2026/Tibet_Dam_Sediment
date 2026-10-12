"""Step 7. VIF, minimum detectable effect, lagged/3-yr temperature, trends, exhaustion test, low-frequency b, threshold sensitivity."""
# %%
import os, sys
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:  # running inside a Jupyter notebook
    sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "src")))
from config import *  # sets paths; working directory becomes results/

# %%
import pandas as pd,numpy as np,statsmodels.api as sm,json,warnings; warnings.filterwarnings('ignore')
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from scipy.optimize import least_squares
S=pd.read_csv('tp_series.csv'); A=pd.read_csv('tp_station_results.csv').set_index('station')
out=[]
for st,d in S.groupby('station',sort=False):
  d=d.sort_values('yr').reset_index(drop=True); n=len(d)
  y=np.log(d.S.values); lq=np.log(d.Q.values); T=d['T'].values-d['T'].mean(); lP=np.log(d.P/d.P.mean()).values
  X=np.c_[lq-lq.mean(),T,lP]; Xc=sm.add_constant(X)
  o=sm.OLS(y,Xc).fit(cov_type='HAC',cov_kwds={'maxlags':2})
  v=[vif(Xc,i) for i in (1,2,3)]
  mde=2.8*o.bse[2]  # 80% power, alpha .05
  # lagged & cumulative T
  T3=pd.Series(T).rolling(3,min_periods=3).mean().values; Tl=np.r_[np.nan,T[:-1]]
  r={}
  for nm,extra in [('Tlag1',Tl),('T3yr',T3)]:
    m=~np.isnan(extra); Xe=sm.add_constant(np.c_[lq[m]-lq.mean(),T[m],lP[m],extra[m]]); oe=sm.OLS(y[m],Xe).fit(cov_type='HAC',cov_kwds={'maxlags':2}); r[nm]=(oe.params[4],oe.pvalues[4])
  # long-term availability trend at fixed Q,T,P
  Xt=sm.add_constant(np.c_[lq-lq.mean(),T,lP,d.yr-d.yr.mean()]); ot=sm.OLS(y,Xt).fit(cov_type='HAC',cov_kwds={'maxlags':2})
  # supply exhaustion: lagged Q at fixed current Q
  m=np.arange(1,n); Xl=sm.add_constant(np.c_[lq[m],lq[m-1]]); ol=sm.OLS(y[m],Xl).fit(cov_type='HAC',cov_kwds={'maxlags':2})
  res=sm.OLS(y,sm.add_constant(lq)).fit().resid; ac=np.corrcoef(res[1:],res[:-1])[0,1]
  # low-frequency vs high-frequency b
  bd=sm.OLS(np.diff(y),sm.add_constant(np.diff(lq))).fit().params[1]
  k=5; nb=n//k; Ym=np.array([y[i*k:(i+1)*k].mean() for i in range(nb)]); Qm=np.array([lq[i*k:(i+1)*k].mean() for i in range(nb)])
  bl=np.polyfit(Qm,Ym,1)[0] if nb>=4 else np.nan
  out.append(dict(station=st,R2=A.loc[st,'R2_rating'],n=n,b=round(o.params[1],2),VIF_Q=round(v[0],2),VIF_T=round(v[1],2),VIF_P=round(v[2],2),corr_TQ=round(np.corrcoef(T,lq)[0,1],2),cT=round(o.params[2],3),cT_se=round(o.bse[2],3),MDE_cT=round(mde,3),
    Tlag=round(r['Tlag1'][0],3),pTlag=round(r['Tlag1'][1],3),T3=round(r['T3yr'][0],3),pT3=round(r['T3yr'][1],3),trend_dec=round(10*ot.params[4],3),p_trend=round(ot.pvalues[4],3),
    lagQ=round(ol.params[2],2),p_lagQ=round(ol.pvalues[2],3),res_ac1=round(ac,2),b_diff=round(bd,2),b_5yr=round(bl,2),nblocks=nb))
B=pd.DataFrame(out); pd.set_option('display.width',300); print(B.to_string()); B.to_csv('rev3_station_tests.csv',index=False)
G=B[B.R2>=0.3]
summ=dict(VIF_T_max=float(G.VIF_T.max()),VIF_T_med=float(G.VIF_T.median()),MDE_med=float(G.MDE_cT.median()),MDE_range=[float(G.MDE_cT.min()),float(G.MDE_cT.max())],
 n_Tlag_sig=int((G.pTlag<0.05).sum()),n_T3_sig=int((G.pT3<0.05).sum()),n_trend_pos=int(((G.p_trend<0.05)&(G.trend_dec>0)).sum()),n_trend_neg=int(((G.p_trend<0.05)&(G.trend_dec<0)).sum()),
 n_lagQ_neg_sig=int(((G.p_lagQ<0.05)&(G.lagQ<0)).sum()),n_lagQ_pos_sig=int(((G.p_lagQ<0.05)&(G.lagQ>0)).sum()),b_med=float(G.b.median()),bdiff_med=float(G.b_diff.median()),b5_med=float(G.b_5yr.median()))
for thr in (0.3,0.5,0.7):
  g=A[A.R2_rating>=thr]; summ[f'thr{thr}']=dict(n=len(g),b_med=round(float(g.b.median()),2),cT_detect=int(((g.cT_lo>0)|(g.cT_hi<0)).sum()),hydP_over_totP_med=round(float((g.hydP/g.totP)[g.totP_lo>0].median()),2) if (g.totP_lo>0).any() else None)
print(json.dumps(summ,indent=1)); json.dump(summ,open('rev3_summary.json','w'),indent=1)
