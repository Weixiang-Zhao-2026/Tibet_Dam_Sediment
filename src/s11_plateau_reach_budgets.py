"""Step 11. Exploratory plateau reach budgets (Tarim, Shule, Indus, Yarlung)."""
# %%
import os, sys
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:  # running inside a Jupyter notebook
    sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "src")))
from config import *  # sets paths; working directory becomes results/

# %%
import pandas as pd,numpy as np,statsmodels.api as sm,warnings; warnings.filterwarnings('ignore')
S=pd.read_csv('tp_series.csv')
def g(st): d=S[S.station==st][['yr','Q','S','T','P']].set_index('yr'); return d
rng=np.random.default_rng(1)
def test(out,ins,name):
  o=g(out); I=[g(s) for s in ins]; yrs=o.index
  for d in I: yrs=yrs.intersection(d.index)
  So=o.loc[yrs,'S']; Qo=o.loc[yrs,'Q']; Si=sum(d.loc[yrs,'S'] for d in I); Qi=sum(d.loc[yrs,'Q'] for d in I)
  J=So-Si; frac=-J/Si
  X=sm.add_constant(np.c_[np.log(Qo),np.log(Si)]); m=sm.OLS(np.log(So.values),X).fit(cov_type='HAC',cov_kwds={'maxlags':2})
  X2=sm.add_constant(np.c_[np.log(Qi),np.log(Si)]); m2=sm.OLS(np.log(So.values),X2).fit(cov_type='HAC',cov_kwds={'maxlags':2})
  from scipy.stats import kendalltau
  tf,pf=kendalltau(yrs,frac)
  print(f'{name}: n={len(yrs)} {yrs.min()}-{yrs.max()}  mean Sin={Si.mean():.0f} Sout={So.mean():.0f}  mean trapped frac={frac.mean():.2f}  trend tau={tf:.2f} p={pf:.3f}')
  print(f'   lnSout = b lnQout + g lnSin : b={m.params[1]:.2f}±{m.bse[1]:.2f}  gamma={m.params[2]:.2f}±{m.bse[2]:.2f} (p={m.pvalues[2]:.3f}); with Qin: gamma={m2.params[2]:.2f}±{m2.bse[2]:.2f}; Qout/Qin={ (Qo/Qi).mean():.2f}')
  early=frac[yrs<yrs.min()+ (yrs.max()-yrs.min())/2].mean(); late=frac[yrs>=yrs.min()+(yrs.max()-yrs.min())/2].mean(); print(f'   trapped frac early {early:.2f} late {late:.2f}')
  return pd.DataFrame({'yr':yrs,'Sin':Si.values,'Sout':So.values,'Qin':Qi.values,'Qout':Qo.values})
a=test('Alar',['Kaqun','Tongguziluoke','Xindahe'],'Tarim: 3 headwaters -> Alar')
b=test('Panjiazhuang',['Changmabao'],'Shule: Changmabao -> Panjiazhuang')
c=test('Besham Qila',['Partab Bridge'],'Indus: Partab Bridge -> Besham Qila')
d=test('Nuxia',['Lhasa'],'Yarlung: Lhasa(trib) -> Nuxia (not closed)')
a.to_csv('tarim_budget.csv',index=False); b.to_csv('shule_budget.csv',index=False)
