"""Step 6. Sub-reach decay robustness (Table / Section 5.1)."""
# %%
import os, sys
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:  # running inside a Jupyter notebook
    sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "src")))
from config import *  # sets paths; working directory becomes results/

# %%
import pandas as pd,numpy as np,json,warnings; warnings.filterwarnings('ignore')
import statsmodels.api as sm
from scipy.optimize import least_squares
from scipy.stats import kendalltau
D=pd.read_csv('reach_budget.csv'); post=D[D.yr>=2004].copy(); post['t']=post.yr-2003; QM=post.Q_ZC.mean(); post['q']=np.log(post.Q_ZC/QM); post['qs']=np.log(post.Q_SS/post.Q_SS.mean())
R={}
X=sm.add_constant(np.c_[post.q,post.t]); rng=np.random.default_rng(11)
for k,qc in [('up','q'),('lo','qs')]:
  Xk=sm.add_constant(np.c_[post[qc],post.t]); o=sm.OLS(np.log(post['J_'+k].values),Xk).fit()
  fv=o.fittedvalues; rs=o.resid; n=len(rs); out={}
  ac=np.corrcoef(rs[:-1],rs[1:])[0,1]; out['lag1']=round(ac,2)
  oh=sm.OLS(np.log(post['J_'+k].values),Xk).fit(cov_type='HAC',cov_kwds={'maxlags':3}); out['p_hac']=float(oh.pvalues[2]); out['slope_ci_hac']=np.round(oh.conf_int()[2],4).tolist()
  for L in (3,4):
    tb=[]
    for _ in range(4000):
      idx=np.concatenate([np.arange(s,s+L) for s in rng.integers(0,n-L+1,int(np.ceil(n/L)))])[:n]
      s=sm.OLS(fv+rs[idx],Xk).fit().params[2]; tb.append(s)
    tb=np.array(tb); out['slope_boot_L%d'%L]=np.round(np.percentile(tb,[2.5,50,97.5]),4).tolist(); out['tau_boot_L%d'%L]=[round(-1/v,1) if v<0 else None for v in np.percentile(tb,[2.5,50,97.5])]
  tt,pp=kendalltau(post.yr,post['J_'+k]); out['MK']=[round(tt,2),round(pp,3)]
  out['mean_0414']=round(post[post.yr<=2014]['J_'+k].mean(),1); out['mean_1522']=round(post[post.yr>=2015]['J_'+k].mean(),1)
  out['m']=[round(o.params[1],2),round(o.bse[1],2)]; out['tau']=round(-1/o.params[2],1) if o.params[2]<0 else None
  R[k]=out
# anticorrelation from shared Shashi term
ru=sm.OLS(np.log(post.J_up.values),X).fit().resid; rl=sm.OLS(np.log(post.J_lo.values),X).fit().resid
R['corr_resid_up_lo']=round(np.corrcoef(ru,rl)[0,1],2)
R['corr_Jup_Jlo']=round(np.corrcoef(post.J_up,post.J_lo)[0,1],2)
# mixture model for whole reach: J=(A e^{-t/tau}+C)(Q/QM)^m
y=np.log(post.J_all.values); t=post.t.values; q=post.q.values; n=len(y)
def aicc(rss,k): return n*np.log(rss/n)+2*k+2*k*(k+1)/(n-k-1)
mix=least_squares(lambda p:np.log(np.exp(p[0])*np.exp(-t/p[1])+np.exp(p[2]))+p[3]*q-y,[3,11,3,1],bounds=([-5,1,-5,-1],[6,500,6,4]))
ex=least_squares(lambda p:p[0]+p[1]*q-t/p[2]-y,[3.8,1.2,28])
R['mix']=dict(A=round(np.exp(mix.x[0]),1),tau=round(mix.x[1],1),C=round(np.exp(mix.x[2]),1),m=round(mix.x[3],2),AICc=round(aicc(np.sum(mix.fun**2),5),1))
R['exp_AICc']=round(aicc(np.sum(ex.fun**2),4),1)
# joint sub-reach model: J_up=A e^{-t/tau} (Q)^m1 ; J_lo=C (Q)^m2 -> sum ; compare with fits
json.dump(R,open('sub_results.json','w'),indent=1,default=float); print(json.dumps(R,indent=1,default=float))
