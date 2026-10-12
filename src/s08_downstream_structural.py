"""Step 8. Structural model S_out = lambda*S_in + B0 (Q/Qm)^m exp(-t/tau) for the Jingjiang reach."""
# %%
import os, sys
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:  # running inside a Jupyter notebook
    sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "src")))
from config import *  # sets paths; working directory becomes results/

# %%
import pandas as pd,numpy as np,statsmodels.api as sm,json
from scipy.optimize import least_squares
D=pd.read_csv('reach_budget.csv'); p=D[D.yr>=2004].copy(); p['t']=p.yr-2003
R={}
def fit(so,si,Q,lab):
  QM=Q.mean()
  # log regression with and without trend
  X=sm.add_constant(np.c_[np.log(Q),np.log(si)]); o=sm.OLS(np.log(so.values),X).fit(cov_type='HAC',cov_kwds={'maxlags':2})
  Xt=sm.add_constant(np.c_[np.log(Q),np.log(si),p.t]); ot=sm.OLS(np.log(so.values),Xt).fit(cov_type='HAC',cov_kwds={'maxlags':2})
  ratio=si/so
  # direct nonlinear model S_out = lam*S_in + B0 (Q/QM)^m exp(-t/tau)
  f=lambda q: q[0]*si.values+np.exp(q[1])*(Q.values/QM)**q[2]*np.exp(-p.t.values/q[3])-so.values
  r=least_squares(f,[1,np.log(30),1,30],bounds=([0,-5,-3,1],[3,8,6,1e4]))
  Jm=r.jac; cov=np.linalg.inv(Jm.T@Jm)*np.sum(r.fun**2)/(len(so)-4); se=np.sqrt(np.diag(cov))
  # local-elasticity implied by fitted model, averaged
  B=np.exp(r.x[1])*(Q.values/QM)**r.x[2]*np.exp(-p.t.values/r.x[3]); el=(r.x[0]*si.values)/(r.x[0]*si.values+B)
  R[lab]=dict(betaS=round(o.params[2],3),se=round(o.bse[2],3),betaS_trend=round(ot.params[2],3),se_t=round(ot.bse[2],3),corr_lnQ_lnSin=round(np.corrcoef(np.log(Q),np.log(si))[0,1],2),
    ratio_of_means=round(si.mean()/so.mean(),2),mean_ratio=round(ratio.mean(),2),median_ratio=round(ratio.median(),2),ratio_range=[round(ratio.min(),2),round(ratio.max(),2)],
    lam=round(r.x[0],2),lam_se=round(se[0],2),B0=round(np.exp(r.x[1]),1),m=round(r.x[2],2),tau=round(r.x[3],1),implied_mean_elasticity=round(el.mean(),2),
    corr_J_Sin=round(np.corrcoef(so-si,si)[0,1],2))
fit(p.Sout_all,p.S_ZC,p.Q_ZC,'whole')
fit(p.S_SS+p.S_XJK+p.S_SDG+p.S_MTS,p.S_ZC,p.Q_ZC,'upper')
fit(p.S_JL+p.S_KJG+p.S_GJP,p.S_SS,p.Q_SS,'lower')
print(json.dumps(R,indent=1)); json.dump(R,open('rev3_downstream.json','w'),indent=1)
