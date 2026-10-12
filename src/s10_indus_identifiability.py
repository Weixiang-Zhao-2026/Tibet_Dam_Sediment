"""Step 10. Identifiability and stability of lambda in the upper Indus reach; long-term residuals R_i."""
# %%
import os, sys
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:  # running inside a Jupyter notebook
    sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "src")))
from config import *  # sets paths; working directory becomes results/

# %%
import pandas as pd,numpy as np,json
from scipy.optimize import least_squares
S=pd.read_csv('tp_series.csv'); rng=np.random.default_rng(9)
a=S[S.station=='Partab Bridge'].set_index('yr'); b=S[S.station=='Besham Qila'].set_index('yr'); yy=np.array(sorted(a.index.intersection(b.index)))
si=a.loc[yy,'S'].values; so=b.loc[yy,'S'].values; Q=b.loc[yy,'Q'].values
def fit(si,so,Q):
  QM=Q.mean(); f=lambda q:q[0]*si+np.exp(q[1])*(Q/QM)**q[2]-so
  return least_squares(f,[0.3,np.log(so.mean()/2),1],bounds=([-3,-10,-5],[5,10,8]))
r=fit(si,so,Q); n=len(so); k=3
J=r.jac; cov=np.linalg.inv(J.T@J)*np.sum(r.fun**2)/(n-k); se=np.sqrt(np.diag(cov)); corr=cov/np.outer(se,se)
rss=np.sum(r.fun**2); R2=1-rss/np.sum((so-so.mean())**2); aicc=n*np.log(rss/n)+2*(k+1)+2*(k+1)*(k+2)/(n-k-2)
# alt models: no inflow term, inflow only
f0=least_squares(lambda q:np.exp(q[0])*(Q/Q.mean())**q[1]-so,[np.log(so.mean()),1]); rss0=np.sum(f0.fun**2); aicc0=n*np.log(rss0/n)+2*3+2*3*4/(n-4)
f1=least_squares(lambda q:q[0]*si+q[1]-so,[1,0]); rss1=np.sum(f1.fun**2); aicc1=n*np.log(rss1/n)+2*3+2*3*4/(n-4)
out=dict(n=int(n),lam=round(r.x[0],3),lam_se=round(se[0],3),m=round(r.x[2],2),B0=float(np.exp(r.x[1])),R2=round(R2,2),RMSE=float(np.sqrt(rss/n)),meanSout=float(so.mean()),AICc=round(aicc,1),AICc_noInflow=round(aicc0,1),AICc_inflowOnly=round(aicc1,1),
 corr_lam_m=round(corr[0,2],2),corr_lam_lnB0=round(corr[0,1],2))
loo=[fit(np.delete(si,i),np.delete(so,i),np.delete(Q,i)).x[0] for i in range(n)]
out['loo']=[round(min(loo),3),round(np.median(loo),3),round(max(loo),3)]
h=n//2; out['first_half']=round(fit(si[:h],so[:h],Q[:h]).x[0],3); out['second_half']=round(fit(si[h:],so[h:],Q[h:]).x[0],3)
for L in (1,2,3,4,5):
  v=[]
  for _ in range(1500):
    idx=np.concatenate([np.arange(s,s+L) for s in rng.integers(0,n-L+1,int(np.ceil(n/L)))])[:n]; v.append(fit(si[idx],so[idx],Q[idx]).x[0])
  out[f'boot_L{L}']=[round(float(np.percentile(v,p)),2) for p in (2.5,50,97.5)]
# R2 of rating at each station for context
print(json.dumps(out,indent=1))
# P2 station residuals using full-record b
A=pd.read_csv('tp_station_results.csv').set_index('station'); rows=[]
for st,d in S.groupby('station',sort=False):
  d=d.sort_values('yr'); h=len(d)//2; d1,d2=d.iloc[:h],d.iloc[h:]
  bb=A.loc[st,'b']; dQ=np.log(d2.Q).mean()-np.log(d1.Q).mean(); dS=np.log(d2.S).mean()-np.log(d1.S).mean()
  rows.append(dict(station=st,good=A.loc[st,'R2_rating']>=0.3,b=bb,dlnQ=dQ,dlnS=dS,Ri=dS-bb*dQ))
X=pd.DataFrame(rows); G=X[X.good]
out2=dict(n=len(G),median_abs_R=round(float(G.Ri.abs().median()),2),n_absR_lt_0p2=int((G.Ri.abs()<0.2).sum()),n_absR_lt_0p3=int((G.Ri.abs()<0.3).sum()),median_R=round(float(G.Ri.median()),2),
 R=G[['station','Ri']].round(2).to_dict('records'),median_abs_dlnS=round(float(G.dlnS.abs().median()),2),
 n_sig=None)
# significance of R_i via block bootstrap of halves means
print(json.dumps(out2,indent=1))
json.dump(dict(indus=out,residuals=out2),open('rev5.json','w'),indent=1,default=float)
