"""Step 9. Cumulative-forcing model, block-bootstrap Delta-lambda, out-of-period test."""
# %%
import os, sys
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:  # running inside a Jupyter notebook
    sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "src")))
from config import *  # sets paths; working directory becomes results/

# %%
import pandas as pd,numpy as np,json,warnings; warnings.filterwarnings('ignore')
from scipy.optimize import least_squares
rng=np.random.default_rng(42)
D=pd.read_csv('reach_budget.csv'); p=D[D.yr>=2004].reset_index(drop=True); p['t']=p.yr-2003
S=pd.read_csv('tp_series.csv')
def aicc(rss,n,k): return n*np.log(rss/n)+2*k+2*k*(k+1)/(n-k-1)
def fit_cal(si,so,Q,t):
  QM=Q.mean(); f=lambda q:q[0]*si+np.exp(q[1])*(Q/QM)**q[2]*np.exp(-t/q[3])-so
  return least_squares(f,[1,np.log(30),1,30],bounds=([-3,-5,-3,1],[5,8,6,1e4]))
def fit_cum(si,so,Q):
  QM=Q.mean()
  def f(q):
    fq=(Q/QM)**q[2]; F=np.cumsum(fq)-fq/2   # cumulative forcing to mid-year
    return q[0]*si+np.exp(q[1])*fq*np.exp(-F/q[3])-so
  return least_squares(f,[1,np.log(30),1,30],bounds=([-3,-5,-3,0.5],[5,8,6,1e4]))
def fit_nodecay(si,so,Q):
  QM=Q.mean(); f=lambda q:q[0]*si+np.exp(q[1])*(Q/QM)**q[2]-so
  return least_squares(f,[1,np.log(30),1],bounds=([-3,-10,-5],[5,10,8]))
def se(r,n,k):
  J=r.jac; return np.sqrt(np.diag(np.linalg.inv(J.T@J)*np.sum(r.fun**2)/(n-k)))
R={}
reaches={'whole':(p.S_ZC.values,p.Sout_all.values,p.Q_ZC.values),
 'upper':(p.S_ZC.values,(p.S_SS+p.S_XJK+p.S_SDG+p.S_MTS).values,p.Q_ZC.values),
 'lower':(p.S_SS.values,(p.S_JL+p.S_KJG+p.S_GJP).values,p.Q_SS.values)}
t=p.t.values.astype(float)
for k,(si,so,Q) in reaches.items():
  rc=fit_cal(si,so,Q,t); ru=fit_cum(si,so,Q); n=len(so)
  R[k]=dict(cal=dict(lam=round(rc.x[0],3),lam_se=round(se(rc,n,4)[0],3),tau=round(rc.x[3],1),m=round(rc.x[2],2),AICc=round(aicc(np.sum(rc.fun**2),n,5),1)),
            cum=dict(lam=round(ru.x[0],3),lam_se=round(se(ru,n,4)[0],3),tauF=round(ru.x[3],1),m=round(ru.x[2],2),AICc=round(aicc(np.sum(ru.fun**2),n,5),1)))
# Indus
a=S[S.station=='Partab Bridge'].set_index('yr'); b=S[S.station=='Besham Qila'].set_index('yr'); yy=a.index.intersection(b.index)
isi,iso,iQ=a.loc[yy,'S'].values,b.loc[yy,'S'].values,b.loc[yy,'Q'].values
ri=fit_nodecay(isi,iso,iQ); R['indus']=dict(lam=round(ri.x[0],3),lam_se=round(se(ri,len(iso),3)[0],3))
# block bootstrap (pairs, moving blocks L=3) for lambda in each and difference
def mbb(n,L=3):
  return np.concatenate([np.arange(s,s+L) for s in rng.integers(0,n-L+1,int(np.ceil(n/L)))])[:n]
si,so,Q=reaches['whole']; lw=[]; li=[]
for _ in range(2000):
  i=mbb(len(so)); r=fit_cal(si[i],so[i],Q[i],t[i]); lw.append(r.x[0])
  j=mbb(len(iso)); r2=fit_nodecay(isi[j],iso[j],iQ[j]); li.append(r2.x[0])
lw=np.array(lw); li=np.array(li); dl=lw-li; np.save('boot_dlam.npy',dl)
q=lambda v:[round(float(np.percentile(v,x)),2) for x in (2.5,50,97.5)]
R['boot']=dict(lam_whole=q(lw),lam_indus=q(li),dlam=q(dl),p_one_sided=float(np.mean(dl<=0)),p_two=float(2*min(np.mean(dl<=0),np.mean(dl>=0))))
dz=(R['whole']['cal']['lam']-R['indus']['lam'])/np.sqrt(R['whole']['cal']['lam_se']**2+R['indus']['lam_se']**2); R['z']=round(float(dz),2)
# cross-period prediction at plateau stations
A=pd.read_csv('tp_station_results.csv').set_index('station'); rows=[]
for st,d in S.groupby('station',sort=False):
  d=d.sort_values('yr'); n=len(d); h=n//2; d1=d.iloc[:h]; d2=d.iloc[h:]
  b1,a1=np.polyfit(np.log(d1.Q),np.log(d1.S),1)
  dQ=np.log(d2.Q).mean()-np.log(d1.Q).mean(); dSo=np.log(d2.S).mean()-np.log(d1.S).mean(); dSp=b1*dQ
  pred=a1+b1*np.log(d2.Q.values); obs=np.log(d2.S.values); r2=1-np.sum((obs-pred)**2)/np.sum((obs-obs.mean())**2)
  cr=np.corrcoef(pred,obs)[0,1]
  # reverse: train on second half predict first
  b2,a2=np.polyfit(np.log(d2.Q),np.log(d2.S),1); dSp2=b2*(-dQ)
  rows.append(dict(station=st,good=A.loc[st,'R2_rating']>=0.3,b_train=round(b1,2),dlnQ=round(dQ,3),dlnS_obs=round(dSo,3),dlnS_pred=round(dSp,3),dlnS_pred_rev=round(dSp2,3),r_annual=round(cr,2),R2_oos=round(r2,2)))
X=pd.DataFrame(rows); X.to_csv('rev4_crossperiod.csv',index=False); print(X.to_string())
G=X[X.good]
R['cross']=dict(n=len(G),corr_pred_obs=round(float(np.corrcoef(G.dlnS_pred,G.dlnS_obs)[0,1]),2),rmse=round(float(np.sqrt(np.mean((G.dlnS_pred-G.dlnS_obs)**2))),3),
 mean_obs=round(float(G.dlnS_obs.mean()),3),mean_pred=round(float(G.dlnS_pred.mean()),3),median_r_annual=round(float(G.r_annual.median()),2),n_r_sig=int((G.r_annual>0.36).sum()),
 resid=G.assign(res=G.dlnS_obs-G.dlnS_pred)[['station','res']].round(2).to_dict('records'))
print(json.dumps(R,indent=1,default=float)); json.dump(R,open('rev4.json','w'),indent=1,default=float)
