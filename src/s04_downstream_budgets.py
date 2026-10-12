"""Step 4. Jingjiang reach budgets, rate-law decay models, robustness tests; supplementary figures S1-S5. Run again after s05 to draw Fig. S4."""
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
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':'0.92','grid.linewidth':.6,'axes.axisbelow':True,'xtick.direction':'out','ytick.direction':'out','legend.frameon':False,'font.size':8.5,'font.family':'DejaVu Sans','axes.linewidth':.7})
D=pd.read_csv('stations_Mt.csv'); D=D[D.yr>=1980].reset_index(drop=True)
R={}
# reach budgets (Mt/yr), signed J = Sout - Sin  (>0 erosion)
D['J_all']=D.S_JL+D.S_XJK+D.S_SDG+D.S_MTS+D.S_KJG+D.S_GJP-D.S_ZC
D['J_up']=D.S_SS+D.S_XJK+D.S_SDG+D.S_MTS-D.S_ZC          # Zhicheng-Shashi (Songzi + Taiping outlets)
D['J_lo']=D.S_JL+D.S_KJG+D.S_GJP-D.S_SS                   # Shashi-Jianli (Ouchi outlet)
D['Sout_all']=D.J_all+D.S_ZC
# uncertainty: eps calibrated on pre-dam residuals of whole reach
pre=D[D.yr<=2002]; post=D[D.yr>=2004].copy()
def mag(d,terms): return np.sqrt(sum(d[t]**2 for t in terms))
T_all=['S_ZC','S_JL','S_XJK','S_SDG','S_MTS','S_KJG','S_GJP']; T_up=['S_ZC','S_SS','S_XJK','S_SDG','S_MTS']; T_lo=['S_SS','S_JL','S_KJG','S_GJP']
eps=np.sqrt(np.mean(pre.J_all**2/mag(pre,T_all)**2)); R['eps']=round(eps,3)
for k,Tm in [('all',T_all),('up',T_up),('lo',T_lo)]: D['sig_'+k]=eps*mag(D,Tm)
pre=D[D.yr<=2002]; post=D[D.yr>=2004].copy()
for k in ['all','up','lo']:
  R['pre_'+k]=[round(pre['J_'+k].mean(),1),round(pre['J_'+k].std(),1)]
  R['post_'+k]=[round(post['J_'+k].mean(),1),round(post['J_'+k].std(),1),int((post['J_'+k]>0).sum())]
  R['cum0322_'+k]=round(D[(D.yr>=2003)&(D.yr<=2022)]['J_'+k].sum(),0); R['cum0320_'+k]=round(D[(D.yr>=2003)&(D.yr<=2020)]['J_'+k].sum(),0)
R['sig_post_all']=[round(post.sig_all.min(),1),round(post.sig_all.max(),1)]
R['Jall_0414']=round(post[post.yr<=2014].J_all.mean(),1); R['Jall_1522']=round(post[post.yr>=2015].J_all.mean(),1)
R['Q_0414']=round(post[post.yr<=2014].Q_ZC.mean()); R['Q_1522']=round(post[post.yr>=2015].Q_ZC.mean())
# historical rating benchmark (HRB) for each reach outlet
for k,qcol,out in [('all','Q_ZC','Sout_all')]:
  b,a=np.polyfit(np.log(pre[qcol]),np.log(pre[out]),1); R['hrb_b']=round(b,2); R['hrb_R2']=round(np.corrcoef(np.log(pre[qcol]),np.log(pre[out]))[0,1]**2,2)
  D['Sb']=np.exp(a)*D[qcol]**b
D['Pi']=(D.S_ZC+D.J_all.clip(lower=0))/D.Sb   # supply ratio wrt benchmark
D['resid_cap']=D.Sb-D.S_ZC; D['ratio']=D.J_all/D.resid_cap
pre=D[D.yr<=2002]; post=D[D.yr>=2004].copy()
R['ratio_post']=[round(post.ratio.min(),3),round(post.ratio.max(),3)]
R['Pi_pre']=[round(pre.Pi.median(),2)]; R['Pi_post']=[round(post.Pi.min(),3),round(post.Pi.max(),3)]
# signed-J consistency pre-dam: corr between observed J and (Sb - Sin)
r=np.corrcoef(pre.J_all,pre.Sb-pre.S_ZC)[0,1]; R['pre_corr_J_vs_excess']=round(r,2)
R['pre_sign_agree']=round(np.mean(np.sign(pre.J_all)==np.sign(pre.Sb-pre.S_ZC)),2)
# ---- decay models on post-dam data
post['t']=post.yr-2003; QM=post.Q_ZC.mean(); post['q']=np.log(post.Q_ZC/QM)
def ols(dd,col):
  X=sm.add_constant(np.c_[dd.q,dd.t]); return sm.OLS(np.log(dd[col].values),X).fit()
fits={}
for k in ['all','up','lo']:
  ok=post['J_'+k]>0; dd=post[ok]
  o=ols(dd,'J_'+k); oh=sm.OLS(np.log(dd['J_'+k].values),sm.add_constant(np.c_[dd.q,dd.t])).fit(cov_type='HAC',cov_kwds={'maxlags':2})
  b=o.params[2]; fits[k]=o
  R['fit_'+k]=dict(n=int(len(dd)),tau=round(-1/b,1) if b<0 else None,slope=round(b,4),p=round(o.pvalues[2],4),p_hac=round(oh.pvalues[2],5),m=round(o.params[1],2),m_se=round(o.bse[1],2),DW=round(sm.stats.durbin_watson(o.resid),2))
  o2=ols(dd[dd.yr<=2021],'J_'+k); R['fit_'+k]['tau_no2022']=round(-1/o2.params[2],1) if o2.params[2]<0 else None
# rolling (whole reach)
roll=[]
for end in range(2014,2023):
  dd=post[post.yr<=end]; o=ols(dd,'J_all'); b=o.params[2]; lo,hi=o.conf_int()[2]
  roll.append(dict(end=end,n=len(dd),tau=(-1/b if b<0 else np.nan),tau_lo=(-1/lo if lo<0 else np.nan),tau_hi=(-1/hi if hi<0 else np.inf),p=o.pvalues[2],m=o.params[1]))
RO=pd.DataFrame(roll); R['rolling']=RO.round(3).to_dict('records')
# block bootstrap (whole reach)
o=fits['all']; X=sm.add_constant(np.c_[post.q,post.t]); fv=o.fittedvalues; rs=o.resid; n=len(rs); rng=np.random.default_rng(7)
for L in (1,3):
  tb=[]
  for _ in range(4000):
    idx=np.concatenate([np.arange(s,s+L) for s in rng.integers(0,n-L+1,int(np.ceil(n/L)))])[:n]
    s=sm.OLS(fv+rs[idx],X).fit().params[2]; tb.append(-1/s if s<0 else np.inf)
  R['boot_L%d'%L]=[round(float(np.percentile(tb,q)),1) for q in (2.5,50,97.5)]
# MC measurement error
tm=[]
for _ in range(2000):
  Jp=np.clip(post.J_all+rng.normal(0,post.sig_all),0.5,None); s=sm.OLS(np.log(Jp.values),X).fit().params[2]; tm.append(-1/s if s<0 else np.inf)
R['mc_tau']=[round(float(np.percentile(tm,q)),1) for q in (2.5,50,97.5)]
R['corr_q_t']=round(np.corrcoef(post.q,post.t)[0,1],2)
for a,b in [(2004,2013),(2014,2022)]:
  dd=post[(post.yr>=a)&(post.yr<=b)]; oo=ols(dd,'J_all'); R['m_%d_%d'%(a,b)]=[round(oo.params[1],2),round(oo.bse[1],2)]
oi=sm.OLS(np.log(post.J_all.values),sm.add_constant(np.c_[post.q,post.t,post.q*post.t])).fit(); R['p_interaction']=round(oi.pvalues[3],3)
# competing forms (AICc)
def aicc(rss,n,k): return n*np.log(rss/n)+2*k+2*k*(k+1)/(n-k-1)
y=np.log(post.J_all.values); n=len(y); forms={}
def fitf(f,p0): r=least_squares(lambda p:f(p)-y,p0); return r
forms['No decay']=(fitf(lambda p:p[0]+p[1]*post.q.values,[3,1]),3)
forms['Exponential']=(fitf(lambda p:p[0]+p[1]*post.q.values-post.t.values/p[2],[3.8,1,30]),4)
forms['Power law']=(fitf(lambda p:p[0]+p[1]*post.q.values-p[2]*np.log(post.t.values),[3.8,1,.2]),4)
forms['Stretched exponential']=(fitf(lambda p:p[0]+p[1]*post.q.values-(post.t.values/p[2])**p[3],[3.8,1,25,1.5]),5)
amin=min(aicc(np.sum(v[0].fun**2),n,v[1]) for v in forms.values())
tab=[]
tr=post.yr.values<=2015
for k,(r,kp) in forms.items():
  f={'No decay':lambda p,d:p[0]+p[1]*d.q.values,'Exponential':lambda p,d:p[0]+p[1]*d.q.values-d.t.values/p[2],'Power law':lambda p,d:p[0]+p[1]*d.q.values-p[2]*np.log(d.t.values),'Stretched exponential':lambda p,d:p[0]+p[1]*d.q.values-(d.t.values/p[2])**p[3]}[k]
  r2=least_squares(lambda p:f(p,post[tr])-y[tr],r.x); hold=np.sqrt(np.mean((f(r2.x,post[~tr])-y[~tr])**2))
  loo=[]
  for i in range(n):
    m=np.arange(n)!=i; ri=least_squares(lambda p:f(p,post[m])-y[m],r.x); loo.append((f(ri.x,post.iloc[[i]])[0]-y[i])**2)
  tab.append(dict(model=k,k=kp,AICc=round(aicc(np.sum(r.fun**2),n,kp),1),dAICc=round(aicc(np.sum(r.fun**2),n,kp)-amin,1),hold=round(hold,3),loo=round(np.sqrt(np.mean(loo)),3),params=np.round(r.x,3).tolist()))
TB=pd.DataFrame(tab); R['models']=TB.to_dict('records')
# discriminator: J_all = a + b Q + c S_in + d t
ol=sm.OLS(post.J_all.values,sm.add_constant(np.c_[post.Q_ZC,post.S_ZC,post.t])).fit(); R['disc_c']=[round(ol.params[2],3),round(ol.bse[2],3)]
# effective store identity
E0=np.exp(fits['all'].params[0]); tau=-1/fits['all'].params[2]; R['E0']=round(E0,1); R['E0tau_Mt']=round(E0*tau)
# survey comparison (MWR 2020 bulletin: Jingjiang bankfull erosion Oct2002-Oct2020 = 12.29e8 m3)
V=12.29e8; R['survey_Gt']=[round(V*rb/1e9,2) for rb in (1.3,1.6)]; R['budget_survey_ratio']=[round(R['cum0320_all']/(V*rb/1e6),2) for rb in (1.6,1.3)]
# upstream supply chronology
D.to_csv('reach_budget.csv',index=False); TB.to_csv('model_table.csv',index=False); RO.to_csv('rolling_tau.csv',index=False)
json.dump(R,open('results.json','w'),indent=1,default=float)
print(json.dumps(R,indent=0,default=float)[:6000])
# ================= FIGURES =================
Y=pd.read_csv('stations_Mt.csv')
# Fig2 supply chronology
fig,ax=plt.subplots(figsize=(7.2,3.6))
for c,l in [('S_SG','Shigu (upper Jinsha, undammed)'),('S_PS','Pingshan/Xiangjiaba (Jinsha outlet)'),('S_CT','Cuntan (TGR inflow)'),('S_YC','Yichang (TGD outflow)'),('S_ZC','Zhicheng (Jingjiang inlet)')]:
  d=Y[(Y[c]>0)&(Y.yr>=1955)]; ax.semilogy(d.yr,d[c],'-',lw=1.1,marker='o',ms=1.8,label=l)
for yv,l in [(2003.4,'TGD 2003'),(2012.8,'XJB/XLD 2012–13'),(2020.0,'Wudongde 2020'),(2021.3,'Baihetan 2021')]:
  ax.axvline(yv,color='grey',ls='--',lw=.6); ax.text(yv-.25,1400,l,rotation=90,fontsize=5.8,ha='right',va='top',color='dimgrey')
ax.set_ylim(.4,1500); ax.set_xlabel('Year'); ax.set_ylabel('Suspended sediment load (Mt yr$^{-1}$)'); ax.legend(fontsize=6.5,frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(0.5,-0.2))
plt.tight_layout(); plt.savefig(os.path.join(FIGS,'FigS1_supply_chronology.png'),dpi=300); plt.close()
# Fig3 reach budgets
fig,ax=plt.subplots(1,2,figsize=(7.2,3.1),gridspec_kw={'width_ratios':[1.25,1]})
a=ax[0]; a.errorbar(D.yr,D.J_all,yerr=D.sig_all,fmt='o',ms=3,color='k',ecolor='0.6',lw=.6,label='Zhicheng–Jianli')
a.axhline(0,color='k',lw=.5); a.axvline(2003,color='C3',ls='--',lw=.8); a.text(2003.5,-120,'TGD',color='C3',fontsize=7)
a.set_xlabel('Year'); a.set_ylabel(r'Net bed exchange $J$ (Mt yr$^{-1}$)'); a.set_title('(a) Whole reach, 1980–2022',fontsize=8.5)
b=ax[1]; p=D[D.yr>=2003]
b.plot(p.yr,p.J_up,'s-',ms=3,lw=1,color='C0',label='Zhicheng–Shashi'); b.plot(p.yr,p.J_lo,'^-',ms=3,lw=1,color='C1',label='Shashi–Jianli')
b.axhline(0,color='k',lw=.5); b.set_xlabel('Year'); b.set_ylabel(r'$J$ (Mt yr$^{-1}$)'); b.legend(fontsize=7,frameon=False); b.set_title('(b) Sub-reaches, 2003–2022',fontsize=8.5)
plt.tight_layout(); plt.savefig(os.path.join(FIGS,'FigS2_reach_budgets.png'),dpi=300); plt.close()
# Fig4 flow-normalized erosion + fits + rolling tau
o=fits['all']; post['Jn']=post.J_all*np.exp(-o.params[1]*post.q)
fig,ax=plt.subplots(1,2,figsize=(7.2,3.1))
a=ax[0]; a.errorbar(post.yr,post.Jn,yerr=post.sig_all*np.exp(-o.params[1]*post.q),fmt='o',ms=3.5,color='k',ecolor='0.6',lw=.6,label=r'$J_n=J\,(Q/\bar Q)^{-m}$')
tt=np.linspace(1,19,100); a.plot(2003+tt,np.exp(o.params[0]-tt/(-1/o.params[2])),'C0-',label=r'exponential, $\tau$=%.0f yr'%(-1/o.params[2]))
a.plot(2003+tt,np.full_like(tt,post.Jn.mean()),'C7--',lw=.8,label='no decay (mean)')
a.set_xticks(range(2004,2023,3)); a.set_xlabel('Year'); a.set_ylabel(r'Flow-normalized $J_n$ (Mt yr$^{-1}$)'); a.legend(fontsize=6.6,frameon=False); a.set_title('(a) Discharge-normalized erosion',fontsize=8.5)
b=ax[1]; RR=RO.copy(); hi=RR.tau_hi.replace(np.inf,300)
b.fill_between(RR.end,RR.tau_lo,hi,color='C0',alpha=.2,label='95% CI'); b.plot(RR.end,RR.tau,'o-',color='C0',ms=3.5,label=r'$\hat\tau$')
b.set_yscale('log'); b.set_ylim(8,300); b.set_xlabel('Last year of fitting window (start 2004)'); b.set_ylabel(r'$\tau$ (yr)'); b.legend(fontsize=6.8,frameon=False); b.set_title('(b) Expanding-window estimates',fontsize=8.5)
plt.tight_layout(); plt.savefig(os.path.join(FIGS,'FigS3_erosion_decay.png'),dpi=300); plt.close()
# Fig5 benchmark vs physics-based scoping
import json as _j
if not os.path.exists('eh_results.json'):
    EH={'cap_Mt_pct':[np.nan,np.nan,np.nan]}
else:
    EH=_j.load(open('eh_results.json'))
fig,ax=plt.subplots(figsize=(4.6,3.3))
ax.axhspan(EH['cap_Mt_pct'][0],EH['cap_Mt_pct'][2],color='C2',alpha=.2,lw=0,label='Engelund–Hansen scoping capacity,\nlower reach at mean Q (5–95%)')
ax.loglog(post.resid_cap,post.J_all,'o',color='C0',ms=4,label='Whole reach J, 2004–2022')
ax.loglog(post.resid_cap,post.J_lo,'^',color='C1',ms=4,label='Shashi–Jianli J, 2004–2022')
l=[3,1500]; ax.plot(l,l,'k--',lw=.7); ax.text(9,12,'1:1',fontsize=6.5,rotation=37)
ax.set_xlim(l); ax.set_ylim(3,1500); ax.set_xlabel(r'Residual rating benchmark $S_b(Q)-S_{in}$ (Mt yr$^{-1}$)'); ax.set_ylabel(r'$J$ or capacity (Mt yr$^{-1}$)'); ax.legend(fontsize=6.3,frameon=False,loc='upper left')
plt.tight_layout(); plt.savefig(os.path.join(FIGS,'FigS4_capacity_benchmark.png'),dpi=300); plt.close()
# Fig6 budget closure
fig,ax=plt.subplots(figsize=(4.4,3.2)); p=D[D.yr>=2003]
c=p.J_all.cumsum(); cu=p.J_up.cumsum(); ax.plot(p.yr,c,'k-o',ms=2.5,label='Suspended-load budget, Zhicheng–Jianli')
ax.fill_between(p.yr,(p.J_all-p.sig_all).cumsum(),(p.J_all+p.sig_all).cumsum(),color='0.8')
ax.errorbar([2020],[np.mean(R['survey_Gt'])*1000],yerr=[[np.mean(R['survey_Gt'])*1000-R['survey_Gt'][0]*1000],[R['survey_Gt'][1]*1000-np.mean(R['survey_Gt'])*1000]],fmt='s',color='C3',ms=5,capsize=3,label='Survey, Zhicheng–Chenglingji\n(12.29×10$^8$ m$^3$, ρ$_b$=1.3–1.6 t m$^{-3}$)')
ax.set_xticks(range(2003,2023,4)); ax.set_xlabel('Year'); ax.set_ylabel('Cumulative erosion since 2003 (Mt)'); ax.legend(fontsize=6.3,frameon=False,loc='upper left'); ax.set_ylim(0,2200)
plt.tight_layout(); plt.savefig(os.path.join(FIGS,'FigS5_budget_closure.png'),dpi=300); plt.close()
