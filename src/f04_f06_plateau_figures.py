"""Figs. 4-6. Warming and rating exponents (Fig. 4), climate pathways (Fig. 5), availability change and out-of-period test (Fig. 6)."""
# %%
import os, sys
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:  # running inside a Jupyter notebook
    sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "src")))
from config import *  # sets paths; working directory becomes results/

# %%
import pandas as pd,numpy as np,json,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':'0.92','grid.linewidth':.6,'axes.axisbelow':True,'legend.frameon':False,'font.size':8,'font.family':'DejaVu Sans'})
A=pd.read_csv('tp_station_results.csv'); I=pd.read_csv('tp_Is_table.csv'); S=pd.read_csv('tp_series.csv')
A['good']=A.R2_rating>=0.3
COL={'Lancang':'#1b9e77','Shule':'#d95f02','Tarim':'#7570b3','Nu':'#e7298a','Yarlung Tsangpo':'#66a61e','Indus':'#e6ab02'}
order=A.sort_values(['river','station']).station.tolist()
A=A.set_index('station').loc[order].reset_index()
lab=[f"{s}" for s in A.station]; yy=np.arange(len(A))[::-1]
# ---------- Fig3: warming + rating exponent
fig,ax=plt.subplots(1,2,figsize=(7.2,3.9),sharey=True)
c=[COL[r] for r in A.river]
ax[0].barh(yy,A.T_sen_dec,color=c,alpha=[1 if p<.05 else .35 for p in A.T_trend_p] if False else 1)
for y,p,v in zip(yy,A.T_trend_p,A.T_sen_dec):
  if p>=.05: ax[0].barh(y,v,color='white',hatch='////',edgecolor='0.5',lw=.3)
ax[0].set_yticks(yy); ax[0].set_yticklabels(lab,fontsize=7); ax[0].set_xlabel('Warming trend (°C decade$^{-1}$)'); ax[0].set_title('(a) Air-temperature trend',fontsize=8.5,loc='left')
for y,b,lo,hi,g,cc in zip(yy,A.b,A.b_lo,A.b_hi,A.good,c):
  ax[1].plot([lo,hi],[y,y],color=cc,lw=1.2,alpha=1 if g else .4); ax[1].plot(b,y,'o',ms=5,mfc=cc if g else 'white',mec=cc,mew=1.2)
ax[1].axvspan(0,1,color='0.9',zorder=0); ax[1].text(0.5,len(A)-0.3,'supply-\nlimited',ha='center',va='top',fontsize=6.5,color='0.35')
ax[1].axvline(1,color='0.5',lw=.6,ls=':'); ax[1].axvline(np.median(A[A.good].b),color='k',lw=.7,ls='--'); ax[1].text(np.median(A[A.good].b)+.07,0.2,'median,\nwell-rated',fontsize=6.3)
ax[1].set_xlim(-0.2,4.9); ax[1].set_xlabel('Rating exponent $b$ in $S\\propto Q^{b}$ (95% CI)'); ax[1].set_title('(b) Sediment-rating exponent',fontsize=8.5,loc='left')
from matplotlib.patches import Patch
h=[Patch(color=v,label=k) for k,v in COL.items()]; fig.legend(handles=h,loc='lower center',ncol=6,fontsize=6.8,bbox_to_anchor=(0.55,0.0))
ax[1].plot([],[],'o',mfc='white',mec='k',label='rating $R^2<0.3$'); ax[1].legend(loc='upper right',fontsize=6.5)
plt.tight_layout(rect=(0,0.05,1,1)); plt.savefig(os.path.join(FIG,'Fig4_warming_ratings.png'),dpi=300); plt.close()
# ---------- Fig4: elasticity decomposition
G=A[A.good].copy()
fig,ax=plt.subplots(1,2,figsize=(7.2,3.4),gridspec_kw={'width_ratios':[1,1.25]})
a=ax[0]
for _,r in G.iterrows():
  a.errorbar(r.hydP,r.totP,xerr=[[r.hydP-r.hydP_lo],[r.hydP_hi-r.hydP]],yerr=[[r.totP-r.totP_lo],[r.totP_hi-r.totP]],fmt='o',ms=5,color=COL[r.river],elinewidth=.6,capsize=0)
  off={'Nuxia':(.12,-.35),'Daojieba':(-1.25,.15),'Kaqun':(.1,.3),'Tongguziluoke':(.15,-.35),'Yanqi':(.15,-.05),'Xindahe':(-.2,-.55),'Dainyor':(.1,.2)}.get(r.station,(.08,.12))
  a.text(r.hydP+off[0],r.totP+off[1],r.station,fontsize=5.8,color='0.25')
l=[-2.7,5.6]; a.plot(l,l,'k--',lw=.7); a.set_xlim(l); a.set_ylim(-2.2,5.8)
a.set_xlabel(r'Runoff pathway $b\,\varepsilon^{Q}_{P}$'); a.set_ylabel(r'Total elasticity $\varepsilon^{S}_{P}$'); a.set_title('(a) Precipitation: $\\varepsilon^S_P$ vs runoff pathway',fontsize=8.5,loc='left')
b=ax[1]; y=np.arange(len(G))[::-1]; w=.36
b.barh(y+w/2,G.hydT,height=w,color='#2c7fb8',label='runoff pathway $b\\,\\partial\\ln Q/\\partial T$')
b.errorbar(G.hydT,y+w/2,xerr=[G.hydT-G.hydT_lo,G.hydT_hi-G.hydT],fmt='none',ecolor='k',elinewidth=.6)
b.barh(y-w/2,G.cT,height=w,color='#fdae61',label='direct (fixed-$Q$) pathway $c_T$')
b.errorbar(G.cT,y-w/2,xerr=[G.cT-G.cT_lo,G.cT_hi-G.cT],fmt='none',ecolor='k',elinewidth=.6)
b.axvline(0,color='k',lw=.6); b.set_yticks(y); b.set_yticklabels(G.station,fontsize=7); b.set_xlabel('Sensitivity of ln $S$ to temperature (°C$^{-1}$)'); b.legend(fontsize=6.5,loc='lower right'); b.set_title('(b) Temperature: two pathways',fontsize=8.5,loc='left')
plt.tight_layout(); plt.savefig(os.path.join(FIG,'Fig5_climate_pathways.png'),dpi=300); plt.close()
# ---------- Fig5: long-term anomaly, rating change, cross-period prediction
X=pd.read_csv('rev4_crossperiod.csv')
fig,ax=plt.subplots(1,3,figsize=(7.4,2.9),gridspec_kw={'width_ratios':[1.35,1,1]})
a=ax[0]
for st,cc in [('Lhasa','#66a61e'),('Daojieba','#e7298a'),('Changmabao','#d95f02'),('Yanqi','#7570b3'),('Dainyor','#e6ab02')]:
  d=S[S.station==st].sort_values('yr'); h=int(np.ceil(len(d)/2)); bq,aq=np.polyfit(np.log(d.Q[:h]),np.log(d.S[:h]),1)
  Is=np.log(d.S)-(aq+bq*np.log(d.Q)); sm=pd.Series(Is.values).rolling(7,center=True,min_periods=4).mean()
  a.plot(d.yr,Is,'.',color=cc,ms=2.5,alpha=.4); a.plot(d.yr,sm,'-',color=cc,lw=1.6,label=st)
a.axhline(0,color='k',lw=.6); a.set_xlabel('Year'); a.set_ylabel(r'$I_s=\ln S-\ln\hat S(Q)$'); a.legend(fontsize=6,ncol=2,loc='lower left'); a.set_title('(a) Discharge-conditioned anomaly',fontsize=8.5,loc='left')
b=ax[1]
for _,r in A.iterrows():
  sig=r.p_db<0.05
  b.plot(r.b_early,r.b_late,'o',ms=6 if sig else 4.5,mfc=COL[r.river] if sig else 'white',mec=COL[r.river],mew=1.2)
  if sig: b.text(r.b_early+.07,r.b_late+.07,r.station,fontsize=5.8)
l=[0,3.8]; b.plot(l,l,'k--',lw=.7); b.set_xlim(l); b.set_ylim(l); b.set_xlabel('$b$, first half'); b.set_ylabel('$b$, second half')
b.text(2.2,0.3,'flattening',fontsize=6.3,color='0.35'); b.text(0.15,3.3,'steepening',fontsize=6.3,color='0.35')
b.set_title('(b) Rating-exponent change',fontsize=8.5,loc='left')
c=ax[2]; G=X[X.good]
riv=A.set_index('station').river
for _,r in G.iterrows():
  c.plot(r.dlnS_pred,r.dlnS_obs,'o',ms=5.5,color=COL[riv[r.station]])
  if abs(r.dlnS_obs-r.dlnS_pred)>0.3: c.text(r.dlnS_pred+.04,r.dlnS_obs,r.station,fontsize=5.8)
l=[-1.7,1.0]; c.plot(l,l,'k--',lw=.7); c.set_xlim(l); c.set_ylim(l)
c.set_xlabel(r'Predicted $\Delta\ln S=b_{\rm train}\,\Delta\ln Q$'); c.set_ylabel(r'Observed $\Delta\ln S$'); c.set_title('(c) Out-of-period test',fontsize=8.5,loc='left')
plt.tight_layout(); plt.savefig(os.path.join(FIG,'Fig6_availability.png'),dpi=300); plt.close()
