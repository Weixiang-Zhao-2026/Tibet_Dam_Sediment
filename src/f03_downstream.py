"""Fig. 3. Plateau sediment intercepted by reservoirs and Jingjiang sub-reach erosion."""
# %%
import os, sys
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:  # running inside a Jupyter notebook
    sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "src")))
from config import *  # sets paths; working directory becomes results/

# %%
import numpy as np,pandas as pd,json,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import statsmodels.api as sm
plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':'0.92','grid.linewidth':.6,'axes.axisbelow':True,'legend.frameon':False,'font.size':8,'font.family':'DejaVu Sans'})
# ---- Fig6 downstream contrast
Y=pd.read_csv('stations_Mt.csv'); D=pd.read_csv('reach_budget.csv')
fig,ax=plt.subplots(1,2,figsize=(7.2,3.1))
a=ax[0]
for c,l,cl in [('S_SG','Shigu (plateau outlet, undammed)','#1f4e9c'),('S_PS','Pingshan (below Jinsha cascade)','#c0392b'),('S_ZC','Zhicheng (below TGD)','#7f7f7f')]:
  d=Y[(Y[c]>0)&(Y.yr>=1960)]; a.semilogy(d.yr,d[c],'-o',ms=1.8,lw=1.1,color=cl,label=l)
for yv in [2003.4,2012.8,2020.0,2021.3]: a.axvline(yv,color='0.6',ls='--',lw=.6)
a.set_ylim(.4,1500); a.set_xlabel('Year'); a.set_ylabel('Suspended load (Mt yr$^{-1}$)'); a.legend(fontsize=6.3,loc='lower left'); a.set_title('(a) Plateau sediment intercepted by reservoirs',fontsize=8.5,loc='left')
b=ax[1]; p=D[D.yr>=2004].copy(); p['t']=p.yr-2003
for col,cl,lab in [('J_up','#e67e22','Zhicheng–Shashi'),('J_lo','#8e44ad','Shashi–Jianli')]:
  b.plot(p.yr,p[col],'o',ms=3.5,color=cl,label=lab)
  o=sm.OLS(np.log(p[col].values),sm.add_constant(p.t.values)).fit(); tt=np.linspace(1,19,50); b.plot(2003+tt,np.exp(o.params[0]+o.params[1]*tt),'-',color=cl,lw=1.3)
b.text(2014.5,4.5,r'$\tau\approx11$ yr',color='#e67e22',fontsize=7); b.text(2015,25.5,'no decay',color='#8e44ad',fontsize=7)
b.set_xticks(range(2004,2023,3)); b.set_xlabel('Year'); b.set_ylabel(r'Bed erosion $J$ (Mt yr$^{-1}$)'); b.legend(fontsize=6.5,loc='upper right'); b.set_title('(b) Sediment-starved reach below the TGD',fontsize=8.5,loc='left')
plt.tight_layout(); plt.savefig(os.path.join(FIG,'Fig3_downstream.png'),dpi=300); plt.close(); print('ok')
