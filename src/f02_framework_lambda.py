"""Fig. 2. Climate pathways, regime diagram, transmission coefficients and bootstrap difference."""
# %%
import os, sys
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:  # running inside a Jupyter notebook
    sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "src")))
from config import *  # sets paths; working directory becomes results/

# %%
import numpy as np,json,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':'0.92','grid.linewidth':.6,'axes.axisbelow':True,'legend.frameon':False,'font.size':8,'font.family':'DejaVu Sans'})
R=json.load(open('rev4.json'))
fig=plt.figure(figsize=(7.4,5.6))
a=fig.add_axes([0.02,0.52,0.56,0.44]); a.axis('off'); a.set_xlim(0,10); a.set_ylim(0,6.2)
def box(x,y,w,h,t,c,fs=7.2):
  a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.1',fc=c,ec='k',lw=.7)); a.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs)
box(0.2,3.9,2.2,1.3,'Climate\n$T$, $P$','#fde0c5'); box(4.0,3.9,2.2,1.3,'Runoff $Q$','#dbe9f6'); box(7.6,2.2,2.2,1.3,'Sediment\nflux $S$','#e8e0d0'); box(4.0,0.5,2.2,1.3,'Sediment\navailability','#e2f0d9')
a.annotate('',(4.0,4.55),(2.4,4.55),arrowprops=dict(arrowstyle='-|>',lw=1.2)); a.text(3.2,4.75,r'$\varepsilon^{Q}_{X}$',ha='center',fontsize=8)
a.annotate('',(7.6,3.1),(6.2,4.3),arrowprops=dict(arrowstyle='-|>',lw=2.0,color='#2c7fb8')); a.text(7.2,4.15,r'$\times\,b$',fontsize=8.5,color='#2c7fb8')
a.annotate('',(4.0,1.15),(1.4,3.85),arrowprops=dict(arrowstyle='-|>',lw=1.2,color='#e08214',ls='--')); a.annotate('',(7.6,2.6),(6.2,1.2),arrowprops=dict(arrowstyle='-|>',lw=1.2,color='#e08214',ls='--'))
a.text(1.5,2.2,'glacier retreat,\npermafrost thaw,\nrainfall erosion',fontsize=6.3,color='#a35a0b'); a.text(7.1,1.3,r'$c_X$',fontsize=8.5,color='#a35a0b')
a.text(5.0,5.75,r'$\varepsilon^{S}_{X}\;=\;b\,\varepsilon^{Q}_{X}\;+\;c_X$',ha='center',fontsize=9)
a.set_title('(a) Two pathways from climate to sediment flux',fontsize=8.5,loc='left')
# (b) regime diagram
b=fig.add_axes([0.66,0.58,0.32,0.36])
pi=np.logspace(-1.5,1,400)
b.plot(pi,np.where(pi<1,1,0),'k-',lw=1.6,label='single reach, Eq. (4)')
for w,ls in [(0.35,'--'),(0.8,':')]:
  lam=1-0.5*(1+np.tanh(np.log(pi)/w)); b.plot(pi,lam,color='0.45',ls=ls,lw=1.1,label=f'heterogeneous reach (spread {w})')
b.axvline(1,color='0.7',lw=.6); b.set_xscale('log'); b.set_ylim(-0.1,1.15)
b.text(0.08,0.62,'sediment-\nstarved:\nflux follows\nsupply',fontsize=6.2,ha='center',color='#7b1d12'); b.text(3.3,0.62,'sediment-\nrich:\nflux follows\nwater',fontsize=6.2,ha='center',color='#0b3d91')
b.set_xlabel(r'Supply ratio $\Pi_s=S_{in}/S_c$'); b.set_ylabel(r'$\lambda=\partial S_{out}/\partial S_{in}$'); b.legend(fontsize=5.8,loc='lower left')
b.set_title('(b) Regime diagram',fontsize=8.5,loc='left')
# (c) observed lambda
c=fig.add_axes([0.08,0.08,0.52,0.36])
labs=['Jingjiang\nwhole reach','Zhicheng–\nShashi','Shashi–\nJianli','Upper Indus\nPartab–Besham']
vals=[R['whole']['cal']['lam'],R['upper']['cal']['lam'],R['lower']['cal']['lam'],R['indus']['lam']]
ses=[R['whole']['cal']['lam_se'],R['upper']['cal']['lam_se'],R['lower']['cal']['lam_se'],R['indus']['lam_se']]
cols=['#c0392b','#e67e22','#8e44ad','#e6ab02']
for i,(v,s_,cl) in enumerate(zip(vals,ses,cols)):
  c.errorbar(i,v,yerr=1.96*s_,fmt='o',ms=7,color=cl,capsize=4,lw=1.3)
c.axhline(1,color='#7b1d12',ls='--',lw=.8); c.text(3.45,1.03,'starved limit',fontsize=6.3,color='#7b1d12',ha='right')
c.axhline(0,color='#0b3d91',ls='--',lw=.8); c.text(3.45,0.03,'capacity limit',fontsize=6.3,color='#0b3d91',ha='right')
c.set_xticks(range(4)); c.set_xticklabels(labs,fontsize=6.6); c.set_ylabel(r'Estimated $\lambda$ ($\pm$95% CI)'); c.set_ylim(-0.25,1.35); c.set_xlim(-0.5,3.5)
c.axvspan(2.5,3.5,color='#fff3cd',alpha=.5,zorder=0); c.text(3,1.25,'plateau',ha='center',fontsize=6.5); c.text(1,1.25,'below the dams',ha='center',fontsize=6.5)
c.set_title('(c) Inflow transmission coefficient',fontsize=8.5,loc='left')
# (d) bootstrap difference
d=fig.add_axes([0.70,0.08,0.28,0.36])
import numpy.random as nr
bt=R['boot']; lo,med,hi=bt['dlam']
dl=np.load('boot_dlam.npy'); hh,ed=np.histogram(dl,bins=40,range=(-0.3,1.5)); d.bar(ed[:-1],hh/hh.max(),width=ed[1]-ed[0],align='edge',color='#2c7fb8',alpha=.45)
d.axvline(0,color='k',lw=.7); d.axvline(med,color='#2c7fb8',lw=1.2); d.axvspan(lo,hi,ymin=0,ymax=0.06,color='#2c7fb8')
d.text(med+0.05,0.92,r'$\Delta\lambda$='+f'{med:.2f}\n95% CI {lo:.2f}–{hi:.2f}\n$p$={bt["p_two"]:.3f}',fontsize=6.4)
d.set_yticks([]); d.set_xlabel(r'$\Delta\lambda=\lambda_{\rm Jingjiang}-\lambda_{\rm Indus}$'); d.set_title('(d) Block-bootstrap difference',fontsize=8.5,loc='left')
fig.savefig(os.path.join(FIG,'Fig2_framework_lambda.png'),dpi=300); print('ok')
