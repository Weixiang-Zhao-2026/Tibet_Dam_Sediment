"""Step 5. Engelund-Hansen scoping estimate of sand transport capacity, lower Jingjiang (Appendix A)."""
# %%
import os, sys
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:  # running inside a Jupyter notebook
    sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "src")))
from config import *  # sets paths; working directory becomes results/

# %%
# Engelund-Hansen (1967) scoping estimate of bed-material transport capacity, lower Jingjiang (sand bed)
import numpy as np,pandas as pd,json
rng=np.random.default_rng(3); N=20000
D=pd.read_csv('reach_budget.csv'); post=D[D.yr>=2004]
Qm=post.Q_SS.mean()*1e8/3.15576e7     # mean discharge m3/s at Shashi
h=rng.uniform(8,12,N); B=rng.uniform(1000,1500,N); S=rng.uniform(3e-5,6e-5,N); d=rng.uniform(0.15e-3,0.25e-3,N)
g=9.81; s=2.65; U=Qm/(B*h); f=2*g*h*S/U**2; th=h*S/((s-1)*d); Phi=0.1*th**2.5/f
qs=Phi*np.sqrt((s-1)*g*d**3); Qs=qs*B*2650*3.15576e7/1e9   # Mt/yr
out=dict(Qmean_m3s=round(Qm),cap_Mt_pct=[round(float(np.percentile(Qs,p)),1) for p in (5,50,95)],
 Jianli_load_post=[round(post.S_JL.min(),1),round(post.S_JL.median(),1),round(post.S_JL.max(),1)],J_lo_mean=round(post.J_lo.mean(),1),
 benchmark_resid_median=round((post.Sb-post.S_ZC).median(),1))
json.dump(out,open('eh_results.json','w'),indent=1); print(out)

# %%
# Redraw the downstream supplementary figures now that the capacity estimate exists (Fig. S4 uses it).
import runpy
runpy.run_path(os.path.join(os.path.dirname(RES), 'src', 's04_downstream_budgets.py'), run_name='__main__')
