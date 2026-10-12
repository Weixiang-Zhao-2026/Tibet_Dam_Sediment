"""Step 12. LaTeX rows for Appendix Table B1."""
# %%
import os, sys
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:  # running inside a Jupyter notebook
    sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "src")))
from config import *  # sets paths; working directory becomes results/

# %%
import pandas as pd
A=pd.read_csv('tp_station_results.csv'); I=pd.read_csv('tp_Is_table.csv').set_index('Station')
out=[]; prev=None
for _,r in A.iterrows():
  dag='' if r.T_trend_p<0.05 else '$^{\\dagger}$'
  ss='$^*$' if r.S_trend_p<0.05 else ''; qs='$^*$' if r.Q_trend_p<0.05 else ''
  isv=I.loc[r.station]; iss='$^*$' if isv.p_Is<0.05 else ''
  riv=r.river if r.river!=prev else ''; prev=r.river
  out.append(f"{riv} & {r.station} & {r.period.replace(chr(8211),'--')} & {r.n} & {r.R2_rating:.2f} & {r.b:.2f} [{r.b_lo:.2f}, {r.b_hi:.2f}] & {r.T_sen_dec:.2f}{dag} & {r.Q_trend_tau:+.2f}{qs} & {r.S_trend_tau:+.2f}{ss} & {isv.tau_Is:+.2f}{iss} \\\\")
for st in ['Batang','Shigu']:
  isv=I.loc[st]; iss='$^*$' if isv.p_Is<0.05 else ''
  b={'Batang':2.45,'Shigu':2.97}[st]; r2={'Batang':0.63,'Shigu':0.81}[st]; per={'Batang':'1964--2022','Shigu':'1958--2022'}[st]; n={'Batang':47,'Shigu':63}[st]
  out.append(f"{'Jinsha' if st=='Batang' else ''} & {st} & {per} & {n} & {r2:.2f} & {b:.2f} & -- & -- & -- & {isv.tau_Is:+.2f}{iss} \\\\")
open('tableB1_rows.tex','w').write('\n'.join(out).replace('+','$+$').replace('& -0','& $-$0').replace(' -0.',' $-$0.'))
