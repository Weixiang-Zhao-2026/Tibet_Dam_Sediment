"""Step 3. Discharge-conditioned anomaly I_s for 18 stations (incl. Batang, Shigu)."""
# %%
import os, sys
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:  # running inside a Jupyter notebook
    sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "src")))
from config import *  # sets paths; working directory becomes results/

# %%
import pandas as pd,numpy as np,json,warnings; warnings.filterwarnings('ignore')
from scipy.stats import kendalltau,pearsonr,spearmanr
from scipy.optimize import least_squares
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans'})
R={}
def sen(x,y):
  s=[(y[j]-y[i])/(x[j]-x[i]) for i in range(len(x)) for j in range(i+1,len(x)) if x[j]!=x[i]]; return np.median(s)
# ---------- A. headwater stations ----------
EN={'澜沧江':'Lancang','疏勒河':'Shule','塔里木河':'Tarim','怒江':'Nu','雅鲁藏布江':'Yarlung Tsangpo','印度河':'Indus','金沙江':'Jinsha'}
STN={'香达':'Xiangda','旧州':'Jiuzhou','昌马堡':'Changmabao','潘家庄':'Panjiazhuang','卡群':'Kaqun','同古孜洛克':'Tongguziluoke','新大河':'Xindahe','焉耆':'Yanqi','阿拉尔':'Alar','道街坝':'Daojieba','拉萨':'Lhasa','奴下':'Nuxia','巴塘':'Batang','石鼓':'Shigu'}
x=pd.read_excel(XLS_TP,sheet_name=None,header=None)
H={}
for sh,df in x.items():
  for c in range(df.shape[1]):
    if str(df.iat[0,c]).startswith('站点名'):
      d=df.iloc[2:,c:c+5].apply(pd.to_numeric,errors='coerce'); d.columns=['yr','Q','S','Tm','P']; d=d.dropna(subset=['yr','Q','S']); d=d[(d.Q>0)&(d.S>0)]
      H[(EN[sh],STN.get(df.iat[0,c+1],df.iat[0,c+1]))]=d.reset_index(drop=True)
y=pd.read_excel(XLS_YZ,sheet_name='原始数据',header=None)
yrs=pd.to_numeric(y[2],errors='coerce'); mm=(yrs>1940)&(yrs<=2022)
def col(n,k=1):
  c=[i for i in range(y.shape[1]) if y.iat[1,i]==n][0]; return pd.to_numeric(y.loc[mm,c+k],errors='coerce').values
YR=yrs[mm].astype(int).values
for n in ['巴塘','石鼓']:
  d=pd.DataFrame({'yr':YR,'Q':col(n,0),'S':col(n)}).dropna(); d=d[(d.Q>0)&(d.S>0)]; H[('Jinsha',STN[n])]=d.reset_index(drop=True)
rows=[]; IS={}
for (rv,st),d in H.items():
  h=int(np.ceil(len(d)/2)); base=d.iloc[:h]; late=d.iloc[h:]
  b,a=np.polyfit(np.log(base.Q),np.log(base.S),1); r2=np.corrcoef(np.log(base.Q),np.log(base.S))[0,1]**2
  Is=np.log(d.S)-(a+b*np.log(d.Q)); IS[(rv,st)]=(d.yr.values,Is.values,h)
  t,p=kendalltau(d.yr,Is); ss=sen(d.yr.values,Is.values)*10
  r={'River':rv,'Station':st,'Period':f'{int(d.yr.min())}–{int(d.yr.max())}','n':len(d),'b0':round(b,2),'R2_0':round(r2,2),
     'dlnQ':round(np.log(late.Q.mean()/base.Q.mean()),2),'dlnS':round(np.log(late.S.mean()/base.S.mean()),2),
     'Is_late':round(late.index.map(lambda i:Is[i]).to_series().mean(),2),'tau_Is':round(t,2),'p_Is':round(p,3),'sen_Is_dec':round(ss,3)}
  if 'Tm' in d and d.Tm.notna().sum()>5:
    r['dT']=round(late.Tm.mean()-base.Tm.mean(),2); r['dP_pct']=round(100*(late.P.mean()/base.P.mean()-1),1)
    tt,pp=kendalltau(d.yr,d.Tm); r['tau_T']=round(tt,2); r['p_T']=round(pp,3)
  rows.append(r)
A=pd.DataFrame(rows); print(A.to_string()); A.to_csv('tp_Is_table.csv',index=False)
Ac=A.dropna(subset=['dT'])
R['nsig_T']=int((Ac.p_T<0.05).sum()); R['n_T']=len(Ac)
R['n_Is_pos_sig']=int(((A.p_Is<0.05)&(A.tau_Is>0)).sum()); R['n_Is_neg_sig']=int(((A.p_Is<0.05)&(A.tau_Is<0)).sum()); R['nst']=len(A)
for v in ['dT','dP_pct']:
  rs,ps=spearmanr(Ac[v],Ac.Is_late); R['rho_Is_'+v]=(round(rs,2),round(ps,3))
print(R)
