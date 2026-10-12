"""Step 1. Read Yangtze station data (runoff 1e8 m3/yr, load converted to Mt/yr)."""
# %%
import os, sys
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:  # running inside a Jupyter notebook
    sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "src")))
from config import *  # sets paths; working directory becomes results/

# %%
import pandas as pd,numpy as np,warnings; warnings.filterwarnings('ignore')
y=pd.read_excel(XLS_YZ,sheet_name='原始数据',header=None)
yrs=pd.to_numeric(y[2],errors='coerce'); mm=(yrs>1940)&(yrs<=2022)
def col(n,k=1):
  c=[i for i in range(y.shape[1]) if y.iat[1,i]==n][0]; return pd.to_numeric(y.loc[mm,c+k],errors='coerce').values
YR=yrs[mm].astype(int).values
D=pd.DataFrame({'yr':YR})
for n,k in [('宜昌','YC'),('枝城','ZC'),('沙市','SS'),('监利','JL'),('新江口','XJK'),('沙道观','SDG'),('弥陀寺','MTS'),('藕池(康三)','KJG'),('藕池(管)','GJP'),('屏山(向家坝)','PS'),('寸滩','CT'),('石鼓','SG'),('华弹(白鹤滩)','HT'),('攀枝花','PZH')]:
  D['S_'+k]=col(n)/100.0   # Mt/yr
  D['Q_'+k]=col(n,0)       # 1e8 m3/yr
D.to_csv('stations_Mt.csv',index=False)
