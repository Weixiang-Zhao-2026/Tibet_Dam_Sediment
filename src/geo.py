"""Readers for GSHHG/WDB-II binaries shipped with basemap-data (pip install basemap-data)."""
import numpy as np
import mpl_toolkits.basemap_data as _b
P=list(_b.__path__)[0]+'/'
def read(name,res='i',bbox=None):
  meta=open(P+f'{name}meta_{res}.dat').read().splitlines(); raw=open(P+f'{name}_{res}.dat','rb').read(); out=[]
  for ln in meta:
    f=ln.split()
    if len(f)<7: continue
    lev=int(f[0]); n=int(f[2]); s,nn=float(f[3]),float(f[4]); off,nb=int(f[5]),int(f[6])
    if bbox and (nn<bbox[2] or s>bbox[3]): continue
    a=np.frombuffer(raw[off:off+nb],dtype='<f8' if nb==n*16 else '<f4').reshape(-1,2)
    if bbox and (a[:,0].max()<bbox[0] or a[:,0].min()>bbox[1]): continue
    out.append((lev,a))
  return out
if __name__=='__main__':
  r=read('rivers','i',(88,123,22,37)); print(len(r), r[0][1][:3]); g=read('gshhs','i',(88,123,22,37)); print(len(g),set(l for l,_ in g)); c=read('countries','i',(70,140,15,55)); print(len(c))
