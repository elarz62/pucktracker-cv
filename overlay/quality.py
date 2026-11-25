import numpy as np, json, sys
from scipy.ndimage import map_coordinates
sys.argv=[sys.argv[0],sys.argv[1]]
src=open('snap4.py').read(); exec(src[:src.index('poses={}')])
P=np.load(cfg['pout']); q=[]
for i in range(1,N+1):
    dt,occ=masks(i); H=Hcam(P[i-1][:3],P[i-1][3],C); row=[]
    for k in ('red','blue','yel'):
        Pk=M[k] if k!='red' else M['red'][:171]   # center line only
        p,ok=img(H,Pk); qq=p*S; d=dt[k]
        ins=ok&(qq[:,0]>=0)&(qq[:,0]<d.shape[1]-1)&(qq[:,1]>=0)&(qq[:,1]<d.shape[0]-1)
        if ins.sum()<10: row.append(np.nan); continue
        qi=qq[ins]; v=map_coordinates(d,[qi[:,1],qi[:,0]],order=1); v=v[~occ[qi[:,1].astype(int),qi[:,0].astype(int)]]
        row.append(np.median(v)/S if len(v) else np.nan)
    q.append(row)
q=np.array(q); np.save(cfg['pout'].replace('.npy','_q.npy'),q)
print('median px err (full res) red,blue,yellow:',np.nanmedian(q,0).round(1),' p90:',np.nanpercentile(q,90,axis=0).round(1))
bad=np.flatnonzero(np.nanmax(q,1)>6)+1; print('frames with any >6px:',len(bad),bad[:40])
