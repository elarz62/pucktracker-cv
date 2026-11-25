import numpy as np, pandas as pd, json, sys, cv2
from scipy.spatial.transform import Rotation as Rot, Slerp
from cam4 import Hcam, Pcam, dist, undist
cfg=json.load(open(sys.argv[1])); N=360
P=np.load(cfg['pout']); Q=np.load(cfg['pout'].replace('.npy','_q.npy'))
cal=np.load(cfg['cal']); C=cal[:3]; b,dy,g=cal[3:6]; k1=cal[6]; W=42.5
ts=np.arange(N)/6.0; sl=Slerp(ts,Rot.from_rotvec(P[:,:3]))
red,blue,yel=Q[:,0],Q[:,1],Q[:,2]
good=~((np.nan_to_num(red)>6)|(np.nan_to_num(yel)>6))
def pose(t):
    t=float(np.clip(t,0,ts[-1])); return sl(t).as_rotvec(), np.interp(t,ts,P[:,3])
def good_at(t): i=int(np.clip(round(t*6),0,N-1)); return bool(good[i])
# --- standard-rink stretch: measured -> standard diagram (blue 25, goal 89, dots y 22, boards 42.5)
def to_std(X,Y):
    ax=np.abs(X); s=np.sign(X)
    xs=np.where(ax<=b, ax*25/b, np.where(ax<=g, 25+(ax-b)*64/(g-b), 89+(ax-g)*64/(g-b)))
    ay=np.abs(Y); ys=np.where(ay<=dy, ay*22/dy, 22+(ay-dy)*(20.5/(W-dy)))
    return s*xs, np.sign(Y)*ys
tr=pd.read_csv(cfg['tracks'])
tr['t']=tr.frame/cfg['fps']-cfg['T0']+cfg['lag']; tr=tr[(tr.t>=0)&(tr.t<60)].copy()
teams=pd.read_csv(cfg['teams']).set_index('track_id')['team']; tr['team0']=tr.track_id.map(teams).fillna('unknown')
tr=tr[tr.team0!='stands']
foot=undist(np.c_[(tr.x1+tr.x2)/2,tr.y2].astype(float),k1); X=np.zeros((len(tr),2))
for t,idx in tr.groupby('t').indices.items():
    r,f=pose(t); Hi=np.linalg.inv(Hcam(r,f,C)); w=np.c_[foot[idx],np.ones(len(idx))]@Hi.T; X[idx]=w[:,:2]/w[:,2:]
tr['X'],tr['Y']=X[:,0],X[:,1]
L=g+11*(g-b)/64
tr=tr[(tr.X.abs()<L+2)&(tr.Y.abs()<W+2)]
tr=tr[[good_at(t) for t in tr.t]]
# team per track from this clip: blue jersey fraction, then height for refs
feat={}
for i in range(1,N+1):
    t=(i-1)/6; rows=tr[np.abs(tr.t-t)<0.5/cfg['fps']]
    if not len(rows): continue
    im=cv2.imread(f"{cfg['frdir']}/%04d.jpg"%i); hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV); r,f=pose(t); Pm=Pcam(r,f,C)
    for rw in rows.itertuples():
        x1,y1,x2,y2=int(rw.x1),int(rw.y1),int(rw.x2),int(rw.y2); top=hsv[y1:y1+max(2,int((y2-y1)*0.55)),x1:x2]
        if top.size==0: continue
        bl=(((top[...,0]>=cfg.get('hue_lo',95))&(top[...,0]<=cfg.get('hue_hi',130))) if not cfg.get('colored_any') else True)
        bl=(bl&(top[...,1]>cfg.get('smin',90))&(top[...,2]>cfg.get('vmin',50))).mean()
        Z=np.linspace(0,9,91); w=np.c_[np.full(91,rw.X),np.full(91,rw.Y),Z,np.ones(91)]@Pm.T; yy=dist(w[:,:2]/w[:,2:],k1)[:,1]
        feat.setdefault(rw.track_id,[]).append((bl,Z[np.argmin(np.abs(yy-y1))],rw.Y))
hmed=np.median([np.median(np.array(v)[:,1]) for v in feat.values() if len(v)>=3])
print('median kid height',round(hmed,2)); print('bl dist',np.round(np.percentile([np.median(np.array(v)[:,0]) for v in feat.values()],[10,25,50,75,90]),3))
team={}
for tid,v in feat.items():
    v=np.array(v); bl=np.median(v[:,0]); near=v[v[:,2]<10]; h=np.median(near[:,1]) if len(near) else np.median(v[:,1])
    team[tid]='b' if bl>cfg.get('blue_frac',0.10) else ('r' if h>cfg.get('ref_ratio',1.4)*hmed else 'w')
import collections
H_=sorted([(round(float(np.median(np.array(v)[:,1])),1),len(v),tid,team[tid]) for tid,v in feat.items()],reverse=True); print(H_[:14])
tr['team']=tr.track_id.map(team).fillna('u')
out=[]
for tid,gq in tr.sort_values('t').groupby('track_id'):
    gq=gq.copy()
    for c in ('X','Y'): gq[c]=gq[c].rolling(5,center=True,min_periods=1).median().rolling(5,center=True,min_periods=1).mean()
    out.append(gq)
tr=pd.concat(out); xs,ys=to_std(tr.X.values,tr.Y.values); tr['sx'],tr['sy']=xs,ys
step=round(cfg['fps']/12); tr=tr[(tr.frame%step)==0]
frames=[[round(t,3),[[int(r.track_id),round(r.sx,1),round(r.sy,1),r.team] for r in gq.itertuples()]] for t,gq in tr.groupby('t')]
# per 1/6 s: homography (measured rink -> video px, pre-distortion) and camera footprint in standard coords
Hs=[];fp=[]
bx=np.r_[np.c_[np.linspace(0,1919,30),np.full(30,1079)],np.c_[np.full(12,1919),np.linspace(1079,0,12)],np.c_[np.linspace(1919,0,30),np.zeros(30)],np.c_[np.zeros(12),np.linspace(0,1079,12)]]
for i in range(N):
    H=Hcam(P[i,:3],P[i,3],C); H=H/H[2,2]; Hs.append([round(v,7) for v in H.ravel()] if good[i] else None)
    w=np.c_[undist(bx,k1),np.ones(len(bx))]@np.linalg.inv(H).T; ok=w[:,2]>0
    pts=w[ok,:2]/w[ok,2:]; pts=pts[(np.abs(pts[:,0])<300)&(np.abs(pts[:,1])<200)]
    if good[i] and len(pts)>2:
        sx,sy=to_std(np.clip(pts[:,0],-150,150),np.clip(pts[:,1],-80,80)); fp.append(np.round(np.c_[sx,sy],1).tolist())
    else: fp.append([])
dims={'W':W,'blue':float(b),'doty':float(dy),'goal':float(g)}
json.dump({'game':cfg['title'],'dims':dims,'k1':float(k1),'frames':frames,'H':Hs,'foot':fp,'hstep':1/6,'good':[int(x) for x in good]},open(cfg['out'],'w'),separators=(',',':'))
print('teams',pd.Series(team).value_counts().to_dict(),'frames',len(frames),'mean dots',np.mean([len(f[1]) for f in frames]).round(1),'good %',round(good.mean()*100))
