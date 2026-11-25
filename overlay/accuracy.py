import numpy as np, json, sys, cv2, pickle
from cam4 import Hcam, dist, undist
cfg=json.load(open(sys.argv[1])); P=np.load(cfg['pout']); Q=np.nan_to_num(np.load(cfg['pout'].replace('.npy','_q.npy')))
cal=np.load(cfg['cal']); C=cal[:3]; b,dy,g=cal[3:6]; k1=cal[6]; N=360
good=(Q[:,0]<=6)&(Q[:,2]<=6)
def H(i): return Hcam(P[i-1,:3],P[i-1,3],C)
def to_world(i,pts):
    w=np.c_[undist(np.asarray(pts,float),k1),np.ones(len(pts))]@np.linalg.inv(H(i)).T; return w[:,:2]/w[:,2:]
def to_img(i,X):
    w=np.c_[X,np.ones(len(X))]@H(i).T; return dist(w[:,:2]/w[:,2:],k1), w[:,2]
# 1) faceoff dots: find red blob near predicted spot, compare in feet
dots=np.array([[sx*(b-5),sy*dy] for sx in(-1,1) for sy in(-1,1)])
errs=[]
for i in range(1,N+1,3):
    if not good[i-1]: continue
    im=cv2.imread(f"{cfg['frdir']}/%04d.jpg"%i).astype(np.float32); bb,gg,rr=cv2.split(im); red=cv2.GaussianBlur(rr-(gg+bb)/2,(7,7),0)
    p,z=to_img(i,dots)
    for d,(x,y),zz in zip(dots,p,z):
        if zz<=0 or not(60<x<1860 and 60<y<1020): continue
        x0,y0=int(x)-45,int(y)-30; c=red[max(y0,0):y0+60,max(x0,0):x0+90]
        if c.size==0 or c.max()<cfg.get('dot_t',15): continue
        m=c>c.max()*0.6; ys,xs=np.nonzero(m)
        if len(xs)<15 or xs.std()>20: continue
        obs=np.array([[max(x0,0)+xs.mean(),max(y0,0)+ys.mean()]]); wd=to_world(i,obs)[0]
        errs.append((i,d[0],d[1],wd[0]-d[0],wd[1]-d[1],np.hypot(*(obs[0]-[x,y]))))
E=np.array(errs)
if len(E):
    e=np.hypot(E[:,3],E[:,4]); print('faceoff dots: n',len(E),'median err ft',round(np.median(e),1),'p90',round(np.percentile(e,90),1),'median px',round(np.median(E[:,5]),1))
    for d in np.unique(E[:,1:3],axis=0):
        s=E[(E[:,1]==d[0])&(E[:,2]==d[1])]; print('  dot',d.round(1),'n',len(s),'mean offset ft',s[:,3:5].mean(0).round(1),'spread ft',s[:,3:5].std(0).round(1))
# 2) repeatability: same ice point seen from two frames, mapped by each frame's mapping
pairs=pickle.load(open(cfg.get('pairs','pairs.pkl'),'rb')); rows=[]
for i,j,a,bq in pairs:
    if j-i<6 or not(good[i-1] and good[j-1]): continue
    wa=to_world(i,a); wb=to_world(j,bq)
    on=(np.abs(wa[:,0])<g+8)&(np.abs(wa[:,1])<42)
    for p,q in zip(wa[on],wb[on]): rows.append((p[0],p[1],np.hypot(*(p-q))))
R=np.array(rows)
if len(R):
    print('repeatability: n',len(R),'median ft',round(np.median(R[:,2]),2),'p90',round(np.percentile(R[:,2],90),2))
    for name,m in [('near half (camera side)',R[:,1]<0),('far half',R[:,1]>=0),('center zone |x|<b',np.abs(R[:,0])<b),('end zones |x|>b',np.abs(R[:,0])>=b)]:
        if m.sum(): print(f'  {name}: n {m.sum()} median {np.median(R[m,2]):.2f} p90 {np.percentile(R[m,2],90):.2f}')
# 3) feet per pixel at typical spots (how much 1 px of box error moves a dot)
i=np.flatnonzero(good)[len(np.flatnonzero(good))//2]+1
for name,X in [('near center ice',(0,-25)),('far boards center',(0,40)),('far end zone',(g-15,30)),('near end zone',(g-15,-20))]:
    p,z=to_img(i,np.array([X],float))
    if z[0]<=0: continue
    w0=to_world(i,p); w1=to_world(i,p+[0,1]); w2=to_world(i,p+[1,0])
    print(f'  1 px at {name} (frame {i}): down {np.hypot(*(w1-w0)[0]):.2f} ft, across {np.hypot(*(w2-w0)[0]):.2f} ft')
print('dims measured: blue',round(b,1),'dot y',round(dy,1),'goal',round(g,1),'camera at',C.round(1),'k1',round(k1,3),'good frames',good.mean().round(2))
