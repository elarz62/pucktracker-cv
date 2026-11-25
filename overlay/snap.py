"""Per-frame rink homography by snapping a rink model to painted lines, seeded by frame-to-frame motion."""
import cv2, numpy as np, pandas as pd, json, sys
from scipy.optimize import least_squares
from scipy.ndimage import map_coordinates
cfg=json.load(open(sys.argv[1])); N=360; S=0.5   # work at half resolution
tr=pd.read_csv(cfg['tracks'])
def boxes(i):
    f=int(round(((i-1)/6.0+cfg['T0']-cfg['lag'])*cfg['fps'])); return tr[tr.frame==f][['x1','y1','x2','y2']].values
Hc=np.load(cfg['hchain'])  # frame1 px -> frame i px
dims=cfg['dims']; W=42.5; b=dims['blue']; dy=dims['doty']; g=dims['goal']; L=g+11
def model(delta):
    pts={'red':[], 'blue':[], 'yel':[]}
    ys=np.arange(-W,W+.01,0.5); pts['red'].append(np.c_[np.zeros_like(ys),ys])
    a=np.linspace(0,np.pi,60); pts['red'].append(np.c_[10*np.cos(a),-W+10*np.sin(a)])
    for sx in(-1,1):
        for sy in(-1,1):
            a=np.linspace(0,2*np.pi,10,endpoint=False); pts['red'].append(np.c_[sx*(b-5)+0.9*np.cos(a),sy*dy+0.9*np.sin(a)])
        pts['blue'].append(np.c_[np.full_like(ys,sx*b),ys])
    xs=np.arange(-(L-28),(L-28)+.01,0.5); pts['yel'].append(np.c_[xs,np.full_like(xs,W+delta)])
    return {k:np.vstack(v) for k,v in pts.items()}
def masks(i):
    im=cv2.imread(f"{cfg['frdir']}/%04d.jpg"%i); im=cv2.resize(im,None,fx=S,fy=S,interpolation=cv2.INTER_AREA).astype(np.float32)
    bb,gg,rr=cv2.split(im)
    m={'red':rr-(gg+bb)/2>cfg.get('tred',25),'blue':bb-(rr+gg)/2>cfg.get('tblue',8),'yel':(rr+gg)/2-bb>cfg.get('tyel',40)}
    if cfg.get('far_blue'): m['yel']=m['blue'].copy()
    occ=np.zeros(m['red'].shape,bool)
    for x1,y1,x2,y2 in boxes(i): occ[int(y1*S):int(y2*S)+2,int(x1*S):int(x2*S)+2]=True
    dt={}
    for k,v in m.items():
        v=(v&~occ).astype(np.uint8)
        dt[k]=cv2.distanceTransform(1-v,cv2.DIST_L2,3).astype(np.float32)
    return dt,occ
WGT={'red':1.0,'blue':0.7,'yel':1.0}
GRID=np.array([[x,y] for x in (200,960,1720) for y in (250,600,1000)],float)
def proj(H,w): w=np.c_[w,np.ones(len(w))]@H.T; return w[:,:2]/w[:,2:], w[:,2]
def refine(H0,dt,occ,M,prior=0.3,cap=12.0):
    h0=(H0/H0[2,2]).ravel()[:8]; g0,_=proj(H0,GRID)
    sel={}
    for k,P in M.items():
        p,w=proj(H0,P); ok=(w>0)&(p[:,0]>-200)&(p[:,0]<2120)&(p[:,1]>-200)&(p[:,1]<1280)
        sel[k]=P[ok]
    def r(h):
        H=np.append(h,1).reshape(3,3); out=[]
        for k,P in sel.items():
            if not len(P): continue
            p,w=proj(H,P); q=p*S
            inside=(q[:,0]>=0)&(q[:,0]<dt[k].shape[1]-1)&(q[:,1]>=0)&(q[:,1]<dt[k].shape[0]-1)&(w>0)
            v=np.full(len(P),cap)
            if inside.any():
                qi=q[inside]; v[inside]=np.minimum(map_coordinates(dt[k],[qi[:,1],qi[:,0]],order=1),cap)
                oc=occ[qi[:,1].astype(int),qi[:,0].astype(int)]; vv=v[inside]; vv[oc]=0; v[inside]=vv
            v[~inside]=0
            out.append(WGT[k]*v)
        gp,_=proj(H,GRID); out.append(prior*(gp-g0).ravel()*S)
        return np.concatenate(out)
    s=least_squares(r,h0,loss='soft_l1',f_scale=2.0,diff_step=1e-4)
    H=np.append(s.x,1).reshape(3,3); res=r(s.x)
    return H, s
def run(Hk,k,delta):
    M=model(delta); Hs={k:Hk}; stats={}
    for direction in (1,-1):
        H=Hk; i=k
        while 1<=i+direction<=N:
            j=i+direction
            step=Hc[j-1]@np.linalg.inv(Hc[i-1])   # frame i px -> frame j px
            H0=step@H; dt,occ=masks(j); H,s=refine(H0,dt,occ,M)
            Hs[j]=H; i=j
    return Hs
if __name__=='__main__':
    ann=json.load(open(cfg['ann']))
    Hk=np.array(ann['H']); k=ann['frame']; delta=ann.get('delta',0.0)
    # refine keyframe too
    dt,occ=masks(k); Hk,_=refine(Hk,dt,occ,model(delta),prior=0.05)
    Hs=run(Hk,k,delta)
    np.save(cfg['hout'],np.array([Hs[i]/Hs[i][2,2] for i in range(1,N+1)]))
    print('done',len(Hs))
