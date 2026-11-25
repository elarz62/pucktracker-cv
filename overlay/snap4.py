import numpy as np, json, sys, cv2
from scipy.optimize import least_squares
from scipy.ndimage import map_coordinates
from cam4 import Hcam, dist, undist
exec(open('snap.py').read().split("dims=cfg['dims']")[0])   # cfg, tr, boxes, Hc, N, S
exec("def _m():\n  pass")
src=open('snap.py').read(); exec(src[src.index('def masks'):src.index('WGT=')])
cal=np.load(cfg['cal']); C=cal[:3]; W=42.5; b,dy,g=cal[3:6]; k1=cal[6]; kfs=cfg['kfs']
def model():
    P={'red':[],'blue':[],'yel':[]}
    ys=np.arange(-W,W+.01,0.5); P['red'].append(np.c_[np.zeros_like(ys),ys])
    a=np.linspace(0,np.pi,50); P['red'].append(np.c_[10*np.cos(a),-W+10*np.sin(a)])
    for sx in(-1,1):
        P['blue'].append(np.c_[np.full_like(ys,sx*b),ys])
        for sy in(-1,1):
            a=np.linspace(0,2*np.pi,10,endpoint=False); P['red'].append(np.c_[sx*(b-5)+0.9*np.cos(a),sy*dy+0.9*np.sin(a)])
    xs=np.arange(-40,40.01,0.5); P['yel'].append(np.c_[xs,np.full_like(xs,W)])
    return {k:np.vstack(v) for k,v in P.items()}
M=model(); WG={'red':1.0,'blue':0.7,'yel':1.0}
GRID=np.array([[x,y] for x in (200,960,1720) for y in (300,650,1000)],float)
def img(H,P):
    w=np.c_[P,np.ones(len(P))]@H.T; ok=w[:,2]>0; p=np.full((len(P),2),-1e4); p[ok]=w[ok,:2]/w[ok,2:]
    p[ok]=dist(p[ok],k1); return p,ok
def refine(x0,Hpred,dt,occ,prior=0.3,cap=12.0):
    gpred=dist(GRID,0) if Hpred is None else None
    Hinv_pred=None
    # grid prior: undistorted grid positions predicted by step homography
    H0=Hcam(x0[:3],x0[3],C); SEL={}
    for k,P in M.items():
        p,ok=img(H0,P); q=p*S; d=dt[k]
        ins=ok&(q[:,0]>=2)&(q[:,0]<d.shape[1]-3)&(q[:,1]>=2)&(q[:,1]<d.shape[0]-3)
        if ins.any():
            qi=q[ins]; keep=~occ[qi[:,1].astype(int),qi[:,0].astype(int)]; idx=np.flatnonzero(ins)[keep]
        else: idx=np.array([],int)
        SEL[k]=P[idx]
    def r(x):
        H=Hcam(x[:3],x[3],C); out=[]
        for k,P in SEL.items():
            if not len(P): continue
            p,ok=img(H,P); q=p*S; d=dt[k]
            ins=ok&(q[:,0]>=0)&(q[:,0]<d.shape[1]-1)&(q[:,1]>=0)&(q[:,1]<d.shape[0]-1)
            v=np.full(len(P),cap)
            if ins.any(): qi=q[ins]; v[ins]=np.minimum(map_coordinates(d,[qi[:,1],qi[:,0]],order=1),cap)
            out.append(WG[k]*v*np.sqrt(200/max(len(P),200)))
        if Hpred is not None:
            # where the predicted pose maps rink points that sit under the grid, vs this pose
            out.append(prior*S*(img(H,Gw)[0]-Gimg).ravel())
        return np.concatenate(out)
    global Gw,Gimg
    if Hpred is not None:
        Hp_inv=np.linalg.inv(Hpred); w=np.c_[undist(GRID,k1),np.ones(len(GRID))]@Hp_inv.T; Gw=w[:,:2]/w[:,2:]; Gimg=GRID
    s=least_squares(r,x0,loss='soft_l1',f_scale=2.0,x_scale=[0.01,0.01,0.01,20])
    return s.x
poses={}
for kf in kfs:
    poses[kf['frame']]=np.r_[kf['r'],kf['f']]
start=kfs[0]['frame']
dt,occ=masks(start); poses[start]=refine(poses[start],Hcam(poses[start][:3],poses[start][3],C),dt,occ,prior=1.0)
for direction in (1,-1):
    i=start
    while 1<=i+direction<=N:
        j=i+direction; step=Hc[j-1]@np.linalg.inv(Hc[i-1])
        Hi=Hcam(poses[i][:3],poses[i][3],C)
        # predicted homography in undistorted coords (approx: step applied in image space)
        Hpred=step@Hi
        dt,occ=masks(j); poses[j]=refine(poses[i],Hpred,dt,occ); i=j
P=np.array([poses[i] for i in range(1,N+1)]); np.save(cfg['pout'],P)
# check vs other keyframes
for kf in kfs[1:]:
    print('keyframe',kf['frame'],'annot pose f',round(kf['f']),'snapped f',round(P[kf['frame']-1][3]),'rot diff deg',round(np.degrees(np.linalg.norm(P[kf['frame']-1][:3]-np.array(kf['r']))),2))
print('f range',P[:,3].min().round(),P[:,3].max().round())
