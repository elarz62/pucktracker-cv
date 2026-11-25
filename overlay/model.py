import numpy as np, json
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot, Slerp
class Cam:
    def __init__(self, ba_file, N=360, tilt=20):
        p=np.load(ba_file); self.N=N
        self.f=np.exp(p[:N]); self.R=Rot.from_rotvec(np.vstack([np.zeros(3),p[N:].reshape(-1,3)]))
        t=np.radians(tilt); c,s=np.cos(t),np.sin(t); self.base=np.array([[1,0,0],[0,-s,-c],[0,c,-s]])
        self.ts=np.arange(N)/6.0; self.sl=Slerp(self.ts,self.R)
    def K(self,f): return np.array([[f,0,960],[0,f,540],[0,0,1.]])
    def G(self,q):
        R0=Rot.from_rotvec(q[:3]).as_matrix()@self.base; C=q[3:6]
        return R0@np.array([[1,0,-C[0]],[0,1,-C[1]],[0,0,-C[2]]])
    def H(self,q,i): return self.K(self.f[i-1])@self.R[i-1].as_matrix()@self.G(q)
    def H_t(self,q,t):
        t=float(np.clip(t,0,self.ts[-1])); f=np.interp(t,self.ts,self.f)
        return self.K(f)@self.sl(t).as_matrix()@self.G(q)
    def P_t(self,q,t):  # 3x4 projection for 3D rink points
        t=float(np.clip(t,0,self.ts[-1])); f=np.interp(t,self.ts,self.f)
        R0=Rot.from_rotvec(q[:3]).as_matrix()@self.base; C=q[3:6]
        return self.K(f)@self.sl(t).as_matrix()@R0@np.c_[np.eye(3),-C[:,None]]
def proj(H,w): w=np.c_[w,np.ones(len(w))]@H.T; return w[:,:2]/w[:,2:]
def world(n,d):
    W,b,dy,g=d
    return {'far0':(0,W),'farBm':(-b,W),'farBp':(b,W),'nzp-':(b-5,-dy),'nzm-':(-(b-5),-dy),'nzm+':(-(b-5),dy),'nzp+':(b-5,dy),
            'crease':(0,-W+10),'goalR':(g,0),'goalL':(-g,0),'center':(0,0)}[n]
def linedef(n,d): return {'x0':(0,0),'xb':(0,d[1]),'x-b':(0,-d[1]),'yfar':(1,d[0])}[n]
def residuals(cam,q,d,PTS,LINES):
    r=[]
    for i,pp in PTS.items():
        H=cam.H(q,i); r+=list((proj(H,np.array([world(n,d) for a,n in pp]))-np.array([a for a,n in pp],float)).ravel())
    for i,ls in LINES.items():
        H=cam.H(q,i); Hi=np.linalg.inv(H)
        for nm,ips in ls:
            ax,v=linedef(nm,d)
            for ip in ips:
                w=proj(Hi,np.array([ip],float))[0]; w[ax]=v; r+=list(proj(H,w[None])[0]-ip)
    return np.array(r)
def calibrate(cam,PTS,LINES,starts):
    best=None
    for q0 in starts:
        fn=lambda x: residuals(cam,x[:6],(42.5,x[6],x[7],x[8]),PTS,LINES)
        try: s=least_squares(fn,q0,loss='soft_l1',f_scale=4)
        except Exception as e: continue
        e=np.sqrt(np.mean(fn(s.x)**2))
        if best is None or e<best[0]: best=(e,s.x,fn(s.x))
    return best
