import numpy as np, json, sys, cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
from model import proj, linedef
def Hcam(r,f,C):
    K=np.array([[f,0,960],[0,f,540],[0,0,1.]]); R=Rot.from_rotvec(r).as_matrix()
    return K@R@np.array([[1,0,-C[0]],[0,1,-C[1]],[0,0,-C[2]]])
def Pcam(r,f,C):
    K=np.array([[f,0,960],[0,f,540],[0,0,1.]]); R=Rot.from_rotvec(r).as_matrix(); return K@R@np.c_[np.eye(3),-np.asarray(C)[:,None]]
def r_init(yaw,tilt):
    t=np.radians(tilt); c,s=np.cos(t),np.sin(t); base=np.array([[1,0,0],[0,-s,-c],[0,c,-s]])
    return (Rot.from_rotvec([0,yaw,0])*Rot.from_matrix(base)).as_rotvec()
def world(n,d):
    W,b,dy,g=d
    return {'far0':(0,W),'farBm':(-b,W),'farBp':(b,W),'nzp-':(b-5,-dy),'nzm-':(-(b-5),-dy),'nzm+':(-(b-5),dy),'nzp+':(b-5,dy),
            'crease':(0,-W+10),'goalR':(g,0),'goalL':(-g,0)}[n]
def ann_res(H,pp,ls,d):
    out=list((proj(H,np.array([world(n,d) for a,n in pp]))-np.array([a for a,n in pp],float)).ravel()); Hi=np.linalg.inv(H)
    for nm,ips in ls:
        ax,v=linedef(nm,d)
        for ip in ips:
            w=proj(Hi,np.array([ip],float))[0]; w[ax]=v; out+=list(proj(H,w[None])[0]-ip)
    return np.array(out)
def calibrate(PTS,LINES,frames,use_goal=True):
    def fn(x):
        C=x[0:3]; d=(42.5,x[3],x[4],x[5]); out=[]
        for k,i in enumerate(frames):
            r=x[6+4*k:9+4*k]; f=x[9+4*k]
            pp=[p for p in PTS[i] if use_goal or not p[1].startswith('goal')]
            out.append(ann_res(Hcam(r,f,C),pp,LINES[i],d))
        return np.concatenate(out)
    best=None
    for C0 in ([0,-45,10],[0,-50,18],[0,-42,7],[0,-55,30]):
        for tilt in (10,20):
            for yaws in ([0.0]*len(frames),):
                x0=np.r_[C0,20,22,85]
                for i in frames: x0=np.r_[x0,r_init(0,tilt),950]
                # try several yaw per frame by coarse search
                for k,i in enumerate(frames):
                    bestk=None
                    for yaw in np.linspace(-0.8,0.8,9):
                        xx=x0.copy(); xx[6+4*k:9+4*k]=r_init(yaw,tilt)
                        try: e=np.nanmean(np.abs(fn(xx)))
                        except: continue
                        if bestk is None or e<bestk[0]: bestk=(e,yaw)
                    x0[6+4*k:9+4*k]=r_init(bestk[1],tilt)
                try: s=least_squares(fn,x0,loss='soft_l1',f_scale=4,x_scale='jac')
                except Exception as ex: continue
                e=np.sqrt(np.mean(fn(s.x)**2))
                if best is None or e<best[0]: best=(e,s.x,fn(s.x))
    return best
if __name__=='__main__':
    from kf2 import PTS, LINES

def undist(p,k1,S=960.0):
    p=np.asarray(p,float); q=(p-[960,540])/S; u=q.copy()
    for _ in range(15): u=q/(1+k1*(u**2).sum(-1,keepdims=True))
    return u*S+[960,540]
def dist(p,k1,S=960.0):
    p=np.asarray(p,float); u=(p-[960,540])/S; return u*(1+k1*(u**2).sum(-1,keepdims=True))*S+[960,540]
def calibrate_k(PTS,LINES,frames):
    def U(i,k1):
        P={i:[(tuple(undist(a,k1)),n) for a,n in PTS[i]]}
        Lq=[(nm,[tuple(undist(ip,k1)) for ip in ips]) for nm,ips in LINES[i]]
        return P[i],Lq
    def fn(x):
        C=x[0:3]; d=(42.5,x[3],x[4],x[5]); k1=x[6]; out=[]
        for k,i in enumerate(frames):
            r=x[7+4*k:10+4*k]; f=x[10+4*k]; pp,ls=U(i,k1)
            out.append(ann_res(Hcam(r,f,C),pp,ls,d))
        return np.concatenate(out)
    return fn
