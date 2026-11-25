import numpy as np, pickle, sys
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation as Rot
tag=sys.argv[1]; pairs=pickle.load(open(sys.argv[2],'rb')); Hc=np.load(sys.argv[3]); N=len(Hc)
f0=float(sys.argv[4]) if len(sys.argv)>4 else 950
def Kf(f): return np.array([[f,0,960],[0,f,540],[0,0,1.]])
def init(f):
    rv=[]
    for H in Hc:
        M=np.linalg.inv(Kf(f))@H@Kf(f); U,S,Vt=np.linalg.svd(M); R=U@Vt
        if np.linalg.det(R)<0: R=-R
        rv.append(Rot.from_rotvec(Rot.from_matrix(R).as_rotvec()).as_rotvec())
    return np.array(rv)
# params: logf[N], rotvec[N-1]
def unpack(p):
    f=np.exp(p[:N]); R=Rot.from_rotvec(np.vstack([np.zeros(3),p[N:].reshape(-1,3)])).as_matrix(); return f,R
def res(p):
    f,R=unpack(p); out=[]
    for i,j,a,b in pairs:
        Ki=np.linalg.inv(Kf(f[i-1])); w=(np.c_[a,np.ones(len(a))]@Ki.T)@R[i-1]; q=w@R[j-1].T@Kf(f[j-1]).T
        out.append((q[:,:2]/q[:,2:]-b).ravel())
    # weak smoothness prior on zoom
    out.append(5*np.diff(p[:N]))
    return np.concatenate(out)
nr=sum(2*len(a) for _,_,a,_ in pairs)+N-1
p0=np.r_[np.full(N,np.log(f0)),init(f0)[1:].ravel()]
J=lil_matrix((nr,len(p0)),dtype=int); r=0
for i,j,a,b in pairs:
    n=2*len(a)
    for k in (i,j):
        J[r:r+n,k-1]=1
        if k>1: J[r:r+n,N+3*(k-2):N+3*(k-1)]=1
    r+=n
for k in range(N-1): J[r+k,k]=1; J[r+k,k+1]=1
s=least_squares(res,p0,jac_sparsity=J,loss='huber',f_scale=1.5,x_scale='jac',max_nfev=100)
e=np.linalg.norm(res(s.x)[:-(N-1)].reshape(-1,2),axis=1)
f,R=unpack(s.x)
print(tag,'median',round(np.median(e),2),'p90',round(np.percentile(e,90),2),'f range',round(f.min()),round(f.max()),'f1',round(f[0]))
np.save(f'ba3_{tag}.npy',s.x)
