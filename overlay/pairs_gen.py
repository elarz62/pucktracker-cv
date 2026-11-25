import cv2, numpy as np, pandas as pd, pickle, sys
tag, frdir, trk, FPS_SRC, T0, LAG = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])
N=360
tr=pd.read_csv(trk)
def boxes(i):
    f=int(round(((i-1)/6.0+T0-LAG)*FPS_SRC)); return tr[tr.frame==f][['x1','y1','x2','y2']].values
sift=cv2.SIFT_create(3000); bf=cv2.BFMatcher()
def feats(i):
    im=cv2.imread(f'{frdir}/%04d.jpg'%i,0); m=np.full(im.shape,255,np.uint8)
    for x1,y1,x2,y2 in boxes(i): cv2.rectangle(m,(int(x1)-10,int(y1)-10),(int(x2)+10,int(y2)+10),0,-1)
    return sift.detectAndCompute(im,m)
F=[None]+[feats(i) for i in range(1,N+1)]
pairs=[]; rng=np.random.default_rng(0); Hs=[np.eye(3)]
for i in range(1,N+1):
    for d in (1,3,6,12,24):
        j=i+d
        if j>N: continue
        ka,da=F[i]; kb,db=F[j]
        ms=[m for m,n in bf.knnMatch(da,db,k=2) if m.distance<0.75*n.distance]
        if len(ms)<30: continue
        pa=np.float32([ka[m.queryIdx].pt for m in ms]); pb=np.float32([kb[m.trainIdx].pt for m in ms])
        H,inl=cv2.findHomography(pa,pb,cv2.RANSAC,2.0); inl=inl.ravel().astype(bool)
        if d==1: Hs.append(H@Hs[-1])
        if inl.sum()<40: continue
        idx=np.flatnonzero(inl); idx=rng.choice(idx,min(60,len(idx)),replace=False)
        pairs.append((i,j,pa[idx],pb[idx]))
pickle.dump(pairs,open(f'pairs_{tag}.pkl','wb')); np.save(f'Hchain_{tag}.npy',np.array(Hs))
print('pairs',len(pairs),'chain',len(Hs))
