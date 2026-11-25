import numpy as np, cv2, sys, json
from scipy.spatial.transform import Rotation as Rot
from cam4 import Hcam, dist
cfg=json.load(open(sys.argv[1])); P=np.load(cfg['pout']); cal=np.load(cfg['cal']); C=cal[:3]; b,dy,g=cal[3:6]; k1=cal[6]; W=42.5
for kf in cfg['kfs'][1:]:
    a=Rot.from_rotvec(P[kf['frame']-1][:3]); bb=Rot.from_rotvec(kf['r']); print('kf',kf['frame'],'angle diff deg',round(np.degrees((a.inv()*bb).magnitude()),2))
ims=[]
for i in [int(a) for a in sys.argv[2:]]:
    H=Hcam(P[i-1][:3],P[i-1][3],C); im=cv2.imread(f"{cfg['frdir']}/%04d.jpg"%i)
    F=[('r',[[0,-W],[0,W]]),('b',[[b,-W],[b,W]]),('b',[[-b,-W],[-b,W]]),('r',[[g,-30],[g,30]]),('r',[[-g,-30],[-g,30]]),('y',[[-60,W],[60,W]])]
    a=np.linspace(0,2*np.pi,60); F.append(('b',np.c_[15*np.cos(a),15*np.sin(a)].tolist()))
    a=np.linspace(0,np.pi,30); F.append(('r',np.c_[10*np.cos(a),-W+10*np.sin(a)].tolist()))
    for sx in(-1,1):
        for sy in(-1,1):
            F.append(('r',np.c_[sx*(b-5)+1*np.cos(np.linspace(0,6.3,12)),sy*dy+1*np.sin(np.linspace(0,6.3,12))].tolist()))
    for k,pl in F:
        pl=np.array(pl,float); pl=np.vstack([np.linspace(pl[j],pl[j+1],40) for j in range(len(pl)-1)])
        w=np.c_[pl,np.ones(len(pl))]@H.T; ok=w[:,2]>0; p=w[:,:2]/w[:,2:]; p[ok]=dist(p[ok],k1)
        col={'r':(0,0,255),'b':(255,0,0),'y':(0,220,255)}[k]
        for a1,b1,o1,o2 in zip(p[:-1],p[1:],ok[:-1],ok[1:]):
            if o1 and o2 and abs(a1).max()<4000 and abs(b1).max()<4000: cv2.line(im,tuple(map(int,a1)),tuple(map(int,b1)),col,2)
    cv2.putText(im,str(i),(20,60),0,2,(0,0,255),3); ims.append(cv2.resize(im,(960,540)))
while len(ims)%2: ims.append(ims[0]*0)
cv2.imwrite(sys.argv[1].replace('.json','_ovl.jpg'),np.vstack([np.hstack(ims[k:k+2]) for k in range(0,len(ims),2)]))
