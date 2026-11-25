import numpy as np, json
from scipy.optimize import least_squares
from cam4 import *
import cam4
_w=cam4.world
def world(n,d):
    if n=='ctop': return (0,15)
    if n=='cbot': return (0,-15)
    return _w(n,d)
cam4.world=world
PTS={307:[((349,692),'nzm-'),((1343.5,659),'nzp-'),((869,497),'ctop'),((865,612),'cbot')]}
LINES={307:[('x0',[(872.5,480),(868.5,580),(868,640),(864.5,690)]),('x-b',[(50,755),(150,704),(250,658),(350,605)]),
            ('xb',[(1300,547),(1400,601),(1600,713),(1700,764),(1800,812)]),
            ('yfar',[(100,446),(300,443),(500,440),(700,439),(900,435),(1300,431),(1500,430),(1700,432),(1850,440)])]}
fn=calibrate_k(PTS,LINES,[307])
best=None
for C0 in ([0,-60,25],[0,-70,35],[0,-55,15],[0,-80,45]):
  for tilt in (15,25):
    for k0 in (0,-0.1):
      x0=np.r_[C0,25,22,89,k0,r_init(0,tilt),1000]
      for fz in (800,1200):
        x0[-1]=fz
        lo=np.r_[-30,-150,3,15,15,80,-0.3,-4,-4,-4,500]; hi=np.r_[30,-43,80,30,26,95,0.3,4,4,4,3000]
        x0=np.clip(x0,lo+1e-6,hi-1e-6)
        try: s=least_squares(lambda x: np.r_[fn(x), 2*(x[5]-89)],x0,loss='soft_l1',f_scale=3,bounds=(lo,hi))
        except Exception as e: continue
        e=np.sqrt(np.mean(fn(s.x)**2))
        if best is None or e<best[0]: best=(e,s.x)
e,x=best; print('rms',round(e,2),'C',x[:3].round(1),'blue,doty',x[3:5].round(1),'k1',round(x[6],3),'f',round(x[10]))
print(np.round(fn(x).reshape(-1,2),1).tolist())
np.save('cal19.npy',x)
