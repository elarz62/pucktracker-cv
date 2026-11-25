import numpy as np
from cam4 import *
from ann26 import PTS, LINES
x4=np.load('cal4_26.npy')
fn=calibrate_k(PTS,LINES,[1,289])
best=None
for k0 in (0,-0.05,-0.1,-0.15,-0.2):
    x0=np.r_[x4[:6],k0,x4[6:]]
    s=least_squares(fn,x0,loss='soft_l1',f_scale=4,x_scale='jac')
    e=np.sqrt(np.mean(fn(s.x)**2))
    print('k0',k0,'-> rms',round(e,2),'k1',round(s.x[6],3),'C',np.round(s.x[:3],1),'dims',np.round(s.x[3:6],1),'f',np.round(s.x[10::4]))
    if best is None or e<best[0]: best=(e,s.x)
np.save('cal5_26.npy',best[1]); print(np.round(fn(best[1]).reshape(-1,2),1).tolist())
