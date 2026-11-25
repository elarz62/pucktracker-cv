import numpy as np, json, sys
from cam4 import *
from ann26 import PTS, LINES
b=calibrate(PTS,LINES,[1,289])
e,x,r=b; print('rms',round(e,2),'C',np.round(x[:3],1),'blue,doty,goal',np.round(x[3:6],1),'f',np.round(x[9::4],0))
print(np.round(r.reshape(-1,2),1).tolist())
np.save('cal4_26.npy',x)
