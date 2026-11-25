import cv2, numpy as np, pandas as pd, sys
FPS_SRC, T0, LAG = 24.0, 330.0, 0.242
N = 360
tr = pd.read_csv('tracks26.csv')
def boxes(i):  # frame i (1-based) of 6fps clip
    t = (i - 1) / 6.0
    f = int(round((t + T0 - LAG) * FPS_SRC))
    return tr[tr.frame == f][['x1','y1','x2','y2']].values
sift = cv2.SIFT_create(3000)
def feats(i):
    im = cv2.imread('fr/%04d.jpg' % i, 0)
    m = np.full(im.shape, 255, np.uint8)
    for x1,y1,x2,y2 in boxes(i): cv2.rectangle(m,(int(x1)-10,int(y1)-10),(int(x2)+10,int(y2)+10),0,-1)
    return sift.detectAndCompute(im, m)
bf = cv2.BFMatcher()
def match(a, b):
    ka, da = a; kb, db = b
    ms = [m for m, n in bf.knnMatch(da, db, k=2) if m.distance < 0.75 * n.distance]
    pa = np.float32([ka[m.queryIdx].pt for m in ms]); pb = np.float32([kb[m.trainIdx].pt for m in ms])
    H, inl = cv2.findHomography(pa, pb, cv2.RANSAC, 2.0)
    return H, int(inl.sum()), len(ms)
F = [None] + [feats(i) for i in range(1, N + 1)]
Hs = {1: np.eye(3)}
log = []
for i in range(2, N + 1):
    H, ni, nm = match(F[i-1], F[i])
    Hs[i] = H @ Hs[i-1]; Hs[i] /= Hs[i][2,2]
    log.append((i, ni, nm))
np.save('Hchain.npy', np.array([Hs[i] for i in range(1, N + 1)]))
print('min inliers', min(l[1] for l in log), 'median', np.median([l[1] for l in log]))
# loop check: direct 1 -> k
for k in (60, 120, 200, 250, 343):
    H, ni, nm = match(F[1], F[k])
    if H is None: print(k, 'no direct'); continue
    pts = np.float32([[400,300],[960,600],[1500,300],[960,1000]]).reshape(-1,1,2)
    d = cv2.perspectiveTransform(pts, H) - cv2.perspectiveTransform(pts, Hs[k])
    print(k, 'direct inliers', ni, 'chain-vs-direct px', np.round(np.linalg.norm(d, axis=2).ravel(), 1))
