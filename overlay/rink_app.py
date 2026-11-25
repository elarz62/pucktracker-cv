"""Full-game rink view: map every frame to the ice, then turn tracker boxes into top-down dots.

    modal run --detach overlay/rink_app.py::build --game-id 2026-09-26

Per rink, overlay/refs/<game>/rink.json holds the fixed camera (position, lens k1), the measured rink
dims and a set of reference frames with known pose. Each 60 s chunk of each game file is mapped in its
own container: anchor the first frame to the best-matching reference frame, then follow the camera
frame to frame (6 fps), snapping a 4-parameter camera (pan/tilt/roll + zoom) to the painted lines, and
re-anchor every 2 s when that fits the lines better.

Output on the volume, master timeline (t=0 at puck drop):
    /data/games/<id>/rink/meta.json        dims, lens, minutes
    /data/games/<id>/rink/<minute>.json     dots at 12 fps, camera per 1/6 s, footprint, line-check flags
"""
import json
import pathlib
import subprocess

import modal

app = modal.App("pucktracker-rink")
vol = modal.Volume.from_name("pucktracker-videos")
DATA = pathlib.Path("/data")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("opencv-python-headless==4.10.0.84", "numpy==1.26.4", "scipy==1.13.1", "pandas==2.2.3", "scikit-learn==1.5.2")
    .add_local_dir(pathlib.Path(__file__).parent / "refs", "/root/refs")
    .add_local_dir(pathlib.Path(__file__).parent.parent / "games", "/root/games")
)
W = 42.5
S = 0.5  # mask scale


# ---------------------------------------------------------------- camera model
def _np():
    import numpy as np
    return np


def Hcam(r, f, C):
    import numpy as np
    from scipy.spatial.transform import Rotation as Rot
    K = np.array([[f, 0, 960], [0, f, 540], [0, 0, 1.0]])
    return K @ Rot.from_rotvec(r).as_matrix() @ np.array([[1, 0, -C[0]], [0, 1, -C[1]], [0, 0, -C[2]]])


def Pcam(r, f, C):
    import numpy as np
    from scipy.spatial.transform import Rotation as Rot
    K = np.array([[f, 0, 960], [0, f, 540], [0, 0, 1.0]])
    return K @ Rot.from_rotvec(r).as_matrix() @ np.c_[np.eye(3), -np.asarray(C)[:, None]]


def dist(p, k1):
    import numpy as np
    u = (np.asarray(p, float) - [960, 540]) / 960.0
    return u * (1 + k1 * (u ** 2).sum(-1, keepdims=True)) * 960 + [960, 540]


def undist(p, k1):
    import numpy as np
    q = (np.asarray(p, float) - [960, 540]) / 960.0
    u = q.copy()
    for _ in range(15):
        u = q / (1 + k1 * (u ** 2).sum(-1, keepdims=True))
    return u * 960 + [960, 540]


class Mapper:
    def __init__(self, cfg, refdir):
        import cv2
        import numpy as np
        self.cfg, self.C, self.k1 = cfg, np.array(cfg["C"]), cfg["k1"]
        b, dy = cfg["blue"], cfg["doty"]
        P = {"red": [], "blue": [], "yel": []}
        ys = np.arange(-W, W + .01, 0.5)
        P["red"].append(np.c_[np.zeros_like(ys), ys])
        a = np.linspace(0, np.pi, 50)
        P["red"].append(np.c_[10 * np.cos(a), -W + 10 * np.sin(a)])
        for sx in (-1, 1):
            P["blue"].append(np.c_[np.full_like(ys, sx * b), ys])
            for sy in (-1, 1):
                a = np.linspace(0, 2 * np.pi, 10, endpoint=False)
                P["red"].append(np.c_[sx * (b - 5) + 0.9 * np.cos(a), sy * dy + 0.9 * np.sin(a)])
        xs = np.arange(-40, 40.01, 0.5)
        P["yel"].append(np.c_[xs, np.full_like(xs, W)])
        self.M = {k: np.vstack(v) for k, v in P.items()}
        self.WG = {"red": 1.0, "blue": 0.7, "yel": 1.0}
        self.GRID = np.array([[x, y] for x in (200, 960, 1720) for y in (300, 650, 1000)], float)
        self.sift = cv2.SIFT_create(3000)
        self.bf = cv2.BFMatcher()
        self.refs = []
        for r in cfg["refs"]:
            im = cv2.imread(str(refdir / r["file"]), 0)
            k, d = self.sift.detectAndCompute(im, None)
            self.refs.append((np.float32([p.pt for p in k]), d, np.array(r["r"]), r["f"]))

    def feats(self, gray, boxes):
        import cv2
        import numpy as np
        m = np.full(gray.shape, 255, np.uint8)
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(m, (int(x1) - 10, int(y1) - 10), (int(x2) + 10, int(y2) + 10), 0, -1)
        k, d = self.sift.detectAndCompute(gray, m)
        return np.float32([p.pt for p in k]), d

    def match(self, fa, fb, min_inl=40):
        import cv2
        import numpy as np
        (pa, da), (pb, db) = fa, fb
        if da is None or db is None or len(da) < 10 or len(db) < 10:
            return None, 0
        ms = [m for m, n in (x for x in self.bf.knnMatch(da, db, k=2) if len(x) == 2) if m.distance < 0.75 * n.distance]
        if len(ms) < 20:
            return None, 0
        a = pa[[m.queryIdx for m in ms]]
        b = pb[[m.trainIdx for m in ms]]
        H, inl = cv2.findHomography(a, b, cv2.RANSAC, 2.0)
        n = int(inl.sum()) if inl is not None else 0
        return (H if n >= min_inl else None), n

    def masks(self, bgr, boxes):
        import cv2
        import numpy as np
        cfg = self.cfg
        im = cv2.resize(bgr, None, fx=S, fy=S, interpolation=cv2.INTER_AREA).astype(np.float32)
        bb, gg, rr = cv2.split(im)
        m = {"red": rr - (gg + bb) / 2 > cfg["tred"], "blue": bb - (rr + gg) / 2 > cfg["tblue"],
             "yel": (rr + gg) / 2 - bb > cfg["tyel"]}
        if cfg.get("far_blue"):
            m["yel"] = m["blue"].copy()
        occ = np.zeros(m["red"].shape, bool)
        for x1, y1, x2, y2 in boxes:
            occ[int(y1 * S):int(y2 * S) + 2, int(x1 * S):int(x2 * S) + 2] = True
        dt = {k: cv2.distanceTransform(1 - (v & ~occ).astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32) for k, v in m.items()}
        return dt, occ

    def img(self, H, P):
        import numpy as np
        w = np.c_[P, np.ones(len(P))] @ H.T
        ok = w[:, 2] > 0
        p = np.full((len(P), 2), -1e4)
        p[ok] = w[ok, :2] / w[ok, 2:]
        p[ok] = dist(p[ok], self.k1)
        return p, ok

    def pose_from_H(self, x0, Himg):
        """4-parameter pose whose ice mapping best matches an image homography (rink -> image px)."""
        import numpy as np
        from scipy.optimize import least_squares
        Gi = np.linalg.inv(Himg)
        w = np.c_[undist(self.GRID, self.k1), np.ones(len(self.GRID))] @ Gi.T
        ok = w[:, 2] > 0
        Gw = w[ok, :2] / w[ok, 2:]
        tgt = self.GRID[ok]
        if len(Gw) < 4:
            return x0
        s = least_squares(lambda x: (self.img(Hcam(x[:3], x[3], self.C), Gw)[0] - tgt).ravel(), x0,
                          x_scale=[0.01, 0.01, 0.01, 20], loss="soft_l1", f_scale=5)
        return s.x

    def refine(self, x0, Hpred, dt, occ, prior=0.3, cap=12.0):
        import numpy as np
        from scipy.ndimage import map_coordinates
        from scipy.optimize import least_squares
        H0 = Hcam(x0[:3], x0[3], self.C)
        SEL = {}
        for k, P in self.M.items():
            p, ok = self.img(H0, P)
            q = p * S
            d = dt[k]
            ins = ok & (q[:, 0] >= 2) & (q[:, 0] < d.shape[1] - 3) & (q[:, 1] >= 2) & (q[:, 1] < d.shape[0] - 3)
            idx = np.flatnonzero(ins)
            if len(idx):
                qi = q[idx]
                idx = idx[~occ[qi[:, 1].astype(int), qi[:, 0].astype(int)]]
            SEL[k] = P[idx]
        if Hpred is not None:
            w = np.c_[undist(self.GRID, self.k1), np.ones(len(self.GRID))] @ np.linalg.inv(Hpred).T
            okg = w[:, 2] > 0
            Gw, Gimg = w[okg, :2] / w[okg, 2:], self.GRID[okg]

        def r(x):
            H = Hcam(x[:3], x[3], self.C)
            out = []
            for k, P in SEL.items():
                if not len(P):
                    continue
                p, ok = self.img(H, P)
                q = p * S
                d = dt[k]
                ins = ok & (q[:, 0] >= 0) & (q[:, 0] < d.shape[1] - 1) & (q[:, 1] >= 0) & (q[:, 1] < d.shape[0] - 1)
                v = np.full(len(P), cap)
                if ins.any():
                    qi = q[ins]
                    v[ins] = np.minimum(map_coordinates(d, [qi[:, 1], qi[:, 0]], order=1), cap)
                out.append(self.WG[k] * v * np.sqrt(200 / max(len(P), 200)))
            if Hpred is not None and len(Gw):
                out.append(prior * S * (self.img(H, Gw)[0] - Gimg).ravel())
            return np.concatenate(out) if out else np.zeros(1)

        s = least_squares(r, x0, loss="soft_l1", f_scale=2.0, x_scale=[0.01, 0.01, 0.01, 20])
        return s.x

    def quality(self, x, dt, occ):
        """Median line distance in full-res px for center line, blue lines, far boards (nan if not in view)."""
        import numpy as np
        from scipy.ndimage import map_coordinates
        H = Hcam(x[:3], x[3], self.C)
        row = []
        for k in ("red", "blue", "yel"):
            Pk = self.M[k] if k != "red" else self.M["red"][:171]
            p, ok = self.img(H, Pk)
            q = p * S
            d = dt[k]
            ins = ok & (q[:, 0] >= 0) & (q[:, 0] < d.shape[1] - 1) & (q[:, 1] >= 0) & (q[:, 1] < d.shape[0] - 1)
            if ins.sum() < 10:
                row.append(float("nan"))
                continue
            qi = q[ins]
            v = map_coordinates(d, [qi[:, 1], qi[:, 0]], order=1)
            v = v[~occ[qi[:, 1].astype(int), qi[:, 0].astype(int)]]
            row.append(float(np.median(v) / S) if len(v) else float("nan"))
        return row

    def anchor(self, f, x_prev=None):
        """Best reference match -> pose guess."""
        best = None
        for rp, rd, rr, rf in self.refs:
            H, n = self.match((rp, rd), f)
            if H is not None and (best is None or n > best[0]):
                best = (n, H, rr, rf)
        if best is None:
            return None
        n, Href2new, rr, rf = best
        Hn = Href2new @ Hcam(rr, rf, self.C)
        x0 = x_prev if x_prev is not None else __import__("numpy").r_[rr, rf]
        return self.pose_from_H(__import__("numpy").r_[rr, rf], Hn), Hn


def score(q):
    import numpy as np
    r, b, y = [0 if np.isnan(v) else v for v in q]
    return r + y + 0.3 * b


@app.function(image=image, volumes={str(DATA): vol}, cpu=2, memory=6144, timeout=3600)
def map_chunk(game_id: str, part: int, file: str, t0: float, t1: float, fps_src: float) -> dict:
    import cv2
    import numpy as np
    import pandas as pd
    cfg = json.loads((pathlib.Path("/root/refs") / game_id / "rink.json").read_text())
    mp = Mapper(cfg, pathlib.Path("/root/refs") / game_id)
    stem = pathlib.Path(file).stem
    tr = pd.read_csv(DATA / "out" / stem / "tracks.csv", usecols=["frame", "track_id", "x1", "y1", "x2", "y2"])
    tr = tr[(tr.frame >= (t0 - 1) * fps_src) & (tr.frame <= (t1 + 1) * fps_src)]
    byf = {f: g for f, g in tr.groupby("frame")}
    cmd = ["ffmpeg", "-v", "error", "-ss", str(t0), "-i", str(DATA / "videos" / file), "-t", str(t1 - t0),
           "-vf", "fps=6", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = 0
    poses, quals, src = [], [], []
    feats_rows = []
    x = None
    f_prev = None
    since_anchor = 99
    while True:
        buf = proc.stdout.read(1920 * 1080 * 3)
        if len(buf) < 1920 * 1080 * 3:
            break
        bgr = np.frombuffer(buf, np.uint8).reshape(1080, 1920, 3)
        t = t0 + n / 6.0
        fr = int(round(t * fps_src))
        g = byf.get(fr)
        boxes = g[["x1", "y1", "x2", "y2"]].values if g is not None else np.zeros((0, 4))
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        f = mp.feats(gray, boxes)
        dt, occ = mp.masks(bgr, boxes)
        cands = []
        if x is not None and f_prev is not None:
            step, _ = mp.match(f_prev, f, min_inl=30)
            if step is not None:
                Hp = step @ Hcam(x[:3], x[3], mp.C)
                xc = mp.refine(x, Hp, dt, occ)
                cands.append(("chain", xc))
        if x is None or since_anchor >= 12 or not cands or score(mp.quality(cands[0][1], dt, occ)) > 6:
            a = mp.anchor(f)
            if a is not None:
                xa = mp.refine(a[0], a[1], dt, occ)
                cands.append(("anchor", xa))
            since_anchor = 0
        if not cands:
            poses.append([np.nan] * 4)
            quals.append([np.nan] * 3)
            src.append("none")
            x = None
        else:
            qs = [mp.quality(c[1], dt, occ) for c in cands]
            k = int(np.argmin([score(q) for q in qs]))
            x = cands[k][1]
            poses.append(list(map(float, x)))
            quals.append(qs[k])
            src.append(cands[k][0])
            # team features for boxes in this frame
            if g is not None and len(g):
                hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                Pm = Pcam(x[:3], x[3], mp.C)
                Hi = np.linalg.inv(Hcam(x[:3], x[3], mp.C))
                lo, hi = cfg["hue"]
                for rw in g.itertuples():
                    x1, y1, x2, y2 = int(rw.x1), int(rw.y1), int(rw.x2), int(rw.y2)
                    top = hsv[max(y1, 0):y1 + max(2, int((y2 - y1) * 0.55)), max(x1, 0):x2]
                    if top.size == 0:
                        continue
                    hue_ok = ((top[..., 0] >= lo) & (top[..., 0] <= hi)) if not cfg.get("colored_any") else True
                    bl = float((hue_ok & (top[..., 1] > cfg["smin"]) & (top[..., 2] > cfg["vmin"])).mean())
                    w = Hi @ np.r_[undist([[(x1 + x2) / 2, y2]], mp.k1)[0], 1]
                    X, Y = w[:2] / w[2]
                    Z = np.linspace(0, 9, 91)
                    ww = np.c_[np.full(91, X), np.full(91, Y), Z, np.ones(91)] @ Pm.T
                    yy = dist(ww[:, :2] / ww[:, 2:], mp.k1)[:, 1]
                    feats_rows.append((int(rw.track_id), bl, float(Z[np.argmin(np.abs(yy - y1))]), float(Y)))
        f_prev = f
        since_anchor += 1
        n += 1
    proc.wait()
    return {"part": part, "t0": t0, "poses": poses, "quals": quals, "src": src, "feats": feats_rows}


def color_feat(hsv, lab):
    """Jersey color signature of a torso crop: hue x saturation histogram of lit pixels, mean Lab,
    share of dark and of white pixels."""
    import numpy as np
    h, sat, v = hsv[..., 0].ravel(), hsv[..., 1].ravel(), hsv[..., 2].ravel()
    lit = v > 50
    hist = np.histogram2d(h[lit], np.minimum(sat[lit], 255), bins=[12, 3], range=[[0, 180], [0, 256]])[0].ravel()
    hist = hist / max(hist.sum(), 1)
    L = lab.reshape(-1, 3).mean(0) / 255.0
    return np.r_[hist, L, (v < 60).mean(), ((sat < 40) & (v > 150)).mean()].astype(np.float32)


@app.function(image=image, volumes={str(DATA): vol}, cpu=2, memory=4096, timeout=3600)
def color_chunk(game_id: str, part: int, file: str, t0: float, t1: float, fps_src: float) -> dict:
    """Jersey color samples for every tracker box, 2 frames a second, straight from the source video."""
    import cv2
    import numpy as np
    import pandas as pd
    stem = pathlib.Path(file).stem
    tr = pd.read_csv(DATA / "out" / stem / "tracks.csv", usecols=["frame", "track_id", "x1", "y1", "x2", "y2"])
    tr = tr[(tr.frame >= (t0 - 1) * fps_src) & (tr.frame <= (t1 + 1) * fps_src)]
    byf = {f: g for f, g in tr.groupby("frame")}
    cmd = ["ffmpeg", "-v", "error", "-ss", str(t0), "-i", str(DATA / "videos" / file), "-t", str(t1 - t0),
           "-vf", "fps=2", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    out, n = {}, 0
    while True:
        buf = proc.stdout.read(1920 * 1080 * 3)
        if len(buf) < 1920 * 1080 * 3:
            break
        bgr = np.frombuffer(buf, np.uint8).reshape(1080, 1920, 3)
        g = byf.get(int(round((t0 + n / 2.0) * fps_src)))
        n += 1
        if g is None:
            continue
        for rw in g.itertuples():
            x1, y1, x2, y2 = max(int(rw.x1), 0), max(int(rw.y1), 0), min(int(rw.x2), 1919), min(int(rw.y2), 1079)
            h, w = y2 - y1, x2 - x1
            if h < 24 or w < 6:
                continue
            crop = bgr[y1 + int(0.12 * h):y1 + int(0.55 * h), x1 + int(0.2 * w):x2 - int(0.2 * w)]
            if crop.size == 0:
                continue
            f = color_feat(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV), cv2.cvtColor(crop, cv2.COLOR_BGR2LAB))
            out.setdefault((part, int(rw.track_id)), []).append(f)
    return out


@app.function(image=image, volumes={str(DATA): vol}, cpu=4, memory=16384, timeout=4 * 3600)
def build(game_id: str, chunk: float = 60.0, link_s: float = 1.5, hold_end_s: float = 1.0, only_produce: bool = False) -> dict:
    import numpy as np
    import pickle
    game = json.loads((pathlib.Path("/root/games") / f"{game_id}.json").read_text())
    out = DATA / "games" / game_id / "rink"
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "chunks.pkl"
    fps = {}
    for i, p in enumerate(game["parts"]):
        pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
                             "-of", "csv=p=0", str(DATA / "videos" / p["file"])], capture_output=True, text=True).stdout.strip()
        a, b = pr.split("/")
        fps[i] = float(a) / float(b)
    if only_produce and cache.exists():
        res = pickle.loads(cache.read_bytes())
    else:
        jobs = []
        for i, p in enumerate(game["parts"]):
            t = float(p["start"])
            while t < p["end"]:
                jobs.append((game_id, i, p["file"], t, min(t + chunk, float(p["end"])), fps[i]))
                t += chunk
        res = list(map_chunk.starmap(jobs, order_outputs=True))
        cache.write_bytes(pickle.dumps(res))
        vol.commit()
    cpath = out / "colors.pkl"
    if not cpath.exists():
        jobs = []
        for i, p in enumerate(game["parts"]):
            t = float(p["start"])
            while t < p["end"]:
                jobs.append((game_id, i, p["file"], t, min(t + chunk, float(p["end"])), fps[i]))
                t += chunk
        colors = {}
        for d in color_chunk.starmap(jobs, order_outputs=True):
            for k, v in d.items():
                colors.setdefault(k, []).extend(v)
        colors = {k: np.median(np.array(v), 0) for k, v in colors.items() if len(v)}
        cpath.write_bytes(pickle.dumps(colors))
        vol.commit()
    colors = pickle.loads(cpath.read_bytes())
    stats = produce(game_id, game, res, fps, out, link_s, hold_end_s, colors=colors)
    vol.commit()
    return stats


def learn_teams(game_id, team, fixes, feats, hmed, colors, out):
    """Team-color model trained on every player's jersey colors.

    Labels, weakest to strongest: the color rules (weight 1), jersey-number tags from the Shift Marker
    (all our team, weight 10), and click-fixes from the rink page (weight 30). Players the model is sure
    about (>= 0.75) and that disagree with the rules get re-colored; click-fixes are applied on top by
    the caller, so they always win. Scored honestly: 5-fold, each tagged player is predicted by a model
    that never saw its tag."""
    import numpy as np
    import pickle
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    keys = [k for k in colors if k in feats and k in team]
    if not keys:
        return {"changed": {}, "fixes": len(fixes), "note": "no color samples"}
    game = json.loads((pathlib.Path("/root/games") / f"{game_id}.json").read_text())
    ours = "b" if str(game.get("jersey", game.get("team_color", "white"))).startswith("b") else "w"

    def xrow(k):
        v = np.array(feats[k])
        return np.r_[colors[k], np.median(v[:, 0]), np.log(max(np.median(v[:, 1]), 0.5) / hmed)]
    X = np.array([xrow(k) for k in keys])
    pk = lambda a: (int(a.split(":")[0]), int(a.split(":")[1]))
    fx = {pk(a): t for a, t in fixes.items()}
    tp = DATA / "games" / game_id / "tags.json"
    tags = {pk(a): ours for a, v in (json.loads(tp.read_text()) if tp.exists() else {}).items() if str(v).isdigit()}
    rules = np.array([team[k] for k in keys])
    y = np.array([fx.get(k, tags.get(k, team[k])) for k in keys])
    wt = np.array([30.0 if k in fx else 10.0 if k in tags else 1.0 for k in keys])

    def fit(mask):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced"))
        m.fit(X[mask], y[mask], logisticregression__sample_weight=wt[mask])
        return m

    def relabel(m, idx):
        P = m.predict_proba(X[idx])
        cls = np.array(m.classes_)
        j = P.argmax(1)
        pred = cls[j]
        sure = (P[np.arange(len(idx)), j] >= 0.75) & np.isin(pred, ["w", "b", "r"])
        return np.where(sure, pred, rules[idx])

    tag_idx = np.array([i for i, k in enumerate(keys) if k in tags and k not in fx])
    ev = {}
    if len(tag_idx) >= 20:
        rng = np.random.default_rng(0)
        fold = rng.integers(0, 5, len(tag_idx))
        got = np.empty(len(tag_idx), dtype=object)
        for f_ in range(5):
            hide = tag_idx[fold == f_]
            mask = np.ones(len(keys), bool)
            mask[hide] = False
            got[fold == f_] = relabel(fit(mask), hide)
        n_ = len(tag_idx)
        ev = {"tagged_players": int(n_), "rules_right_pct": round(100 * float((rules[tag_idx] == ours).mean()), 1),
              "learned_right_pct": round(100 * float((got == ours).mean()), 1)}
    m = fit(np.ones(len(keys), bool))
    new = relabel(m, np.arange(len(keys)))
    changed = {k: str(new[i]) for i, k in enumerate(keys) if new[i] != rules[i]}
    (out / "team_model.pkl").write_bytes(pickle.dumps({"model": m, "classes": list(m.classes_)}))
    # keep every labeled color signature so later games can start from all of them
    lib = DATA / "models" / "team_fixes"
    lib.mkdir(parents=True, exist_ok=True)
    (lib / f"{game_id}.pkl").write_bytes(pickle.dumps({k: (X[i], y[i], wt[i]) for i, k in enumerate(keys) if wt[i] > 1}))
    info = {"fixes": len(fx), "tags": len(tags), "players": len(keys), "recolored": len(changed),
            "recolored_to": {c: sum(v == c for v in changed.values()) for c in "wbr"}, **ev}
    (out / "learn_changes.json").write_text(json.dumps({f"{k[0]}:{k[1]}": [str(rules[keys.index(k)]), v] for k, v in changed.items()}))
    (out / "learn.json").write_text(json.dumps({**info, "built": __import__("time").time()}))
    return {**info, "changed": changed}


def produce(game_id, game, res, fps, out, link_s, hold_end_s, colors=None, min_life=0.6, merge_ft=3.0, dup_ft=2.0, dup_share=0.9, dup_min_s=1.0, cap={'w': 6, 'b': 6, 'r': 3}):
    import numpy as np
    import pandas as pd
    from scipy.spatial.transform import Rotation as Rot, Slerp
    cfg = json.loads((pathlib.Path("/root/refs") / game_id / "rink.json").read_text())
    C, k1, b, dy, g = np.array(cfg["C"]), cfg["k1"], cfg["blue"], cfg["doty"], cfg["goal"]
    offs, t = [], 0.0
    for p in game["parts"]:
        offs.append(t)
        t += p["end"] - p["start"]
    duration = t

    def to_std(X, Y):
        ax, s = np.abs(X), np.sign(X)
        xs = np.where(ax <= b, ax * 25 / b, np.where(ax <= g, 25 + (ax - b) * 64 / (g - b), 89 + (ax - g) * 64 / (g - b)))
        ay = np.abs(Y)
        ys = np.where(ay <= dy, ay * 22 / dy, 22 + (ay - dy) * (20.5 / (W - dy)))
        return s * xs, np.sign(Y) * ys

    # master-timeline camera samples
    cam = []  # (mt, pose(4), good)
    feats = {}
    for r in res:
        p = game["parts"][r["part"]]
        for n, (x, q) in enumerate(zip(r["poses"], r["quals"])):
            mt = offs[r["part"]] + r["t0"] + n / 6.0 - p["start"]
            qq = [0 if (v is None or np.isnan(v)) else v for v in q]
            good = (not np.isnan(x[0])) and qq[0] <= 6 and qq[2] <= 6
            cam.append((mt, x, good, r["part"]))
        for tid, bl, h, Y in r["feats"]:
            feats.setdefault((r["part"], tid), []).append((bl, h, Y))
    cam.sort(key=lambda c: c[0])
    cam_t = np.array([c[0] for c in cam])
    cam_good = np.array([c[2] for c in cam])
    cam_x = np.array([c[1] for c in cam], float)
    # team per track
    hs = [np.median(np.array(v)[:, 1]) for v in feats.values() if len(v) >= 3]
    hmed = float(np.median(hs)) if hs else 4.0
    old = {}
    for i, p in enumerate(game["parts"]):
        tmo = pd.read_csv(DATA / "out" / pathlib.Path(p["file"]).stem / "teams.csv")
        for r_ in tmo.itertuples():
            old[(i, int(r_.track_id))] = (r_.team, float(getattr(r_, "colored", 0) or 0))
    team = {}
    for key, v in feats.items():
        v = np.array(v)
        bl, h = np.median(v[:, 0]), np.median(v[:, 1])
        ot, oc = old.get(key, ("unknown", 0.0))
        is_b = bl > cfg["blue_frac"] or ot == "blue" or oc > 0.2
        team[key] = "b" if is_b else ("r" if h > cfg["ref_ratio"] * hmed and len(v) >= 3 else "w")
    # team fixes made by clicking dots on the /rink page (track key "part:tid" -> team)
    fixp = DATA / "games" / game_id / "rink" / "team_fix.json"
    fixes = json.loads(fixp.read_text()) if fixp.exists() else {}
    learn = learn_teams(game_id, team, fixes, feats, hmed, colors or {}, out)
    for k_, v_ in learn.pop("changed").items():
        team[k_] = v_
    for k_, v_ in fixes.items():
        pp, tt = k_.split(":")
        team[(int(pp), int(tt))] = v_
    # median rink depth per track (to tell far-boards players from fans in the stands)
    ymed = {key: float(np.median(np.array(v)[:, 2])) for key, v in feats.items()}
    # project all tracks
    segs = []
    for i, p in enumerate(game["parts"]):
        stem = pathlib.Path(p["file"]).stem
        tr = pd.read_csv(DATA / "out" / stem / "tracks.csv", usecols=["frame", "track_id", "x1", "y1", "x2", "y2"])
        tm = pd.read_csv(DATA / "out" / stem / "teams.csv").set_index("track_id")["team"]
        st = tr.track_id.map(tm).fillna("u") == "stands"
        on_ice = tr.track_id.map(lambda t_: ymed.get((i, t_), 99.0) < W - 0.5)
        tr = tr[~st | on_ice]
        step = max(1, round(fps[i] / 12))
        tr = tr[tr.frame % step == 0]
        tr["t"] = tr.frame / fps[i]
        tr = tr[(tr.t >= p["start"]) & (tr.t < p["end"])].copy()
        tr["mt"] = offs[i] + tr.t - p["start"]
        sel = (cam_t >= offs[i] - 1) & (cam_t <= offs[i] + p["end"] - p["start"] + 1) & np.array([c[3] == i for c in cam]) & cam_good
        ct, cx = cam_t[sel], cam_x[sel]
        if len(ct) < 2:
            continue
        sl = Slerp(ct, Rot.from_rotvec(cx[:, :3]))
        # drop rows far from any good camera sample (mapping lost)
        idx = np.searchsorted(ct, tr.mt.values).clip(1, len(ct) - 1)
        near = np.minimum(np.abs(ct[idx] - tr.mt.values), np.abs(ct[idx - 1] - tr.mt.values))
        tr = tr[near <= 0.34]
        foot = undist(np.c_[(tr.x1 + tr.x2) / 2, tr.y2].astype(float), k1)
        mts = tr.mt.values.clip(ct[0], ct[-1])
        R = sl(mts).as_matrix()
        fz = np.interp(mts, ct, cx[:, 3])
        XY = np.zeros((len(tr), 2))
        for j in range(len(tr)):
            K = np.array([[fz[j], 0, 960], [0, fz[j], 540], [0, 0, 1.0]])
            H = K @ R[j] @ np.array([[1, 0, -C[0]], [0, 1, -C[1]], [0, 0, -C[2]]])
            w = np.linalg.solve(H, np.r_[foot[j], 1])
            XY[j] = w[:2] / w[2]
        tr["X"], tr["Y"] = XY[:, 0], XY[:, 1]
        tr = tr[(tr.X.abs() < g + 13) & (tr.Y.abs() < W + 2)]
        for tid, gq in tr.sort_values("mt").groupby("track_id"):
            if len(gq) < 2:
                continue
            X = gq.X.rolling(5, center=True, min_periods=1).median().rolling(5, center=True, min_periods=1).mean().values
            Y = gq.Y.rolling(5, center=True, min_periods=1).median().rolling(5, center=True, min_periods=1).mean().values
            if team.get((i, tid)) == "x":  # marked "not a player" on the page
                continue
            segs.append({"team": team.get((i, tid), "w"), "t": gq.mt.values, "X": X, "Y": Y, "key": f"{i}:{tid}"})
    # link short dropouts: a dot that disappears continues if a same-color detection shows up near where it was heading
    segs.sort(key=lambda s: s["t"][0])
    ends = []
    for k, s in enumerate(segs):
        m = s["t"] >= s["t"][-1] - 0.5
        dtt = s["t"][-1] - s["t"][m][0]
        v = np.array([s["X"][-1] - s["X"][m][0], s["Y"][-1] - s["Y"][m][0]]) / dtt if dtt > 0.15 else np.zeros(2)
        speed = np.hypot(*v)
        if speed > 25:
            v *= 25 / speed
        ends.append(v)
    starts = np.array([s["t"][0] for s in segs])
    cand = []
    for k, s in enumerate(segs):
        te = s["t"][-1]
        lo, hi = np.searchsorted(starts, te - 0.25), np.searchsorted(starts, te + link_s)
        for j in range(lo, hi):
            if j <= k or segs[j]["team"] != s["team"]:  # j > k: links only go forward, so no cycles
                continue
            gap = segs[j]["t"][0] - te
            pred = np.array([s["X"][-1], s["Y"][-1]]) + ends[k] * max(gap, 0)
            d = np.hypot(segs[j]["X"][0] - pred[0], segs[j]["Y"][0] - pred[1])
            if d < 4 + 6 * max(gap, 0):
                cand.append((d + 2 * max(gap, 0), k, j))
    cand.sort()
    nxt, prv = {}, {}
    for c, k, j in cand:
        if k in nxt or j in prv:
            continue
        nxt[k], prv[j] = j, k
    dot = {}
    did = 0
    for k in range(len(segs)):
        if k in prv:
            continue
        j = k
        while True:
            dot[j] = did
            if j not in nxt:
                break
            j = nxt[j]
        did += 1
    life = {}
    for k, s_ in enumerate(segs):
        life[dot[k]] = life.get(dot[k], 0.0) + (s_["t"][-1] - s_["t"][0])
    keep_dot = {d_ for d_, L in life.items() if L >= min_life}
    # sample everything at 12 fps on the master clock
    grid = np.round(np.arange(0, duration, 1 / 12), 3)
    frames = {}

    def put(t, row):
        frames.setdefault(int(round(t * 12)), []).append(row)

    for k, s in enumerate(segs):
        if dot[k] not in keep_dot:
            continue
        sx, sy = to_std(s["X"], s["Y"])
        d = dot[k]
        for t_, x_, y_ in zip(s["t"], sx, sy):
            put(t_, [d, round(float(x_), 1), round(float(y_), 1), s["team"], 0])
        if k in nxt:  # bridge the gap
            j = nxt[k]
            nx, ny = to_std(segs[j]["X"][:1], segs[j]["Y"][:1])
            t_a, t_b = s["t"][-1], segs[j]["t"][0]
            for q in range(int(round(t_a * 12)) + 1, int(round(t_b * 12))):
                u = (q / 12 - t_a) / max(t_b - t_a, 1e-6)
                put(q / 12, [d, round(float(sx[-1] + (nx[0] - sx[-1]) * u), 1), round(float(sy[-1] + (ny[0] - sy[-1]) * u), 1), s["team"], 1])
        elif hold_end_s > 0:  # hold the last spot briefly
            for q in range(int(round(s["t"][-1] * 12)) + 1, int(round((s["t"][-1] + hold_end_s) * 12))):
                put(q / 12, [d, round(float(sx[-1]), 1), round(float(sy[-1]), 1), s["team"], 2])
    # one row per dot per frame (prefer the measured row over a held one)
    for q, rows in frames.items():
        best = {}
        for r in rows:
            if r[0] not in best or r[4] < best[r[0]][4]:
                best[r[0]] = r
        frames[q] = list(best.values())
    # true duplicates only: two measured dots of the same team that sit on top of each other
    # (< dup_ft) for nearly all the time they coexist, for at least dup_min_s. Kids crossing in
    # front of each other are close only briefly, so both are kept.
    co, close = {}, {}
    for rows in frames.values():
        ms = [r for r in rows if r[4] == 0]
        for a in range(len(ms)):
            for c in range(a + 1, len(ms)):
                ra, rc = ms[a], ms[c]
                if ra[3] != rc[3]:
                    continue
                key = (min(ra[0], rc[0]), max(ra[0], rc[0]))
                co[key] = co.get(key, 0) + 1
                if np.hypot(ra[1] - rc[1], ra[2] - rc[2]) < dup_ft:
                    close[key] = close.get(key, 0) + 1
    dup_pairs = {k for k, n in close.items() if n >= dup_min_s * 12 and n >= dup_share * co[k]}
    drop = {}  # dot -> set of dots it duplicates (drop the shorter-lived one where both show)
    for a_, c_ in dup_pairs:
        lo_, hi_ = (a_, c_) if life[a_] < life[c_] else (c_, a_)
        drop.setdefault(lo_, set()).add(hi_)
    n_dup_rows = 0
    for q, rows in frames.items():
        here = {r[0]: r for r in rows}
        kept = []
        for r in sorted(rows, key=lambda r: (r[4] > 0, -life[r[0]])):
            o = next((here[d_] for d_ in drop.get(r[0], ()) if d_ in here), None)
            if o is not None and np.hypot(o[1] - r[1], o[2] - r[2]) < merge_ft:
                n_dup_rows += 1
                continue
            kept.append(r)
        count, out_ = {}, []
        for r in kept:
            if count.get(r[3], 0) >= cap.get(r[3], 6):
                continue
            count[r[3]] = count.get(r[3], 0) + 1
            out_.append(r)
        frames[q] = out_
    # which tracks make up each dot (so a team fix on the page can be saved per track)
    dots_tracks = {}
    for k, s_ in enumerate(segs):
        if dot[k] in keep_dot:
            dots_tracks.setdefault(str(dot[k]), []).append(s_["key"])
    (out / "dots.json").write_text(json.dumps(dots_tracks))
    (out / "dot_fix.json").unlink(missing_ok=True)  # page fixes are now baked in through team_fix.json
    # write per-minute files
    bx = np.r_[np.c_[np.linspace(0, 1919, 16), np.full(16, 1079)], np.c_[np.full(6, 1919), np.linspace(1079, 0, 6)],
               np.c_[np.linspace(1919, 0, 16), np.zeros(16)], np.c_[np.zeros(6), np.linspace(0, 1079, 6)]]
    nmin = int(np.ceil(duration / 60))
    for m in range(nmin):
        fr = [[q / 12, frames[q]] for q in range(m * 720, (m + 1) * 720) if q in frames]
        sel = (cam_t >= m * 60) & (cam_t < (m + 1) * 60)
        Hs, fp, gd, ts = [], [], [], []
        for t_, x_, good in zip(cam_t[sel], cam_x[sel], cam_good[sel]):
            ts.append(round(float(t_), 3))
            gd.append(int(good))
            if not good:
                Hs.append(None)
                fp.append([])
                continue
            H = Hcam(x_[:3], x_[3], C)
            H = H / H[2, 2]
            Hs.append([round(float(v), 7) for v in H.ravel()])
            w = np.c_[undist(bx, k1), np.ones(len(bx))] @ np.linalg.inv(H).T
            ok = w[:, 2] > 0
            pts = w[ok, :2] / w[ok, 2:]
            pts = pts[(np.abs(pts[:, 0]) < 300) & (np.abs(pts[:, 1]) < 200)]
            if len(pts) > 2:
                sx, sy = to_std(np.clip(pts[:, 0], -150, 150), np.clip(pts[:, 1], -80, 80))
                fp.append(np.round(np.c_[sx, sy], 1).tolist())
            else:
                fp.append([])
        (out / f"{m}.json").write_text(json.dumps({"frames": fr, "cam_t": ts, "H": Hs, "foot": fp, "good": gd}, separators=(",", ":")))
    meta = {"game": game_id, "duration": duration, "minutes": nmin, "k1": k1,
            "dims": {"W": W, "blue": b, "doty": dy, "goal": g}, "link_s": link_s, "hold_end_s": hold_end_s, "built": __import__("time").time()}
    (out / "meta.json").write_text(json.dumps(meta))
    src = [s for r in res for s in r["src"]]
    return {"camera_samples": len(cam), "good_pct": round(100 * float(cam_good.mean()), 1),
            "anchor_pct": round(100 * src.count("anchor") / max(len(src), 1), 1), "segments": len(segs),
            "links": len(nxt), "dots": did, "learn": learn, "dup_pairs": len(dup_pairs), "dup_rows_dropped": n_dup_rows, "teams": {t: list(team.values()).count(t) for t in "wbr"},
            "bad_minutes": [int(m) for m in range(nmin) if ((cam_t >= m * 60) & (cam_t < (m + 1) * 60)).any()
                            and cam_good[(cam_t >= m * 60) & (cam_t < (m + 1) * 60)].mean() < 0.8]}


@app.local_entrypoint()
def test_chunk(t0: float = 330, t1: float = 345):
    import numpy as np
    r = map_chunk.remote("2026-09-26", 1, "Somerville_Veterans_Memorial_Rink_Rink_1_2026-09-26T142957.mp4", t0, t1, 24.0)
    print("frames", len(r["poses"]), "src", {s: r["src"].count(s) for s in set(r["src"])})
    q = np.array(r["quals"], float)
    print("median line px red,blue,far", np.nanmedian(q, 0).round(1), "bad", int(((np.nan_to_num(q[:, 0]) > 6) | (np.nan_to_num(q[:, 2]) > 6)).sum()))
    np.save("/tmp/chunk_poses.npy", np.array(r["poses"]))
