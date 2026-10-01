"""Learn each kid's look from parent tags, then guess who is in the untagged boxes.

    modal run reid_app.py::embed --game-id 2026-09-26      # crops + DINOv2 embeddings per track
    modal run reid_app.py::evaluate --game-id 2026-09-26   # hold out one file, report accuracy
    modal run reid_app.py::guess --game-id 2026-09-26      # write guesses.json for the Shift Marker

Labels come from tags.json (parent clicks) plus tracks whose number the reader saw clearly.
Track ids are "<part>:<tracker id>", the same ids the Shift Marker uses.
"""
import json
import pathlib

import modal

app = modal.App("pucktracker-reid")
vol = modal.Volume.from_name("pucktracker-videos")
DATA = pathlib.Path("/data")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "git")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "opencv-python-headless", "pandas", "scikit-learn", "numpy<2")
    .add_local_dir("games", "/root/games")
)
TEAMS_KEPT = {"white", "unknown", "stands"}


def game_dir(game_id):
    return DATA / "games" / game_id


def load_game(game_id):
    return json.loads((game_dir(game_id) / "game.json").read_text())


def known_numbers(game_id):
    p = pathlib.Path("/root/games") / f"{game_id}.drafts.json"
    return {k: v[0] for k, v in json.loads(p.read_text()).get("known", {}).items() if v[1] == "a"}


@app.function(image=image, gpu="L4", volumes={str(DATA): vol}, cpu=8, memory=32768, timeout=3 * 60 * 60)
def embed(game_id: str, per_track: int = 12, min_frames: int = 12) -> dict:
    """Embed up to per_track crops of every home-team-colored track (and every tagged track) in the game window."""
    import cv2
    import numpy as np
    import pandas as pd
    import torch

    game = load_game(game_id)
    tags = json.loads((game_dir(game_id) / "tags.json").read_text()) if (game_dir(game_id) / "tags.json").exists() else {}
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").cuda().eval().half()
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1).half()
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1).half()
    ids, embs, sizes = [], [], []
    for i, p in enumerate(game["parts"]):
        stem = pathlib.Path(p["file"]).stem
        tr = pd.read_csv(DATA / "out" / stem / "tracks.csv")
        teams = pd.read_csv(DATA / "out" / stem / "teams.csv").set_index("track_id")["team"]
        cap = cv2.VideoCapture(str(DATA / "videos" / p["file"]))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        tr = tr[(tr.frame >= p["start"] * fps) & (tr.frame < p["end"] * fps)]
        tagged = {int(k.split(":")[1]) for k in tags if k.startswith(f"{i}:")}
        n = tr.groupby("track_id").size()
        keep = [t for t in n.index if t in tagged or (teams.get(t) in TEAMS_KEPT and n[t] >= min_frames)]
        tr = tr[tr.track_id.isin(keep)]
        # Pick evenly spaced frames per track, then decode the file once in order.
        want = {}
        for t, g in tr.groupby("track_id"):
            rows = g.iloc[np.linspace(0, len(g) - 1, min(per_track, len(g))).astype(int)]
            for r in rows.itertuples():
                want.setdefault(int(r.frame), []).append((t, r.x1, r.y1, r.x2, r.y2))
        crops = {t: [] for t in keep}
        frames = sorted(want)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
        f = frames[0]
        for target in frames:
            while f < target:
                cap.grab(); f += 1
            ok, im = cap.read(); f += 1
            if not ok:
                break
            for t, x1, y1, x2, y2 in want[target]:
                x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), int(x2), int(y2)
                c = im[y1:y2, x1:x2]
                if c.size and c.shape[0] >= 16 and c.shape[1] >= 8:
                    crops[t].append(cv2.cvtColor(cv2.resize(c, (112, 224)), cv2.COLOR_BGR2RGB))
        for t, cs in crops.items():
            if not cs:
                continue
            x = torch.from_numpy(np.stack(cs)).cuda().permute(0, 3, 1, 2).half() / 255
            with torch.no_grad():
                e = model((x - mean) / std).float()
            e = torch.nn.functional.normalize(e, dim=1)
            ids.append(f"{i}:{t}"); embs.append(e.mean(0).cpu().numpy()); sizes.append(len(cs))
        print(f"part {i}: {len(keep)} tracks", flush=True)
    np.savez(game_dir(game_id) / "reid_embeddings.npz", ids=np.array(ids), embs=np.stack(embs), sizes=np.array(sizes))
    vol.commit()
    return {"tracks": len(ids)}


def labels_for(game_id):
    tags = json.loads((game_dir(game_id) / "tags.json").read_text())
    lab = dict(known_numbers(game_id))
    lab.update(tags)   # parent tags win over the number reader
    return lab, tags


def fit(X, y):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=2.0, max_iter=3000, class_weight="balanced").fit(X, y)


@app.function(image=image, volumes={str(DATA): vol}, cpu=4, memory=8192, timeout=1800)
def evaluate(game_id: str, test_part: int = 2) -> dict:
    """Train without the parent's tags in one file, then check the guesses on exactly those tags."""
    import numpy as np
    z = np.load(game_dir(game_id) / "reid_embeddings.npz")
    ids, X = list(z["ids"]), z["embs"]
    idx = {t: k for k, t in enumerate(ids)}
    lab, tags = labels_for(game_id)
    test = [t for t in tags if t.startswith(f"{test_part}:") and t in idx]
    train = [t for t in lab if t in idx and t not in set(test)]
    clf = fit(X[[idx[t] for t in train]], [lab[t] for t in train])
    P = clf.predict_proba(X[[idx[t] for t in test]])
    order = np.argsort(-P, axis=1)
    cls = clf.classes_
    top1 = [cls[o[0]] for o in order]
    truth = [tags[t] for t in test]
    conf = P.max(1)
    res = {"train": len(train), "test": len(test), "classes": len(cls),
           "top1": round(float(np.mean([a == b for a, b in zip(top1, truth)])), 3),
           "top3": round(float(np.mean([b in cls[o[:3]] for o, b in zip(order, truth)])), 3),
           "chance": round(1 / len(cls), 3)}
    for th in (0.5, 0.7, 0.85):
        m = conf >= th
        res[f"conf>={th}"] = {"share": round(float(m.mean()), 2),
                              "top1": round(float(np.mean([a == b for a, b, k in zip(top1, truth, m) if k])), 3) if m.any() else None}
    res["mistakes"] = [(t, b, a, round(float(c), 2)) for t, a, b, c in zip(test, top1, truth, conf) if a != b][:25]
    return res


@app.function(image=image, volumes={str(DATA): vol}, cpu=4, memory=8192, timeout=1800)
def guess(game_id: str) -> dict:
    """Train on every label and write guesses.json: {track: [number, confidence]} for untagged tracks."""
    import numpy as np
    z = np.load(game_dir(game_id) / "reid_embeddings.npz")
    ids, X = list(z["ids"]), z["embs"]
    lab, tags = labels_for(game_id)
    tr = [k for k, t in enumerate(ids) if t in lab]
    clf = fit(X[tr], [lab[ids[k]] for k in tr])
    P = clf.predict_proba(X)
    out = {}
    for k, t in enumerate(ids):
        if t in tags:
            continue
        j = int(P[k].argmax())
        out[t] = [str(clf.classes_[j]), round(float(P[k, j]), 3)]
    (game_dir(game_id) / "guesses.json").write_text(json.dumps(out, separators=(",", ":")))
    vol.commit()
    return {"guessed": len(out), "labels": len(tr)}


def stripe_score(crop):
    """Referee shirts alternate black and white across the chest.
    Returns (median light/dark flips per torso row, dark share, row agreement)."""
    import cv2
    import numpy as np
    h, w = crop.shape[:2]
    torso = crop[int(h * 0.15):int(h * 0.5), int(w * 0.1):int(w * 0.9)]
    if torso.size == 0 or torso.shape[1] < 10:
        return None
    g = cv2.cvtColor(cv2.resize(torso, (64, 24), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(g, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    flips = np.abs(np.diff(b.astype(int), axis=1)).sum(1)
    # stripes are vertical: columns agree from row to row
    agree = float((b[1:] == b[:-1]).mean())
    return float(np.median(flips)), float(1 - b.mean()), agree


@app.function(image=image, volumes={str(DATA): vol}, cpu=8, memory=16384, timeout=3 * 60 * 60)
def find_refs(game_id: str, per_track: int = 8) -> dict:
    """Score every track for referee stripes; writes refs.json {track: [flips, dark, light]}."""
    import cv2
    import numpy as np
    import pandas as pd
    game = load_game(game_id)
    out = {}
    for i, p in enumerate(game["parts"]):
        stem = pathlib.Path(p["file"]).stem
        tr = pd.read_csv(DATA / "out" / stem / "tracks.csv")
        cap = cv2.VideoCapture(str(DATA / "videos" / p["file"]))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        tr = tr[(tr.frame >= p["start"] * fps) & (tr.frame < p["end"] * fps) & ((tr.y2 - tr.y1) >= 40)]
        want = {}
        for t, g in tr.groupby("track_id"):
            for r in g.iloc[np.linspace(0, len(g) - 1, min(per_track, len(g))).astype(int)].itertuples():
                want.setdefault(int(r.frame), []).append((t, r.x1, r.y1, r.x2, r.y2))
        scores = {}
        frames = sorted(want); cap.set(cv2.CAP_PROP_POS_FRAMES, frames[0]); f = frames[0]
        for target in frames:
            while f < target:
                cap.grab(); f += 1
            ok, im = cap.read(); f += 1
            if not ok:
                break
            for t, x1, y1, x2, y2 in want[target]:
                s = stripe_score(im[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)])
                if s:
                    scores.setdefault(t, []).append(s)
        for t, s in scores.items():
            out[f"{i}:{t}"] = [round(float(v), 3) for v in np.median(np.array(s), axis=0)]
        print(f"part {i}: {len(scores)} tracks", flush=True)
    (game_dir(game_id) / "refs.json").write_text(json.dumps(out, separators=(",", ":")))
    vol.commit()
    return {"tracks": len(out)}
