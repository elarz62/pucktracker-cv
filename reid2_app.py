"""Teach a model what each kid looks like from the parent's tags, and test it honestly.

    modal run reid2_app.py --game-id 2026-09-26            # crops, then 3-fold test
    modal run reid2_app.py --game-id 2026-09-26 --skip-crops

Crops: up to PER_TRACK frames of every tagged track (numbers only) plus every white or unsure track,
saved to /data/games/<id>/reid_crops_<part>.npz.
Test: train on two files' tags, predict the third file's tagged tracks (crop votes averaged per track).
Two models are compared: frozen DINOv2 features + logistic regression (what failed before with 299 tags),
and DINOv2 with its last blocks fine-tuned on the crops.
"""
import json
import pathlib

import modal

app = modal.App("pucktracker-reid2")
vol = modal.Volume.from_name("pucktracker-videos")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "git")
    .pip_install("torch==2.4.1", "torchvision==0.19.1", "opencv-python-headless", "pandas", "scikit-learn", "numpy<2")
)
DATA = pathlib.Path("/data")
PER_TRACK = 24
H, W = 224, 112


def game_dir(game_id):
    return DATA / "games" / game_id


def tags_for(game_id):
    f = game_dir(game_id) / "tags.json"
    t = json.loads(f.read_text()) if f.exists() else {}
    return {k: v for k, v in t.items() if v.isdigit() and v != "10"}


@app.function(image=image, volumes={str(DATA): vol}, cpu=8, memory=32768, timeout=3 * 60 * 60)
def crops(game_id: str, part: int, src: str = "out") -> dict:
    import cv2
    import numpy as np
    import pandas as pd

    game = json.loads((game_dir(game_id) / "game.json").read_text())
    p = game["parts"][part]
    stem = pathlib.Path(p["file"]).stem
    tr = pd.read_csv(DATA / src / stem / "tracks.csv")
    teams = pd.read_csv(DATA / src / stem / "teams.csv").set_index("track_id")["team"]
    cap = cv2.VideoCapture(str(DATA / "videos" / p["file"]))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    tr = tr[(tr.frame >= p["start"] * fps) & (tr.frame < p["end"] * fps)]
    tagged = {int(k.split(":")[1]) for k in tags_for(game_id) if k.startswith(f"{part}:")} if src == "out" else set()
    n = tr.groupby("track_id").size()
    keep = [t for t in n.index if t in tagged or (teams.get(t) in ("white", "unknown") and n[t] >= 12)]
    tr = tr[tr.track_id.isin(keep)]
    want = {}
    for t, g in tr.groupby("track_id"):
        rows = g.iloc[np.linspace(0, len(g) - 1, min(PER_TRACK, len(g))).astype(int)]
        for r in rows.itertuples():
            want.setdefault(int(r.frame), []).append((t, r.x1, r.y1, r.x2, r.y2))
    frames = sorted(want)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
    f = frames[0]
    ids, ims, hs = [], [], []
    for target in frames:
        while f < target:
            cap.grab(); f += 1
        ok, im = cap.read(); f += 1
        if not ok:
            break
        for t, x1, y1, x2, y2 in want[target]:
            # a little context around the box, so skates and stick are in the crop
            px, py = 0.1 * (x2 - x1), 0.05 * (y2 - y1)
            a, b = max(0, int(x1 - px)), max(0, int(y1 - py))
            c = im[b:int(y2 + py), a:int(x2 + px)]
            if c.size and c.shape[0] >= 16 and c.shape[1] >= 8:
                ids.append(f"{part}:{t}"); hs.append(int(y2 - y1))
                ims.append(cv2.cvtColor(cv2.resize(c, (W, H), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB))
    np.savez(game_dir(game_id) / f"reid_crops_{src}_{part}.npz", ids=np.array(ids), ims=np.stack(ims), hs=np.array(hs))
    vol.commit()
    return {"part": part, "tracks": len(set(ids)), "crops": len(ids)}


def load(game_id, src="out"):
    import numpy as np
    ids, ims = [], []
    n = len(json.loads((game_dir(game_id) / "game.json").read_text())["parts"])
    for part in range(n):
        z = np.load(game_dir(game_id) / f"reid_crops_{src}_{part}.npz")
        ids += list(z["ids"]); ims.append(z["ims"])
    return np.array(ids), np.concatenate(ims)


def backbone():
    import torch
    m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    return m


def track_scores(ids, P):
    """Average crop probabilities per track."""
    import numpy as np
    out = {}
    for t in np.unique(ids):
        out[t] = P[ids == t].mean(0)
    return out


def finetune(m, Xtr, y, n_classes, epochs, prep):
    """Fine-tune the last 4 DINOv2 blocks plus a linear head on crops Xtr with labels y (ints)."""
    import numpy as np
    import torch
    import torch.nn as nn
    dev = "cuda"
    for p in m.parameters():
        p.requires_grad = False
    for blk in m.blocks[-4:]:
        for p in blk.parameters():
            p.requires_grad = True
    for p in m.norm.parameters():
        p.requires_grad = True
    head = nn.Linear(768, n_classes).to(dev)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": params, "lr": 2e-5}, {"params": head.parameters(), "lr": 1e-3}], weight_decay=0.05)
    ytr = torch.tensor(y, device=dev)
    # balance kids: sample each crop with weight 1 / (crops of that kid)
    cnt = np.bincount(np.array(y), minlength=n_classes)
    w = 1.0 / cnt[np.array(y)]
    w = w / w.sum()
    steps = epochs * len(Xtr) // 64
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[2e-5, 1e-3], total_steps=steps)
    rng = np.random.default_rng(0)
    loss_f = nn.CrossEntropyLoss(label_smoothing=0.1)
    m.train()
    for s in range(steps):
        b = rng.choice(len(Xtr), 64, p=w)
        x = prep(Xtr[b])
        if rng.random() < 0.5:
            x = x.flip(3)
        # brightness/contrast jitter and a random shift
        x = x * (1 + 0.2 * (torch.rand(len(b), 1, 1, 1, device=dev) - 0.5)) + 0.2 * (torch.rand(len(b), 1, 1, 1, device=dev) - 0.5)
        dy, dx = rng.integers(-12, 13), rng.integers(-8, 9)
        x = torch.roll(x, shifts=(int(dy), int(dx)), dims=(2, 3))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = loss_f(head(m(x)), ytr[b])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    m.eval()
    return head


@app.function(image=image, gpu="L4", volumes={str(DATA): vol}, cpu=8, memory=49152, timeout=3 * 60 * 60)
def fold(game_id: str, test_part: int, epochs: int = 8) -> dict:
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.linear_model import LogisticRegression

    vol.reload()
    tags = tags_for(game_id)
    ids, ims = load(game_id)
    lab = np.array([tags.get(t, "") for t in ids])
    has = lab != ""
    test = has & np.array([t.startswith(f"{test_part}:") for t in ids])
    train = has & ~test
    classes = sorted(set(lab[train]), key=int)
    ci = {c: k for k, c in enumerate(classes)}
    dev = "cuda"
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    def prep(x):
        x = torch.from_numpy(x).to(dev).permute(0, 3, 1, 2).float() / 255
        return (x - mean) / std

    def feats(m, X, bs=256):
        out = []
        m.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for k in range(0, len(X), bs):
                out.append(m(prep(X[k:k + bs])).float().cpu())
        return torch.cat(out).numpy()

    def score(P, cls_list):
        ts = track_scores(ids[test], P)
        truth = {t: tags[t] for t in ts}
        top1 = {t: cls_list[int(p.argmax())] for t, p in ts.items()}
        conf = {t: float(p.max()) for t, p in ts.items()}
        r = {"tracks": len(ts), "top1": round(np.mean([top1[t] == truth[t] for t in ts]), 3),
             "top3": round(np.mean([truth[t] in [cls_list[j] for j in np.argsort(-p)[:3]] for t, p in ts.items()]), 3)}
        for th in (0.5, 0.7, 0.9):
            sel = [t for t in ts if conf[t] >= th]
            r[f"conf>={th}"] = [round(len(sel) / len(ts), 2), round(np.mean([top1[t] == truth[t] for t in sel]), 3) if sel else None]
        return r, top1, conf

    res = {"test_part": test_part, "train_crops": int(train.sum()), "test_crops": int(test.sum()),
           "classes": len(classes), "chance": round(1 / len(classes), 3)}

    # 1) frozen features + logistic regression
    m = backbone().to(dev)
    F_tr, F_te = feats(m, ims[train]), feats(m, ims[test])
    clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(F_tr, lab[train])
    res["frozen"], _, _ = score(clf.predict_proba(F_te), list(clf.classes_))

    # 2) fine-tune the last 4 blocks + a linear head, with augmentation
    head = finetune(m, ims[train], [ci[c] for c in lab[train]], len(classes), epochs, prep)
    m.eval()
    with torch.no_grad():
        F = torch.from_numpy(feats(m, ims[test])).to(dev)
        P = torch.softmax(head(F), 1).cpu().numpy()
    res["finetuned"], top1, conf = score(P, classes)
    res["finetuned_mistakes"] = [(t, tags[t], top1[t], round(conf[t], 2)) for t in top1 if top1[t] != tags[t]][:20]
    return res


@app.function(image=image, gpu="L4", volumes={str(DATA): vol}, cpu=8, memory=49152, timeout=3 * 60 * 60)
def fill(game_id: str, epochs: int = 8) -> dict:
    """Train on every tag, then guess every untagged track: writes guesses.json {track: [number, confidence]}
    and saves the model (reid_model.pt) for the next game."""
    import numpy as np
    import torch

    vol.reload()
    tags = tags_for(game_id)
    ids, ims = load(game_id)
    lab = np.array([tags.get(t, "") for t in ids])
    train = lab != ""
    classes = sorted(set(lab[train]), key=int)
    ci = {c: k for k, c in enumerate(classes)}
    dev = "cuda"
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    def prep(x):
        x = torch.from_numpy(x).to(dev).permute(0, 3, 1, 2).float() / 255
        return (x - mean) / std

    m = backbone().to(dev)
    head = finetune(m, ims[train], [ci[c] for c in lab[train]], len(classes), epochs, prep)
    every = set(json.loads((game_dir(game_id) / "tags.json").read_text()))   # includes x and ?
    rest = np.array([t not in every for t in ids])
    R, P = ims[rest], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for k in range(0, len(R), 256):
            P.append(torch.softmax(head(m(prep(R[k:k + 256]))).float(), 1).cpu())
    P = torch.cat(P).numpy() if P else np.zeros((0, len(classes)))
    out = {}
    for t, p in track_scores(ids[rest], P).items():
        j = int(p.argmax())
        out[str(t)] = [classes[j], round(float(p[j]), 3)]
    (game_dir(game_id) / "guesses.json").write_text(json.dumps(out, separators=(",", ":")))
    torch.save({"classes": classes, "head": head.state_dict(), "blocks": [b.state_dict() for b in m.blocks[-4:]],
                "norm": m.norm.state_dict()}, game_dir(game_id) / "reid_model.pt")
    vol.commit()
    return {"trained_on_tracks": int(len(set(ids[train]))), "guessed": len(out),
            "conf>=0.5": sum(1 for v in out.values() if v[1] >= 0.5)}


@app.function(image=image, gpu="L4", volumes={str(DATA): vol}, cpu=8, memory=49152, timeout=3 * 60 * 60)
def predict(game_id: str, model_game: str) -> dict:
    """Use the model trained on model_game to guess every track in game_id: writes guesses.json."""
    import numpy as np
    import torch
    import torch.nn as nn

    vol.reload()
    ck = torch.load(game_dir(model_game) / "reid_model.pt", map_location="cuda")
    m = backbone().cuda()
    for b, sd in zip(m.blocks[-4:], ck["blocks"]):
        b.load_state_dict(sd)
    m.norm.load_state_dict(ck["norm"])
    head = nn.Linear(768, len(ck["classes"])).cuda()
    head.load_state_dict(ck["head"])
    m.eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    ids, ims = load(game_id)
    P = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for k in range(0, len(ims), 256):
            x = torch.from_numpy(ims[k:k + 256]).cuda().permute(0, 3, 1, 2).float() / 255
            P.append(torch.softmax(head(m((x - mean) / std)).float(), 1).cpu())
    P = torch.cat(P).numpy()
    out = {}
    for t, p in track_scores(ids, P).items():
        j = int(p.argmax())
        out[str(t)] = [ck["classes"][j], round(float(p[j]), 3)]
    (game_dir(game_id) / "guesses.json").write_text(json.dumps(out, separators=(",", ":")))
    vol.commit()
    return {"guessed": len(out), "conf>=0.5": sum(1 for v in out.values() if v[1] >= 0.5)}


@app.local_entrypoint()
def main(game_id: str = "2026-09-26", skip_crops: bool = False, epochs: int = 8, mode: str = "test",
         model_game: str = "2026-09-26"):
    if not skip_crops:
        n = len(json.load(open(f"games/{game_id}.json"))["parts"])
        for r in crops.starmap([(game_id, k) for k in range(n)]):
            print("CROPS", r, flush=True)
    if mode == "predict":
        print("PREDICT", json.dumps(predict.remote(game_id, model_game)), flush=True)
        return
    if mode == "fill":
        print("FILL", json.dumps(fill.remote(game_id, epochs)), flush=True)
        return
    out = list(fold.starmap([(game_id, k, epochs) for k in range(3)]))
    for r in out:
        print("FOLD", json.dumps(r), flush=True)
    pathlib.Path("reid2_result.json").write_text(json.dumps(out, indent=1))
