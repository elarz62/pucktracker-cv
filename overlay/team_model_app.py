"""Team model: is this player ours, the other team, a referee, or not a player at all?

Trained once per jersey color (ours in white, ours in blue) and reused for every game in that jersey.

    modal run overlay/team_model_app.py::main --jersey white --train 2026-09-26 --test 2026-09-19

Steps:
  1. crops: up to 6 crops of every track in the current tracks.csv, per game part (CPU, cached).
  2. train (GPU): DINOv2 ViT-B/14 with its last 4 blocks fine-tuned plus a 4-way head.
     Labels from the training game(s):
       ours  = jersey-number tags from the Shift Marker, plus rink-page fixes to our color
       them  = tracks both the old and the new color rules call the other color (not tagged)
       ref   = tracks both the old classifier and the height rule call a referee, plus ref fixes
       other = "not a player" tags and fixes, plus a sample of crowd tracks off the ice
  3. predict: per-track probabilities for any game, saved to /data/games/<id>/rink/team_pred.json.
     The rink build reads them (see learn_teams in rink_app.py).
Honest test: the held-out game's jersey tags (all ours) are never used in training.
"""
import json
import pathlib

import modal

app = modal.App("pucktracker-teammodel")
vol = modal.Volume.from_name("pucktracker-videos")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "git", "ffmpeg")
    .pip_install("torch==2.4.1", "torchvision==0.19.1", "opencv-python-headless==4.10.0.84", "pandas==2.2.3",
                 "scikit-learn==1.5.2", "numpy==1.26.4")
)
DATA = pathlib.Path("/data")
CH, CW = 168, 84          # crop size (multiples of the 14 px ViT patch)
PER_TRACK = 6
CLASSES = ["ours", "them", "ref", "other"]


def gdir(g):
    return DATA / "games" / g


@app.function(image=image, volumes={str(DATA): vol}, cpu=4, memory=16384, timeout=2 * 3600)
def crops(game_id: str, part: int) -> dict:
    import cv2
    import numpy as np
    import pandas as pd
    game = json.loads((gdir(game_id) / "game.json").read_text())
    p = game["parts"][part]
    stem = pathlib.Path(p["file"]).stem
    tr = pd.read_csv(DATA / "out" / stem / "tracks.csv", usecols=["frame", "track_id", "x1", "y1", "x2", "y2"])
    cap = cv2.VideoCapture(str(DATA / "videos" / p["file"]))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    tr = tr[(tr.frame >= p["start"] * fps) & (tr.frame < p["end"] * fps) & (tr.y2 - tr.y1 >= 20)]
    want = {}
    for t, g in tr.groupby("track_id"):
        if len(g) < 4:
            continue
        rows = g.iloc[np.linspace(0, len(g) - 1, min(PER_TRACK, len(g))).astype(int)]
        for r in rows.itertuples():
            want.setdefault(int(r.frame), []).append((int(t), r.x1, r.y1, r.x2, r.y2))
    ids, ims, hs = [], [], []
    f = -1
    for target in sorted(want):
        if target - f > 48:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            f = target
        while f < target:
            cap.grab()
            f += 1
        ok, im = cap.read()
        f += 1
        if not ok:
            break
        for t, x1, y1, x2, y2 in want[target]:
            px, py = 0.1 * (x2 - x1), 0.05 * (y2 - y1)
            a, b = max(0, int(x1 - px)), max(0, int(y1 - py))
            c = im[b:int(y2 + py), a:int(x2 + px)]
            if c.size and c.shape[0] >= 16 and c.shape[1] >= 8:
                ids.append(f"{part}:{t}")
                hs.append(int(y2 - y1))
                ims.append(cv2.cvtColor(cv2.resize(c, (CW, CH), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    np.savez(gdir(game_id) / f"team_crops_{part}.npz", ids=np.array(ids), ims=np.stack(ims), hs=np.array(hs))
    vol.commit()
    return {"game": game_id, "part": part, "tracks": len(set(ids)), "crops": len(ids)}


def load_crops(game_id):
    import numpy as np
    n = len(json.loads((gdir(game_id) / "game.json").read_text())["parts"])
    ids, ims = [], []
    for part in range(n):
        z = np.load(gdir(game_id) / f"team_crops_{part}.npz")
        ids.append(z["ids"])
        ims.append(z["ims"])
    return np.concatenate(ids), np.concatenate(ims)


def labels(game_id, jersey):
    """Track -> class for a training game (see module doc)."""
    import numpy as np
    import pandas as pd
    import pickle
    game = json.loads((gdir(game_id) / "game.json").read_text())
    ours_code = "w" if jersey == "white" else "b"
    them_old = "blue" if jersey == "white" else "white"
    lab = {}
    # rule labels from the rink build (team per track before fixes), when the game has one
    rink = gdir(game_id) / "rink"
    rule = {}
    if (rink / "rules_team.json").exists():
        rule = json.loads((rink / "rules_team.json").read_text())
    for i, p in enumerate(game["parts"]):
        tm = pd.read_csv(DATA / "out" / pathlib.Path(p["file"]).stem / "teams.csv")
        n_stands = 0
        for r in tm.itertuples():
            k = f"{i}:{int(r.track_id)}"
            if r.team == them_old and rule.get(k, "b" if ours_code == "w" else "w") != ours_code:
                lab[k] = "them"
            elif r.team == "ref" and rule.get(k, "r") == "r":
                lab[k] = "ref"
            elif r.team == "stands" and n_stands < 400 and rule.get(k) is None:
                lab[k] = "other"
                n_stands += 1
    tags = json.loads((gdir(game_id) / "tags.json").read_text()) if (gdir(game_id) / "tags.json").exists() else {}
    for k, v in tags.items():
        if str(v).isdigit():
            lab[k] = "ours"
        elif v == "x":
            lab[k] = "other"
    fx = json.loads((rink / "team_fix.json").read_text()) if (rink / "team_fix.json").exists() else {}
    for k, v in fx.items():
        lab[k] = {"x": "other", "r": "ref"}.get(v, "ours" if v == ours_code else "them")
    return lab


def backbone():
    import torch
    return torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")


def make_prep(dev):
    import torch
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    def prep(x):
        x = torch.as_tensor(x, device=dev).permute(0, 3, 1, 2).float() / 255.0
        return (x - mean) / std
    return prep


def finetune(m, X, y, epochs, prep, dev="cuda"):
    import numpy as np
    import torch
    import torch.nn as nn
    for p in m.parameters():
        p.requires_grad = False
    for blk in m.blocks[-4:]:
        for p in blk.parameters():
            p.requires_grad = True
    for p in m.norm.parameters():
        p.requires_grad = True
    head = nn.Linear(768, len(CLASSES)).to(dev)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": params, "lr": 2e-5}, {"params": head.parameters(), "lr": 1e-3}], weight_decay=0.05)
    y = np.asarray(y)
    yt = torch.tensor(y, device=dev)
    cnt = np.bincount(y, minlength=len(CLASSES)).astype(float)
    w = 1.0 / np.maximum(cnt[y], 1)
    w /= w.sum()
    steps = max(200, epochs * len(X) // 64)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[2e-5, 1e-3], total_steps=steps)
    rng = np.random.default_rng(0)
    loss_f = nn.CrossEntropyLoss(label_smoothing=0.1)
    m.train()
    for s in range(steps):
        b = rng.choice(len(X), 64, p=w)
        x = prep(X[b])
        if rng.random() < 0.5:
            x = x.flip(3)
        x = x * (1 + 0.3 * (torch.rand(len(b), 1, 1, 1, device=dev) - 0.5)) + 0.3 * (torch.rand(len(b), 1, 1, 1, device=dev) - 0.5)
        dy, dx = rng.integers(-10, 11), rng.integers(-6, 7)
        x = torch.roll(x, shifts=(int(dy), int(dx)), dims=(2, 3))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = loss_f(head(m(x)), yt[b])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    m.eval()
    return head


def predict_tracks(m, head, ids, X, prep):
    import numpy as np
    import torch
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for k in range(0, len(X), 512):
            out.append(torch.softmax(head(m(prep(X[k:k + 512]))).float(), 1).cpu().numpy())
    P = np.concatenate(out)
    res = {}
    order = np.argsort(ids, kind="stable")
    ids_s, P_s = ids[order], P[order]
    cuts = np.flatnonzero(ids_s[1:] != ids_s[:-1]) + 1
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(ids_s)]):
        res[str(ids_s[a])] = [round(float(v), 4) for v in P_s[a:b].mean(0)]
    return res


@app.function(image=image, gpu="L4", volumes={str(DATA): vol}, cpu=8, memory=65536, timeout=3 * 3600)
def train(jersey: str, train_games: list, test_games: list, epochs: int = 6) -> dict:
    import numpy as np
    import torch
    dev = "cuda"
    prep = make_prep(dev)
    Xs, ys = [], []
    stats = {}
    for g in train_games:
        ids, X = load_crops(g)
        lab = labels(g, jersey)
        sel = np.array([lab.get(str(i)) is not None for i in ids])
        Xs.append(X[sel])
        ys.append([CLASSES.index(lab[str(i)]) for i in ids[sel]])
        stats[g] = {c: int(sum(v == c for v in lab.values())) for c in CLASSES}
    X, y = np.concatenate(Xs), np.concatenate(ys)
    m = backbone().to(dev)
    head = finetune(m, X, y, epochs, prep)
    torch.save({"model": m.state_dict(), "head": head.state_dict(), "classes": CLASSES, "jersey": jersey,
                "trained_on": train_games}, DATA / "models" / f"team_{jersey}.pt")
    report = {"train_labels": stats}
    for g in list(train_games) + list(test_games):
        ids, Xg = load_crops(g)
        pred = predict_tracks(m, head, ids, Xg, prep)
        (gdir(g) / "rink").mkdir(parents=True, exist_ok=True)
        (gdir(g) / "rink" / "team_pred.json").write_text(json.dumps({"jersey": jersey, "classes": CLASSES, "tracks": pred}))
        if g in test_games:
            tags = json.loads((gdir(g) / "tags.json").read_text())
            ours = [k for k, v in tags.items() if str(v).isdigit() and k in pred]
            xs = [k for k, v in tags.items() if v == "x" and k in pred]
            top = {k: CLASSES[int(np.argmax(p))] for k, p in pred.items()}
            report[g] = {"tracks": len(pred), "tagged_ours": len(ours),
                         "ours_called_ours_pct": round(100 * np.mean([top[k] == "ours" for k in ours]), 1) if ours else None,
                         "not_player_tags": len(xs),
                         "not_player_called_ours_pct": round(100 * np.mean([top[k] == "ours" for k in xs]), 1) if xs else None,
                         "split": {c: sum(v == c for v in top.values()) for c in CLASSES}}
    vol.commit()
    return report


@app.local_entrypoint()
def main(jersey: str = "white", train: str = "2026-09-26", test: str = "2026-09-19", skip_crops: bool = False, epochs: int = 6):
    games = [g for g in (train.split(",") + test.split(",")) if g]
    if not skip_crops:
        jobs = []
        for g in games:
            n = len(json.loads((pathlib.Path(__file__).parent.parent / "games" / f"{g}.json").read_text())["parts"])
            jobs += [(g, i) for i in range(n)]
        for r in crops.starmap(jobs):
            print(r)
    print(json.dumps(globals()["train"].remote(jersey, train.split(","), [g for g in test.split(",") if g], epochs), indent=1))
