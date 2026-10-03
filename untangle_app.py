"""Re-check who is who wherever two players cross, instead of trusting the tracker's guess.

    modal run untangle_app.py --game-id 2026-09-26                 # measure: events, cuts, swaps, #21 score
    modal run untangle_app.py --game-id 2026-09-26 --apply --margin 0.05

When two boxes overlap, the tracker keeps the ids it had, and after the players separate it can hand a
kid's id to the player he crossed (often an opponent). For every crossing we take clean looks at each
player just before and just after, embed them with the look-alike model, and decide:
  keep  - each player clearly looks like himself afterwards;
  swap  - they clearly traded ids; trade them back;
  unsure - neither is clear by `margin`; cut both tracks there and let the labels stop at the crossing.
A player whose jersey turns from ours to the other team's across a crossing is always cut.
Labels on a cut track stay only on the pieces that still look like that kid, so "unsure" shows as an
unlabeled box instead of a wrong number.
"""
import json
import pathlib

import modal

app = modal.App("pucktracker-untangle")
vol = modal.Volume.from_name("pucktracker-videos")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "git")
    .pip_install("torch==2.4.1", "torchvision==0.19.1", "opencv-python-headless", "pandas", "numpy<2")
    .add_local_dir("games", "/root/games")
)
DATA = pathlib.Path("/data")
H, W = 224, 112
MODEL = "2026-09-19"          # look-alike model trained on both games, with a "not ours" class
OVERLAP = 0.3                 # share of the smaller box covered by the other: the players are crossing
WINDOW = 1.5                  # seconds before/after a crossing to take clean looks
LOOKS = 3                     # looks per player per side
PIECE_BASE = 3_000_000


def game_dir(game_id):
    return DATA / "games" / game_id


def overlaps(tr, step):
    """Rows (frame, a, b, ov) for box pairs in the same frame covering each other by more than OVERLAP."""
    import numpy as np
    d = tr[tr.frame % step == 0][["frame", "track_id", "x1", "y1", "x2", "y2"]]
    m = d.merge(d, on="frame", suffixes=("_a", "_b"))
    m = m[m.track_id_a < m.track_id_b]
    iw = np.clip(np.minimum(m.x2_a, m.x2_b) - np.maximum(m.x1_a, m.x1_b), 0, None)
    ih = np.clip(np.minimum(m.y2_a, m.y2_b) - np.maximum(m.y1_a, m.y1_b), 0, None)
    area_a = (m.x2_a - m.x1_a) * (m.y2_a - m.y1_a)
    area_b = (m.x2_b - m.x1_b) * (m.y2_b - m.y1_b)
    m["ov"] = iw * ih / np.minimum(area_a, area_b)
    return m[m.ov > OVERLAP][["frame", "track_id_a", "track_id_b", "ov"]]


@app.function(image=image, gpu="L4", volumes={str(DATA): vol}, cpu=8, memory=49152, timeout=3 * 60 * 60)
def analyze(game_id: str, part: int) -> dict:
    """Find crossings in one file and score each player's look before vs after."""
    import cv2
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn

    game = json.loads((game_dir(game_id) / "game.json").read_text())
    p = game["parts"][part]
    stem = pathlib.Path(p["file"]).stem
    cap = cv2.VideoCapture(str(DATA / "videos" / p["file"]))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    tr = pd.read_csv(DATA / "out" / stem / "tracks.csv")
    tr = tr[(tr.frame >= p["start"] * fps) & (tr.frame < p["end"] * fps)]
    step = 2
    ov = overlaps(tr, step)
    busy = set(zip(ov.frame, ov.track_id_a)) | set(zip(ov.frame, ov.track_id_b))
    span = tr.groupby("track_id").frame.agg(["min", "max"])

    # group overlap frames into crossing events per pair
    events = []
    for (a, b), g in ov.groupby(["track_id_a", "track_id_b"]):
        fr = g.frame.values
        s = fr[0]
        for k in range(1, len(fr) + 1):
            if k == len(fr) or fr[k] - fr[k - 1] > fps * 0.5:
                events.append((int(a), int(b), int(s), int(fr[k - 1])))
                if k < len(fr):
                    s = fr[k]

    # clean looks before/after for each player in each event
    rows = {t: g for t, g in tr.groupby("track_id")}
    win = int(WINDOW * fps)
    want, looks = {}, {}
    for e, (a, b, f0, f1) in enumerate(events):
        for t in (a, b):
            g = rows[t]
            for side, lo, hi in (("pre", f0 - win, f0 - 2), ("post", f1 + 2, f1 + win)):
                c = g[(g.frame >= lo) & (g.frame <= hi) & ((g.y2 - g.y1) >= 24)]
                c = c[[(f, t) not in busy and (f - f % step, t) not in busy for f in c.frame]]
                if len(c) == 0:
                    continue
                c = c.iloc[np.linspace(0, len(c) - 1, min(LOOKS, len(c))).astype(int)]
                for r in c.itertuples():
                    want.setdefault(int(r.frame), []).append((e, int(t), side, r.x1, r.y1, r.x2, r.y2))
    keys, ims = [], []
    frames = sorted(want)
    if frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
        f = frames[0]
        for target in frames:
            while f < target:
                cap.grab(); f += 1
            ok, im = cap.read(); f += 1
            if not ok:
                break
            for e, t, side, x1, y1, x2, y2 in want[target]:
                px, py = 0.1 * (x2 - x1), 0.05 * (y2 - y1)
                x0, y0 = max(0, int(x1 - px)), max(0, int(y1 - py))
                c = im[y0:int(y2 + py), x0:int(x2 + px)]
                if c.size and c.shape[0] >= 16 and c.shape[1] >= 8:
                    keys.append((e, t, side))
                    ims.append(cv2.cvtColor(cv2.resize(c, (W, H), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB))

    # embed with the look-alike model (fine-tuned DINOv2 + head with a "not ours" class)
    ck = torch.load(game_dir(MODEL) / "reid_model.pt", map_location="cuda")
    m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").cuda()
    for blk, sd in zip(m.blocks[-4:], ck["blocks"]):
        blk.load_state_dict(sd)
    m.norm.load_state_dict(ck["norm"])
    head = nn.Linear(768, len(ck["classes"])).cuda()
    head.load_state_dict(ck["head"])
    m.eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    F, P = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for k in range(0, len(ims), 256):
            x = torch.from_numpy(np.stack(ims[k:k + 256])).cuda().permute(0, 3, 1, 2).float() / 255
            f = m((x - mean) / std).float()
            F.append(torch.nn.functional.normalize(f, dim=1).cpu())
            P.append(torch.softmax(head(f), 1).cpu())
    F = torch.cat(F).numpy() if F else np.zeros((0, 768))
    P = torch.cat(P).numpy() if P else np.zeros((0, len(ck["classes"])))
    groups = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    xi = ck["classes"].index("x") if "x" in ck["classes"] else None

    def look(e, t, side):
        ix = groups.get((e, t, side))
        if not ix:
            return None
        v = F[ix].mean(0)
        return v / (np.linalg.norm(v) + 1e-9), P[ix].mean(0)

    out = []
    for e, (a, b, f0, f1) in enumerate(events):
        L = {(t, s): look(e, t, s) for t in (a, b) for s in ("pre", "post")}
        rec = {"a": a, "b": b, "f0": f0, "f1": f1, "cut": (f0 + f1) // 2}
        for t, n in ((a, "a"), (b, "b")):
            rec[f"{n}_pre"] = L[(t, "pre")] is not None
            rec[f"{n}_post"] = L[(t, "post")] is not None
            if xi is not None:
                for s in ("pre", "post"):
                    if L[(t, s)] is not None:
                        rec[f"{n}_{s}_x"] = round(float(L[(t, s)][1][xi]), 3)
        def sim(t1, s1, t2, s2):
            u, v = L[(t1, s1)], L[(t2, s2)]
            return None if u is None or v is None else round(float(u[0] @ v[0]), 4)
        rec["aa"], rec["bb"] = sim(a, "pre", a, "post"), sim(b, "pre", b, "post")
        rec["ab"], rec["ba"] = sim(a, "pre", b, "post"), sim(b, "pre", a, "post")
        out.append(rec)
    # per-piece class probabilities for every look (used to decide which pieces keep a tag)
    piece_looks = {}
    for (e, t, s), ix in groups.items():
        f = events[e][2] if s == "pre" else events[e][3]
        piece_looks.setdefault(int(t), []).append([int(f), s, [round(float(v), 3) for v in P[ix].mean(0)]])
    res = {"part": part, "fps": fps, "events": out, "looks": piece_looks, "classes": ck["classes"],
           "span": {int(t): [int(r["min"]), int(r["max"])] for t, r in span.iterrows()}}
    (game_dir(game_id) / "untangle").mkdir(exist_ok=True)
    (game_dir(game_id) / "untangle" / f"events_{part}.json").write_text(json.dumps(res))
    vol.commit()
    return {"part": part, "events": len(out), "looks": len(ims)}


def decide(ev, margin, xflip=0.5):
    """Return ("keep" | "swap" | "cut_a" | "cut_b" | "cut_both", relinks) for one crossing."""
    a_flip = "a_pre_x" in ev and "a_post_x" in ev and abs(ev["a_pre_x"] - ev["a_post_x"]) > xflip
    b_flip = "b_pre_x" in ev and "b_post_x" in ev and abs(ev["b_pre_x"] - ev["b_post_x"]) > xflip
    aa, bb, ab, ba = ev["aa"], ev["bb"], ev["ab"], ev["ba"]
    if None not in (aa, bb, ab, ba):
        keep, swap = aa + bb, ab + ba
        if swap - keep > 2 * margin:
            return "swap"
        if keep - swap > 2 * margin and not (a_flip or b_flip):
            return "keep"
        return "cut_both"
    # only one player seen on both sides: does he still look like himself rather than the other one?
    out = []
    if aa is not None:
        other = ba if ba is not None else None
        if a_flip or (other is not None and aa - other < margin):
            out.append("a")
    if bb is not None:
        other = ab if ab is not None else None
        if b_flip or (other is not None and bb - other < margin):
            out.append("b")
    if not out:
        return "keep"
    return "cut_both" if len(out) == 2 else f"cut_{out[0]}"


def build(game_id, margin, tags, known, refs):
    """Cut and relink tracks for every part; returns per part (piece map frames->new id), new labels."""
    import bisect
    import numpy as np
    import pandas as pd
    game = json.loads((game_dir(game_id) / "game.json").read_text())
    parts_out = []
    new_tags, new_known, new_refs, stats = {}, {}, {}, {"keep": 0, "swap": 0, "cut": 0, "events": 0}
    for i, p in enumerate(game["parts"]):
        ev = json.loads((game_dir(game_id) / "untangle" / f"events_{i}.json").read_text())
        classes = ev["classes"]
        cuts, links = {}, []
        for e in ev["events"]:
            stats["events"] += 1
            d = decide(e, margin)
            a, b, c = e["a"], e["b"], e["cut"]
            if d == "keep":
                stats["keep"] += 1
                continue
            if d == "swap":
                stats["swap"] += 1
                cuts.setdefault(a, set()).add(c); cuts.setdefault(b, set()).add(c)
                links.append(((a, c, "pre"), (b, c, "post")))
                links.append(((b, c, "pre"), (a, c, "post")))
                continue
            stats["cut"] += 1
            if d in ("cut_both", "cut_a"):
                cuts.setdefault(a, set()).add(c)
            if d in ("cut_both", "cut_b"):
                cuts.setdefault(b, set()).add(c)
        cuts = {t: sorted(v) for t, v in cuts.items()}
        span = {int(k): v for k, v in ev["span"].items()}

        def seg(t, c, side):
            cs = cuts.get(t, [])
            return (t, bisect.bisect_right(cs, c - 1 if side == "pre" else c))
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for u, v in links:
            su, sv = seg(*u), seg(*v)
            # only relink pieces that exist (the track runs on that side of the cut)
            parent.setdefault(su, su); parent.setdefault(sv, sv)
            parent[find(su)] = find(sv)
        # pieces and their frame ranges
        pieces = {}
        for t, cs in cuts.items():
            lo, hi = span[t]
            bounds = [lo] + cs + [hi + 1]
            for k in range(len(bounds) - 1):
                if bounds[k] < bounds[k + 1]:
                    pieces[(t, k)] = (bounds[k], bounds[k + 1] - 1)
        groups = {}
        for pc in pieces:
            groups.setdefault(find(pc), []).append(pc)
        # a group whose pieces overlap in time can't be one player: break it up
        final = []
        for g in groups.values():
            g = sorted(g, key=lambda x: pieces[x][0])
            if any(pieces[g[k]][0] <= pieces[g[k - 1]][1] for k in range(1, len(g))):
                final += [[x] for x in g]
            else:
                final.append(g)
        nid = PIECE_BASE
        piece_id = {}
        for g in final:
            for pc in g:
                piece_id[pc] = nid
            nid += 1
        # labels: a piece keeps its track's tag unless the model is sure it's someone else
        looks = {int(k): v for k, v in ev["looks"].items()}

        def piece_probs(t, rng):
            ps = [np.array(pr) for f, s, pr in looks.get(t, []) if rng[0] <= f <= rng[1]]
            return np.mean(ps, 0) if ps else None

        def keeps(lbl, pr):
            if pr is None or lbl == "?":
                return True
            top = classes[int(pr.argmax())]
            if lbl == "x":
                return not (top != "x" and pr.max() > 0.6 and pr[classes.index("x")] < 0.1)
            if lbl not in classes:
                return True
            if top == "x" and pr.max() > 0.5:
                return False
            return not (top != lbl and pr.max() > 0.6 and pr[classes.index(lbl)] < 0.1)
        # A tag was given to a whole track, but after cutting we can't tell which piece the parent clicked.
        # It goes to the track's longest piece (and whatever that piece is relinked with); the other
        # pieces start unlabeled rather than inherit a number that may belong to the player crossed.
        longest = {}
        for (t, k), (a, b) in pieces.items():
            if t not in longest or b - a > pieces[(t, longest[t])][1] - pieces[(t, longest[t])][0]:
                longest[t] = k
        for g in final:
            gid = piece_id[g[0]]
            votes = {}
            for t, k in g:
                rng = pieces[(t, k)]
                key = f"{i}:{t}"
                w = rng[1] - rng[0] + 1
                lbl = tags.get(key) if longest[t] == k else None
                if lbl is not None and keeps(lbl, piece_probs(t, rng)):
                    votes[lbl] = votes.get(lbl, 0) + w
                if key in known and known[key][1] == "a" and longest[t] == k and keeps(known[key][0], piece_probs(t, rng)):
                    new_known.setdefault(f"{i}:{gid}", known[key])
                if key in refs:
                    new_refs[f"{i}:{gid}"] = refs[key]
            if votes:
                best = max(votes, key=votes.get)
                if votes[best] >= 0.8 * sum(votes.values()):   # tags from two different kids: leave it open
                    new_tags[f"{i}:{gid}"] = best
        for t, l in tags.items():
            pt, tid = t.split(":")
            if int(pt) == i and int(tid) not in cuts:
                new_tags[t] = l
        for t, v in known.items():
            pt, tid = t.split(":")
            if int(pt) == i and int(tid) not in cuts:
                new_known[t] = v
        for t, v in refs.items():
            pt, tid = t.split(":")
            if int(pt) == i and int(tid) not in cuts:
                new_refs[t] = v
        parts_out.append({"cuts": cuts, "piece_id": {f"{t}:{k}": v for (t, k), v in piece_id.items()},
                          "pieces": {f"{t}:{k}": v for (t, k), v in pieces.items()}, "fps": ev["fps"]})
    return parts_out, new_tags, new_known, new_refs, stats


def relabel_tracks(tr, cuts, piece_id):
    import bisect
    import numpy as np
    tr = tr.copy()
    ids = tr.track_id.values.copy()
    for t, cs in cuts.items():
        mk = ids == int(t)
        fr = tr.frame.values[mk]
        k = np.array([bisect.bisect_right(cs, f) for f in fr])
        ids[mk] = [piece_id[f"{t}:{kk}"] for kk in k]
    tr["track_id"] = ids
    return tr


def score(game_id, tags, known, use_tracks):
    """#21 (and any marked kid) from tags + sure reads, plus seconds where one number is in two places."""
    import numpy as np
    game = json.loads((game_dir(game_id) / "game.json").read_text())
    marks = {}
    md = game_dir(game_id) / "marks"
    if md.exists():
        for f in md.glob("*.json"):
            s = [[x["on"], x["off"]] for x in json.loads(f.read_text())["shifts"] if x.get("off") is not None]
            if sum(b - a for a, b in s) >= 300:
                marks[f.stem] = s
    vis, two = {}, 0.0
    for i, p in enumerate(game["parts"]):
        tr, fps = use_tracks[i]
        lab = {}
        for k, v in tags.items():
            if k.startswith(f"{i}:") and v.isdigit() and v != "10":
                lab[int(k.split(":")[1])] = v
        for k, v in known.items():
            tid = int(k.split(":")[1])
            if k.startswith(f"{i}:") and v[1] == "a" and v[0] != "10" and tid not in lab:
                lab[tid] = v[0]
        t = tr[tr.track_id.isin(lab)].copy()
        t["num"] = t.track_id.map(lab)
        t = t[t.frame % 6 == 0]
        t["cx"] = (t.x1 + t.x2) / 2
        t["w"] = t.x2 - t.x1
        g = t.groupby(["frame", "num"])
        spread = g.cx.max() - g.cx.min()
        two += float(((spread > 1.5 * g.w.mean()) & (g.size() > 1)).sum()) * 6 / fps
        for n, d in t.groupby("num"):
            ts = np.sort(game["offsets"][i] + d.frame.unique() / fps - p["start"])
            iv = []
            for x in ts:
                if iv and x - iv[-1][1] <= 2:
                    iv[-1][1] = x
                else:
                    iv.append([x, x])
            vis.setdefault(n, []).extend(iv)
    res = {"two_places_s": round(two)}
    for n, mk in marks.items():
        iv = sorted(vis.get(n, []))
        u = []
        for a, b in iv:
            if u and a - u[-1][1] <= 40:
                u[-1][1] = max(u[-1][1], b)
            else:
                u.append([a, b])
        u = [x for x in u if x[1] - x[0] >= 3]
        ov = sum(max(0, min(b, d) - max(a, c)) for a, b in u for c, d in mk)
        tot = sum(b - a for a, b in u)
        res[n] = {"right": round(ov), "wrong": round(tot - ov), "missed": round(sum(b - a for a, b in mk) - ov)}
    return res


@app.function(image=image, volumes={str(DATA): vol}, cpu=8, memory=65536, timeout=2 * 60 * 60)
def plan(game_id: str, margins: list, apply_margin: float = -1.0) -> dict:
    import shutil

    import pandas as pd
    vol.reload()
    g = game_dir(game_id)
    game = json.loads((g / "game.json").read_text())
    tags = json.loads((g / "tags.json").read_text()) if (g / "tags.json").exists() else {}
    known = json.loads(pathlib.Path(f"/root/games/{game_id}.drafts.json").read_text()).get("known", {})
    refs = json.loads((g / "refs.json").read_text()) if (g / "refs.json").exists() else {}
    base = []
    for p in game["parts"]:
        stem = pathlib.Path(p["file"]).stem
        tr = pd.read_csv(DATA / "out" / stem / "tracks.csv")
        ev = json.loads((g / "untangle" / f"events_{len(base)}.json").read_text())
        base.append((tr, ev["fps"]))
    out = {"before": score(game_id, tags, known, base)}
    for mg in margins:
        parts, nt, nk, nr, st = build(game_id, mg, tags, known, refs)
        use = [(relabel_tracks(tr, pt["cuts"], pt["piece_id"]), fps) for (tr, fps), pt in zip(base, parts)]
        out[f"margin {mg}"] = {**st, **score(game_id, nt, nk, use), "tags": len(nt)}
        if mg == apply_margin:
            keep = g / "untangle" / "before"
            keep.mkdir(exist_ok=True)
            for f in ("tags.json", "refs.json"):
                if (g / f).exists() and not (keep / f).exists():
                    shutil.copy(g / f, keep / f)
            (keep / "known.json").write_text(json.dumps(known))
            for (tr, fps), pt, p in zip(use, parts, game["parts"]):
                stem = pathlib.Path(p["file"]).stem
                o = DATA / "out" / stem
                if not (o / "tracks_before_untangle.csv").exists():
                    shutil.copy(o / "tracks.csv", o / "tracks_before_untangle.csv")
                    shutil.copy(o / "teams.csv", o / "teams_before_untangle.csv")
                tr.to_csv(o / "tracks.csv", index=False)
                # pieces keep the team of the track they came from
                teams = pd.read_csv(o / "teams_before_untangle.csv")
                tm = teams.set_index("track_id").team
                extra = [{"track_id": v, "team": tm.get(int(k.split(":")[0]), "unknown")} for k, v in pt["piece_id"].items()]
                pd.concat([teams, pd.DataFrame(extra).drop_duplicates("track_id")]).to_csv(o / "teams.csv", index=False)
            (g / "refs.json").write_text(json.dumps(nr))
            (g / "untangle" / "tags.json").write_text(json.dumps(nt))
            (g / "untangle" / "known.json").write_text(json.dumps(nk))
            out["applied"] = mg
    vol.commit()
    return out


@app.local_entrypoint()
def main(game_id: str = "2026-09-26", margins: str = "0.02,0.05,0.1", apply: bool = False, margin: float = 0.05,
         skip_analyze: bool = False):
    if not skip_analyze:
        n = len(json.load(open(f"games/{game_id}.json"))["parts"])
        for r in analyze.starmap([(game_id, k) for k in range(n)]):
            print("ANALYZE", r, flush=True)
    ms = [float(x) for x in margins.split(",")]
    if apply and margin not in ms:
        ms.append(margin)
    r = plan.remote(game_id, ms, margin if apply else -1.0)
    for k, v in r.items():
        print("PLAN", k, json.dumps(v), flush=True)
