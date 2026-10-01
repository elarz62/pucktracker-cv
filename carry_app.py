"""Move a game's labels from the original tracks onto the hold-tracker run, and score both against marks.

    modal run carry_app.py --game-id 2026-09-26            # dry run: writes /data/games/<id>/v2/, changes nothing live
    modal run carry_app.py --game-id 2026-09-26 --apply    # swap the new tracks and labels in

Each new box takes the label of the old tagged box it overlaps (IoU > 0.5) in the same frame. A new track
whose boxes carry two different kids is cut where the label changes, so holding players longer never
merges two tagged kids. Number reads ("known") and referee scores are carried the same way.
"""
import json
import pathlib

import modal

app = modal.App("pucktracker-carry")
vol = modal.Volume.from_name("pucktracker-videos")
image = modal.Image.debian_slim(python_version="3.11").pip_install("pandas", "numpy").add_local_dir("games", "/root/games")
DATA = pathlib.Path("/data")
SPLIT_BASE = 1_000_000   # ids for cut pieces of a new track


def iou(a, b):
    import numpy as np
    x1 = np.maximum(a[:, 0], b[:, 0]); y1 = np.maximum(a[:, 1], b[:, 1])
    x2 = np.minimum(a[:, 2], b[:, 2]); y2 = np.minimum(a[:, 3], b[:, 3])
    i = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return i / ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - i + 1e-9)


def carry_part(old, new, labels):
    """old/new: tracks frames. labels: {old track id: label}. Returns (new tracks with cuts, {new id: label},
    {new id: old id it overlaps most})."""
    import numpy as np
    m = old.merge(new, on="frame", suffixes=("_o", ""))
    m["iou"] = iou(m[["x1_o", "y1_o", "x2_o", "y2_o"]].values, m[["x1", "y1", "x2", "y2"]].values)
    m = m[m.iou > 0.5].sort_values("iou", ascending=False)
    m = m.drop_duplicates(["frame", "track_id"]).drop_duplicates(["frame", "track_id_o"])   # one to one per frame
    best_old = m.groupby("track_id").track_id_o.agg(lambda s: s.value_counts().idxmax())
    m["lab"] = m.track_id_o.map(labels)
    lab = m.dropna(subset=["lab"])[["frame", "track_id", "lab"]]
    new = new.copy()
    new["piece"] = new.track_id
    new["orig"] = new.track_id
    out_labels, out_best = {}, {}
    next_id = SPLIT_BASE
    lab_by_track = {t: g.sort_values("frame") for t, g in lab.groupby("track_id")}
    for t, g in new.groupby("track_id"):
        runs = []
        if t in lab_by_track:
            for f, l in zip(lab_by_track[t].frame.values, lab_by_track[t].lab.values):
                if runs and runs[-1][2] == l:
                    runs[-1][1] = f
                else:
                    runs.append([f, f, l])
        # short label blips (< 1 s) inside a longer run are box swaps at a crossing; drop them
        runs = [r for r in runs if r[1] - r[0] >= 24 or len(runs) == 1] or runs
        merged = []
        for r in runs:
            if merged and merged[-1][2] == r[2]:
                merged[-1][1] = r[1]
            else:
                merged.append(list(r))
        if len(merged) <= 1:
            if merged:
                out_labels[int(t)] = merged[0][2]
            if t in best_old.index:
                out_best[int(t)] = int(best_old[t])
            continue
        # cut halfway between runs of different kids
        cuts = [(a[1] + b[0]) // 2 for a, b in zip(merged, merged[1:])]
        frames = g.frame.values
        piece = np.searchsorted(cuts, frames, side="right")
        ids = []
        for k in range(len(merged)):
            ids.append(next_id); next_id += 1
            out_labels[ids[-1]] = merged[k][2]
        new.loc[g.index, "piece"] = [ids[k] for k in piece]
        sub = m[m.track_id == t]
        for k, pid in enumerate(ids):
            lo = cuts[k - 1] if k else -1
            hi = cuts[k] if k < len(cuts) else 10 ** 9
            s = sub[(sub.frame > lo) & (sub.frame <= hi)]
            if len(s):
                out_best[pid] = int(s.track_id_o.value_counts().idxmax())
    new["track_id"] = new.piece
    return new.drop(columns="piece"), out_labels, out_best


def spans(tr, ids, off, start, gap=2.0):
    """Game-time intervals where any of the tracks in ids is visible."""
    fr = sorted(set(tr[tr.track_id.isin(ids)].frame.values))
    out = []
    for f in fr:
        t = off + f / 24 - start
        if out and t - out[-1][1] <= gap:
            out[-1][1] = t
        else:
            out.append([t, t])
    return out


def union(iv, gap):
    iv = sorted(iv)
    out = []
    for a, b in iv:
        if out and a - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def overlap(a, b):
    return sum(max(0, min(y, q) - max(x, p)) for x, y in a for p, q in b)


def length(a):
    return sum(y - x for x, y in a)


@app.function(image=image, volumes={str(DATA): vol}, cpu=8, memory=65536, timeout=2 * 60 * 60)
def run(game_id: str, apply: bool = False) -> dict:
    import shutil

    import pandas as pd

    vol.reload()
    g = DATA / "games" / game_id
    game = json.loads((g / "game.json").read_text())
    tags = json.loads((g / "tags.json").read_text())
    drafts = json.loads(pathlib.Path(f"/root/games/{game_id}.drafts.json").read_text())
    known = drafts.get("known", {})
    refs = json.loads((g / "refs.json").read_text()) if (g / "refs.json").exists() else {}
    marks = {p.stem: [[s["on"], s["off"]] for s in json.loads(p.read_text())["shifts"] if s.get("off") is not None]
             for p in (g / "marks").glob("*.json")}
    v2 = g / "v2"
    v2.mkdir(exist_ok=True)
    new_tags, new_known, new_refs = {}, {}, {}
    report = {"parts": []}
    cover = {"old": {}, "new": {}}
    for i, p in enumerate(game["parts"]):
        stem = pathlib.Path(p["file"]).stem
        off = game["offsets"][i]
        old = pd.read_csv(DATA / "out" / stem / "tracks.csv")
        new = pd.read_csv(DATA / "out_hold" / stem / "tracks.csv")
        lo, hi = p["start"] * 24, p["end"] * 24
        old = old[(old.frame >= lo) & (old.frame < hi)]
        new = new[(new.frame >= lo) & (new.frame < hi)]
        t_old = {int(k.split(":")[1]): v for k, v in tags.items() if k.startswith(f"{i}:")}
        new2, lab, best = carry_part(old, new, t_old)
        for t, l in lab.items():
            new_tags[f"{i}:{t}"] = l
        for t, o in best.items():
            k = f"{i}:{o}"
            if k in known and f"{i}:{t}" not in new_tags:
                new_known[f"{i}:{t}"] = known[k]
            if k in refs:
                new_refs[f"{i}:{t}"] = refs[k]
        # teams for cut pieces: same as the track they were cut from
        teams = pd.read_csv(DATA / "out_hold" / stem / "teams.csv")
        pmap = new2[new2.track_id >= SPLIT_BASE][["track_id", "orig"]].drop_duplicates("track_id")
        extra = pmap.merge(teams, left_on="orig", right_on="track_id", suffixes=("", "_t"))
        extra = extra.drop(columns=["orig", "track_id_t"])
        new2 = new2.drop(columns="orig")
        new2.to_csv(DATA / "out_hold" / stem / "tracks_carried.csv", index=False)
        pd.concat([teams, extra]).to_csv(DATA / "out_hold" / stem / "teams_carried.csv", index=False)
        report["parts"].append({"part": i, "old_tracks": int(old.track_id.nunique()), "new_tracks": int(new.track_id.nunique()),
                                "after_cuts": int(new2.track_id.nunique()), "cut_tracks": int(sum(1 for t in lab if t >= SPLIT_BASE)),
                                "tagged_old": len(t_old), "tagged_new": len(lab)})
        for name, tr, lb in (("old", old, t_old), ("new", new2, lab)):
            by = {}
            for t, l in lb.items():
                by.setdefault(l, []).append(t)
            for l, ids in by.items():
                if l.isdigit():
                    cover[name].setdefault(l, []).extend(spans(tr, ids, off, p["start"]))
    # seconds each kid is visible under a tag, and how #21 lines up with the parent's marks
    report["seconds_tagged"] = {n: {"old": round(length(union(cover["old"].get(n, []), 2))), "new": round(length(union(cover["new"].get(n, []), 2)))}
                                for n in sorted(set(cover["old"]) | set(cover["new"]), key=int)}
    report["vs_marks"] = {}
    for n, mk in marks.items():
        if length(mk) < 300:
            continue
        r = {"marked": round(length(mk))}
        for name in ("old", "new"):
            for gap in (5, 20):
                d = [x for x in union(cover[name].get(n, []), gap) if x[1] - x[0] >= 3]
                r[f"{name}_gap{gap}"] = {"draft": round(length(d)), "right": round(overlap(d, mk)), "wrong": round(length(d) - overlap(d, mk)),
                                         "missed": round(length(mk) - overlap(d, mk))}
        report["vs_marks"][n] = r
    (v2 / "tags.json").write_text(json.dumps(new_tags, separators=(",", ":")))
    (v2 / "known.json").write_text(json.dumps(new_known, separators=(",", ":")))
    (v2 / "refs.json").write_text(json.dumps(new_refs, separators=(",", ":")))
    (v2 / "report.json").write_text(json.dumps(report, indent=1))
    if apply:
        backup = g / "v1"
        backup.mkdir(exist_ok=True)
        for f in ("tags.json", "refs.json"):
            if (g / f).exists():
                shutil.copy(g / f, backup / f)
        for i, p in enumerate(game["parts"]):
            stem = pathlib.Path(p["file"]).stem
            o = DATA / "out" / stem
            keep = DATA / "out_v1" / stem
            if not keep.exists():
                shutil.copytree(o, keep)
            shutil.copy(DATA / "out_hold" / stem / "tracks_carried.csv", o / "tracks.csv")
            shutil.copy(DATA / "out_hold" / stem / "teams_carried.csv", o / "teams.csv")
        shutil.copy(v2 / "tags.json", g / "tags.json")
        shutil.copy(v2 / "refs.json", g / "refs.json")
        report["applied"] = True
    vol.commit()
    return report


@app.function(image=image, volumes={str(DATA): vol}, cpu=8, memory=32768, timeout=3600)
def drafts(game_id: str, thrs: list = (1.1, 0.7, 0.5, 0.3), gaps: list = (10, 20, 40), min_len: float = 3) -> dict:
    """Shift drafts for every kid from tags, sure number reads and the model's guesses (at or above thr),
    scored against every marked player. thr 1.1 means tags and reads only."""
    import pandas as pd

    vol.reload()
    g = DATA / "games" / game_id
    game = json.loads((g / "game.json").read_text())
    tags = json.loads((g / "tags.json").read_text())
    known = json.loads((g / "v2" / "known.json").read_text()) if (g / "v2" / "known.json").exists() else {}
    guesses = json.loads((g / "guesses.json").read_text()) if (g / "guesses.json").exists() else {}
    marks = {p.stem: [[s["on"], s["off"]] for s in json.loads(p.read_text())["shifts"] if s.get("off") is not None]
             for p in (g / "marks").glob("*.json")}
    marks = {n: m for n, m in marks.items() if length(m) >= 300}
    vis = {}   # (source, number) -> visible intervals
    for i, p in enumerate(game["parts"]):
        stem = pathlib.Path(p["file"]).stem
        tr = pd.read_csv(DATA / "out" / stem / "tracks.csv", usecols=["frame", "track_id"])
        tr = tr[(tr.frame >= p["start"] * 24) & (tr.frame < p["end"] * 24)]
        sp = {}
        for t, fr in tr.groupby("track_id").frame:
            sp[f"{i}:{t}"] = (fr.values, )
        for k, (fr,) in sp.items():
            src = None
            if tags.get(k, "").isdigit():
                src, n, c = "tag", tags[k], 1.0
            elif k in tags:
                continue
            elif k in known and known[k][1] == "a":
                src, n, c = "read", known[k][0], 1.0
            elif k in guesses:
                src, n, c = "model", guesses[k][0], guesses[k][1]
            if not src or n == "10":
                continue
            ts = sorted(set(fr))
            iv = []
            for f in ts:
                t = game["offsets"][i] + f / 24 - p["start"]
                if iv and t - iv[-1][1] <= 2:
                    iv[-1][1] = t
                else:
                    iv.append([t, t])
            vis.setdefault(n, []).extend([(a, b, c) for a, b in iv])
    table, best = [], None
    for thr in thrs:
        for gap in gaps:
            row = {"thr": thr, "gap": gap}
            for n, mk in marks.items():
                d = [x for x in union([(a, b) for a, b, c in vis.get(n, []) if c >= thr], gap) if x[1] - x[0] >= min_len]
                ov = overlap(d, mk)
                row[n] = {"draft": round(length(d)), "right": round(ov), "wrong": round(length(d) - ov), "missed": round(length(mk) - ov)}
            table.append(row)
    out = {}
    for thr in thrs:
        for gap in gaps:
            out[f"{thr}_{gap}"] = {n: [[round(a, 1), round(b, 1)] for a, b in union([(a, b) for a, b, c in v if c >= thr], gap) if b - a >= min_len]
                                   for n, v in vis.items()}
    return {"table": table, "drafts": out}


@app.local_entrypoint()
def main(game_id: str = "2026-09-26", apply: bool = False, mode: str = "carry"):
    if mode == "drafts":
        r = drafts.remote(game_id)
        for row in r["table"]:
            print("SCORE", json.dumps(row))
        pathlib.Path(f"/tmp/drafts_{game_id}.json").write_text(json.dumps(r["drafts"]))
        return
    r = run.remote(game_id, apply)
    print("REPORT", json.dumps(r))
