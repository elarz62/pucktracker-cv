"""Stage 3: follow one tagged player across track breaks.

Start from a seed track (the one the parent clicks), then chain tracks forward and
backward: same team, starts soon after the previous one ends, and begins near where
the previous one left off after correcting for camera pan.

Usage: python -m pucktracker.follow <tracks.csv> <teams.csv> <seed_track_id> <out.csv>
"""
import sys

import pandas as pd

MAX_GAP = 48      # frames (2 s at 24 fps)
MAX_OVERLAP = 6   # frames the tracker may briefly double-count


def summarize(df):
    df = df.assign(fx=(df.x1 + df.x2) / 2 - df.cam_dx, fy=df.y2, h=df.y2 - df.y1)
    g = df.sort_values("frame").groupby("track_id")
    first, last = g.first(), g.last()
    return pd.DataFrame({
        "f0": first.frame, "f1": last.frame,
        "x0": first.fx, "y0": first.fy, "h0": first.h,
        "x1": last.fx, "y1": last.fy, "h1": last.h,
    })


def link_score(a_end, b_start):
    """Lower is better; None when the two ends can't be the same player."""
    (fa, xa, ya, ha), (fb, xb, yb, hb) = a_end, b_start
    gap = fb - fa
    if gap < -MAX_OVERLAP or gap > MAX_GAP:
        return None
    if not 0.55 < hb / ha < 1.8:
        return None
    radius = ha * (0.6 + max(gap, 0) / 12)
    dist = ((xa - xb) ** 2 + (ya - yb) ** 2) ** 0.5
    return dist / radius if dist < radius else None


def follow(tracks_csv, teams_csv, seed):
    df = pd.read_csv(tracks_csv)
    teams = pd.read_csv(teams_csv).set_index("track_id").team
    s = summarize(df)
    s["team"] = teams
    team = s.loc[seed, "team"]
    pool = s[(s.team == team)]
    chain = [seed]
    # Forward
    cur = seed
    while True:
        a = pool.loc[cur]
        cands = []
        for tid, b in pool.iterrows():
            if tid in chain or b.f0 <= a.f0:
                continue
            sc = link_score((a.f1, a.x1, a.y1, a.h1), (b.f0, b.x0, b.y0, b.h0))
            if sc is not None:
                cands.append((sc + max(b.f0 - a.f1, 0) / MAX_GAP, tid))
        if not cands:
            break
        cur = min(cands)[1]
        chain.append(cur)
    # Backward
    cur = seed
    while True:
        a = pool.loc[cur]
        cands = []
        for tid, b in pool.iterrows():
            if tid in chain or b.f1 >= a.f1:
                continue
            sc = link_score((b.f1, b.x1, b.y1, b.h1), (a.f0, a.x0, a.y0, a.h0))
            if sc is not None:
                cands.append((sc + max(a.f0 - b.f1, 0) / MAX_GAP, tid))
        if not cands:
            break
        cur = min(cands)[1]
        chain.insert(0, cur)
    out = df[df.track_id.isin(chain)].sort_values(["frame", "conf"]).drop_duplicates("frame", keep="last")
    return chain, pool.loc[chain, ["f0", "f1"]], out


if __name__ == "__main__":
    chain, spans, out = follow(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    print(spans.to_string())
    out.to_csv(sys.argv[4], index=False)
    print(f"covered {len(out)} frames")
