"""Stage 5: find one player's shifts from tracks + jersey reads.

1. Anchor tracks: this team's tracks where OCR read the target number.
2. Extend each anchor forward/backward through track breaks (follow.py rules),
   never through a track that clearly reads as a different number.
3. Merge presence into shifts, bridging gaps up to MERGE_GAP seconds.

Usage: python -m pucktracker.shifts <tracks.csv> <teams.csv> <jersey.csv> <number> <fps> <out_prefix>
"""
import sys

import pandas as pd

from pucktracker.follow import MAX_GAP, link_score, summarize

MERGE_GAP_S = 20
MIN_SHIFT_S = 8


def is_target(row, number):
    counts = dict(kv.split(":") for kv in str(row["all"]).split() if ":" in kv)
    hits = int(counts.get(number, 0))
    total = sum(int(v) for v in counts.values())
    return hits >= 2 or (hits >= 1 and hits * 2 >= total)


def other_number(row, number):
    return row.votes >= 2 and str(row.number) not in ("", "nan", number)


def extend(seed, pool, exclude):
    chain = [seed]
    for direction in (1, -1):
        cur = seed
        while True:
            a = pool.loc[cur]
            best = None
            for tid, b in pool.iterrows():
                if tid in chain or tid in exclude:
                    continue
                if direction == 1 and b.f0 > a.f0:
                    sc = link_score((a.f1, a.x1, a.y1, a.h1), (b.f0, b.x0, b.y0, b.h0))
                    gap = b.f0 - a.f1
                elif direction == -1 and b.f1 < a.f1:
                    sc = link_score((b.f1, b.x1, b.y1, b.h1), (a.f0, a.x0, a.y0, a.h0))
                    gap = a.f0 - b.f1
                else:
                    continue
                if sc is not None:
                    score = sc + max(gap, 0) / MAX_GAP
                    if best is None or score < best[0]:
                        best = (score, tid)
            if best is None:
                break
            cur = best[1]
            chain.append(cur)
    return chain


def main(tracks_csv, teams_csv, jersey_csv, number, fps, prefix):
    fps = float(fps)
    df = pd.read_csv(tracks_csv)
    teams = pd.read_csv(teams_csv).set_index("track_id").team
    jer = pd.read_csv(jersey_csv, dtype={"number": str})
    s = summarize(df)
    s["team"] = teams
    anchors = [r.track_id for _, r in jer.iterrows() if is_target(r, number)]
    team = s.loc[anchors, "team"].mode()[0]
    exclude = {r.track_id for r in jer.itertuples() if other_number(r, number)}
    pool = s[s.team == team]
    mine = set()
    for a in anchors:
        mine.update(extend(a, pool, exclude))
    pres = df[df.track_id.isin(mine)].frame.drop_duplicates().sort_values().tolist()
    shifts, start, prev = [], pres[0], pres[0]
    for f in pres[1:]:
        if (f - prev) / fps > MERGE_GAP_S:
            shifts.append((start, prev))
            start = f
        prev = f
    shifts.append((start, prev))
    out = pd.DataFrame([(a / fps, b / fps) for a, b in shifts if (b - a) / fps >= MIN_SHIFT_S],
                       columns=["start_s", "end_s"])
    out["length_s"] = (out.end_s - out.start_s).round(1)
    out.index = out.index + 1
    out.to_csv(prefix + "_shifts.csv", index_label="shift")
    pd.Series(sorted(mine)).to_csv(prefix + "_tracks.csv", index=False, header=["track_id"])
    print(f"anchors {len(anchors)}, tracks {len(mine)}, presence {len(pres)} frames")
    print(out.to_string())
    print(f"total {out.length_s.sum()/60:.1f} min over {len(out)} shifts")


if __name__ == "__main__":
    main(*sys.argv[1:])
