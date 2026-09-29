"""Stage 2b: read jersey numbers per track by OCR on torso crops, vote per track.

Usage: python -m pucktracker.jersey <video.mp4> <tracks.csv> <teams.csv> <out.csv> [team=white] [min_h=80]
Output: one row per track with every number read and how often.
"""
import re
import sys
from collections import Counter, defaultdict

import cv2
import pandas as pd
from rapidocr_onnxruntime import RapidOCR


def main(video, tracks_csv, teams_csv, out_csv, team="white", min_h=80):
    min_h = int(min_h)
    ocr = RapidOCR()
    df = pd.read_csv(tracks_csv)
    teams = pd.read_csv(teams_csv).set_index("track_id").team
    df = df[df.track_id.map(teams) == team]
    df = df[(df.y2 - df.y1) >= min_h]
    df = df[df.track_id.map(df.track_id.value_counts()) >= 4]
    # At most 8 samples per track, spread out, largest boxes first within each track.
    picks = (df.assign(h=df.y2 - df.y1).sort_values("h", ascending=False)
               .groupby("track_id").head(8))
    wanted = defaultdict(list)
    for r in picks.itertuples():
        wanted[r.frame].append(r)
    reads = defaultdict(Counter)
    tried = Counter()
    cap = cv2.VideoCapture(video)
    f = 0
    last = max(wanted) if wanted else -1
    while f <= last:
        if f not in wanted:
            if not cap.grab():
                break
            f += 1
            continue
        ok, img = cap.read()
        if not ok:
            break
        for r in wanted[f]:
            h, w = r.y2 - r.y1, r.x2 - r.x1
            crop = img[max(r.y1 + int(h * 0.12), 0):r.y1 + int(h * 0.6), max(r.x1, 0):r.x2]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            res, _ = ocr(crop)
            tried[r.track_id] += 1
            for _, txt, conf in res or []:
                for m in re.findall(r"\d{1,2}", txt):
                    if conf > 0.5:
                        reads[r.track_id][m] += 1
        f += 1
    rows = []
    for tid in tried:
        c = reads[tid]
        top, n = c.most_common(1)[0] if c else ("", 0)
        rows.append(dict(track_id=tid, samples=tried[tid], number=top, votes=n,
                         all=" ".join(f"{k}:{v}" for k, v in c.most_common())))
    pd.DataFrame(rows).sort_values("votes", ascending=False).to_csv(out_csv, index=False)


if __name__ == "__main__":
    main(*sys.argv[1:])
