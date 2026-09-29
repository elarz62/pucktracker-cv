"""Stage 2: drop spectators, label each track as white team, blue team, or referee.

Usage: python -m pucktracker.classify <video.mp4> <tracks.csv> <out.csv> [contact_sheet.jpg]
"""
import sys

import cv2
import numpy as np
import pandas as pd


def on_ice(frame, x1, y1, x2, y2):
    """True when the pixels just below the feet look like ice (bright, unsaturated)."""
    h = y2 - y1
    ya, yb = min(y2 + 2, frame.shape[0] - 1), min(y2 + 2 + max(4, h // 8), frame.shape[0])
    patch = frame[ya:yb, max(x1, 0):x2]
    if patch.size == 0:
        return y2 > frame.shape[0] * 0.5
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    return hsv[..., 2].mean() > 165 and hsv[..., 1].mean() < 60


def jersey_features(frame, x1, y1, x2, y2):
    """Fractions of white, blue/red, and dark pixels on the torso."""
    h, w = y2 - y1, x2 - x1
    torso = frame[y1 + int(h * 0.2):y1 + int(h * 0.55), x1 + int(w * 0.2):x2 - int(w * 0.2)]
    if torso.size == 0:
        return None
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(int)
    hue, sat, val = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    white = ((sat < 50) & (val > 150)).mean()
    colored = ((sat > 80) & (val > 50) & (((hue > 95) & (hue < 135)) | (hue < 10) | (hue > 165))).mean()
    dark = (val < 70).mean()
    return white, colored, dark


def label(white, colored, dark):
    if colored > 0.25:
        return "blue"
    if dark > 0.25 and white > 0.2 and colored < 0.1:
        return "ref"
    if white > 0.35:
        return "white"
    return "unknown"


def main(video, tracks_csv, out_csv, sheet=None):
    df = pd.read_csv(tracks_csv)
    cap = cv2.VideoCapture(video)
    frames = sorted(df.frame.unique())
    feats = {}
    ice = {}
    best_crop = {}
    # Sample every 6th tracked frame to keep it quick; decode sequentially (seeking is slow).
    wanted = set(frames[::6])
    last = max(wanted)
    by_frame = dict(tuple(df.groupby("frame")))
    for f in range(last + 1):
        if f not in wanted:
            cap.grab()
            continue
        ok, img = cap.read()
        if not ok:
            break
        for r in by_frame[f].itertuples():
            x1, y1, x2, y2 = max(r.x1, 0), max(r.y1, 0), r.x2, r.y2
            ice.setdefault(r.track_id, []).append(on_ice(img, x1, y1, x2, y2))
            ft = jersey_features(img, x1, y1, x2, y2)
            if ft:
                feats.setdefault(r.track_id, []).append(ft)
            if (y2 - y1) > best_crop.get(r.track_id, (0, None))[0]:
                best_crop[r.track_id] = (y2 - y1, img[y1:y2, x1:x2].copy())
    counts = df.track_id.value_counts()
    tracks = []
    for tid, vals in feats.items():
        w, c, d = np.median(np.array(vals), axis=0)
        tracks.append(dict(track_id=tid, n=int(counts[tid]),
                           ice=float(np.mean(ice[tid])), white=w, colored=c, dark=d,
                           team=label(w, c, d)))
    t = pd.DataFrame(tracks)
    t.loc[t.ice < 0.5, "team"] = "stands"
    t.to_csv(out_csv, index=False)
    print(t.team.value_counts().to_string())
    if sheet:
        keep = t[t.team == "white"].sort_values("n", ascending=False)
        tiles = []
        for r in keep.itertuples():
            crop = best_crop[r.track_id][1]
            crop = cv2.resize(crop, (100, 200))
            cv2.putText(crop, str(r.track_id), (3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            tiles.append(crop)
        rows = [np.hstack(tiles[i:i + 10] + [np.zeros((200, 100, 3), np.uint8)] * (10 - len(tiles[i:i + 10])))
                for i in range(0, len(tiles), 10)]
        if rows:
            cv2.imwrite(sheet, np.vstack(rows))


if __name__ == "__main__":
    main(*sys.argv[1:])
