"""Stage 4: render a clip with the followed player highlighted, plus a check sheet.

Usage: python -m pucktracker.render <video.mp4> <tracks.csv> <teams.csv> <follow.csv> <label> <out.mp4> <sheet.jpg>
"""
import sys

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd

COLORS = {"white": (200, 200, 200), "blue": (200, 90, 30), "ref": (0, 200, 255)}
TARGET = (0, 215, 255)


def main(video, tracks_csv, teams_csv, follow_csv, label, out_mp4, sheet):
    df = pd.read_csv(tracks_csv)
    teams = pd.read_csv(teams_csv).set_index("track_id").team.to_dict()
    tgt = pd.read_csv(follow_csv).set_index("frame")
    by_frame = dict(tuple(df.groupby("frame")))
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))
    writer = imageio_ffmpeg.write_frames(out_mp4, (w, h), fps=fps, codec="libx264",
                                         pix_fmt_out="yuv420p", quality=7)
    writer.send(None)
    crops, f = [], 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        for r in by_frame.get(f, pd.DataFrame()).itertuples():
            team = teams.get(r.track_id)
            if team in COLORS:
                cv2.rectangle(img, (r.x1, r.y1), (r.x2, r.y2), COLORS[team], 1)
        if f in tgt.index:
            r = tgt.loc[f]
            x1, y1, x2, y2 = int(r.x1), int(r.y1), int(r.x2), int(r.y2)
            cx, bh = (x1 + x2) // 2, y2 - y1
            cv2.ellipse(img, (cx, y2), (int(bh * 0.45), int(bh * 0.15)), 0, 0, 360, TARGET, 3)
            cv2.putText(img, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TARGET, 2)
            if f % 24 == 0:
                c = img[max(y1, 0):y2, max(x1, 0):x2]
                if c.size:
                    crops.append(cv2.resize(c, (100, 200)))
        writer.send(np.ascontiguousarray(img[:, :, ::-1]))
        f += 1
    writer.close()
    if crops:
        rows = [np.hstack(crops[i:i + 10] + [np.zeros((200, 100, 3), np.uint8)] * (10 - len(crops[i:i + 10])))
                for i in range(0, len(crops), 10)]
        cv2.imwrite(sheet, np.vstack(rows))


if __name__ == "__main__":
    main(*sys.argv[1:])
