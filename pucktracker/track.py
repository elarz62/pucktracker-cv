"""Stage 1: detect and track every person in a video, save raw tracks to CSV.

Usage: python -m pucktracker.track <video.mp4> <out.csv> [stride]
stride=3 processes every 3rd frame (8 fps from 24 fps), about 3x faster.
"""
import csv
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO


def pan_dx(prev_gray, gray):
    """Horizontal camera pan between two frames, measured on the stands strip."""
    (dx, _), _ = cv2.phaseCorrelate(prev_gray, gray)
    return dx


def main(video, out_csv, stride=1, model_name="yolo11s.pt", imgsz=1280, tracker="bytetrack.yaml",
         start_s=0.0, end_s=None):
    """start_s/end_s limit tracking to the game (e.g. skip another team's ice time).
    Frame numbers stay absolute, so times still match the source video."""
    model = YOLO(model_name)
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    prev = None
    cum_dx = 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    frame_idx = int(start_s * fps)
    last_frame = int(end_s * fps) if end_s else total
    if frame_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    t0 = time.time()
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "track_id", "x1", "y1", "x2", "y2", "conf", "cam_dx"])
        while frame_idx < last_frame:
            if frame_idx % stride:
                if not cap.grab():
                    break
                frame_idx += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            strip = cv2.cvtColor(frame[0:180], cv2.COLOR_BGR2GRAY).astype(np.float32)
            if prev is not None:
                cum_dx += pan_dx(prev, strip)
            prev = strip
            r = model.track(frame, persist=True, classes=[0], imgsz=imgsz, conf=0.2,
                            tracker=tracker, verbose=False)[0]
            if r.boxes.id is not None:
                for (x1, y1, x2, y2), tid, conf in zip(r.boxes.xyxy.tolist(), r.boxes.id.int().tolist(),
                                                       r.boxes.conf.tolist()):
                    w.writerow([frame_idx, tid, round(x1), round(y1), round(x2), round(y2),
                                round(conf, 3), round(cum_dx, 1)])
            frame_idx += 1
            if (frame_idx - 1) % (stride * 240) == 0:
                el = time.time() - t0
                print(f"frame {frame_idx}/{total} elapsed {el/60:.1f} min "
                      f"eta {el / max(frame_idx - int(start_s * fps), 1) * (last_frame - frame_idx) / 60:.0f} min", flush=True)
    print(f"done: {frame_idx} frames in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1,
         tracker=sys.argv[4] if len(sys.argv) > 4 else "bytetrack.yaml")
