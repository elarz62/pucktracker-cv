# pucktracker-cv

Computer-vision pipeline for PuckTracker: finds one player in LiveBarn rink footage, detects his shifts, and cuts clips. The web app (Replit) will call this as a GPU job.

## Pipeline

| Stage | Module | Output |
|---|---|---|
| 1. Detect and track every person | `pucktracker/track.py` (YOLO11 + ByteTrack/BoT-SORT) | `tracks.csv` |
| 2. Drop spectators, split teams, find refs | `pucktracker/classify.py` (jersey color) | `teams.csv` |
| 3. Read jersey numbers per track | `pucktracker/jersey.py` (OCR, vote per track) | `jersey.csv` |
| 4. Chain one player's tracks and build shifts | `pucktracker/follow.py`, `pucktracker/shifts.py` | `player_shifts.csv` |
| 5. Render a highlighted clip | `pucktracker/render.py` | `.mp4` |

## Run on a GPU (Modal)

```bash
pip install -r requirements.txt
modal token set --token-id $MODAL_TOKEN_ID --token-secret $MODAL_TOKEN_SECRET
modal run modal_app.py --video segment.mp4 --number 21 --team white
```

## Run locally (CPU, slow)

```bash
python -m pucktracker.track segment.mp4 tracks.csv 6 configs/hockey_bytetrack.yaml
python -m pucktracker.classify segment.mp4 tracks.csv teams.csv
python -m pucktracker.jersey segment.mp4 tracks.csv teams.csv jersey.csv white 70
python -m pucktracker.shifts tracks.csv teams.csv jersey.csv 21 24 player
```

## Status (2026-09-29)

First CPU run on a full 30 minute segment (4 fps) found the target player reliably only 3 times and 2 shifts, versus 14 in LiveBarn's report. Tracks fragment (median 3.5 s) and general OCR misreads "21" as "2". Next: full frame rate on GPU, appearance re-ID for stitching, a jersey-number model trained on hockey crops, and a parent review step.

## Notes

- Footage is minors. Keep videos and outputs private; LiveBarn content is licensed for personal, non-commercial use.
- Ultralytics YOLO is AGPL-3.0. Fine for personal use; revisit before any commercial use.
- Videos and model weights are git-ignored.
