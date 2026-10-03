"""Re-track a game's files with the hold tracker (players kept through 10 s gaps) and re-sort teams.

    modal run retrack_app.py --game-id 2026-09-26

Writes /data/out_hold/<stem>/tracks.csv and teams.csv, next to the original /data/out/<stem>/ run,
so parent tags on the old tracks can be carried over by box overlap (see carry_tags in games_app.py).
"""
import json
import pathlib

import modal

app = modal.App("pucktracker-retrack")
vol = modal.Volume.from_name("pucktracker-videos")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("ultralytics==8.4.164", "opencv-python-headless", "pandas", "lap")
    .add_local_dir("configs", "/root/configs")
    .add_local_python_source("pucktracker")
)
DATA = pathlib.Path("/data")


@app.function(image=image, gpu="L4", volumes={str(DATA): vol}, cpu=4, memory=16384, timeout=4 * 60 * 60)
def retrack(file: str, start_s: float, end_s: float, tracker: str = "hockey_bytetrack_hold.yaml", outdir: str = "out_hold") -> dict:
    import time

    from pucktracker import classify, track

    out = DATA / outdir / pathlib.Path(file).stem
    out.mkdir(parents=True, exist_ok=True)
    video = str(DATA / "videos" / file)
    t = time.time()
    track.main(video, str(out / "tracks.csv"), 1, tracker=f"/root/configs/{tracker}", start_s=start_s, end_s=end_s)
    vol.commit()
    t1 = time.time()
    classify.main(video, str(out / "tracks.csv"), str(out / "teams.csv"))
    vol.commit()
    return {"file": file, "track_min": round((t1 - t) / 60, 1), "teams_min": round((time.time() - t1) / 60, 1)}


@app.local_entrypoint()
def main(game_id: str = "2026-09-26", pad: float = 15, outdir: str = "out_hold"):
    game = json.load(open(f"games/{game_id}.json"))
    jobs = [(p["file"], max(0.0, p["start"] - pad), p["end"] + pad, "hockey_bytetrack_hold.yaml", outdir) for p in game["parts"]]
    for r in retrack.starmap(jobs):
        print("DONE", r, flush=True)
