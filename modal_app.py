"""Run the PuckTracker CV pipeline on a Modal GPU.

Setup (once):
    pip install modal
    modal token set --token-id $MODAL_TOKEN_ID --token-secret $MODAL_TOKEN_SECRET

Analyze a 30 minute LiveBarn segment for one player:
    modal run modal_app.py --video path/to/segment.mp4 --number 21 --team white

The video is uploaded to the "pucktracker-videos" volume, processed on an L4 GPU,
and the outputs (tracks, teams, jersey reads, shifts) are downloaded to ./out/<video stem>/.
"""
import pathlib

import modal

app = modal.App("pucktracker-cv")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "ultralytics==8.4.164",
        "opencv-python-headless",
        "pandas",
        "rapidocr-onnxruntime",
        "lap",
        "imageio-ffmpeg",
    )
    .add_local_dir("configs", "/root/configs")
    .add_local_python_source("pucktracker")
)

volume = modal.Volume.from_name("pucktracker-videos", create_if_missing=True)
DATA = pathlib.Path("/data")
OUTPUTS = ["tracks.csv", "teams.csv", "jersey.csv", "player_shifts.csv", "player_tracks.csv"]


@app.function(image=image, gpu="L4", volumes={str(DATA): volume}, timeout=3 * 60 * 60)
def analyze(video_name: str, number: str = "21", team: str = "white", stride: int = 1,
            tracker: str = "hockey_bytetrack.yaml", start_s: float = 0.0, end_s: float = 0.0) -> dict:
    import time

    from pucktracker import classify, jersey, shifts, track

    video = str(DATA / "videos" / video_name)
    out = DATA / "out" / pathlib.Path(video_name).stem
    out.mkdir(parents=True, exist_ok=True)
    timings = {}

    t = time.time()
    track.main(video, str(out / "tracks.csv"), stride, tracker=f"/root/configs/{tracker}",
               start_s=start_s, end_s=end_s or None)
    timings["track_min"] = round((time.time() - t) / 60, 1)

    t = time.time()
    classify.main(video, str(out / "tracks.csv"), str(out / "teams.csv"))
    jersey.main(video, str(out / "tracks.csv"), str(out / "teams.csv"), str(out / "jersey.csv"), team, 70)
    timings["identify_min"] = round((time.time() - t) / 60, 1)

    shifts.main(str(out / "tracks.csv"), str(out / "teams.csv"), str(out / "jersey.csv"),
                number, 24, str(out / "player"))
    volume.commit()
    return {"out_dir": str(out.relative_to(DATA)), "timings": timings}


@app.local_entrypoint()
def main(video: str, number: str = "21", team: str = "white", stride: int = 1,
         tracker: str = "hockey_bytetrack.yaml", start_s: float = 0.0, end_s: float = 0.0):
    src = pathlib.Path(video)
    with volume.batch_upload(force=True) as batch:
        batch.put_file(str(src), f"/videos/{src.name}")
    print(f"uploaded {src.name}")

    result = analyze.remote(src.name, number, team, stride, tracker, start_s, end_s)
    print(result)

    local = pathlib.Path("out") / src.stem
    local.mkdir(parents=True, exist_ok=True)
    for name in OUTPUTS:
        with open(local / name, "wb") as f:
            for chunk in volume.read_file(f"{result['out_dir']}/{name}"):
                f.write(chunk)
    print(f"results in {local}/")
