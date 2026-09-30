"""Game videos: stitch LiveBarn files into one game master, cut per-player videos, and serve the Shift Marker.

Build the master for a game (definition in games/<id>.json):
    modal run --detach games_app.py::make_master --game-id 2026-09-26

Serve the Shift Marker web app:
    modal deploy games_app.py

Everything lives on the "pucktracker-videos" volume under /data/games/<id>/:
    game.json      the game definition (files and where the game starts and ends in each)
    master.mp4     the stitched game, 720p, starting at puck drop
    marks/<n>.json on/off marks for player <n>, in seconds of the master
    players/<n>.mp4 that player's shifts cut from the master
"""
import json
import pathlib
import subprocess

import modal

app = modal.App("pucktracker-games")
vol = modal.Volume.from_name("pucktracker-videos", create_if_missing=True)
DATA = pathlib.Path("/data")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("fastapi[standard]==0.115.12", "aiofiles==24.1.0")
    .add_local_dir("games", "/root/games")
    .add_local_dir("web", "/root/web")
)


def game_dir(game_id: str) -> pathlib.Path:
    return DATA / "games" / game_id


def load_game(game_id: str) -> dict:
    p = game_dir(game_id) / "game.json"
    if p.exists():
        return json.loads(p.read_text())
    return json.loads((pathlib.Path("/root/games") / f"{game_id}.json").read_text())


def offsets(game: dict) -> list:
    """Where each part starts on the master timeline."""
    out, t = [], 0.0
    for p in game["parts"]:
        out.append(t)
        t += p["end"] - p["start"]
    return out


def run(cmd: list):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-3000:])


@app.function(image=image, volumes={str(DATA): vol}, cpu=8, memory=8192, timeout=3 * 60 * 60)
def make_master(game_id: str, height: int = 720, crf: int = 23) -> dict:
    """Trim each file to the game window and join them into one video that starts at puck drop."""
    game = json.loads((pathlib.Path("/root/games") / f"{game_id}.json").read_text())
    d = game_dir(game_id)
    d.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for p in game["parts"]:
        cmd += ["-ss", str(p["start"]), "-to", str(p["end"]), "-i", str(DATA / "videos" / p["file"])]
    n = len(game["parts"])
    chains = "".join(f"[{i}:v]scale=-2:{height},setsar=1,fps=24[v{i}];" for i in range(n))
    joined = "".join(f"[v{i}][{i}:a]" for i in range(n))
    has_audio = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                                "-of", "csv=p=0", str(DATA / "videos" / game["parts"][0]["file"])],
                               capture_output=True, text=True).stdout.strip() != ""
    if has_audio:
        filt = chains + joined + f"concat=n={n}:v=1:a=1[v][a]"
        maps = ["-map", "[v]", "-map", "[a]", "-c:a", "aac", "-b:a", "96k"]
    else:
        filt = chains + "".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[v]"
        maps = ["-map", "[v]"]
    tmp = d / "master.tmp.mp4"
    cmd += ["-filter_complex", filt, *maps, "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
            "-g", "48", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)]
    run(cmd)
    tmp.rename(d / "master.mp4")
    game["offsets"] = offsets(game)
    game["duration"] = sum(p["end"] - p["start"] for p in game["parts"])
    (d / "game.json").write_text(json.dumps(game, indent=2))
    vol.commit()
    return {"master": str(d / "master.mp4"), "mb": round((d / "master.mp4").stat().st_size / 1e6), **game}


def player_shifts(game_id: str, number: str) -> list:
    """Marked shifts if the player has been reviewed, otherwise the tracker draft."""
    p = game_dir(game_id) / "marks" / f"{number}.json"
    if p.exists():
        rows = json.loads(p.read_text()).get("shifts", [])
        return sorted([r["on"], r["off"]] for r in rows if r.get("off") is not None)
    return []


@app.function(image=image, volumes={str(DATA): vol}, cpu=4, memory=4096, timeout=60 * 60)
def make_player_video(game_id: str, number: str, pre: float = 3.0, post: float = 2.0) -> dict:
    """Cut every shift for one player from the master and join them into one video."""
    vol.reload()
    d = game_dir(game_id)
    shifts = player_shifts(game_id, number)
    if not shifts:
        return {"error": f"No marked shifts for #{number} yet."}
    dur = load_game(game_id).get("duration", 1e9)
    # Pad each shift, then merge pads that overlap so no footage repeats.
    spans = []
    for on, off in shifts:
        a, b = max(0.0, on - pre), min(dur, off + post)
        if spans and a <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])
    out = d / "players"
    out.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for a, b in spans:
        cmd += ["-ss", f"{a:.2f}", "-to", f"{b:.2f}", "-i", str(d / "master.mp4")]
    n = len(spans)
    filt = "".join(f"[{i}:v][{i}:a]" for i in range(n)) + f"concat=n={n}:v=1:a=1[v][a]"
    tmp = out / f"{number}.tmp.mp4"
    cmd += ["-filter_complex", filt, "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "23", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(tmp)]
    run(cmd)
    tmp.rename(out / f"{number}.mp4")
    vol.commit()
    return {"number": number, "shifts": len(shifts), "spans": spans,
            "seconds": round(sum(b - a for a, b in spans), 1)}


# ---------------------------------------------------------------- web app

@app.function(image=image, volumes={str(DATA): vol}, secrets=[modal.Secret.from_name("pucktracker-web")],
              cpu=1, memory=1024, timeout=60 * 60, scaledown_window=15 * 60,
              max_containers=1)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def web():
    import os
    import time

    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse

    api = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    key = os.environ["VIEW_KEY"]
    games_src = pathlib.Path("/root/games")

    def check(req: Request):
        if req.cookies.get("pt") != key and req.query_params.get("k") != key:
            raise HTTPException(401, "Open the link you were given.")

    def game_ids() -> list:
        return sorted(p.stem for p in games_src.glob("*.json") if not p.stem.endswith(".drafts"))

    def extras(game_id: str) -> dict:
        p = games_src / f"{game_id}.drafts.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def reload():
        # Picks up files written by the player-video job; skipped while a video is streaming.
        try:
            vol.reload()
        except Exception:
            pass

    def marks_dir(game_id: str) -> pathlib.Path:
        d = game_dir(game_id) / "marks"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @api.get("/")
    def index(req: Request):
        if req.query_params.get("k") == key:
            r = RedirectResponse("/", 303)
            r.set_cookie("pt", key, max_age=365 * 86400, httponly=True, secure=True, samesite="lax")
            return r
        check(req)
        return HTMLResponse((pathlib.Path("/root/web") / "marker.html").read_text())

    @api.get("/api/games")
    def games(req: Request):
        check(req)
        out = []
        for g in game_ids():
            game = load_game(g)
            out.append({"id": g, "title": game.get("title", g), "ready": (game_dir(g) / "master.mp4").exists()})
        return out

    @api.get("/api/games/{game_id}")
    def game(game_id: str, req: Request):
        check(req)
        if game_id not in game_ids():
            raise HTTPException(404)
        reload()
        g, x = load_game(game_id), extras(game_id)
        d = marks_dir(game_id)
        # First open: carry over marks made in the per-file Shift Marker.
        for n, rows in x.get("seed_marks", {}).items():
            p = d / f"{n}.json"
            if not p.exists():
                p.write_text(json.dumps({"shifts": rows, "updated": time.time()}))
                vol.commit()
        marks = {p.stem: json.loads(p.read_text()) for p in d.glob("*.json")}
        players = {}
        pd = game_dir(game_id) / "players"
        if pd.exists():
            players = {p.stem: p.stat().st_mtime for p in pd.glob("*.mp4") if not p.stem.endswith(".tmp")}
        return {"id": game_id, "title": g.get("title"), "team_color": g.get("team_color"),
                "duration": g.get("duration") or x.get("duration"), "parts": g["parts"],
                "offsets": g.get("offsets") or offsets(g), "drafts": x.get("drafts", {}), "ref": x.get("ref", {}),
                "ref_note": x.get("ref_note", {}), "roster": g.get("roster") or x.get("roster") or sorted(x.get("drafts", {}), key=int),
                "marks": marks, "player_videos": players}

    @api.put("/api/games/{game_id}/marks/{number}")
    async def save_marks(game_id: str, number: str, req: Request):
        check(req)
        if game_id not in game_ids() or not number.isdigit() or len(number) > 2:
            raise HTTPException(400)
        body = await req.json()
        rows = [{"on": float(r["on"]), "off": None if r.get("off") is None else float(r["off"]),
                 "src": "mine" if r.get("src") == "mine" else "auto"} for r in body.get("shifts", [])]
        (marks_dir(game_id) / f"{number}.json").write_text(json.dumps({"shifts": rows, "updated": time.time()}))
        vol.commit()
        return {"ok": True}

    @api.post("/api/games/{game_id}/players/{number}/video")
    def build_player(game_id: str, number: str, req: Request):
        check(req)
        if game_id not in game_ids() or not number.isdigit():
            raise HTTPException(400)
        call = make_player_video.spawn(game_id, number)
        return {"call": call.object_id}

    @api.get("/api/calls/{call_id}")
    def call_status(call_id: str, req: Request):
        check(req)
        fc = modal.FunctionCall.from_id(call_id)
        try:
            return {"done": True, "result": fc.get(timeout=0)}
        except TimeoutError:
            return {"done": False}
        except Exception as e:
            return {"done": True, "result": {"error": str(e)[:300]}}

    def send_video(path: pathlib.Path, req: Request, name: str):
        if not path.exists():
            raise HTTPException(404)
        size = path.stat().st_size
        rng = req.headers.get("range")
        headers = {"Accept-Ranges": "bytes", "Content-Type": "video/mp4", "Cache-Control": "private, max-age=3600",
                   "Content-Disposition": f'inline; filename="{name}"'}
        if not rng or not rng.startswith("bytes="):
            return FileResponse(path, media_type="video/mp4", headers=headers)
        a, _, b = rng[6:].split(",")[0].partition("-")
        start = int(a) if a else max(0, size - int(b))
        end = min(int(b), size - 1) if (a and b) else size - 1
        if start >= size:
            raise HTTPException(416)

        def chunks():
            with open(path, "rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    buf = f.read(min(1 << 20, left))
                    if not buf:
                        break
                    left -= len(buf)
                    yield buf

        headers.update({"Content-Range": f"bytes {start}-{end}/{size}", "Content-Length": str(end - start + 1)})
        return StreamingResponse(chunks(), status_code=206, headers=headers)

    @api.get("/video/{game_id}/master.mp4")
    def master(game_id: str, req: Request):
        check(req)
        return send_video(game_dir(game_id) / "master.mp4", req, f"{game_id}-game.mp4")

    @api.get("/video/{game_id}/players/{number}.mp4")
    def player_video(game_id: str, number: str, req: Request):
        check(req)
        if not number.isdigit():
            raise HTTPException(400)
        reload()
        return send_video(game_dir(game_id) / "players" / f"{number}.mp4", req, f"{game_id}-{number}.mp4")

    return api
