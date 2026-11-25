# Rink overlay (proof of concept, 2026-11-25)

Maps every tracked player's skates to a top-down rink, by jersey color only, and plays it beside the video.

Pipeline (run from a work folder holding the clip frames at 6 fps, tracks.csv and teams.csv from the Modal volume):
1. `pairs_gen.py` matches features frame to frame (players masked) and writes the frame-to-frame homography chain.
2. `run_cal5.py` / `cal19.py` calibrate one rink from a few hand-marked landmarks in 1 or 2 frames: fixed camera position, lens k1, per-frame pan/tilt/roll and zoom, measured rink dims.
3. `snap4.py` walks the clip from the keyframe, predicting each frame from the motion chain and snapping a 4-parameter camera (rotation + zoom) to the painted lines (red, blue, far boards).
4. `quality.py` scores each frame by line fit; frames failing the check are hidden in the page.
5. `produce2.py` projects box bottom-centers to the ice, assigns team by jersey color (refs by height), smooths, stretches to a standard 200 x 85 diagram, and writes `pos<game>.json`.
6. `accuracy.py` reports faceoff-dot error and frame-to-frame repeatability in feet.

`web/index.html` is the two-pane player (video + rink, line overlay, camera view, trails, pen and arrow).

Key finding: LiveBarn cameras pan AND zoom, so a fixed-focal model drifts; the per-frame zoom is required.
