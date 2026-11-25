"""Standard 200 x 85 ft rink model in feet. Origin at center ice, x along the length
(+x to the right of the camera), y across (+y = far boards, -y = camera side)."""
import numpy as np

L, W, R = 100.0, 42.5, 28.0          # half length, half width, corner radius
BLUE, GOAL = 25.0, 89.0
DOT_Y, NZ_X, EZ_X = 22.0, 20.0, 69.0


def arc(cx, cy, r, a0, a1, n=60):
    a = np.radians(np.linspace(a0, a1, n))
    return np.c_[cx + r * np.cos(a), cy + r * np.sin(a)]


def boards(n=40):
    pts = []
    for cx, cy, a0 in [(L - R, W - R, 0), (-L + R, W - R, 90), (-L + R, -W + R, 180), (L - R, -W + R, 270)]:
        pts.append(arc(cx, cy, R, a0, a0 + 90, n))
    p = np.vstack(pts)
    return np.vstack([p, p[:1]])


def goal_line(x):
    # goal line stops where it meets the rounded corner
    yy = W - R + np.sqrt(max(R ** 2 - (abs(x) - (L - R)) ** 2, 0))
    return np.array([[x, -yy], [x, yy]])


def features():
    """List of (kind, polyline Nx2). kind is the paint color: red, blue, boards."""
    f = [("boards", boards()), ("red", np.array([[0, -W], [0, W]])),
         ("blue", np.array([[-BLUE, -W], [-BLUE, W]])), ("blue", np.array([[BLUE, -W], [BLUE, W]])),
         ("red", goal_line(GOAL)), ("red", goal_line(-GOAL)),
         ("blue", arc(0, 0, 15, 0, 360)), ("red", arc(0, -W, 10, 0, 180))]
    for sx in (-1, 1):
        for sy in (-1, 1):
            f.append(("red", arc(sx * EZ_X, sy * DOT_Y, 15, 0, 360)))
    return f


def dots():
    d = [(0.0, 0.0)]
    for sx in (-1, 1):
        for sy in (-1, 1):
            d += [(sx * NZ_X, sy * DOT_Y), (sx * EZ_X, sy * DOT_Y)]
    return np.array(d)
