"""Tie the near-end and far-end calibrations through one shared centre circle.

WHY RETRY SOMETHING THAT ALREADY FAILED. 3.56 tried exactly this and it dragged
t=4200's clicked lines from sub-pixel out to 10px. The cause was identified there:
t=4200 measured the circle at field x~54m while its own validity stopped at 50.7m.
t=4200 has since been retired entirely (3.59, its halfway line was clicked on bare
grass). The three calibrations that remain -- t=2935, t=3020 near, t=3625 far --
are each verified against real paint with an offset control (3.63) and each fits
its own lines sub-pixel. They also already agree on the circle centre to 1.59m in x
and 1.96m in y without being asked to.

So the question is sharp: can one circle explain all three WITHOUT degrading any
frame's lines? If yes, the frames are tied and the near and far ends share a
coordinate system. If the lines degrade, they genuinely disagree, and that is the
answer rather than a failure of the method.

Unknowns: three homographies (8 each) + one shared circle (cx, cy, r) = 27.
Residuals: each frame's clicked lines, scored IN ITS OWN FRAME, plus each frame's
circle clicks against the single shared circle. Nothing is transported between
frames; only the circle is shared.

The guard from 3.60 applies: a tiny circle residual is not evidence of a good fit,
because a large circle far away matches a shallow arc beautifully. The lines are
the direct evidence and the fit is refused if any of them degrades past 5px.

Run:  python3 run_fit_bundle.py [--w W]
"""
import json
import sys
import numpy as np
from scipy.optimize import least_squares

from frisbee_analysis import UltimatePitchConfiguration
from frisbee_analysis.calibration import Calibration

cfg = UltimatePitchConfiguration()
FRAMES = [("2935", "near"), ("3020", "near"), ("3625", "far")]
EDGE_FOR = {
    "near": {"near sideline": (1, 7), "far sideline": (2, 8),
             "goal line": (3, 4), "end line": (1, 2)},
    "far": {"near sideline": (1, 7), "far sideline": (2, 8),
            "goal line": (5, 6), "end line": (7, 8)},
}
NAMES = {(1, 7): "near sideline y=0", (2, 8): "far sideline y=37",
         (1, 2): "end line x=0", (3, 4): "goal line x=18",
         (5, 6): "goal line x=82", (7, 8): "end line x=100"}


def fline(e):
    return np.cross(np.array([*cfg.vertices[e[0] - 1], 1.0]),
                    np.array([*cfg.vertices[e[1] - 1], 1.0]))


data, starts = [], []
for t, end in FRAMES:
    d = json.load(open(f"outputs/diagnostics/clicked_{t}.json"))
    tab = EDGE_FOR[end]
    lines = {tab[L["name"]]: np.array(L["points"], float)
             for L in d["lines"] if L["name"] in tab and len(L["points"]) >= 2}
    circ = np.array(next(L["points"] for L in d["lines"]
                         if L["name"] == "soccer centre circle"), float)
    c = Calibration.load(f"outputs/calibration/calibration_t{t}_circletied.json")
    H = np.linalg.inv(np.asarray(c.H())); H /= H[2, 2]
    data.append((t, lines, circ))
    starts.append(H.ravel()[:8])
    print(f"t={t}: {len(lines)} lines, {len(circ)} circle points")

C0 = np.mean([np.load(f"outputs/calibration/calibration_t{t}_circle.npy")
              for t, _ in FRAMES], axis=0)
print(f"\nstarting circle (mean of the three separate fits): "
      f"({C0[0]/100:.2f}, {C0[1]/100:.2f}) m  r={C0[2]/100:.3f} m\n")

W = float(sys.argv[sys.argv.index("--w") + 1]) if "--w" in sys.argv else None
n_line_pts = sum(len(P) for _, L, _ in data for P in L.values())
n_circ_pts = sum(len(c) for _, _, c in data)
w = W if W else np.sqrt(n_line_pts / n_circ_pts)
print(f"circle weighted at w={w:.2f} ({n_line_pts} line points, {n_circ_pts} circle)")


def split(v):
    return [np.append(v[i * 8:(i + 1) * 8], 1.0).reshape(3, 3) for i in range(3)], v[24:27]


def resid(v):
    Hs, (cx, cy, r) = split(v)
    out = []
    for Hf, (_, lines, circ) in zip(Hs, data):
        Hi = np.linalg.inv(Hf)
        for e, P in sorted(lines.items()):
            l = Hi.T @ fline(e)
            out.append((P @ l[:2] + l[2]) / np.hypot(l[0], l[1]))
        q = Hi @ np.column_stack([circ, np.ones(len(circ))]).T
        fx, fy = q[0] / q[2], q[1] / q[2]
        p0 = Hf @ np.stack([fx, fy, np.ones_like(fx)]); p0 = p0[:2] / p0[2]
        p1 = Hf @ np.stack([fx + 1, fy, np.ones_like(fx)]); p1 = p1[:2] / p1[2]
        out.append(w * (np.hypot(fx - cx, fy - cy) - r) * np.hypot(*(p1 - p0)))
    return np.concatenate(out)


v0 = np.concatenate(starts + [C0])
lo = np.concatenate([np.full(24, -np.inf), [3000., -2000., 500.]])
hi = np.concatenate([np.full(24, np.inf), [8000., 6000., 1500.]])
sol = least_squares(resid, v0, bounds=(lo, hi), max_nfev=60000)
Hs, (cx, cy, r) = split(sol.x)
print(f"\nshared circle: centre ({cx/100:.2f}, {cy/100:.2f}) m  radius {r/100:.3f} m\n")

worst = 0.0
for Hf, (t, lines, circ) in zip(Hs, data):
    Hi = np.linalg.inv(Hf)
    for e, P in sorted(lines.items()):
        l = Hi.T @ fline(e)
        dd = np.abs(P @ l[:2] + l[2]) / np.hypot(l[0], l[1])
        worst = max(worst, dd.max())
        print(f"  t={t}  {NAMES[e]:20s} max={dd.max():6.2f}px")
    q = Hi @ np.column_stack([circ, np.ones(len(circ))]).T
    fx, fy = q[0] / q[2], q[1] / q[2]
    e = np.abs(np.hypot(fx - cx, fy - cy) - r)
    print(f"  t={t}  {'centre circle':20s} mean={e.mean():5.1f}cm max={e.max():6.1f}cm")

print(f"\nworst line residual anywhere: {worst:.2f}px")
if worst > 5.0:
    print("REFUSED: one shared circle cannot explain all three frames without\n"
          "         distorting their lines. The frames genuinely disagree.")
else:
    print("ACCEPTED: one circle explains all three with the lines still clean --\n"
          "          the near and far ends share a coordinate system.")
    for Hf, (t, _, _) in zip(Hs, data):
        Calibration(homography=np.linalg.inv(Hf).tolist(), keypoint_labels=[],
                    image_points=[], field_points=[],
                    image_path=f"torovcanyon.mp4 t={t}s (3-frame bundle, shared circle)",
                    line_details=[{"edge": list(e), "points": P.tolist(),
                                   "frame_indices": None}
                                  for e, P in sorted(data[[d[0] for d in data].index(t)][1].items())]
                    ).save(f"outputs/calibration/calibration_t{t}_bundle.json")
        print(f"  saved outputs/calibration/calibration_t{t}_bundle.json")
    np.save("outputs/calibration/circle_bundle.npy", np.array([cx, cy, r]))
