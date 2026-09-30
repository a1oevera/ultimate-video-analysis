"""Turn a run_click_lines.py file into a calibration.

This step existed only as interactive ad-hoc code -- `calibration_t4200_clicked.json`
was produced by hand, which means it could not be re-derived after the fact or
repeated for a second frame without redoing the same manual work. This is that step,
written down.

It carries over the guards that were paid for elsewhere:

  * BOTH SIDELINES OR NO FIT. Measured on t=1700 with 2px click noise: 4-line
    subsets containing both sidelines predict the lines you did NOT click to
    7-47px; subsets with only one sideline, 284-1769px. Every both-sidelines
    subset beat every one-sideline subset. Clicking more of the other lines does
    not make up for a missing sideline. --force overrides.

  * NO ANGLE TEST. An earlier version refused cross-field lines drawn within 25deg
    of the sidelines. That is invalid here: near the horizon a cross line genuinely
    flattens toward the sideline direction, and the rule would have refused 9 of 13
    valid lines across four real calibrations. Bad labelling is caught by its
    RESULT -- a collapsed validity region -- not by how the strokes look.

  * A 4-LINE FIT IS EXACT, so its reprojection error says how straight your clicks
    are and NOTHING about whether the labelling is right. Five lines is the first
    number that means something.

  * A BROKEN FIT (validity under 20%) is saved under a _REJECTED name so it cannot
    be picked up by mistake.

Run:  python3 run_fit_clicked.py [clicked.json] [--out NAME] [--force]
Then: python3 run_ground_check.py <the frame>   -- the scale-independent check
      that the far lines land on turf and not in the trees.
"""
import os
import re
import sys
import json
import cv2
import numpy as np

from frisbee_analysis import UltimatePitchConfiguration
from frisbee_analysis.calibration import (fit_calibration_with_lines,
                                          check_line_reprojection_error,
                                          field_validity_region)

# The two ends are 82m apart in the model and the picture does not say which one
# you were looking at, so run_click_lines.py records it and this honours it.
EDGE_FOR = {
    "near": {"near sideline": (1, 7), "far sideline": (2, 8),
             "goal line": (3, 4), "end line": (1, 2), "halfway": (11, 12)},
    "far":  {"near sideline": (1, 7), "far sideline": (2, 8),
             "goal line": (5, 6), "end line": (7, 8), "halfway": (11, 12)},
}
NAMES = {(1, 7): "near sideline y=0", (2, 8): "far sideline y=37",
         (1, 2): "end line x=0", (3, 4): "goal line x=18",
         (5, 6): "goal line x=82", (7, 8): "end line x=100",
         (11, 12): "halfway x=50"}

args = [a for a in sys.argv[1:] if not a.startswith("--")]
src = args[0] if args else "outputs/diagnostics/clicked_lines.json"
force = "--force" in sys.argv
out_name = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None

data = json.load(open(src))
end = data.get("end", "near")
table = EDGE_FOR[end]
tag = data.get("source", src)

lines = {}
for L in data["lines"]:
    if L["name"] not in table:
        continue                       # the centre circle is not a straight line
    if len(L["points"]) >= 2:
        lines[table[L["name"]]] = [(float(x), float(y)) for x, y in L["points"]]

print(f"\n{tag}   endzone: {end.upper()}")
for e in sorted(lines):
    print(f"  {NAMES[e]:22s} {len(lines[e])} points")
circle = next((L["points"] for L in data["lines"]
               if L["name"] == "soccer centre circle"), [])
if circle:
    print(f"  ({len(circle)} centre-circle points present; not used by this fit)")

if len(lines) < 4:
    sys.exit(f"\nneed >= 4 lines with 2+ points each, have {len(lines)}")
missing = [NAMES[e] for e in ((1, 7), (2, 8)) if e not in lines]
if missing and not force:
    sys.exit("\nFIT REFUSED -- both sidelines are needed to pin the perspective:\n"
             + "\n".join(f"    no {m} clicked" for m in missing)
             + "\n\n  Measured: 4 lines WITH both sidelines predicts the rest to 7-47px;"
               "\n  with only one sideline, 284-1769px. Clicking more of the other lines"
               "\n  does NOT make up for a missing sideline. Pass --force to fit anyway.")

cfg = UltimatePitchConfiguration()
edges = sorted(lines)
calib = fit_calibration_with_lines(
    image_points=[], keypoint_indices=[], image_path=tag,
    line_image_points=[lines[e] for e in edges], line_field_edges=edges, cfg=cfg)

print(f"\nfitted from {len(edges)} clicked lines")
for r in check_line_reprojection_error(calib, cfg):
    print(f"  {NAMES.get(tuple(r['edge']), str(r['edge'])):22s} "
          f"max={r['max']:6.2f}px mean={r['mean']:5.2f}px over {len(r['errors'])} pts")
if len(edges) == 4:
    print("  NOTE 4 lines determine a homography EXACTLY -- these numbers say how\n"
          "  straight your clicks are, not whether the labelling is right. Add a 5th.")

vr = field_validity_region(calib, cfg)
frac = vr["usable_fraction"]
print(f"  valid to x={vr['usable_x_max'] / 100:.1f}m ({frac * 100:.0f}%)")

broken = frac < 0.20
if broken:
    print("\n  *** THIS FIT IS BROKEN. The horizon lands on the field, so almost\n"
          "  *** nothing is legal to draw. That is the symptom, not the cause.")
    if len(edges) >= 5:
        print("\n  dropping each line in turn (a big jump = that line is the liar):")
        for e in edges:
            sub = [x for x in edges if x != e]
            try:
                c2 = fit_calibration_with_lines(
                    image_points=[], keypoint_indices=[], image_path=tag,
                    line_image_points=[lines[x] for x in sub],
                    line_field_edges=sub, cfg=cfg)
                f2 = field_validity_region(c2, cfg)["usable_fraction"]
            except Exception:
                f2 = 0.0
            print(f"    without {NAMES[e]:22s} -> {f2 * 100:5.1f}% usable")

stem = out_name or ("clicked_" + re.sub(r"\W+", "_", tag).strip("_"))
out = (f"outputs/calibration/{stem}_REJECTED.json" if broken
       else f"outputs/calibration/{stem}.json")
os.makedirs("outputs/calibration", exist_ok=True)
calib.save(out)
print(f"\n  saved {out}"
      + ("   (kept out of the good filename so it can't be used by mistake)"
         if broken else ""))

# Draw the whole model back on the frame. Every wrong calibration in this project
# was caught by looking at one of these, and none by a statistic.
m = re.search(r"([\w.]+\.(?:mp4|mov|avi|mkv))\s+t=([\d.]+)", tag)
frame = None
if m:
    cap = cv2.VideoCapture(f"videos/{m.group(1)}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(float(m.group(2)) * (cap.get(cv2.CAP_PROP_FPS) or 30)))
    ok, frame = cap.read()
    cap.release()
    frame = frame if ok else None
if frame is not None:
    Hf = np.linalg.inv(np.asarray(calib.H()))       # field -> image
    ref = Hf @ np.array([cfg.length / 2, cfg.width / 2, 1.0])
    for (a, b) in cfg.edges + [(11, 12)]:
        A, B = cfg.vertices[a - 1], cfg.vertices[b - 1]
        P = np.stack([np.linspace(A[0], B[0], 400), np.linspace(A[1], B[1], 400),
                      np.ones(400)])
        Q = Hf @ P
        uv = (Q[:2] / Q[2]).T[np.sign(Q[2]) == np.sign(ref[2])]
        keep = [p for p in uv if -3000 < p[0] < 5000 and -3000 < p[1] < 4000]
        col = (0, 255, 0) if (a, b) in lines or (a, b) in edges else (0, 200, 255)
        for i in range(1, len(keep)):
            cv2.line(frame, tuple(np.int32(keep[i - 1])), tuple(np.int32(keep[i])), col, 3)
    for e in edges:
        for (x, y) in lines[e]:
            cv2.circle(frame, (int(x), int(y)), 6, (0, 0, 255), -1)
    p = f"outputs/diagnostics/{stem}_overlay.jpg"
    cv2.imwrite(p, frame)
    print(f"  overlay {p}   green = lines you clicked, orange = predicted from them")
    print("  LOOK AT IT. The orange lines landing on real paint is the check that matters.")
