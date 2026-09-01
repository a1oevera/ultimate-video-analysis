"""
Click the field corners on a window's pan canvas, fit, and check immediately.

WHY CORNERS AND NOT LINES: labelling lines on this footage keeps failing in
ways no score catches (results/field_lines_finding.md 3.17, 3.25, 3.28-3.30) --
a halfway line taken for a back line, two near-coincident lines assigned to
edges 32m apart, a 4-point registration artifact picked as a sideline because
its EXTENSION passed through the right place. Four points fit the homography
through the points-only path, which carries none of that: no direction
inference, no vanishing points, no near/far end to get backwards.

A corner does NOT have to be painted. Estimate where two lines would meet;
cones usually mark it.

CONTROLS
  1..8      choose which field vertex you are placing (see the legend)
  9 / 0     the HALFWAY line's two ends (vertices 11 and 12). PREFER THESE:
            one endzone's corners pin the along-field direction with only 18m
            of baseline (~100px here), so a 2px click error rotates the whole
            field. Halfway gives 50m of baseline for the same two clicks.
  n / space skip this vertex and move to the next; p goes back. You only ever
            place the corners you can actually see -- any FOUR of the eight
            determine the homography, they do not have to be one endzone's set.
  click     place it (click again with the same vertex selected to move it)
  u         undo the last placement
  c         clear all
  +/-       zoom the view;  arrow keys or drag with right button to pan
  f         fit from >=4 corners, save, and rectify this frame as a check
  q         quit

Run:  python3 run_click_corners.py <video> <start_sec> <duration_sec> [sample_fps]
      (uses the cached pan canvas for that window -- run the propose tool first)
"""
import sys
import os
import glob
import json
import cv2
import numpy as np

from frisbee_analysis import UltimatePitchConfiguration
from frisbee_analysis.calibration import (fit_calibration, field_to_pixel,
                                           field_validity_region, Calibration)

os.makedirs("outputs/calibration", exist_ok=True)
os.makedirs("outputs/diagnostics", exist_ok=True)

video = sys.argv[1] if len(sys.argv) > 1 else "videos/torovcanyon.mp4"
start = float(sys.argv[2]) if len(sys.argv) > 2 else 1700.0
dur = float(sys.argv[3]) if len(sys.argv) > 3 else 90.0
fps_s = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0

stem = f"outputs/field_lines_cache/{os.path.basename(video)}_{start:.0f}_{dur:.0f}_{fps_s:g}"
cands = sorted(glob.glob(stem + "*_pan.jpg"), key=os.path.getmtime)
if not cands:
    sys.exit(f"no cached pan canvas for that window ({stem}*_pan.jpg).\n"
             f"run run_field_lines_propose.py on it first.")
pan_path = cands[-1]
meta_path = pan_path.replace("_pan.jpg", ".json")
canvas = cv2.imread(pan_path)
meta = json.load(open(meta_path))
T = np.asarray(meta["canvas_T"])
offx, offy = float(T[0, 2]), float(T[1, 2])
CH, CW = canvas.shape[:2]
print(f"canvas {CW}x{CH} from {pan_path}")

cfg = UltimatePitchConfiguration()
NAMES = {11: "HALFWAY x NEAR(blue) sideline   (50m, 0m)  [key 9]",
         12: "HALFWAY x FAR sideline          (50m,37m)  [key 0]",
         1: "near end x NEAR(blue) sideline   (0m, 0m)",
         2: "near end x FAR sideline          (0m,37m)",
         3: "near goal x NEAR(blue) sideline (18m, 0m)",
         4: "near goal x FAR sideline        (18m,37m)",
         5: "far goal x NEAR(blue) sideline  (82m, 0m)",
         6: "far goal x FAR sideline         (82m,37m)",
         7: "far end x NEAR(blue) sideline  (100m, 0m)",
         8: "far end x FAR sideline         (100m,37m)"}
COLS = {1: (80, 200, 255), 2: (80, 255, 200), 3: (255, 120, 0), 4: (0, 255, 0),
        5: (255, 0, 255), 6: (200, 0, 200), 7: (0, 255, 255), 8: (255, 255, 255),
        11: (0, 165, 255), 12: (255, 200, 0)}
# MEASURED, and why the halfway line matters more than a second endzone corner:
# clicking one endzone's four corners pins the ALONG-FIELD direction with only
# 18m of baseline, which at this camera angle is ~100px. A 2px click error then
# rotates the whole field by degrees -- the first two attempts came out visibly
# rotated (near endzone 1096x134px of span, far endzone 941x116px). Vertices 11
# and 12 are the halfway line's ends, giving a 50m baseline instead: five times
# the leverage, on the most clearly painted marking these pitches have.

WINW, WINH = 1500, 820
state = {"zoom": min(WINW / CW, WINH / CH), "cx": CW / 2.0, "cy": CH / 2.0,
         "sel": 1, "mouse": (0, 0), "drag": None}
corners, order = {}, []


def view_to_canvas(mx, my):
    z = state["zoom"]
    return (state["cx"] + (mx - WINW / 2) / z, state["cy"] + (my - WINH / 2) / z)


def canvas_to_view(x, y):
    z = state["zoom"]
    return (int(WINW / 2 + (x - state["cx"]) * z), int(WINH / 2 + (y - state["cy"]) * z))


def on_mouse(ev, x, y, flags, _):
    state["mouse"] = (x, y)
    if ev == cv2.EVENT_LBUTTONDOWN:
        cx, cy = view_to_canvas(x, y)
        if 0 <= cx < CW and 0 <= cy < CH:
            v = state["sel"]
            if v not in corners:
                order.append(v)
            corners[v] = (cx, cy)
            print(f"  vertex {v} at canvas ({cx:.0f},{cy:.0f})  -- {NAMES[v]}")
    elif ev == cv2.EVENT_RBUTTONDOWN:
        state["drag"] = (x, y, state["cx"], state["cy"])
    elif ev == cv2.EVENT_RBUTTONUP:
        state["drag"] = None
    elif ev == cv2.EVENT_MOUSEMOVE and state["drag"] is not None:
        ox, oy, bx, by = state["drag"]
        state["cx"] = bx - (x - ox) / state["zoom"]
        state["cy"] = by - (y - oy) / state["zoom"]
    elif ev == cv2.EVENT_MOUSEWHEEL:
        state["zoom"] *= 1.2 if flags > 0 else 1 / 1.2


def do_fit():
    if len(corners) < 4:
        print(f"need >= 4 corners, have {len(corners)}")
        return
    ks = sorted(corners)
    pix = [(corners[k][0] - offx, corners[k][1] - offy) for k in ks]
    try:
        calib = fit_calibration(image_points=pix, keypoint_indices=ks,
                                 image_path=f"{video} t={start:.0f}-{start+dur:.0f}s "
                                            f"(corners clicked)", cfg=cfg)
    except Exception as exc:
        print(f"FIT REFUSED: {exc}")
        return
    err = [np.hypot(*(np.array(field_to_pixel(*cfg.vertices[k - 1], calib)) - np.array(p)))
           for k, p in zip(ks, pix)]
    vr = field_validity_region(calib, cfg)
    print(f"\nfitted from {len(ks)} corners {ks}")
    print(f"  reprojection: max {max(err):.2f}px mean {sum(err)/len(err):.2f}px"
          + ("   (4 corners fit EXACTLY -- add a 5th for a number that means something)"
             if len(ks) == 4 else ""))
    print(f"  valid to x={vr['usable_x_max']/100:.1f}m ({vr['usable_fraction']*100:.0f}%)")
    out = "outputs/calibration/calibration_corners.json"
    calib.save(out)
    print(f"  saved {out}")
    # rectify THIS canvas as the check: content must land on the yellow box
    ppm, M = 14.0, 10.0
    OW = int((cfg.length / 100.0 + 2 * M) * ppm)
    OH = int((cfg.width / 100.0 + 2 * M) * ppm)
    F2C = np.array([[ppm / 100.0, 0, M * ppm], [0, ppm / 100.0, M * ppm], [0, 0, 1.0]])
    Hc = np.linalg.inv(np.asarray(calib.H())) @ np.linalg.inv(F2C)
    src = canvas.copy()
    rect = cv2.warpPerspective(src, Hc, (OW, OH), flags=cv2.WARP_INVERSE_MAP)
    gx, _ = np.meshgrid(np.arange(OW), np.arange(OH))
    rect[(gx - M * ppm) * 100.0 / ppm > vr["usable_x_max"]] = 0

    def fp(xc, yc):
        p = F2C @ np.array([xc, yc, 1.0])
        return int(p[0] / p[2]), int(p[1] / p[2])
    for a, b in (((0, 0), (0, cfg.width)), ((cfg.length, 0), (cfg.length, cfg.width)),
                 ((0, 0), (cfg.length, 0)), ((0, cfg.width), (cfg.length, cfg.width)),
                 ((1800, 0), (1800, cfg.width)), ((5000, 0), (5000, cfg.width)),
                 ((8200, 0), (8200, cfg.width))):
        cv2.line(rect, fp(*a), fp(*b), (0, 220, 255), 2)
    cv2.putText(rect, "content must sit on the yellow box", (14, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    cv2.imwrite("outputs/diagnostics/corners_rectified.jpg", rect)
    cv2.imshow("rectified check", cv2.resize(rect, None, fx=0.85, fy=0.85))
    print("  -> outputs/diagnostics/corners_rectified.jpg  (also shown in a window)")
    print("  --corner " + " ".join(f"{k}={corners[k][0]:.0f},{corners[k][1]:.0f}" for k in ks))


cv2.namedWindow("click corners", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("click corners", on_mouse)
print("\nkeys: 1-8 pick vertex | click place | u undo | c clear | +/- zoom | "
      "arrows pan | f fit | q quit\n")
while True:
    z = state["zoom"]
    Mx = np.array([[z, 0, WINW / 2 - state["cx"] * z], [0, z, WINH / 2 - state["cy"] * z]])
    view = cv2.warpAffine(canvas, Mx, (WINW, WINH))
    for k, (x, y) in corners.items():
        p = canvas_to_view(x, y)
        cv2.drawMarker(view, p, COLS[k], cv2.MARKER_CROSS, 26, 2)
        cv2.circle(view, p, 13, COLS[k], 2)
        for c2, t2 in (((0, 0, 0), 5), (COLS[k], 2)):
            cv2.putText(view, str(k), (p[0] + 16, p[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, c2, t2)
    # magnifier so a corner can be placed precisely
    mx, my = state["mouse"]
    cx, cy = view_to_canvas(mx, my)
    R = 60
    x0, y0 = int(cx) - R, int(cy) - R
    patch = np.zeros((2 * R, 2 * R, 3), np.uint8)
    xs0, ys0 = max(0, x0), max(0, y0)
    xs1, ys1 = min(CW, x0 + 2 * R), min(CH, y0 + 2 * R)
    if xs1 > xs0 and ys1 > ys0:
        patch[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0] = canvas[ys0:ys1, xs0:xs1]
    loupe = cv2.resize(patch, (240, 240), interpolation=cv2.INTER_NEAREST)
    cv2.line(loupe, (120, 100), (120, 140), (0, 0, 255), 1)
    cv2.line(loupe, (100, 120), (140, 120), (0, 0, 255), 1)
    cv2.rectangle(loupe, (0, 0), (239, 239), (255, 255, 255), 1)
    view[10:250, WINW - 250:WINW - 10] = loupe
    missing = [v for v in (1, 2, 3, 4, 11, 12, 5, 6, 7, 8) if v not in corners]
    bar = (f"[{state['sel']}] {NAMES[state['sel']]}"
           f"   placed {sorted(corners)}  ({len(corners)}/4 needed)")
    cv2.rectangle(view, (0, WINH - 62), (WINW, WINH), (0, 0, 0), -1)
    cv2.putText(view, bar, (12, WINH - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(view, f"canvas ({cx:.0f},{cy:.0f})   zoom {z:.2f}x   "
                      f"1-8 pick | n/space skip | click place | u undo | f fit | q quit",
                (12, WINH - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 220, 255), 1)
    cv2.imshow("click corners", view)
    k = cv2.waitKey(20) & 0xFF
    if k == ord('q'):
        break
    elif ord('1') <= k <= ord('8'):
        state["sel"] = k - ord('0')
    elif k == ord('9'):
        state["sel"] = 11          # halfway x near(blue) sideline
    elif k == ord('0'):
        state["sel"] = 12          # halfway x far sideline
    elif k in (ord('n'), ord(' ')):          # skip this one, go to the next
        seq = [1, 2, 3, 4, 11, 12, 5, 6, 7, 8]
        state["sel"] = seq[(seq.index(state["sel"]) + 1) % len(seq)]
    elif k == ord('p'):
        seq = [1, 2, 3, 4, 11, 12, 5, 6, 7, 8]
        state["sel"] = seq[(seq.index(state["sel"]) - 1) % len(seq)]
    elif k == ord('u') and order:
        v = order.pop()
        corners.pop(v, None)
        print(f"  undo vertex {v}")
    elif k == ord('c'):
        corners.clear(); order.clear(); print("  cleared")
    elif k in (ord('+'), ord('=')):
        state["zoom"] *= 1.25
    elif k in (ord('-'), ord('_')):
        state["zoom"] /= 1.25
    elif k == 81: state["cx"] -= 60 / z
    elif k == 83: state["cx"] += 60 / z
    elif k == 82: state["cy"] -= 60 / z
    elif k == 84: state["cy"] += 60 / z
    elif k == ord('f'):
        do_fit()
cv2.destroyAllWindows()
