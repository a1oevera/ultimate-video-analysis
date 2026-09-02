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
  m         switch between CORNER mode and LINE mode
  a s d g h j k   in LINE mode, choose which line you are clicking:
            a near(blue) sideline   s far sideline   d near goal   g far goal
            h near end line         j far end line   k halfway
            Click 2+ points anywhere ALONG the line -- you never need to find
            a corner. Four lines determine the homography.
  1..8      choose which field vertex you are placing (see the legend)
  9 / 0     the HALFWAY line's two ends (vertices 11 and 12). PREFER THESE:
            one endzone's corners pin the along-field direction with only 18m
            of baseline (~100px here), so a 2px click error rotates the whole
            field. Halfway gives 50m of baseline for the same two clicks.
  n / space skip this vertex and move to the next; p goes back. You only ever
            place the corners you can actually see -- any FOUR of the eight
            determine the homography, they do not have to be one endzone's set.
  click     place it (click again with the same vertex selected to move it)
  z         undo the last point
  c         clear
  +/-       zoom the view;  arrow keys or drag with right button to pan
  f         FIT AND SAVE, then rectify as a check -- this is how you finish
  q or ESC  quit (nothing is saved on quit; press f first)

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
from frisbee_analysis.calibration import (fit_calibration, fit_calibration_with_lines,
                                           field_to_pixel, check_line_reprojection_error,
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

# ---- LINE mode -----------------------------------------------------------
# Clicking CORNERS needs the corner to be identifiable, and on this footage it
# often is not: the far endzone's four corners came out spanning 941x116px,
# too shallow to constrain the along-field direction, and both attempts came
# out visibly rotated. A LINE only needs two points you are confident lie
# somewhere along it -- you never have to find the intersection. That is a much
# lower bar on this footage and it is the path validated exactly correct in
# results/calibration_finding.md (lines-only; point/line MIXING has a known
# unresolved bug, so this tool keeps the two modes separate and never mixes).
# a s d g h j k -- deliberately clear of every command key (m f c q z n p),
# because the first version reused q and u for both a line and a command.
LINE_KEYS = {ord('a'): (1, 7), ord('s'): (2, 8), ord('d'): (3, 4),
             ord('g'): (5, 6), ord('h'): (1, 2), ord('j'): (7, 8),
             ord('k'): (11, 12)}
LINE_NAMES = {(1, 7): "NEAR sideline (the blue one)  y=0m",
              (2, 8): "FAR sideline                  y=37m",
              (3, 4): "near goal line                x=18m",
              (5, 6): "far goal line                 x=82m",
              (1, 2): "near end line                 x=0m",
              (7, 8): "far end line                  x=100m",
              (11, 12): "HALFWAY line                  x=50m"}
LINE_COLS = {(1, 7): (255, 120, 0), (2, 8): (0, 255, 0), (3, 4): (255, 0, 255),
             (5, 6): (200, 0, 200), (1, 2): (0, 255, 255), (7, 8): (0, 200, 200),
             (11, 12): (255, 255, 255)}
lines = {}                 # edge -> [canvas points]
state["mode"] = "corner"   # or "line"
state["edge"] = (1, 7)


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
            if state["mode"] == "line":
                e = state["edge"]
                lines.setdefault(e, []).append((cx, cy))
                print(f"  {LINE_NAMES[e]}: point {len(lines[e])} "
                      f"at canvas ({cx:.0f},{cy:.0f})")
            else:
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


ALONG = {(1, 7), (2, 8)}          # sidelines run down the field
ACROSS = {(3, 4), (5, 6), (1, 2), (7, 8), (11, 12)}   # end/goal/halfway cross it


def _seg_angle(ps):
    dx, dy = np.asarray(ps[-1], float) - np.asarray(ps[0], float)
    a = abs(np.degrees(np.arctan2(dy, dx)))
    return min(a, 180.0 - a)


def conditioning_warnings(usable):
    """Both sidelines or the fit is junk. Measured on t1700 with 2px click noise,
    4-line subsets: WITH both sidelines the lines you did not click land 7-47px
    from truth; with only one sideline, 284-1769px. Every both-sidelines subset
    beat every one-sideline subset -- it matters more than click precision or
    how many points you put on each line."""
    missing = [LINE_NAMES[e] for e in ((1, 7), (2, 8)) if e not in usable]
    if missing:
        return [f"    no {m} clicked" for m in missing]
    return []


def orientation_warnings(usable, min_sep_deg=25.0):
    """RETRACTED -- always returns []. This used to refuse a cross-field line drawn
    within 25deg of the sideline family. That is the same angle-based reasoning that
    fails elsewhere in this project: near the horizon a cross line genuinely flattens
    toward the sideline direction. Measured against four real calibrations it would
    have refused 9 of 13 valid lines -- the t=3235 near goal line sits 6.8deg from its
    sidelines, t=1700 halfway 4.0deg. VP concurrency was tried as a replacement and
    also fails to separate: the bad t=2700 cross lines are concurrent to 1.56deg,
    inside click noise. Bad labelling is caught by its RESULT (a degenerate validity
    region) in _finish, not by how the strokes look."""
    return []


def do_fit():
    usable = {e: p for e, p in lines.items() if len(p) >= 2}
    if state["mode"] == "line" or usable:
        if len(usable) < 4:
            print(f"need >= 4 lines with 2+ points each, have {len(usable)}: "
                  f"{[LINE_NAMES[e].split('(')[0].strip() for e in usable]}")
            return
        cond = conditioning_warnings(usable)
        if cond and not state.get("force"):
            print("\nFIT REFUSED -- both sidelines are needed to pin the perspective:")
            for w in cond:
                print(w)
            print("\n  Measured: 4 lines WITH both sidelines predicts the rest to 7-47px;")
            print("  with only one sideline, 284-1769px. Clicking more of the other lines")
            print("  does NOT make up for a missing sideline. Press F to fit anyway.")
            return
        edges = sorted(usable)
        pts = [[(x - offx, y - offy) for x, y in usable[e]] for e in edges]
        try:
            calib = fit_calibration_with_lines(
                image_points=[], keypoint_indices=[],
                image_path=f"{video} t={start:.0f}-{start+dur:.0f}s (lines clicked)",
                line_image_points=pts, line_field_edges=edges, cfg=cfg)
        except Exception as exc:
            print(f"FIT REFUSED: {exc}")
            return
        print(f"\nfitted from {len(edges)} clicked lines")
        for r in check_line_reprojection_error(calib, cfg):
            print(f"  {str(tuple(r['edge'])):8s} {LINE_NAMES.get(tuple(r['edge']),''):34s} "
                  f"max={r['max']:6.2f}px mean={r['mean']:5.2f}px over {len(r['errors'])} pts")
        if len(edges) == 4:
            print("  NOTE 4 lines determine a homography EXACTLY -- these numbers say how")
            print("  straight your clicks are, not whether the labelling is right. Add a 5th.")
        _finish(calib, "lines")
        return
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
    print(f"\nfitted from {len(ks)} corners {ks}")
    print(f"  reprojection: max {max(err):.2f}px mean {sum(err)/len(err):.2f}px"
          + ("   (4 corners fit EXACTLY -- add a 5th for a number that means something)"
             if len(ks) == 4 else ""))
    _finish(calib, "corners")


ALL_EDGES = ((1, 7), (2, 8), (3, 4), (5, 6), (1, 2), (7, 8), (11, 12))
preview = {"key": None, "calib": None, "vr": None, "on": True}


def preview_calib():
    """Refit continuously from whatever is clicked so far. Once 4 lines are in,
    the remaining lines are DETERMINED -- drawing them turns 'guess where the far
    end line is' into 'check that the predicted line sits on the paint', which is
    the whole reason the t=2700 attempt had to be guessed at."""
    usable = {e: p for e, p in lines.items() if len(p) >= 2}
    key = tuple(sorted((e, len(p), tuple(map(tuple, p))) for e, p in usable.items()))
    if key == preview["key"]:
        return preview["calib"]
    preview["key"] = key
    preview["calib"] = preview["vr"] = None
    if len(usable) < 4 or conditioning_warnings(usable):
        return None
    edges = sorted(usable)
    pts = [[(x - offx, y - offy) for x, y in usable[e]] for e in edges]
    try:
        c2 = fit_calibration_with_lines(
            image_points=[], keypoint_indices=[], image_path="preview",
            line_image_points=pts, line_field_edges=edges, cfg=cfg)
        preview["calib"] = c2
        preview["vr"] = field_validity_region(c2, cfg)
    except Exception:
        pass
    return preview["calib"]


def draw_prediction(view):
    c2 = preview_calib()
    if c2 is None or not preview["on"]:
        return None
    xmax = preview["vr"]["usable_x_max"]
    for e in ALL_EDGES:
        a = np.asarray(cfg.vertices[e[0] - 1], float)
        b = np.asarray(cfg.vertices[e[1] - 1], float)
        run, clicked = [], e in lines and len(lines[e]) >= 2
        for t in np.linspace(0, 1, 60):
            fx, fy = a + (b - a) * t
            if fx > xmax:
                run = []
                continue
            try:
                px, py = field_to_pixel(fx, fy, c2)
            except Exception:
                run = []
                continue
            run.append(canvas_to_view(px + offx, py + offy))
            if len(run) >= 2:
                # solid where you clicked it, dashed-thin where it is PREDICTED
                cv2.line(view, run[-2], run[-1], (90, 255, 90),
                         3 if clicked else 1)
        if run:
            for c3, th in (((0, 0, 0), 4), ((90, 255, 90), 1)):
                cv2.putText(view, ("" if clicked else "predicted ") + LINE_NAMES[e],
                            (run[len(run) // 2][0] + 8, run[len(run) // 2][1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, c3, th)
    return preview["vr"]


def _leave_one_out():
    usable = {e: p for e, p in lines.items() if len(p) >= 2}
    out = []
    for drop in sorted(usable):
        sub = {e: p for e, p in usable.items() if e != drop}
        if len(sub) < 4:
            continue
        edges = sorted(sub)
        pts = [[(x - offx, y - offy) for x, y in sub[e]] for e in edges]
        try:
            c2 = fit_calibration_with_lines(
                image_points=[], keypoint_indices=[], image_path="loo",
                line_image_points=pts, line_field_edges=edges, cfg=cfg)
            out.append((field_validity_region(c2, cfg)["usable_fraction"],
                        LINE_NAMES.get(drop, str(drop))))
        except Exception:
            pass
    return out


def _finish(calib, kind):
    """Save, then rectify THIS canvas as the check -- content must land on the
    yellow box. That visual test is what caught every wrong calibration in this
    project; no statistic here has."""
    vr = field_validity_region(calib, cfg)
    frac = vr["usable_fraction"]
    print(f"  valid to x={vr['usable_x_max']/100:.1f}m ({frac*100:.0f}%)")
    broken = frac < 0.20
    if broken:
        print("\n  *** THIS FIT IS BROKEN. The horizon lands on the field, so almost")
        print("  *** nothing is legal to draw -- the rectified check below will be")
        print("  *** mostly black. That is the symptom, not the cause.")
        if kind == "lines" and len(lines) >= 5:
            print("\n  dropping each line in turn (a big jump = that line is the liar):")
            for f2, nm in sorted(_leave_one_out(), reverse=True):
                print(f"    without {nm:34s} -> {f2*100:5.1f}% usable")
    out = ("outputs/calibration/calibration_clicked_REJECTED.json" if broken
           else "outputs/calibration/calibration_clicked.json")
    calib.save(out)
    print(f"  saved {out}"
          + ("   (kept out of the good filename so it can't be used by mistake)"
             if broken else ""))
    # rectify THIS canvas as the check: content must land on the yellow box
    ppm, M = 14.0, 10.0
    OW = int((cfg.length / 100.0 + 2 * M) * ppm)
    OH = int((cfg.width / 100.0 + 2 * M) * ppm)
    F2C = np.array([[ppm / 100.0, 0, M * ppm], [0, ppm / 100.0, M * ppm], [0, 0, 1.0]])
    Hc = np.linalg.inv(np.asarray(calib.H())) @ np.linalg.inv(F2C)
    src = canvas.copy()
    rect = cv2.warpPerspective(src, Hc, (OW, OH), flags=cv2.WARP_INVERSE_MAP)
    gx, _ = np.meshgrid(np.arange(OW), np.arange(OH))
    beyond = (gx - M * ppm) * 100.0 / ppm > vr["usable_x_max"]
    rect[beyond] = 40          # dark grey, not black -- black reads as "tool crashed"
    if beyond.any():
        cv2.putText(rect, f"grey = past the {vr['usable_x_max']/100:.0f}m validity limit, "
                          f"not painted", (14, OH - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (170, 170, 170), 2)

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
    if kind == "corners":
        print("  --corner " + " ".join(f"{k}={corners[k][0]:.0f},{corners[k][1]:.0f}"
                                       for k in sorted(corners)))
    else:
        for e, ps in sorted(lines.items()):
            if len(ps) >= 2:
                print(f"  --line {e[0]},{e[1]}={ps[0][0]:.0f},{ps[0][1]:.0f},"
                      f"{ps[-1][0]:.0f},{ps[-1][1]:.0f}")


cv2.namedWindow("click corners", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("click corners", on_mouse)
print("\nLINE mode (press m):  a=near sideline(blue)  s=far sideline  d=near goal")
print("                      g=far goal  h=near end  j=far end  k=halfway")
print("CORNER mode:          1-8 pick a vertex, 9/0 the halfway ends")
print("click = place | z = undo | c = clear | v = toggle prediction | +/- zoom | arrows")
print("\nafter 4 good lines the tool DRAWS the rest in green -- don't guess them,")
print("check the green line sits on the paint, then click along it to confirm.")
print("\n*** press  f  to FIT AND SAVE.  q or ESC quits without saving. ***\n")
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
    for e, ps in lines.items():
        col = LINE_COLS.get(e, (255, 255, 255))
        pv = [canvas_to_view(x, y) for x, y in ps]
        for q in pv:
            cv2.circle(view, q, 6, col, -1)
            cv2.circle(view, q, 6, (0, 0, 0), 1)
        if len(pv) >= 2:
            a = np.array(pv[0], float); b = np.array(pv[-1], float)
            d = b - a; n = np.linalg.norm(d)
            if n > 1e-6:
                d /= n
                cv2.line(view, tuple((a - d * 4000).astype(int)),
                         tuple((b + d * 4000).astype(int)), col, 1)
                cv2.line(view, tuple(a.astype(int)), tuple(b.astype(int)), col, 3)
                mid = ((a + b) / 2).astype(int)
                for c2, th in (((0, 0, 0), 5), (col, 2)):
                    cv2.putText(view, f"{e[0]},{e[1]}", tuple(mid),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, c2, th)
    pvr = draw_prediction(view)
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
    if state["mode"] == "line":
        ready = sum(1 for p in lines.values() if len(p) >= 2)
        bar = (f"LINE  [{state['edge'][0]},{state['edge'][1]}] {LINE_NAMES[state['edge']]}"
               f"   {len(lines.get(state['edge'], []))} pts on it, "
               f"{ready}/4 lines ready")
        if pvr is not None:
            bar += (f"   | GREEN = predicted field, valid to "
                    f"{pvr['usable_x_max']/100:.0f}m")
        elif ready >= 4:
            bar += "   | no preview: labels disagree, press f to see why"
        need = [LINE_NAMES[e].split('(')[0].strip() for e in ((1, 7), (2, 8))
                if len(lines.get(e, [])) < 2]
        if need:
            bar += "   | STILL NEED: " + ", ".join(need)
    else:
        bar = (f"CORNER  [{state['sel']}] {NAMES[state['sel']]}"
               f"   placed {sorted(corners)}  ({len(corners)}/4 needed)")
    cv2.rectangle(view, (0, WINH - 62), (WINW, WINH), (0, 0, 0), -1)
    cv2.putText(view, bar, (12, WINH - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(view, f"canvas ({cx:.0f},{cy:.0f})   zoom {z:.2f}x   "
                      f"m mode | asdghjk line | 1-8 corner | click place | "
                      f"z undo | F = FIT AND SAVE | q quit",
                (12, WINH - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 220, 255), 1)
    cv2.imshow("click corners", view)
    k = cv2.waitKey(20) & 0xFF
    if k in (ord('q'), 27):
        break
    elif k == ord('m'):
        state["mode"] = "line" if state["mode"] == "corner" else "corner"
        print(f"  mode: {state['mode'].upper()}")
    elif k in LINE_KEYS:
        state["mode"] = "line"; state["edge"] = LINE_KEYS[k]
        print(f"  placing points on: {LINE_NAMES[state['edge']]}")
    elif ord('1') <= k <= ord('8'):
        state["mode"] = "corner"; state["sel"] = k - ord('0')
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
    elif k == ord('z'):
        if state["mode"] == "line" and lines.get(state["edge"]):
            lines[state["edge"]].pop()
            print(f"  undo -- {len(lines[state['edge']])} left on this line")
        elif order:
            v = order.pop(); corners.pop(v, None); print(f"  undo vertex {v}")
    elif k == ord('c'):
        if state["mode"] == "line":
            lines.clear(); print("  cleared all lines")
        else:
            corners.clear(); order.clear(); print("  cleared corners")
    elif k in (ord('+'), ord('=')):
        state["zoom"] *= 1.25
    elif k in (ord('-'), ord('_')):
        state["zoom"] /= 1.25
    elif k == 81: state["cx"] -= 60 / z
    elif k == 83: state["cx"] += 60 / z
    elif k == 82: state["cy"] -= 60 / z
    elif k == 84: state["cy"] += 60 / z
    elif k == ord('v'):
        preview["on"] = not preview["on"]
    elif k in (ord('f'), ord('F')):
        state["force"] = (k == ord('F'))
        do_fit()
cv2.destroyAllWindows()
