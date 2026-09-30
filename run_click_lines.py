"""Click points along field lines on ONE video frame, and save them.

Built because automatic line finding has failed repeatedly on this footage: the
far sideline is painted only inside the endzone, and every detector run has either
missed it or picked a different line. A few clicked points beat all of it.

Key layout deliberately keeps command keys clear of selection keys -- an earlier
tool bound 'q' to both quit and a line, and 'u' to both undo and a line, so both
were dead. Selection is on the number row only; commands are letters.

Run:
    python3 run_click_lines.py videos/torovcanyon.mp4 4200
    python3 run_click_lines.py outputs/diagnostics/frame_4200.jpg
    python3 run_click_lines.py videos/torovcanyon.mp4 4200 --out mypoints.json
"""
import sys
import os
import json
import numpy as np
import cv2

LINES = [
    ("near sideline", "the blue painted one nearest the camera", (255, 120, 0)),
    ("far sideline", "painted in the endzone only; extend it", (0, 220, 255)),
    ("goal line", "endzone line, 18m from the end", (0, 255, 0)),
    ("end line", "the very back of the endzone", (0, 160, 0)),
    ("halfway", "midfield, 50m", (255, 0, 255)),
    ("soccer centre circle", "click along the painted arc", (0, 0, 255)),
]

args = [a for a in sys.argv[1:] if not a.startswith("--")]
out_path = "outputs/diagnostics/clicked_lines.json"
if "--out" in sys.argv:
    out_path = sys.argv[sys.argv.index("--out") + 1]
if not args:
    sys.exit(__doc__)

src = args[0]
if src.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
    t = float(args[1]) if len(args) > 1 else 0.0
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"could not read {src} at {t}s")
    tag = f"{os.path.basename(src)} t={t:g}s"
else:
    frame = cv2.imread(src)
    if frame is None:
        sys.exit(f"could not read {src}")
    tag = os.path.basename(src)

FH, FW = frame.shape[:2]
WINW, WINH = 1500, 850
state = {"zoom": min(WINW / FW, WINH / FH), "cx": FW / 2.0, "cy": FH / 2.0,
         "sel": 0, "mouse": (0, 0), "drag": None}
pts = {i: [] for i in range(len(LINES))}


def view_to_frame(mx, my):
    z = state["zoom"]
    return (state["cx"] + (mx - WINW / 2) / z, state["cy"] + (my - WINH / 2) / z)


def frame_to_view(x, y):
    z = state["zoom"]
    return (int(WINW / 2 + (x - state["cx"]) * z), int(WINH / 2 + (y - state["cy"]) * z))


def on_mouse(ev, x, y, flags, _):
    state["mouse"] = (x, y)
    if ev == cv2.EVENT_LBUTTONDOWN:
        fx, fy = view_to_frame(x, y)
        if 0 <= fx < FW and 0 <= fy < FH:
            pts[state["sel"]].append((float(fx), float(fy)))
            print(f"  {LINES[state['sel']][0]}: point {len(pts[state['sel']])} "
                  f"at ({fx:.0f},{fy:.0f})")
    elif ev == cv2.EVENT_RBUTTONDOWN:
        state["drag"] = (x, y, state["cx"], state["cy"])
    elif ev == cv2.EVENT_RBUTTONUP:
        state["drag"] = None
    elif ev == cv2.EVENT_MOUSEMOVE and state["drag"]:
        x0, y0, cx0, cy0 = state["drag"]
        state["cx"] = cx0 - (x - x0) / state["zoom"]
        state["cy"] = cy0 - (y - y0) / state["zoom"]


def save():
    """Merge into any existing file for the SAME frame rather than replacing it.
    The first version overwrote, so clicking lines in a second session silently
    destroyed circle points clicked in the first."""
    existing = {}
    if os.path.exists(out_path):
        try:
            old = json.load(open(out_path))
            if old.get("source") == tag:
                existing = {L["name"]: L["points"] for L in old.get("lines", [])}
        except Exception:
            pass
    for i in range(len(LINES)):
        if pts[i]:
            existing[LINES[i][0]] = pts[i]      # this session wins for lines touched
    data = {"source": tag, "frame_size": [FW, FH],
            "lines": [{"name": k, "points": v} for k, v in existing.items()]}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nsaved {out_path} (merged with anything already there for this frame)")
    for L in data["lines"]:
        print(f"  {L['name']}: {len(L['points'])} points")


cv2.namedWindow("click lines", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("click lines", on_mouse)
print(f"\n{tag}  ({FW}x{FH})\n")
for i, (nm, hint, _) in enumerate(LINES):
    print(f"  {i+1}  {nm:22s} {hint}")
print("\nclick = add a point (2+ per line)   z = undo   c = clear this line")
print("+/- = zoom    arrows or right-drag = pan")
print("\n*** press  s  to SAVE, then q to quit ***\n")

while True:
    z = state["zoom"]
    M = np.array([[z, 0, WINW / 2 - state["cx"] * z],
                  [0, z, WINH / 2 - state["cy"] * z]])
    view = cv2.warpAffine(frame, M, (WINW, WINH))
    for i, (nm, _, col) in enumerate(LINES):
        P = pts[i]
        if not P:
            continue
        vp = [frame_to_view(*p) for p in P]
        for q in vp:
            cv2.circle(view, q, 5, col, -1)
            cv2.circle(view, q, 5, (0, 0, 0), 1)
        if len(P) >= 2:
            A = np.asarray(P, float)
            c = A.mean(0)
            _, _, V = np.linalg.svd(A - c)
            d = V[0]
            a = frame_to_view(*(c - d * 4000))
            b = frame_to_view(*(c + d * 4000))
            cv2.line(view, a, b, col, 1)           # extended: check it follows the paint
            cv2.line(view, vp[0], vp[-1], col, 3)
            mid = vp[len(vp) // 2]
            for cc, th in (((0, 0, 0), 5), (col, 2)):
                cv2.putText(view, nm, (mid[0] + 8, mid[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, cc, th)
    mx, my = state["mouse"]
    fx, fy = view_to_frame(mx, my)
    R = 55
    x0, y0 = int(fx) - R, int(fy) - R
    patch = np.zeros((2 * R, 2 * R, 3), np.uint8)
    xs0, ys0 = max(0, x0), max(0, y0)
    xs1, ys1 = min(FW, x0 + 2 * R), min(FH, y0 + 2 * R)
    if xs1 > xs0 and ys1 > ys0:
        patch[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0] = frame[ys0:ys1, xs0:xs1]
    loupe = cv2.resize(patch, (220, 220), interpolation=cv2.INTER_NEAREST)
    cv2.line(loupe, (110, 92), (110, 128), (0, 0, 255), 1)
    cv2.line(loupe, (92, 110), (128, 110), (0, 0, 255), 1)
    cv2.rectangle(loupe, (0, 0), (219, 219), (255, 255, 255), 1)
    view[10:230, WINW - 230:WINW - 10] = loupe
    nm, hint, col = LINES[state["sel"]]
    cv2.rectangle(view, (0, WINH - 58), (WINW, WINH), (0, 0, 0), -1)
    cv2.putText(view, f"[{state['sel']+1}] {nm} -- {hint}   ({len(pts[state['sel']])} pts)",
                (12, WINH - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2)
    cv2.putText(view, f"({fx:.0f},{fy:.0f})  zoom {z:.2f}x   "
                      f"1-6 pick line | click add | z undo | c clear | s SAVE | q quit",
                (12, WINH - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)
    cv2.imshow("click lines", view)
    k = cv2.waitKey(20) & 0xFF
    if k in (ord('q'), 27):
        break
    elif ord('1') <= k <= ord('6'):
        state["sel"] = k - ord('1')
    elif k == ord('z'):
        if pts[state["sel"]]:
            pts[state["sel"]].pop()
    elif k == ord('c'):
        pts[state["sel"]] = []
    elif k in (ord('+'), ord('=')):
        state["zoom"] *= 1.25
    elif k in (ord('-'), ord('_')):
        state["zoom"] /= 1.25
    elif k == 81:
        state["cx"] -= 60 / z
    elif k == 83:
        state["cx"] += 60 / z
    elif k == 82:
        state["cy"] -= 60 / z
    elif k == 84:
        state["cy"] += 60 / z
    elif k == ord('s'):
        save()
cv2.destroyAllWindows()
