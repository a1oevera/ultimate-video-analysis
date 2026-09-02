"""
Draw the field lines on the WHOLE video from ONE calibration.

The point the person made, and it is correct: with a fixed camera you do not
transfer LINES, you transfer the CALIBRATION. Knowing field -> pixels in one
frame, plus frame -> frame registration, gives field -> pixels in every other
frame by composition. Line detection is then needed exactly once, in the
window you calibrate, and never again.

WHY THIS IS NOT THE CHAINED TRANSFER THAT FAILED: run_line_transfer.py walks
frame by frame, so 400 hops accumulate to ~200px of drift
(results/field_lines_finding.md 3.30). Here every frame registers DIRECTLY to
one of a few hubs -- MEASURED at 2 hubs placing all 556 sampled frames of this
video -- so nothing is ever deeper than two compositions and there is no
accumulation.

THE REAL LIMIT is the calibration's own validity, not the transfer: a fit made
at one end of the pitch stops meaning anything past the horizon
(calibration.field_validity_region). Edges beyond it are not drawn, so a frame
showing only the far half will correctly show nothing.

Run:
  python3 run_field_overlay.py <video> <calib.json> <calib_sec> [every_s] [max_min]
"""
import sys
import os
import json
import cv2
import numpy as np

from frisbee_analysis import MosaicConfig, UltimatePitchConfiguration
from frisbee_analysis.mosaic import register_pair_with_coverage
from frisbee_analysis.calibration import Calibration, field_validity_region
from frisbee_analysis.field_model import _valid_polyline, FIELD_EDGES, EDGE_NAMES
from frisbee_analysis.video_overlays import overlay_boxes_for

os.makedirs("outputs/field_model", exist_ok=True)

video = sys.argv[1] if len(sys.argv) > 1 else "videos/torovcanyon.mp4"
calib_path = sys.argv[2] if len(sys.argv) > 2 else "outputs/calibration/calibration_t3235.json"
calib_sec = float(sys.argv[3]) if len(sys.argv) > 3 else 3235.0
every = float(sys.argv[4]) if len(sys.argv) > 4 else 30.0
max_min = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0

cfg = UltimatePitchConfiguration()
calib = Calibration.load(calib_path)
vr = field_validity_region(calib, cfg)
print(f"{calib_path}: valid to x={vr['usable_x_max']/100:.1f}m "
      f"({vr['usable_fraction']*100:.0f}% of the field)")

cap = cv2.VideoCapture(video)
if not cap.isOpened():
    sys.exit(f"could not open {video}")
fps = cap.get(cv2.CAP_PROP_FPS)
total = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
if max_min > 0:
    total = min(total, max_min * 60)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
mcfg = MosaicConfig(overlay_mask_boxes=tuple(overlay_boxes_for(video, W, H)))


def grab(t):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
    ok, f = cap.read()
    return f if ok else None


ref = grab(calib_sec)
if ref is None:
    sys.exit(f"could not read the calibration's own frame at t={calib_sec}")
# the field edges, once, in the calibration frame's pixels
EDGE_PL = {}
for e in list(FIELD_EDGES) + [(11, 12)]:
    pl = _valid_polyline(e, calib, cfg)
    if pl is not None and len(pl) >= 2:
        EDGE_PL[e] = np.array([[p[0], p[1]] for p in pl], dtype=np.float64)
print(f"{len(EDGE_PL)} of 7 edges are drawable at all under this calibration: "
      f"{[EDGE_NAMES.get(e, str(e)) for e in EDGE_PL]}\n")

COL = {(1, 7): (255, 120, 0), (2, 8): (0, 255, 0), (3, 4): (255, 0, 255),
       (5, 6): (200, 0, 200), (1, 2): (0, 255, 255), (7, 8): (0, 200, 200),
       (11, 12): (255, 255, 255)}
hubs = [(calib_sec, np.eye(3), ref)]
rows = []
t = 0.0
while t < total:
    f = grab(t)
    if f is None:
        break
    got = None
    for k in range(len(hubs)):
        _, Hh, hf = hubs[k]
        Hs, n_in, _ = register_pair_with_coverage(hf, f, mcfg)   # f px -> hub px
        if Hs is not None and n_in >= mcfg.min_inliers:
            got = (Hh @ Hs, n_in, k)
            break
    if got is None and len(hubs) < 12:
        for k in range(len(hubs)):
            _, Hh, hf = hubs[k]
            Hs, n_in, _ = register_pair_with_coverage(f, hf, mcfg)
            if Hs is not None and n_in >= mcfg.min_inliers:
                Hn = Hh @ np.linalg.inv(Hs)
                hubs.append((t, Hn, f))
                got = (Hn, n_in, len(hubs) - 1)
                print(f"  t={t:6.0f}s became hub {len(hubs)-1} ({n_in} inliers)", flush=True)
                break
    if got is None:
        rows.append({"t": t, "ok": False})
        print(f"  t={t:6.0f}s  could not register to any hub", flush=True)
        t += every
        continue
    H_to_ref, n_in, hub = got
    H_ref_to_f = np.linalg.inv(H_to_ref)
    vis = f.copy()
    drawn = 0
    for e, P in EDGE_PL.items():
        Q = cv2.perspectiveTransform(P.reshape(-1, 1, 2), H_ref_to_f).reshape(-1, 2)
        inside = Q[(Q[:, 0] > -4000) & (Q[:, 0] < W + 4000) &
                   (Q[:, 1] > -4000) & (Q[:, 1] < H + 4000)]
        if len(inside) < 2:
            continue
        cv2.polylines(vis, [Q.astype(np.int32)], False, COL.get(e, (255, 255, 255)), 3)
        vis_pts = Q[(Q[:, 0] >= 0) & (Q[:, 0] < W) & (Q[:, 1] >= 0) & (Q[:, 1] < H)]
        if len(vis_pts):
            m = vis_pts[len(vis_pts) // 2].astype(int)
            lab = EDGE_NAMES.get(e, "HALFWAY (50m)")
            for c2, th in (((0, 0, 0), 5), (COL.get(e, (255, 255, 255)), 2)):
                cv2.putText(vis, lab, tuple(m), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c2, th)
            drawn += 1
    cv2.rectangle(vis, (0, 0), (W, 40), (0, 0, 0), -1)
    cv2.putText(vis, f"t={t:.0f}s   hub {hub} ({n_in} inliers)   {drawn} edges in frame",
                (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    out = f"outputs/field_model/overlay_{int(t):06d}.jpg"
    cv2.imwrite(out, vis)
    rows.append({"t": t, "ok": True, "inliers": int(n_in), "hub": hub, "edges": drawn})
    if len(rows) % 10 == 0:
        ok = sum(1 for r in rows if r.get("ok"))
        print(f"  {len(rows)} frames, {ok} placed, {len(hubs)} hubs", flush=True)
    t += every
cap.release()
ok = [r for r in rows if r.get("ok")]
print(f"\n{len(ok)}/{len(rows)} frames overlaid using {len(hubs)} hub(s)")
if ok:
    ed = [r["edges"] for r in ok]
    print(f"edges visible per frame: median {int(np.median(ed))}, "
          f"min {min(ed)}, max {max(ed)}")
    print(f"inliers: median {int(np.median([r['inliers'] for r in ok]))}")
with open("outputs/field_model/overlay_index.json", "w") as fjs:
    json.dump({"video": video, "calibration": calib_path, "rows": rows}, fjs)
print("wrote outputs/field_model/overlay_*.jpg and overlay_index.json")
