"""
Carry human-labelled field lines forward and backward through the video,
frame by frame, and report how well the transfer holds up.

WHY: labelling a line is the expensive step and the only one a person can do
reliably (results/field_lines_finding.md 3.10-3.29). The geometry of where a
line IS can be re-measured in every window by the detector; what cannot be
re-derived is WHICH line it is. Registration between frames carries that
labelling. MEASURED (3.30): three of four labels transferred from t=3600 to
t=3235 landed within 3-13px of lines that window had detected independently.

The transfer is used to IDENTIFY, not to measure. Accuracy degrades with
distance from the overlap -- a single hop measured 40px mean / 72px max
disagreement between routes -- which is fine for saying "that is the goal
line" and far too loose to fit a calibration from. So each step re-detects
lines in the new frame and reports whether the transferred label lands on one.

Run:
  python3 run_line_transfer.py <video> <source_sec> [step_sec] [max_steps] [direction]
     direction: -1 backward, +1 forward, 0 both (default)
"""
import sys
import os
import json
import cv2
import numpy as np

from frisbee_analysis import MosaicConfig
from frisbee_analysis.mosaic import register_pair_with_coverage
from frisbee_analysis.field_lines import detect_line_segments, LineConfig
from frisbee_analysis.video_overlays import overlay_boxes_for

# outputs are filed into folders; make sure ours exist
os.makedirs("outputs/transfer", exist_ok=True)

video = sys.argv[1] if len(sys.argv) > 1 else "videos/torovcanyon.mp4"
source_sec = float(sys.argv[2]) if len(sys.argv) > 2 else 3600.0
step_sec = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
max_steps = int(sys.argv[4]) if len(sys.argv) > 4 else 60
direction = int(sys.argv[5]) if len(sys.argv) > 5 else 0

# The lines the person confirmed, in the SOURCE frame's pixel coordinates.
SOURCE_LINES = {
    "near sideline": ((-22.0, 403.0), (640.0, 1076.0)),
    "goal line":     ((1191.0, 375.0), (-35.0, 501.0)),
    "far sideline":  ((1128.0, 375.0), (890.0, 353.0)),
}
_p = np.array([546.0, 377.0])
_v = np.array([np.cos(np.radians(178.11)), np.sin(np.radians(178.11))])
SOURCE_LINES["end line"] = (tuple(_p - _v * 600), tuple(_p + _v * 600))

cap = cv2.VideoCapture(video)
if not cap.isOpened():
    sys.exit(f"could not open {video}")
fps = cap.get(cv2.CAP_PROP_FPS)
total_sec = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
boxes = overlay_boxes_for(video, W, H)
mcfg = MosaicConfig(overlay_mask_boxes=tuple(boxes))
lcfg = LineConfig()


def grab(t):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
    ok, f = cap.read()
    return f if ok else None


def ang(p1, p2):
    return np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0])) % 180


def homogeneous(p1, p2):
    l = np.cross([p1[0], p1[1], 1.0], [p2[0], p2[1], 1.0])
    n = np.hypot(l[0], l[1])
    return l / n if n > 1e-9 else l


def agreement(line, segments, max_deg=12.0):
    """Nearest detected segment to a transferred line: (offset_px, deg, support)."""
    p1, p2 = line
    a = ang(p1, p2)
    mid = (np.asarray(p1) + np.asarray(p2)) / 2.0
    best = None
    for s in segments:
        da = abs(ang(s.p1, s.p2) - a)
        da = min(da, 180 - da)
        if da > max_deg:
            continue
        h = homogeneous(s.p1, s.p2)
        off = abs(h[0] * mid[0] + h[1] * mid[1] + h[2])
        if best is None or off < best[0]:
            best = (off, da, s.support_px)
    return best


def walk(sign):
    """Step away from the source, composing registrations as we go."""
    src = grab(source_sec)
    if src is None:
        return []
    H_acc = np.eye(3)          # source px -> current frame px
    prev = src
    prev_t = source_sec
    rows = []
    for k in range(1, max_steps + 1):
        t = source_sec + sign * step_sec * k
        if t < 0 or t > total_sec - 1:
            break
        cur = grab(t)
        if cur is None:
            break
        Hs, n_in, cov = register_pair_with_coverage(cur, prev, mcfg)
        if Hs is None or n_in < mcfg.min_inliers:
            # A single unregisterable frame is a local problem -- motion blur, a
            # player filling the frame, a hard cut -- not the end of the camera's
            # continuity. MEASURED: the first backward walk covered 161 steps and
            # then stopped dead at ONE failure, 1050s short of the framing that
            # shows the other endzone, which links fine to almost every other
            # part of the video. Stepping over the bad frame recovers the chain.
            bridged = False
            for skip in (2, 3, 4, 6, 8):
                t2 = source_sec + sign * step_sec * (k + skip - 1)
                if t2 < 0 or t2 > total_sec - 1:
                    break
                alt = grab(t2)
                if alt is None:
                    continue
                Hs2, n2, cov2 = register_pair_with_coverage(alt, prev, mcfg)
                if Hs2 is not None and n2 >= mcfg.min_inliers:
                    print(f"  t={t:7.0f}s step {k:3d}: {n_in} inliers -- SKIPPED "
                          f"{skip - 1} frame(s), resumed at t={t2:.0f}s ({n2} inliers)",
                          flush=True)
                    cur, Hs, n_in, cov, t = alt, Hs2, n2, cov2, t2
                    bridged = True
                    break
            if not bridged:
                rows.append({"t": t, "step": k, "ok": False, "inliers": int(n_in or 0),
                             "coverage": float(cov or 0.0)})
                print(f"  t={t:7.0f}s step {k:3d}: REGISTRATION FAILED ({n_in} inliers) "
                      f"and no bridge within 8 frames -- chain ends here", flush=True)
                break
        H_acc = Hs @ H_acc
        segs = detect_line_segments(cur, None, boxes, lcfg, None, frame_idx=k)
        row = {"t": t, "step": k, "ok": True, "inliers": int(n_in),
               "coverage": float(cov), "n_segments": len(segs), "lines": {}}
        parts = []
        for name, (p1, p2) in SOURCE_LINES.items():
            q = cv2.perspectiveTransform(
                np.array([[p1], [p2]], dtype=np.float64), H_acc).reshape(-1, 2)
            row["lines"][name] = {"p1": q[0].tolist(), "p2": q[1].tolist()}
            m = agreement((q[0], q[1]), segs)
            if m is None:
                row["lines"][name]["match"] = None
                parts.append(f"{name.split()[0]:5s} --")
            else:
                row["lines"][name]["match"] = {"offset_px": m[0], "deg": m[1], "support": m[2]}
                parts.append(f"{name.split()[0]:5s}{m[0]:5.0f}px")
        rows.append(row)
        print(f"  t={t:7.0f}s step {k:3d}: {n_in:4d} inliers cov {cov:.2f}, "
              f"{len(segs):3d} segs | " + "  ".join(parts), flush=True)
        prev = cur
    return rows


print(f"source t={source_sec:.0f}s, step {step_sec:g}s, up to {max_steps} steps each way")
print(f"transferring {len(SOURCE_LINES)} human-labelled lines; the number after each")
print(f"name is how far the transferred line sits from the nearest line THAT FRAME")
print(f"detected by itself (small = the label still points at a real marking)\n")
out = {"video": video, "source_sec": source_sec, "step_sec": step_sec,
        "source_lines": {k: [list(v[0]), list(v[1])] for k, v in SOURCE_LINES.items()}}
for sign, label in (((-1), "BACKWARD"), ((1), "FORWARD")):
    if direction != 0 and direction != sign:
        continue
    print(f"--- {label} from t={source_sec:.0f}s")
    out[label.lower()] = walk(sign)
cap.release()
os.makedirs("outputs", exist_ok=True)
with open("outputs/transfer/line_transfer.json", "w") as f:
    json.dump(out, f)
print("\nwrote outputs/transfer/line_transfer.json")
