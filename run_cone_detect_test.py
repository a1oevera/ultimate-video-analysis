"""
Cheap feasibility test: can plain HSV color-filtering find bright orange
field cones on real footage, with NO trained model? Much simpler problem
than the earlier failed line-detection attempt (results/calibration_finding.md)
-- a cone is a small, consistently-colored, compact blob, not an ambiguous
edge/line easily confused with treelines or broadcast overlays.

Scans several frames spread across the video, flags any frame with an
orange blob passing basic size/shape sanity checks (not too big/small, not
wildly elongated -- a cone silhouette is compact), and saves one annotated
example so a candidate can be checked by eye, not just trusted as a count.

*** WRONG-FIELD WARNING (same as calibration_finding.md): *** this venue
has multiple simultaneous games in the background of many wide shots. A
detected orange blob might belong to an ADJACENT field, not the one being
broadcast -- this script only checks "is there an orange blob here", not
which field it's on. Needs a human look before trusting any hit.

Run:  python run_cone_detect_test.py <video_path> [n_samples] [out.jpg]
Needs opencv-python.
"""
import sys
import cv2
import numpy as np

video_path = sys.argv[1] if len(sys.argv) > 1 else "videos/videoplayback.mp4"
n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 20
out_path = sys.argv[3] if len(sys.argv) > 3 else "outputs/cone_detect_sample.jpg"

# WFDF/broadcast cones are typically a saturated orange -- wide-ish hue band
# (some cones lean more red-orange, some more yellow-orange) but always high
# saturation and reasonably bright, unlike grass (low sat, greenish hue) or
# most jerseys/background (rarely this specific hue+sat combo together).
LOWER_ORANGE = np.array([5, 140, 120])
UPPER_ORANGE = np.array([22, 255, 255])

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    sys.exit(f"could not open {video_path}")
fps = cap.get(cv2.CAP_PROP_FPS)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration_s = n_frames / fps
print(f"{video_path}: {duration_s/60:.1f} min, scanning {n_samples} frames spread across it")

best_frame = None
best_boxes = []
best_t = None
hits = []

for i in range(n_samples):
    t_sec = duration_s * (i + 0.5) / n_samples
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_sec * fps))
    ok, frame = cap.read()
    if not ok:
        continue
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frame_area = frame.shape[0] * frame.shape[1]
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 8 or area / frame_area > 0.002:  # too tiny (noise) or too big (not a cone)
            continue
        x, y, w, h = cv2.boundingRect(c)
        extent = area / (w * h)  # how much of the bounding box the blob fills
        aspect = max(w, h) / max(1, min(w, h))
        if extent < 0.35 or aspect > 3.0:  # too sparse/elongated to be a compact cone shape
            continue
        boxes.append((x, y, w, h))

    if boxes:
        hits.append((t_sec, len(boxes)))
        if len(boxes) > len(best_boxes):
            best_frame, best_boxes, best_t = frame.copy(), boxes, t_sec

cap.release()

print(f"\nFrames with candidate orange blobs: {len(hits)}/{n_samples}")
for t, n in hits:
    print(f"  t={int(t//60)}:{t%60:05.2f}  {n} candidate(s)")

if best_frame is not None:
    vis = best_frame.copy()
    for x, y, w, h in best_boxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite(out_path, vis)
    print(f"\nBest frame (t={int(best_t//60)}:{best_t%60:05.2f}, {len(best_boxes)} candidates) "
          f"saved to {out_path} -- CHECK BY EYE before trusting: could be jerseys, "
          f"skin tone, signage, or a cone on the wrong field.")
else:
    print("\nNo candidate orange blobs found in any sampled frame.")
