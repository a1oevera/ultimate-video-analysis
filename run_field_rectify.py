"""
Warp the video into the FIELD, not into another frame: a top-down 100x37m
canvas showing which parts of the pitch were ever actually observed.

WHY THIS AND NOT A MOSAIC. An image-space mosaic stitches frame->frame, so
error compounds and there is no ground truth to check the result against --
four attempts at one (results/field_lines_finding.md 3.31) produced no usable
image and no way to tell misregistration from honest perspective. The pitch,
though, is PLANAR, so every frame maps to it by an exact homography, which is
exactly what a calibration is. Rectifying gives:

  - no chaining: each frame maps straight to the field, errors do not accumulate
  - a canvas with ground truth: the sidelines MUST land at y=0 and y=37m, the
    goal lines at x=18 and 82m. A line in the wrong place is visible instantly.
  - a direct answer to "how much of the field do we ever see", in metres

WHY SEVERAL CALIBRATIONS COMBINE FOR FREE. A calibration is LOCAL -- fitted
from evidence at one end, it can represent that end and not the other
(MEASURED: cal_sub2 sees x=0..44m, calibration_corners sees x=14..100m).
Neither covers the pitch. But both rectify into the SAME field coordinates,
in metres, so combining them is a per-pixel merge and nothing else: no
stitching, no feature matching between them, no error to accumulate. That is
the whole payoff of rectifying to the plane rather than mosaicking. Better
still, where two calibrations overlap they can be CHECKED against each other
-- they were fitted from different frames at different times, so agreement
there is real evidence, and this tool measures it (see the paint-offset
figure it prints).

HOW A FRAME GETS ITS CALIBRATION. Each calibration owns a family of hubs,
seeded by its own reference frame. Every sampled frame is registered against
every family's hubs and goes to whichever family matches it best -- which is
also how a frame gets assigned to the end of the field it is actually
looking at. Registration is direct-to-hub within a family, never chained.

Run:
  python3 run_field_rectify.py <video> <calib.json>@<ref_sec> [<calib2.json>@<ref_sec2> ...]
                               [--every S] [--ppm P] [--out NAME]
  python3 run_field_rectify.py <video> <calib.json> <ref_sec> [every_s] [px_per_m]   (single, legacy form)

Example -- both endzones of torovcanyon into one canvas:
  python3 run_field_rectify.py videos/torovcanyon.mp4 \
      outputs/calibration/cal_sub2.json@3600 \
      outputs/calibration/calibration_corners.json@1700
"""
import sys
import os
import cv2
import numpy as np

from frisbee_analysis import MosaicConfig, UltimatePitchConfiguration
from frisbee_analysis.mosaic import register_pair_with_coverage
from frisbee_analysis.calibration import Calibration, field_visibility
from frisbee_analysis.video_overlays import overlay_boxes_for

os.makedirs("outputs/mosaic", exist_ok=True)

# ---------------------------------------------------------------- arguments
argv = sys.argv[1:]
every, ppm, out_name = 10.0, 12.0, "field_rectified"
positional = []
i = 0
while i < len(argv):
    a = argv[i]
    if a == "--every":
        every = float(argv[i + 1]); i += 2
    elif a == "--ppm":
        ppm = float(argv[i + 1]); i += 2
    elif a == "--out":
        out_name = argv[i + 1]; i += 2
    else:
        positional.append(a); i += 1

video = positional[0] if positional else "videos/torovcanyon.mp4"
specs = []                                  # [(calib_path, ref_sec)]
rest = positional[1:]
if any("@" in a for a in rest):
    for a in rest:
        if "@" not in a:
            sys.exit(f"expected <calib.json>@<ref_sec>, got '{a}'")
        p, _, s = a.rpartition("@")
        specs.append((p, float(s)))
else:                                       # legacy positional form
    p = rest[0] if rest else "outputs/calibration/calibration_t3235.json"
    s = float(rest[1]) if len(rest) > 1 else 3235.0
    specs.append((p, s))
    if len(rest) > 2:
        every = float(rest[2])
    if len(rest) > 3:
        ppm = float(rest[3])

cfg = UltimatePitchConfiguration()

cap = cv2.VideoCapture(video)
if not cap.isOpened():
    sys.exit(f"could not open {video}")
fps = cap.get(cv2.CAP_PROP_FPS)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
mcfg = MosaicConfig(overlay_mask_boxes=tuple(overlay_boxes_for(video, W, H)))


def grab(t):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
    ok, f = cap.read()
    return f if ok else None


# ------------------------------------------------------- the shared canvas
# field (cm) -> canvas px, with a margin so out-of-bounds play is visible.
# Every calibration paints into THIS, which is what makes them combinable.
MARGIN_M = 8.0
OW = int((cfg.length / 100.0 + 2 * MARGIN_M) * ppm)
OH = int((cfg.width / 100.0 + 2 * MARGIN_M) * ppm)
F2C = np.array([[ppm / 100.0, 0, MARGIN_M * ppm],
                [0, ppm / 100.0, MARGIN_M * ppm],
                [0, 0, 1.0]])
gx, gy = np.meshgrid(np.arange(OW), np.arange(OH))
fx_cm = (gx - MARGIN_M * ppm) * 100.0 / ppm
fy_cm = (gy - MARGIN_M * ppm) * 100.0 / ppm
in_padded_field = (fx_cm >= -MARGIN_M * 100) & (fx_cm <= cfg.length + MARGIN_M * 100) & \
                  (fy_cm >= -MARGIN_M * 100) & (fy_cm <= cfg.width + MARGIN_M * 100)
print(f"canvas {OW}x{OH} at {ppm:g} px/m ({MARGIN_M:g}m margin all round)")

# one distinct hue per calibration, for the provenance map
FAMILY_COLOURS = [(90, 200, 90), (200, 140, 60), (90, 110, 220), (200, 90, 200)]


class Family:
    """One calibration, its hub set, and the layer it paints."""

    def __init__(self, path, ref_sec, colour):
        self.path, self.ref_sec, self.colour = path, ref_sec, colour
        self.calib = Calibration.load(path)
        ref = grab(ref_sec)
        if ref is None:
            sys.exit(f"could not read {path}'s reference frame at t={ref_sec}")
        self.hubs = [(ref_sec, np.eye(3), ref)]        # (time, H_hub_to_ref, frame)
        self.last_hub = 0
        self.H_field_to_ref = np.linalg.inv(np.asarray(self.calib.H()))
        # WHERE THIS CALIBRATION MAY PAINT. Not `x <= usable_x_max`: that
        # scalar assumes the camera stands at the x=0 end, and masks a
        # far-endzone calibration onto the near end (MEASURED,
        # calibration_corners.json: usable_x_max=11.7m for a fit anchored on
        # vertices 5-8 at x=82..100m). field_visibility is side-agnostic --
        # it asks which side of the horizon the calibration's own evidence
        # is on -- which is what lets two opposite-end calibrations coexist
        # on one canvas.
        self.visible = field_visibility(self.calib, fx_cm, fy_cm, cfg) & in_padded_field
        self.canvas = np.zeros((OH, OW, 3), np.uint8)
        self.best_q = np.zeros((OH, OW), np.float32)
        self.placed = 0
        xs = fx_cm[self.visible]
        print(f"calibration {os.path.basename(path)} (ref t={ref_sec:.0f}s)")
        print(f"  can represent field x {xs.min()/100:.0f}..{xs.max()/100:.0f}m "
              f"-- {100.0*self.visible.mean():.0f}% of the canvas; nothing is painted outside it")


families = [Family(p, s, FAMILY_COLOURS[k % len(FAMILY_COLOURS)])
            for k, (p, s) in enumerate(specs)]

overlap = np.ones((OH, OW), bool)
for fam in families:
    overlap &= fam.visible
if len(families) > 1:
    xs = fx_cm[overlap]
    if xs.size:
        print(f"\nthe calibrations overlap over field x {xs.min()/100:.0f}..{xs.max()/100:.0f}m "
              f"-- that overlap is a free cross-check, measured at the end")
    else:
        print("\nthe calibrations do not overlap anywhere -- no cross-check is possible")

# MEASURED, and the reason the first version of this tool drew diagonal
# sidelines: registration was gated on inlier COUNT alone, discarding the
# coverage that register_pair_with_coverage returns precisely so it can be
# gated on. Replaying this loop over torovcanyon.mp4 and recording the
# discarded value: 135 of 163 accepted frames (83%) had coverage below
# MosaicConfig.min_coverage=0.15, median 0.096 -- inliers packed into under
# a tenth of the frame. Some were egregious (t=5190s: 233 inliers, coverage
# 0.027), which is exactly the high-count/no-spread failure the docstring on
# register_pair_with_coverage warns about. A homography fit to one clustered
# patch is fine at that patch and sheared everywhere else, and once composed
# with the calibration that shear IS the tilt: white paint that should run
# along the canvas axes came out up to 11 deg off (26 Hough segments over the
# painted canvas, 10 of them >5 deg), against 1.3-2.6 deg for the reference
# frame warped through the calibration alone. So the calibration was never
# the problem; the frames stacked on top of it were.
STRONG_COVERAGE = 0.30    # ~2x the min_coverage bar; nothing beats this by enough to be worth the search


def best_hub_in(fam, frame):
    """Keep the BEST hub match in this family, not the first one that clears
    the bar. The old order (nearest hub index to the last one used) is not a
    quality ordering at all, so with a bar as low as inliers>=15 the first
    acceptable hub routinely beat a far better one that was never tried.

    Searching every hub is quadratic once hubs accumulate, so try the last
    hub that worked first -- consecutive samples usually share a shot -- and
    stop early on a match well clear of the bar. A frame that only ever
    finds marginal matches still gets compared against all of them."""
    best = None
    for k in sorted(range(len(fam.hubs)), key=lambda k: k != fam.last_hub):
        _, Hh, hf = fam.hubs[k]
        Hs, n_in, cov = register_pair_with_coverage(hf, frame, mcfg)
        if Hs is None or n_in < mcfg.min_inliers or cov < mcfg.min_coverage:
            continue
        if best is None or cov > best[1]:
            best = (Hh @ Hs, cov, k)
        if cov >= STRONG_COVERAGE:
            break
    return best


def promote_into(fam, t, frame):
    """Let a frame become a hub if it reaches an existing one the other way
    round; keeps depth at 2 within the family, never a chain."""
    for k, (_, Hh, hf) in enumerate(fam.hubs):
        Hs, n_in, cov = register_pair_with_coverage(frame, hf, mcfg)
        if Hs is not None and n_in >= mcfg.min_inliers and cov >= mcfg.min_coverage:
            Hnew = Hh @ np.linalg.inv(Hs)
            fam.hubs.append((t, Hnew, frame))
            return (Hnew, cov, len(fam.hubs) - 1)
    return None


times = list(np.arange(0, n_frames / fps, every))
skipped = 0
print(f"\n{len(times)} sample times every {every:g}s\n", flush=True)

for n, t in enumerate(times):
    f = grab(t)
    if f is None:
        continue
    # every family bids; the frame goes to whichever one registers it best,
    # which is also how a frame finds the end of the field it is looking at
    bids = [(fam, best_hub_in(fam, f)) for fam in families]
    bids = [(fam, b) for fam, b in bids if b is not None]
    if not bids:
        for fam in families:
            b = promote_into(fam, t, f)
            if b is not None:
                bids = [(fam, b)]
                print(f"  t={t:.0f}s became hub {len(fam.hubs)-1} of "
                      f"{os.path.basename(fam.path)} (coverage {b[1]:.2f})", flush=True)
                break
    if not bids:
        skipped += 1
        continue
    # A frame paints into EVERY family that registered it, not only the best
    # one. Where two calibrations can both see the same ground, that is what
    # gets it painted twice -- once through each calibration, from the same
    # frame -- and two independent placements of one patch of ground is
    # exactly the cross-check the overlap is for. Merging still picks a
    # single owner per pixel afterwards, so the combined image is unaffected;
    # without this the layers tile instead of overlapping and there is
    # nothing to compare.
    for fam, (H_frame_to_ref, quality, hub_k) in bids:
        fam.last_hub = hub_k
        # field -> canvas needs field -> ref -> frame
        H_field_to_frame = np.linalg.inv(H_frame_to_ref) @ fam.H_field_to_ref
        H_canvas_to_frame = H_field_to_frame @ np.linalg.inv(F2C)
        warped = cv2.warpPerspective(f, H_canvas_to_frame, (OW, OH), flags=cv2.WARP_INVERSE_MAP)
        cover = cv2.warpPerspective(np.full((H, W), 255, np.uint8), H_canvas_to_frame,
                                    (OW, OH), flags=cv2.WARP_INVERSE_MAP)
        # First-come-first-served painting (the old `filled == 0`) handed the
        # canvas to whichever frame arrived first, and times start at t=0 --
        # the frames FURTHEST from the calibration reference, so the least
        # trustworthy registrations claimed the canvas and the reference
        # frame itself got only the leftovers. Let a better-covered
        # registration overwrite a worse one instead.
        paint = (cover > 0) & fam.visible & (quality > fam.best_q)
        fam.canvas[paint] = warped[paint]
        fam.best_q[paint] = quality
        fam.placed += 1
    if n % 25 == 0:
        done = np.zeros((OH, OW), bool)
        for g in families:
            done |= g.best_q > 0
        print(f"  {n}/{len(times)} t={t:6.0f}s  "
              + ", ".join(f"{g.placed} in {os.path.basename(g.path)[:18]}" for g in families)
              + f", {skipped} unregistered, {100.0*done.mean():.0f}% of canvas painted", flush=True)
cap.release()

# --------------------------------------------------------------- combining
# Both layers are already in field coordinates, so this is a per-pixel merge
# and nothing else. Each pixel goes to the calibration that placed the
# best-covered frame there.
canvas = np.zeros((OH, OW, 3), np.uint8)
sources = np.zeros((OH, OW, 3), np.uint8)
owner_q = np.zeros((OH, OW), np.float32)
for fam in families:
    take = fam.best_q > owner_q
    canvas[take] = fam.canvas[take]
    sources[take] = fam.colour
    owner_q[take] = fam.best_q[take]
filled = owner_q > 0


def white_paint(img):
    b, g, r = (img[:, :, k].astype(int) for k in range(3))
    return ((b > 140) & (g > 140) & (r > 140) &
            (abs(b - g) < 45) & (abs(g - r) < 45)).astype(np.float32)


# THE CROSS-CHECK. Where two calibrations both painted, they are describing
# the same physical ground from different frames at different times. A
# photometric difference would mostly measure the lighting and the players,
# so compare the FIELD PAINT instead: cross-correlate each layer's white-line
# mask over the overlap and read off the peak offset. Perfect agreement puts
# the peak at zero; anything else is the two calibrations' disagreement about
# where the lines are, in metres of real field.
if len(families) > 1:
    print("\ncross-check in the overlap (both calibrations painted the same ground):")
    for a in range(len(families)):
        for b in range(a + 1, len(families)):
            fa, fb = families[a], families[b]
            both = (fa.best_q > 0) & (fb.best_q > 0)
            if both.sum() < 500:
                print(f"  {os.path.basename(fa.path)} vs {os.path.basename(fb.path)}: "
                      f"only {int(both.sum())} px painted by both -- too little to compare")
                continue
            ys, xs = np.where(both)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            A = white_paint(fa.canvas[y0:y1, x0:x1]) * both[y0:y1, x0:x1]
            B = white_paint(fb.canvas[y0:y1, x0:x1]) * both[y0:y1, x0:x1]
            if A.sum() < 50 or B.sum() < 50:
                print(f"  {os.path.basename(fa.path)} vs {os.path.basename(fb.path)}: "
                      f"no field paint visible in the overlap -- nothing to align")
                continue
            # search a +/-2m window around zero shift
            rad = int(2.0 * ppm)
            pad = cv2.copyMakeBorder(A - A.mean(), rad, rad, rad, rad, cv2.BORDER_CONSTANT, value=0)
            R = cv2.matchTemplate(pad, B - B.mean(), cv2.TM_CCORR)
            _, _, _, loc = cv2.minMaxLoc(R)
            dx, dy = (loc[0] - rad) / ppm, (loc[1] - rad) / ppm
            print(f"  {os.path.basename(fa.path)} vs {os.path.basename(fb.path)}: "
                  f"{int(both.sum())} px overlap, field paint aligns to within "
                  f"dx={dx:+.2f}m dy={dy:+.2f}m (0,0 = perfect agreement)")

# ------------------------------------------------------------ annotate/save
def fp(xc, yc):
    p = F2C @ np.array([xc, yc, 1.0])
    return int(p[0] / p[2]), int(p[1] / p[2])


for (a, b) in (((0, 0), (0, cfg.width)), ((cfg.length, 0), (cfg.length, cfg.width)),
               ((0, 0), (cfg.length, 0)), ((0, cfg.width), (cfg.length, cfg.width)),
               ((1800, 0), (1800, cfg.width)), ((8200, 0), (8200, cfg.width)),
               ((5000, 0), (5000, cfg.width))):
    for im in (canvas, sources):
        cv2.line(im, fp(*a), fp(*b), (0, 220, 255), 2)
for xm in range(0, int(cfg.length / 100) + 1, 10):
    for im in (canvas, sources):
        cv2.putText(im, f"{xm}m", (fp(xm * 100, 0)[0] + 4, OH - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
for k, fam in enumerate(families):
    cv2.putText(sources, f"{os.path.basename(fam.path)} (t={fam.ref_sec:.0f}s)",
                (12, 28 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.65, fam.colour, 2)

cv2.imwrite(f"outputs/mosaic/{out_name}.jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
if len(families) > 1:
    cv2.imwrite(f"outputs/mosaic/{out_name}_sources.jpg", sources,
                [int(cv2.IMWRITE_JPEG_QUALITY), 90])

print(f"\n{sum(f.placed for f in families)} frames rectified, {skipped} could not be registered")
for fam in families:
    got = (fam.best_q > 0).sum()
    print(f"  {os.path.basename(fam.path):38s} {fam.placed:4d} frames, "
          f"painted {100.0*got/max(1,fam.visible.sum()):3.0f}% of what it can represent, "
          f"{len(fam.hubs)} hubs")
union = np.zeros((OH, OW), bool)
for fam in families:
    union |= fam.visible
print(f"together they can represent {100.0*union.mean():.0f}% of the canvas "
      f"and painted {100.0*filled.sum()/max(1,union.sum()):.0f}% of it")
if filled.any():
    print(f"field x actually observed: {fx_cm[filled].min()/100:.0f}.."
          f"{fx_cm[filled].max()/100:.0f}m of 0..{cfg.length/100:.0f}m")
print(f"wrote outputs/mosaic/{out_name}.jpg -- yellow = true field lines from WFDF geometry.")
print("      Painted content should line up with the yellow lines; where it does")
print("      not, the calibration is wrong and you can see it directly.")
if len(families) > 1:
    print(f"wrote outputs/mosaic/{out_name}_sources.jpg -- which calibration owns which pixel.")
