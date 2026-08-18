# Field-Calibration Finding (measured on real footage)

Prompted by a proposed refinement to B2/B3: detect field lines/cones once on
the mosaic (not per-frame), then propagate that calibration to every frame in
a segment via the already-known frame→mosaic homography chain
(`frisbee_analysis/mosaic.py`'s `register_sequence`). The chaining idea is
architecturally sound — this note is about the harder question underneath it:
are lines/cones actually detectable at all on this footage?

## Method

Three attempts, each testing a different confound:

1. **Classic white-line-on-grass Hough line detection on the stitched
   mosaic** (`outputs/mosaic_sample.jpg`): HSV brightness/saturation mask +
   Canny + `HoughLinesP`.
2. Same method **on a single raw frame** instead of the mosaic, to isolate
   whether the mosaic itself was the problem.
3. **Visual scan for cones** across frames spread through the game
   (0-24 min), since every previously-sampled frame was mid-field, not an
   endzone/corner shot where boundary cones would appear.

## Result

1. **Mosaic**: 89 line segments detected. Nearly all are **stitching-seam
   artifacts** — the hard edges where compositing blends different frames'
   warped content meet, not real-world lines. Eyeballing the overlay
   (`outputs/line_detect_test.jpg`, local only) confirms this: the detected
   lines trace the mosaic's own composite boundary, not the sideline.
2. **Raw frame**: 80 line segments detected. Dominated by different
   confounds — the treeline/sky horizon, the scoreboard box border, and the
   closed-captions underline are all higher-contrast than the actual white
   sideline paint on grass. At most one of the 80 looks like a genuine
   sideline segment.
3. **Cones**: a small cone-like marker is visible near mid-field in at least
   one sampled wide shot, but no frame sampled so far shows a clean set of
   the corner/goal-line cones (`ultimate_config.py`'s 8 boundary cones + 2
   brick marks) needed for a real homography calibration.

## Verdict: NOT a quick win, unlike B1/B3

Both B1 (mosaic feasibility) and B3 (zero-training detection) turned out
better than expected once actually measured. Field-line/cone calibration is
the opposite: naive automated detection is genuinely hard on this footage,
for two independent reasons (mosaic seam contamination, raw-frame confounds
from treeline/overlay). A working automated version would need real
preprocessing this test didn't attempt: overlay + non-grass masking before
any edge detection, orientation filtering (sidelines have a narrow expected
angle range), and probably a dedicated cone color detector (WFDF cones are a
consistent bright color) rather than generic line detection.

**The chaining idea itself is still sound and worth keeping**: whatever
calibration method works, doing it once per segment and propagating via the
existing frame→mosaic homography is strictly better than recalibrating every
frame. It's the calibration step itself, not the propagation, that's hard.

## Recommendation

Given `CONTEXT.md` already made the same call once before ("a little manual
input... makes identity tractable" — accepting manual pull-formation tagging
over chasing full automatic identity), the same logic applies here: **rather
than investing more in automated cone/line detection, click the calibration
points once per segment** (a handful of points on one representative frame
per segment — cheap, since a segment can be minutes long) and let the
existing homography chain do the propagation automatically. Revisit
automated detection only if manual calibration turns out to be the actual
bottleneck once the rest of the pipeline is running.

## Addendum: blended mosaic images are the wrong thing to click on

First version of `run_calibrate.py` had the user click keypoints directly on
the composited mosaic image (e.g. `outputs/mosaic_sample.jpg`,
`outputs/mosaic_cones.jpg`). Went looking for calibration-worthy footage by
scanning many timestamps and building several candidate mosaics; one
(`outputs/mosaic_cones.jpg`, t≈26:20–29:20, a real 41-frame continuous
segment) appeared under pixel-level HSV inspection to show a cone at a
specific pixel. **The person checked it in the actual tool and confirmed no
cone was visible there at all.**

Root cause: mosaic building (`run_mosaic_visualize.py`) blends multiple
warped frames together (weighted pixel averaging). A cone is small relative
to a frame, and even sub-pixel registration error smears it across a few
pixels when several frames' cone positions don't perfectly stack — the
averaging can wash it out or shift it, badly enough that automated
pixel-level color analysis found a spurious signal that direct human viewing
correctly rejected. **This is exactly the kind of thing a human eye on a
real image catches and a pixel-difference script can get fooled by** —
trust the person's visual check over the automated one.

**Fix, not a workaround**: `run_calibrate.py` now has the user browse and
click on RAW (unblended) frames within a segment instead of the composited
mosaic image. `frisbee_analysis/mosaic.py`'s `register_sequence` already
computes each raw frame's own homography into the segment's shared
coordinate system — the tool now uses that directly: click a keypoint on
whichever raw frame shows it clearest (different keypoints can come from
different frames), and each click gets transformed through that frame's
homography before fitting the calibration. The mosaic image is still useful
for the visual sanity-check overlay afterward, and for a quick eyeball scan
of roughly where things are — just not as the thing you click precise points
on.

## Addendum 2: wrong-field contamination, a sharper diagnosis than blending alone

The person's own read on the failed attempt: the "cone" pixel-analysis found
in `outputs/mosaic_cones.jpg` was likely a real cone — just on a **different,
adjacent field**, not the one being played/broadcast. This tournament venue
has multiple simultaneous games visible in the background of nearly every
wide shot seen throughout this project (confirmed repeatedly, e.g. in the
very first frames sampled for B1). A cone belonging to a neighboring field's
boundary is geometrically meaningless for calibrating THIS field's homography
— fitting one in produces a transform that can look superficially plausible
per-point but is nonsense overall, consistent with the wildly-off-canvas
projected overlay from that attempt.

This is at least as likely an explanation as mosaic-blending washout, maybe
more so — and it's a risk the raw-frame-browsing fix above does NOT
automatically solve, since raw frames also show the neighboring fields.
Added an explicit on-screen warning in `run_calibrate.py`: only click cones
you can confidently place on the same field boundary as the actively
broadcast game (e.g. in line with that field's own sideline/players), not
just any orange dot visible in a wide shot.

## Addendum 3: line correspondences work standalone, but NOT mixed with points (open bug)

Real segments on the UFA footage kept producing too few *usable* corner
points (registration breaks meant left-side and right-side content landed in
different, non-bridgeable segments — see the live debugging session that
led to `_reclaim_failed_frames` in `mosaic.py`). Point requested: fit a
homography from a combination of point clicks AND line evidence (you don't
need the exact corner, just 2+ pixels you're confident lie somewhere along a
known field line, which is a much easier ask than pinning down an exact
intersection).

Implemented `fit_calibration_with_lines` (combined-DLT with Hartley
normalization, using the standard `l_image = H^T @ l_field` projective
duality for line constraints). Extensive synthetic validation:

- **Points-only** (4 points, own code path): exact recovery, 0px error. ✅
- **Lines-only** (4 lines, zero points): exact recovery, 0px error. ✅
- **Mixed** (any split of points + lines, e.g. 2+2 or 3+1): consistently
  **rank-deficient by exactly 1**, even with fully random synthetic
  correspondences (not a geometry-specific fluke — tested multiple
  configurations, including deliberately non-collinear, non-parallel,
  non-vertex-sharing ones). The row formula itself is individually verified
  correct (evaluates to ~0 at the true homography for every row), the
  normalization is individually verified correct (matches `cv2.findHomography`
  exactly for points alone) — but something about how the two row types
  combine loses one degree of freedom that neither type loses on its own.
  **Root cause not found** despite methodical isolation (point rows, line
  rows, normalization, and denormalization all individually confirmed
  correct in isolation).

**Until this is root-caused, `fit_calibration_with_lines` warns loudly when
you mix points and lines, and hard-fails with a clear error if the resulting
constraint matrix doesn't reach full rank** (checked via the singular value
gap, not just element count) — silently returning an arbitrary wrong
homography from an under-determined system would be far worse than refusing.

**What's actually usable right now: lines-only.** If you can identify 4
different field lines (e.g. both sidelines + both goal lines, or any 4
non-parallel, non-concurrent field lines) with 2+ pixels each, that's a
fully validated path to a calibration — no point clicks needed at all. This
directly solves the original problem (too few reliable corner points) since
lines are a much lower bar than exact corners.

## Addendum 4: cheap color-filter cone detection -- measured, not reliable enough

Prompted by: "couldn't [we] make a model for automatic field detection... large
orange cone and very visible lines" -- a fair question, since a colorful
compact object like a cone is architecturally a much easier target than the
line-detection approach above (which failed on confounds like the treeline).
Tested the cheapest version first, no model training: plain HSV color-
filtering for bright orange (`run_cone_detect_test.py`), scanning 30 frames
spread across the full `videoplayback.mp4` (134.5 min).

Result: 7/30 frames had candidate blobs; checked each by eye.
- **1 real hit**: an actual orange/white striped boundary marker pole,
  correctly found.
- **1 miss**: a frame with a visible cone in the background the filter
  didn't catch.
- **4 false positives**: a mid-broadcast commercial (a soda can ad --
  broadcasts include ad breaks, which naive frame-sampling doesn't know to
  skip), a maroon venue field logo, a yellow football goalpost in the
  background, and — worst — **81 false "candidates" on a single crowd shot,
  all fans' skin tone**, since skin sits in a similar hue/saturation range as
  a bright orange cone.

**Verdict**: not reliable as-is. Skin tone is the binding confound — any
frame with a crowd or close-up player defeats it — and it still missed a
real cone. Same conclusion as the line-detection attempt above, for a
different reason: the naive/cheap version of automated field-marker
detection is genuinely hard on real broadcast footage, not a quick win.
Would need real additional work to be trustworthy (restrict the search to
the grass/field region only -- which is circular without already having a
calibration -- and/or better hue tuning, and/or a trained detector instead
of a fixed color range). Not pursued further for now; the calibration bank
(auto-reusing already-calibrated camera framings, see `NEXT_STEPS.md`) is
the higher-value next investment instead, since it doesn't depend on cone
visibility at all.
