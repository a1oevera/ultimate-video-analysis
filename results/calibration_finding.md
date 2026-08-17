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
