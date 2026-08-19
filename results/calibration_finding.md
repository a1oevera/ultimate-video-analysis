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

## Addendum 4.5: a "not 100% accurate" line in an auto-matched overlay -- traced, not guessed

Person spotted the far-sideline overlay drifting slightly in an auto-matched
calibration and asked what the red dot was too (answer: one of the 12
known field reference vertices `draw_field_overlay` marks -- only one was
visible in that crop, the rest projected outside the frame).

First hypothesis for the drift: the bank-match step's inlier keypoints were
clustered in one region of the frame (low "coverage"), so the fitted
homography was extrapolating badly to the far side of the frame where the
sideline actually is. Built `register_pair_with_coverage` (mosaic.py) to
test this directly rather than assume it -- **disproven**: the actual match
had 1,902+ inliers AND 55.6% frame coverage, both healthy. Not the
explanation.

Traced it properly instead: re-ran `check_line_reprojection_error` on the
ORIGINAL manual calibration (before any bank-matching at all) and found it
already had up to **12.8px error on one line** (the near goal line), with
5.9px on the far sideline itself -- a real, pre-existing imprecision from
the original click session. It was below the 20px "reject, don't add to
bank" threshold, so it passed silently -- and once composed into an
auto-matched calibration (which carries no `line_details` of its own),
that imprecision became invisible: nothing would tell you it was there
short of eyeballing the overlay, which is exactly what caught it.

**Fix**: added `Calibration.source_max_reprojection_px`, stamped by
`run_calibrate.py` right after computing reprojection error, and carried
forward through every bank composition (`_grow_and_composite`,
`try_auto_calibrate`) so it's never lost. `run_calibrate_auto.py` now prints
it explicitly on every auto-match, e.g. "the calibration this canvas was
built from had up to 12.8px reprojection error of its own." The
coverage check (real, if not the explanation here) is also wired in and
printed, with a warning below 15%.

**Honest framing**: this doesn't make the underlying calibration more
accurate -- it makes its actual accuracy visible instead of silently
disappearing at composition time. The right response to a high
`source_max_reprojection_px` is still to redo that line in `run_calibrate.py`,
same as it always was.

## Addendum 4.6: redo attempt 1 made it WORSE, and the bank had a real bug hiding a fix

Person redid the near goal line following advice to click more points
spread apart -- result was worse across nearly every line, not just the one
being fixed (max reprojection error 12.8px -> 29.0px). Root cause traced,
not guessed: the far sideline's points came from 4 different frames spread
1465px apart in the shared coordinate system -- exactly the "homography-
chaining drift risk" `run_calibrate.py` already warns about when clicks
spread across frames, which the earlier advice to "spread points apart"
didn't account for (spreading pixel-wise within one frame is good;
spreading across multiple frames trades that for chaining risk). The near
goal line got worse too despite being single-frame -- likely plain click
imprecision on a foreshortened/partially-blocked view of that line.

**Redo attempt 2** (one frame per line, points spread within that frame
only): every single line improved versus BOTH previous attempts. Max error
dropped to 6.5px (near goal line specifically: 12.8px -> 2.9px). Visually
confirmed: the same corner that landed on a player before now lands right
next to an actual visible cone.

**Real bug found while verifying this got saved into the bank**: it didn't.
`add_entry`'s merge path always kept the EXISTING canvas's calibration as
authoritative when merging a matching new frame in, regardless of which one
-- old canvas or new frame -- was actually more accurate. The improved
calibration matched the existing (worse) canvas, got merged in for its
pixel coverage, and its better accuracy was silently discarded. Fixed:
`add_entry` now compares `source_max_reprojection_px` and re-bases the
canvas on whichever calibration is more accurate (warping the OLD canvas's
content into the NEW, better frame's coordinate system when the incoming
one wins, symmetric to the normal merge). Verified: canvas's stored error
correctly updated 12.8px -> 6.5px, and a fresh `run_calibrate_auto.py` run
now reports 6.5px, not the stale 12.8px.

## Addendum 4.7: a false-positive bank match corrupted the shared canvas

Working through the batch calibration list, person reported the second
auto-match "very wrong." Investigated: it had matched bank canvas_000 with
only **17 RANSAC inliers and 15% coverage** -- barely above the OLD
`min_inliers=15` floor. Re-running the identical command produced a
visibly different result (different dot positions) -- a real match doesn't
wobble between reruns of the same input; a coincidental one does. The two
frames were 26 minutes apart in the game.

**Worse than a one-off bad calibration**: `run_calibrate_auto.py` always
calls `add_entry` at the end regardless of match quality, so this
false-positive got MERGED into the shared canvas -- confirmed visually
(`outputs/debug_canvas_corrupted.jpg`, not committed): two badly mismatched
framings warped together, skewed background, different scoreboard scores
composited on top of each other. This corrupted canvas_000 for EVERY future
match, including the previously-verified good one -- re-testing t=1470
against the corrupted canvas returned "no match" even though it had 2,354+
inliers against the clean version.

**Root cause**: `min_inliers=15` (`MosaicConfig`'s default) was validated
for a different job entirely -- `mosaic.py`'s within-segment frame-to-frame
chaining, where consecutive frames share most of their background. It's far
too permissive for bank matching, which compares potentially unrelated
moments anywhere in the game. Known GOOD bank matches in this project
measured 1,800-2,500+ inliers and 55-63% coverage -- a huge gap above the
old floor.

**Fix**: `BANK_MIN_INLIERS = 50` and `BANK_MIN_COVERAGE = 0.25`, a much
stricter bank-specific gate applied everywhere a bank match gets accepted --
`try_auto_calibrate` (read-only matching) AND both matching loops inside
`add_entry` (which can actually corrupt a canvas if wrong, so needs to be at
least as strict). Grounded in the one real false-positive data point plus
the cluster of known-good ones, not exhaustively tuned.

**Recovery**: rebuilt the bank from scratch, re-seeded from the current
verified-good calibration (2.1px reprojection error). Verified: the same
false-positive timestamp now correctly falls back to manual instead of
matching, and the good match still works (2,354 inliers, 63% coverage) on
the clean bank.

## Addendum 5: the calibration bank -- reuse instead of re-detect

Prompted by the person's own observation: this footage cuts/pans often
enough that ONE manual `run_calibrate.py` session per camera segment doesn't
scale to a full game (could be dozens of segments). Since it's a fixed
mount (pure rotation/zoom only), the broadcast almost certainly reuses a
small set of recurring framings (the standard wide shot, close-ups per
endzone) far more than it invents brand new ones each cut -- worth checking
"have I already calibrated a shot like this?" before asking for more clicks.

Built `frisbee_analysis/calibration_bank.py` + `run_calibrate_auto.py`:
reuses the SAME ORB+RANSAC registration already validated for within-segment
frame chaining (`mosaic.py`'s `register_pair`), just applied BETWEEN a new
segment's reference frame and every already-calibrated bank entry's
reference frame. A match composes the calibration for free
(`new_calib.H = bank_entry.H() @ H_step`) -- no clicking. `run_calibrate.py`
now auto-adds every successful manual calibration to the bank too, so the
bank grows automatically as a byproduct of normal use, no extra step.

Verified end-to-end on real footage: seeded the bank with the existing
calibration (found its true anchor frame -- see the code comment on why
that's not always `seg_idx[0]`, `_reclaim_failed_frames` in `mosaic.py` can
fold an earlier-index frame into a later segment's anchor). Re-running on
the SAME window matched with 2,533 RANSAC inliers and produced a correct
overlay (visually identical quality to the manual one). Running on a
DIFFERENT, unrelated window (t=300s vs. the seed's t=1158s) correctly found
no match and fell back with clear instructions, rather than forcing a bad
match.

Caveat carried over from the wrong-field warning above: an automatic match
is still just a homography composition from a RANSAC inlier count -- a
coincidental match is possible in principle (two genuinely different
framings that happen to share enough background texture). `run_calibrate_auto.py`
saves the same sanity-check overlay image any calibration does; check it by
eye before trusting an auto-match, same as a manual one.
