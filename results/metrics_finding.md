# B5 First-Pass Finding: end-to-end team metrics (measured on real footage)

Closes the first version of `NEXT_STEPS.md` B5 — wiring registration +
calibration + detection + team classification + `is_in_field` together into
actual numbers, not just validated components. `run_metrics_test.py`.

## Method

Same segment used for the real calibration run and the overlay-tracking
video (`videos/videoplayback.mp4`, t=1158–1170s, 3fps, 32 registered frames,
`outputs/calibration.json`). Per sampled frame: YOLO person detection
(`imgsz=1920`) + ByteTrack, foot point (bottom-center of box) projected to
field cm via `calib.H() @ frame_homography`, filtered with `is_in_field`
(100cm margin), team split by torso saturation (per-frame median threshold).
Speed: for any ByteTrack id present in two *consecutive sampled* frames, the
field-space displacement / dt is one instantaneous speed sample — explicitly
not a full-lifespan track, sidestepping the ~90% fragmentation measured in
`results/tracking_finding.md` rather than solving it, per the B2/B4 descope.

## Result

- On-field detections/frame: team A mean 0.5 (0–7), team B mean 0.6 (0–2) —
  sparse, consistent with this being a tight-ish 12s window, not a wide pull
  shot with all 14 players visible.
- 8 usable consecutive-frame speed samples out of 36 total on-field
  detections across the window (most detections don't have a same-ID match
  one sample later — same fragmentation problem, just now visible as a
  yield-rate number instead of a track-lifespan number).
- Team A: mean 1.34 m/s (n=2). Team B: mean 1.50 m/s, max 2.07 m/s (n=6).
  Low for active ultimate play (typical is more like 3–6 m/s when moving,
  1–2 m/s is closer to standing/jogging) — plausible for this specific
  window (players setting up/repositioning) but **n is too small (8 samples
  total) to trust as a real number**, and could also mask real error (foot-
  point noise, calibration drift within the segment, YOLO box jitter).
- Visual spot check (`outputs/metrics_sample.jpg`, local only): the
  `is_in_field` filter behaved sensibly — two people clearly standing on the
  running track outside the sideline were correctly excluded, while on-field
  players got boxed and team-colored correctly.

## Verdict: pipeline wiring works, numbers are not yet trustworthy

This is a **plumbing check, not a metrics result** — it confirms every piece
built so far (B1 registration, calibration, B2 team split, B3 detection)
composes correctly into field-space numbers with sane-looking filtering, on
one real short window. It is NOT yet evidence that the speed/distance numbers
are accurate: sample size is tiny (one 12s window, 8 speed samples), the
calibration in use predates `line_details` so its own reprojection error
can't be re-checked, and there's no ground truth here to compare against
(unlike Track A, which had UFATrack).

## Addendum: ByteTrack was silently dropping most on-field detections (real bug, fixed)

The person spot-checked the very first run against the actual frame and
reported 3 detected out of 9 players visibly on the field — a much bigger
gap than generic-detector recall would explain on its own.

Root cause, found by reading `supervision` 0.30.0's `ByteTrack` source
directly (`tracker/byte_tracker/core.py`): `update_with_detections` silently
**drops** any detection it can't confidently match to an already-existing
track. Concretely, a first-seen (or re-appearing-after-lost) player with YOLO
confidence in `[track_activation_threshold, det_thresh)` — `[0.25, 0.35)` by
default — never gets a track id at all ("Step 4: Init new stracks":
`if track.score < self.det_thresh: continue`) and is discarded from the
returned detections entirely, not just left untracked. The bug in
`run_metrics_test.py`: on-field counting and team classification were reading
the ByteTrack-*filtered* output, when only the speed estimate actually needs
a track id.

**Fix**: keep the raw YOLO detections for counting/classification/in-field
filtering; only consult ByteTrack's output (a same-order subset of the raw
boxes) to look up a track id per box where a match happens to exist, used
solely for consecutive-frame speed matching. Same window, same frame:
on-field counts went from **team A mean 0.5→6.6, team B mean 0.6→4.5** per
frame — the last frame's visual count went from 3 boxed players to 11,
matching the person's manual count of 9 (the extra 2 being genuine detections
outside their count, not filter errors — two people clearly standing on the
running track were still correctly excluded by `is_in_field`). Speed numbers
were barely affected (still ~1.3–1.5 m/s, still only 8 usable samples) since
that estimate was never reading the undercounted path — the bug was specific
to counting/classification, not speed.

## Addendum 2: fixing the ByteTrack drop un-hid a real false positive (cone), also fixed

The person spotted a cone/field-marker getting boxed as a player in the
sample image right after the ByteTrack fix above. Traced it: a 7x13px box at
conf=0.34 — squarely in the `[0.25, 0.35)` band ByteTrack used to silently
eat — in a frame where every real player's box was >=32px tall. The
ByteTrack-drop fix had an unintended side effect: it stopped hiding this
false positive along with the real detections it was wrongly hiding.

Fix: drop any detection under 40% of the frame's own median detection height
(only applied when there are >=3 detections, so the median itself is
meaningful). Verified on the same frame: the cone box disappears, all 11 real
player boxes remain (team B's mean count moved 4.5→4.2, the cone had been
counted as one low-saturation-adjacent "player"). This is a per-frame
heuristic, not a learned classifier — it assumes real players at any depth on
this footage stay roughly proportionate to each other in the same frame,
which held here but isn't guaranteed on a much wider zoom range.

## Addendum 3: three more real bugs from a person spot-check (margin, saturation, occlusion)

Another round of the same pattern: the person looked at the actual output
and found three concrete problems, not noise.

1. **Sideline/bench people counted as on-field.** `is_in_field`'s 100cm
   outward margin (added earlier to be forgiving of foot-point/calibration
   noise) was generous enough that people standing *near* the boundary, not
   on it, got counted as players. Fixed: `margin_cm=0`.
2. **Dark-jersey players tagged as the white team.** The classifier used HSV
   *saturation*, on the assumption (from `results/team_classify_finding.md`)
   that a white-vs-colored matchup splits cleanly on it. Measured directly
   on a real frame: saturation ranged 1–56 with heavy overlap between teams
   — a near-black/neutral dark jersey reads as low-saturation too, since a
   deep, desaturated color isn't the same thing as a vivid one. HSV
   *brightness* (value), on the same frame, split cleanly: 55–131 (dark) vs
   219–255 (white), a real gap with nothing in between. Switched
   classification to brightness. Also switched from a per-frame threshold to
   one Otsu threshold pooled across the whole window, since a per-frame
   split silently assumes each frame shows a ~50/50 mix of both teams.
3. **Overlapping players cross-contaminating each other's team read.** The
   occlusion-exclusion crop from Addendum 1/the ByteTrack fix had too loose
   a floor (30% of torso width, no absolute minimum) — two heavily-
   overlapping, near-duplicate detection boxes a few pixels apart reduced
   each other's usable crop to a few-pixel sliver, caught by two supposed
   detections of the same area returning wildly different brightness
   values. Tightened the floor to 60%-of-width-or-6px, whichever is larger.

Verified on the same real frame used throughout this doc: every dark-jersey
player now boxes red, the white-jersey player boxes blue, matching what's
actually visible.

**Pattern worth naming**: every real bug found in this pipeline so far
(ByteTrack drop, the cone false positive, and these three) was found by a
person looking at the actual saved image, not by the printed summary
numbers looking wrong. The numbers looked *plausible* every single time
before the fix. Keep treating the visual spot-check as load-bearing, not
optional, before trusting an aggregate from this pipeline.

## Addendum 4: double-boxed player (duplicate YOLO detections, not merged by NMS)

Person spotted a literal double box around one dark-jersey player. Confirmed
by direct inspection: box `(534,147,551,190)` and box `(539,147,550,190)` are
the same physical player — the second almost entirely CONTAINED inside the
first (identical y-range, x-range a strict subset) — that YOLO's own NMS
left unmerged. A plain IoU check alone would likely have missed this too:
when one box is much smaller than the other, intersection-over-*union* stays
low even at near-total overlap of the smaller box, since the union is
dominated by the larger box's area.

Fix: `dedupe_boxes()` — for any pair of same-frame boxes with high IoU OR
where the smaller box is mostly *contained* in the larger one (checked
separately, since containment catches what IoU misses), keep only the
higher-confidence box. Verified: the double box disappears on the same real
frame, and — more convincing than one frame — the pooled detection count
across the *entire* 32-frame window dropped from 340 to 290, meaning this
was a widespread duplication issue across many frames, not a single-frame
fluke.

## Recommendation

- Re-run calibration with the current `run_calibrate.py` (saves
  `line_details`) so `check_line_reprojection_error` can confirm the
  homography's own accuracy before trusting anything downstream.
- Run `run_metrics_test.py` over a longer, wider window (ideally a segment
  spanning most of a point, with most players visible) once a matching
  calibration/segment exists — a single 12s tight shot is too small a sample
  to report a real speed number from.
- Consider a sanity ceiling check (e.g. flag any single speed sample above a
  physically implausible threshold, ~8–9 m/s sprint) once volume is higher —
  cheap outlier detection for calibration/tracking errors before trusting an
  aggregate.
