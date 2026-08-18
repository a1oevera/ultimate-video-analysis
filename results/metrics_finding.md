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
