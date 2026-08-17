# Next Steps — prioritised

The project splits into **Track A** (possession logic, coordinate-space, no video
— mostly built, in this package) and **Track B** (the CV pipeline: YOLO +
tracking + homography — not started, needs footage + a GPU).

**Do Track A first. It decides whether Track B is worth building at all.**

---

## Track A — go/no-go DECIDED: NO-GO on possession-from-motion-alone

### A1. ✅ DONE — fixed the turnover blindness
Root cause: `_emission_logprob` gave defense players a permanent `-inf`
emission at every frame — an absolute veto, so no `switch_penalty` value
could ever cross to a defense holder. Fixed: removed the veto, added
`defense_bias` (soft per-frame prior against defense holding, not a hard
rule), made `dist_to_defender` symmetric (renamed `dist_to_marker`, defined
for both teams). Predicted turnovers went 0/5 → 3/5 (in-sample, untuned). See
`results/tuning_finding.md`.

### A2. ✅ DONE — tuned + cross-validated
Stratified 5-fold CV (one of the 5 known turnover tracks per fold) + 400-config
random search over `w_rel_speed, w_accel, w_dist_def, switch_penalty,
defense_bias`, optimising transition F1. Result: **CV transition F1 ≈
0.29–0.35, turnover recall 0–1 of 5 held out.** The random search overfit the
16-track selection set and did not beat the original hand-set weights on
held-out data. Full fold-by-fold numbers + the search-vs-baseline comparison
in `results/tuning_finding.md`. Run `python run_tune.py` to reproduce.

### A3. Disc-position-derived features — not pursued
Deferred per the original plan (only worth it if A1+A2 didn't already clear
~0.5). They didn't get close, and disc position from UFATrack would be
circular for this dataset anyway — see A4's fallback below for the real path
if disc detection is wanted.

### A4. GO / NO-GO — **NO-GO** on possession-from-motion-alone (see `results/tuning_finding.md`)
Cross-validated transition F1 (0.29–0.35) is well under the ~0.5 GO bar, and
under the ~0.4 "insufficient" line even after tuning. The turnover-blindness
*bug* is fixed (turnovers are representable now), but the motion + marking
feature set itself doesn't carry enough signal for possession, and tuning
can't manufacture signal that isn't there — it can actively hurt on this
little data (see the overfitting result in `results/tuning_finding.md`).

**Decision made (2026-08-17): rescope to metrics #3/#4** (play time,
speed/distance) — turnovers/passes are dropped since they needed possession,
which didn't pan out. Disc detection as a primary signal stays parked, not
ruled out; revisit it if priorities change later. Active work is now Track B
below, specifically B1 (mosaic registration) and B3/B4 (identity) — #3/#4
never needed possession, so this isn't gated on A1–A4.

---

## Track B — CV pipeline (rescoped to metrics #3/#4: play time, speed/distance)

**Environment set up (2026-08-17):** `requirements.txt`'s Track B deps
(ultralytics, supervision, umap-learn, transformers, torch, opencv-python,
roboflow/sports) are installed. **This laptop is an Intel Mac with an AMD
GPU — no ROCm on macOS, so `torch` runs CPU-only here.** Fine for wiring and
testing code; real YOLO training needs a different machine (cloud GPU, or an
Apple Silicon / NVIDIA machine) — don't try to train at scale on this one.

B2's `TeamClassifier` (offense/defense split) isn't required for #3/#4 (play
time and speed/distance are per-player, team-agnostic) — deprioritize that
half of B2 unless disc detection / possession comes back into scope. Its
`ViewTransformer` (homography) IS still needed.

Footage reality: a ~90 min broadcast recording is in `videos/` (gitignored —
too large + not ours to redistribute). Confirms the predicted constraint —
fixed elevated camera that pans/tilts/zooms hard, swinging between wide
full-field shots and tight close-ups with no field context. Lined field, rich
treeline/tent background.

### B1. ✅ DONE — mosaic-registration feasibility CONFIRMED, real pipeline built
Measured with real ORB+RANSAC registration on real footage (see
`results/mosaic_finding.md`): wide-to-wide frame pairs register solidly
(74–84 RANSAC inliers at realistic 2–5s gaps, 16+ even 63 min apart) — this
footage's background (treeline, tents) has enough texture. Tight close-up
frames get 0 inliers, as expected — no shared background to match.

Built the real pipeline, not just the feasibility test: `frisbee_analysis/mosaic.py`
(`register_pair`, `register_sequence`, `interpolate_missing`) + `run_mosaic_test.py`.
Running it end to end on real footage caught a real bug the pairwise test
couldn't: `register_sequence` originally anchored every frame against frame 0,
so when frame 0 was a broadcast intro card (not camera footage), EVERY later
gameplay frame failed to match it — even though those frames matched each
other perfectly. Fixed with a `segment_id` concept: when the stale anchor
fails but a frame matches its immediate predecessor, start a new segment
there rather than stalling. Effect on the same test window: 3/120 frames
registered → 116/120, in 4 correctly-identified segments (intro card /
transition / gameplay). `interpolate_missing` only bridges gaps within a
segment now — bridging across a real cut would be meaningless.

Still true and unchanged: mask static broadcast overlay graphics before
feature detection, and expect segments/failures to correspond to real content
boundaries (cuts, close-ups), not registration bugs, once this fix is in.

### B2. Reuse roboflow/sports (MIT) — lift, don't rewrite
- `ViewTransformer` (homography) — use as-is with `ultimate_config.py` keypoints.
- `TeamClassifier` (SigLIP+UMAP+KMeans) — can't run on this machine at all
  (needs `transformers`' torch backend, unavailable — torch capped at 2.2.2
  on Intel macOS). Built a lightweight replacement instead, see below.
- pitch-config + annotator pattern — `ultimate_config.py` already matches it;
  gives the minimap for free.
- Do NOT use their soccer .pt weights or their `BallTracker` (naive; fails on a
  disc — moot anyway now that disc detection is parked).

**Decision (2026-08-17): descope individual player identity for now — team
assignment only.** Per-player tracking measured badly fragmented (see B3
below); team is a per-frame property that doesn't need identity continuity
at all, so this sidesteps that problem rather than trying to solve it. Metric
#3 (play time PER PLAYER) is on hold under this scope; metric #4
(speed/distance) can be reported as team-level aggregates instead of
per-player. B4's re-anchor/appearance-matching design stays as the documented
path if/when individual identity comes back into scope.

**✅ Team classification BUILT and working**: `run_team_classify_test.py`
(see `results/team_classify_finding.md`). Four attempts at joint hue/color
k-means all failed — same-team players kept splitting across clusters,
because a genuinely low-saturation team (white jerseys) gets outvoted by
clustering built around finding hue variation. Fix: split on saturation
FIRST (a threshold, not a cluster) to separate white/grey from colored, THEN
cluster hue only within the colored group. For this game's white-vs-maroon
matchup, confirmed correct by eye — both teams' players now group
consistently, where every earlier attempt split them. Known remaining
contamination: sideline/bench people still get classified into one team or
the other — same fix as B3/B4, filter through `is_in_field()` once
`run_calibrate.py` has been run for real.

**Field-calibration keypoint detection: MEASURED HARD, not automated yet**
(see `results/calibration_finding.md`). Tried Hough line detection on the
mosaic (89 "lines" found, nearly all stitching-seam artifacts, not field
lines) and on a raw frame (80 found, dominated by the treeline horizon and
broadcast overlay, not the sideline). Cones aren't clearly visible in any
sampled frame either. Architecturally, calibrating once per segment and
propagating via `mosaic.py`'s existing frame→mosaic homography chain (instead
of recalibrating every frame) is still the right design — it's automated
*detection* of the calibration points that's hard, not the propagation.
**Recommendation: click calibration points manually, once per segment**
(cheap — a segment can span minutes), same "accept a little manual input"
call already made for identity in `CONTEXT.md`. Revisit automated detection
(grass masking + orientation filtering + cone color detection) only if manual
calibration becomes the actual bottleneck.

**✅ Manual calibration tool BUILT**: `frisbee_analysis/calibration.py`
(`fit_calibration`, `pixel_to_field`, `field_to_pixel`, `is_in_field`,
`draw_field_overlay`) + `run_calibrate.py` (interactive — click keypoints on
a mosaic image, 's' to skip an off-camera one, needs >=4 total). A cone being
off-camera doesn't block calibration: verified with a synthetic-homography
test that fitting from 8 of the 10 keypoints (skipping the two near-back
corners closest to the camera — confirmed the common case on this footage)
recovers the position of the two SKIPPED points to ~0cm error, and correctly
computes the far goal line's position without ever seeing its cones. The
interactive click loop itself needs a real display and hasn't been run for
real yet — that's on you to try with `python run_calibrate.py
outputs/mosaic_sample.jpg`. Next: run it for real, then wire `is_in_field`
into the B3 detections to filter out sideline/bench people.

**Known constraint (from the person, not yet re-verified against this
footage): brick marks aren't painted on most fields**, especially
recreational/coned ones — same unreliability class as the near-back corners.
`run_calibrate.py`'s prompt order already reflects this (goal-line + far-back
corners first, near-back corners and brick marks last, skip freely). Worst
case that leaves 6 reliable points (the 4 goal-line corners + 2 far-back
corners) — still comfortably above the 4-point minimum, but don't assume
brick marks will be there; treat them as a bonus, not a plan.

### B3. Train YOLO on ultimate footage

**✅ Zero-training feasibility test DONE** (see `results/detection_finding.md`,
`run_player_detection_test.py`): COCO-pretrained YOLO11n, no ultimate-specific
training at all, run at native resolution (`imgsz=1920`, NOT YOLO's own
640px default) gets real detection coverage across the whole visible field,
including far-side players. At the 640px default it completely missed the
entire field of play (0 detections there) — a resolution artifact, not a
capability gap; fixed by not downscaling. This de-risks B3: a first working
player-position pipeline may not require the full fine-tuning investment
below before something end-to-end exists — fine-tuning still matters for
production-quality recall/precision and anything beyond generic "person".

- Pretrain person detection on pro footage, fine-tune on the person's own.
- Seed annotation candidates found on Roboflow Universe (all CC-BY-4.0;
  downloading any of them needs a free Roboflow account + API key, not yet
  set up here):
  - [Ultimate Player](https://universe.roboflow.com/ultimetrics/ultimate-player) —
    1,626 images, player class, pretrained model available. Best fit for
    #3/#4 (player-only, no disc/team clutter).
  - [Tracking](https://universe.roboflow.com/frisbee-tracker/tracking-w8biu) —
    423 images, frisbee/observer/player classes.
  - [Frisbee Tracking](https://universe.roboflow.com/tracking-f1jov/frisbee-tracking) —
    521 images, disc/observer/player classes.
  - (Disc-focused ones exist too — [Ultimate Frisbee Disc Detector](https://universe.roboflow.com/eelke-van-foeken-sfimp/ultimate-frisbee-disc-detector),
    509 images — not needed for #3/#4, keep in mind if disc detection is revisited.)
**✅ Multi-frame tracking DONE, confirms B4's plan is necessary, not
optional** (see `results/tracking_finding.md`, `run_tracking_test.py`).
YOLO + ByteTrack over two real 15s/120-frame windows: only 12% (tight,
cluttered scene) and 4% (wide, panning scene) of created track IDs survive
≥80% of the window — ~90% of IDs are fragments, not real persistent
identities. Confirms, on real footage, what `CONTEXT.md` already ruled out as
a dead end ("tag once, track all game" doesn't survive occlusion). Naive
continuous tracking is NOT usable for play-time/identity on its own — this
is exactly why B4's re-anchor + per-tracklet approach exists, not a reason to
reconsider it.

Next: quantify far-side recall properly against ground truth (the detection
test eyeballed one frame; the tracking test measured persistence, not
accuracy) — lower priority now that fragmentation is the confirmed binding
constraint, not raw detection quality.

### B4. Player identity across pans (metric #3) — ON HOLD, descoped for now
Per the B2 decision above, individual identity is parked in favor of
team-level classification only. This design stays documented (and well
motivated by B3's tracking finding — naive tracking fragments ~90% of the
time on real footage) for if/when per-player metric #3 comes back into
scope; not active work right now.
- Re-anchor identity at each pull (wide shot, 7 spaced players — best identity
  frame, ~once per point).
- Per-tracklet appearance voting (SigLIP embeddings, which can't run on this
  machine — see B2 — so this would need a different platform anyway) +
  estimated height from homography + 7-per-team constraint (Hungarian
  assignment).

### B5. Emit Track-schema rows from the CV pipeline
The whole point of `schema.py`: the CV pipeline's job is to output the SAME
`Track` object the loader produces. Then Track A's possession code runs
unchanged on real footage.

---

## Hard constraints to remember
- Pi Zero 2 W CANNOT run this pipeline (no GPU; SigLIP/YOLO need one). Process on
  a laptop/desktop with a GPU. Keep any "run live on the Pi" idea out of scope.
- Existing footage only — can't refilm, can't add a second camera.
- Scope: the person has other summer projects. Track A + a reduced Track B is a
  realistic summer; the full thing is not.
