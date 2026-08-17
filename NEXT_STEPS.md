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
- `TeamClassifier` (SigLIP+UMAP+KMeans) — deprioritized, see note above.
- pitch-config + annotator pattern — `ultimate_config.py` already matches it;
  gives the minimap for free.
- Do NOT use their soccer .pt weights or their `BallTracker` (naive; fails on a
  disc — moot anyway now that disc detection is parked).

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
- Next: track detections across frames (not just per-frame boxes), and
  quantify far-side recall properly (this test eyeballed one frame — a real
  measurement needs ground truth to compare against).

### B4. Player identity across pans (metric #3)
- Re-anchor identity at each pull (wide shot, 7 spaced players — best identity
  frame, ~once per point).
- Per-tracklet appearance voting (SigLIP embeddings) + estimated height from
  homography + 7-per-team constraint (Hungarian assignment).

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
