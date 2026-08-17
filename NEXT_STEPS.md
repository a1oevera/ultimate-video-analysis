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

**Active next step — pick one:**
- **Disc detection as a PRIMARY signal**, not an assist. This is the hard CV
  problem the project originally bet against needing — that bet didn't pay
  off, so this is back on the table if turnovers/passes stay the goal.
- **Rescope to metrics #3/#4** (play time, speed/distance), which don't need
  possession at all. Already scoped as "still a real project" — see Track B
  below, specifically B1 (mosaic registration) and B3/B4 (identity), which
  matter for #3/#4 independent of whether possession ever works.

---

## Track B — CV pipeline (needs footage + GPU)

A4 came back NO-GO on possession (turnovers/passes), so B2 (team classifier
for offense/defense split, used by the possession emission model) is on hold
pending the disc-detection-as-primary-signal decision above. B1/B3/B4 are
NOT gated on that — they serve metrics #3/#4 (play time, speed/distance),
which never needed possession — so they're valid to start now if that's the
direction chosen.

Footage reality (from the person): existing footage only, fixed high mount that
pans/tilts/zooms, **often only part of the field visible per frame**, sometimes
lined / sometimes coned fields.

### B1. Mosaic-registration feasibility test *(second cheap experiment)*
Because only part of the field is visible per frame, per-frame calibration often
has too few field points. The fixed mount saves you: register each frame to a
pre-built mosaic of the whole field via BACKGROUND features.
- Extract ~10 frames from one clip, including grass-only frames.
- Try `cv2` ORB/SIFT feature matching to stitch/register them.
- **If background is too texture-poor to register markerless frames, metrics
  #3/#4 degrade badly.** Find this out before building. (Depends on what
  surrounds the specific fields — treeline/buildings good, open park bad.)

### B2. Reuse roboflow/sports (MIT) — lift, don't rewrite
- `ViewTransformer` (homography) — use as-is with `ultimate_config.py` keypoints.
- `TeamClassifier` (SigLIP+UMAP+KMeans) — use as-is for offense/defense split.
- pitch-config + annotator pattern — `ultimate_config.py` already matches it;
  gives the minimap for free.
- Do NOT use their soccer .pt weights or their `BallTracker` (naive; fails on a
  disc). Train your own YOLO; keep disc as a secondary signal.

### B3. Train YOLO on ultimate footage
- Pretrain person detection on pro footage, fine-tune on the person's own.
- Seed annotation from the two Roboflow Universe *ultimate* datasets.
- Measure far-side-player recall specifically (partial-field framing makes small
  players common).

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
