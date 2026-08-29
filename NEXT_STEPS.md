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

**✅ Manual calibration tool BUILT, then FIXED after real user feedback**:
`frisbee_analysis/calibration.py` (`fit_calibration`, `pixel_to_field`,
`field_to_pixel`, `is_in_field`, `draw_field_overlay`) + `run_calibrate.py`
(interactive). A cone being off-camera doesn't block calibration: verified
with a synthetic-homography test that fitting from 8 of the 10 keypoints
(skipping the two near-back corners closest to the camera — confirmed the
common case on this footage) recovers the position of the two SKIPPED points
to ~0cm error, and correctly computes the far goal line's position without
ever seeing its cones.

First version had the user click on a composited mosaic image — the person
tried it and reported no cone was visible where pixel-level analysis said one
should be. Root cause: mosaic blending washes out/distorts small objects like
cones (see `results/calibration_finding.md`'s addendum) — trust the human
eye on the real image over automated pixel analysis. Fixed: `run_calibrate.py`
now rebuilds the segment registration itself and has the user browse+click on
RAW frames (`n`/`p` to move between them within the segment), transforming
each click through that specific frame's own homography into the shared
segment coordinate system. Different keypoints can come from different
frames. Verified the transform logic end-to-end with simulated clicks on
different frames of a real segment before handing back to the user — still
needs a real interactive run to actually place points, which needs a display
this environment doesn't have. Run: `python run_calibrate.py videos/ojuc.mp4
1580 180 1.0`. Next: run it for real, then wire `is_in_field` into the B3
detections to filter out sideline/bench people.

**✅ DONE: calibration bank -- auto-reuse a calibrated camera framing instead
of re-clicking every segment** (see `results/calibration_finding.md`
Addendum 5, `frisbee_analysis/calibration_bank.py`, `run_calibrate_auto.py`).
Motivated by the person's own observation that one manual session per camera
segment doesn't scale to a full game with many cuts. Reuses `mosaic.py`'s
ORB+RANSAC registration between segments' reference frames instead of just
within one; `run_calibrate.py` now auto-adds every calibration it produces to
the bank. Verified: re-matching the same framing composed a correct
calibration with zero clicking (2,533 inliers); an unrelated window correctly
found no match and fell back to the manual flow instead of forcing a bad
one. Also tested (and rejected) a cheaper alternative first: plain HSV
color-filtering for cones had too many false positives (crowd skin tone
especially) to trust -- see Addendum 4 in the same doc.

**✅ DONE (2026-08-25): scene-aware field line detection + cross-pan
TRACKING** (see `results/field_lines_finding.md`,
`frisbee_analysis/field_lines.py`, `run_field_lines_test.py`,
`run_field_lines_propose.py`). Reverses the earlier "automated detection is
not a quick win" call, for the reason that note itself named: the old
attempts ran over the WHOLE frame with no idea which pixels were field.
Restricting to a colour+texture playing-surface mask (no geometry needed --
so the "circular without a calibration" objection was wrong), detecting
white paint as a multi-scale local-contrast RIDGE rather than a Canny edge,
and pooling detections across a pan in `mosaic.py`'s shared segment
coordinates takes one frame from 888 naive Hough segments to 6 real lines,
and recovers **3 of the 4 lines a human clicked to within 1.4-6.3px and
1.3 degrees, with no clicking at all** (torovcanyon.mp4 t=3945-4005s).
Cone hints are usable too now -- subtracting YOLO person boxes kills the
skin-tone confound that sank Addendum 4 (on-surface false positives 17->0
and 15->1) -- but real cones are only ~3-5px at broadcast distance, so they
stay corroboration, never anchors.

**Output shape (2026-08-25, per the person)**: `frisbee_analysis/field_model.py`
emits exactly SIX connected lines -- two sidelines, two end lines, two goal
lines -- each tagged `observed` (paint was detected) or `estimated` (position
computed from the homography + WFDF dimensions). They connect by construction,
since all six come from one homography applied to `ultimate_config.py`. This
matters on `torovcanyon.mp4`, where half the ultimate boundary is not painted
at all (near sideline is BLUE, far sideline is white only inside the endzones)
so the far sideline across midfield can only ever be estimated. Labelling
search now enforces ORDER PRESERVATION (a homography preserves the order in
which parallel lines cross a transversal), which fixed the blue line being
mislabelled an end line and cut the best model's worst residual 215px -> 29px.

**Camera coverage awareness (2026-08-25, per the person)**: a broadcast shot
almost never frames the whole field. `segment_pan_polygon` gives the mosaic
footprint the camera actually swept; `visible_extent` pushes it through the
fitted homography to say which part of the field is in view, in metres; every
model line is tagged `in_frame`/`in_pan`/`unseen`. Two gates follow from it --
a line the camera never pointed at cannot be "observed", and the implied
coverage must be physically sensible for the known field size (this caught a
fit claiming its pan swept 164m across a 37m-wide field). Also measured:
ranking by "how much paint do these six lines explain" rewards a WRONGLY
SHRUNKEN field, since more of the soccer markings underneath fall inside it.

**CORRECTED (2026-08-25, per the person): canvas_002 is LOCAL, not broken.**
Round-trip conditioning is 0.01-0.04 m per image pixel across x=0-60m -- sharp
everywhere the camera sees. Its horizon is TILTED across the field (vanishing
at x=64.4m along one sideline, 117.5m along the other), so only the unobserved
far end is meaningless. New `calibration.field_validity_region` reports how far
a calibration can be trusted (canvas_002: 54.7m = 54.7% of the field, which
matches the independently-measured 52-56% the camera sweeps).
`calibration_covers_evidence` replaces field_in_front_of_camera as the veto
everywhere -- canvas_002 passes and would now be banked; canvas_004 (27.9m,
fails on its own evidence) does not. The earlier "re-do the torovcanyon
calibrations" action is retracted except for canvas_004.

**Superseded note on canvas_002's "defect"**: not a mislabel and not a duplicate
line -- the fitted camera puts part of the field BEHIND itself, i.e. the
horizon cuts through the playing surface (vertices 5 and 7 have the opposite
homogeneous sign to the rest). New check `calibration.field_in_front_of_camera`
separates the bank perfectly: canvas_000 True, canvas_002/003/004 False, five
synthetic cameras True. Also added: `field_covers_points` (a fit must contain
the pixels it was fitted from -- catches COLLAPSE, which vertices_within_bounds
structurally cannot), `field_lines.no_parallel_crossings` (lines assigned to
parallel edges can't meet inside the paint), and a one-frame-per-calibration
rule in run_calibrate.py. The duplicate-line hypothesis was measured and
refuted: chained vs direct registration between frames 0 and 12 diverges by
only 1.3-2.9px and re-deriving the line directly moves its angle 0.02 deg.

**Search bug FOUND AND FIXED (2026-08-25)**: group_by_vanishing_point returns
ONE split (largest VP cluster vs everything else), making the second "family"
a grab-bag rather than a direction. New `field_lines.direction_hypotheses`
enumerates every plausible VP pair (40 on the test window). Models found 0 ->
3, blue-as-sideline candidates enumerated 672 -> 1,684, runtime 2m30s -> 46s.

**WATCHER-AS-SEARCH-PRIOR: MEASURED, DOES NOT WORK (2026-08-26)**. Sweeping
perpendicular offsets from the watcher row, the paint profile peaks at +32px
(37.9%) while the TRUE far sideline sits at -11.2px with 2.6% density; every
windowed variant peaks at +28..+44px too. Root cause found by sampling each
stage along the person's clicked line: (a) V reads 73-99 against turf 52-78 --
barely a step, so the ridge test fires on 3 of 9 samples and chroma on 1; and
(b) THE SURFACE MASK EXCLUDES x=709-919, the endzone end that actually has
paint, so part of the line isn't even in the search region.
NEXT CONCRETE FIX: (b). The search region is the surface dilated 15px@1920,
not enough where the boundary is ragged -- and a sideline is definitionally AT
that boundary. Upstream of the problem, unlike everything else tried.

**REGISTRATION DRIFT ADDRESSED (2026-08-26)**: `mosaic.refine_to_anchor`.
Measured: chained homographies diverge from a direct anchor registration by a
median 2.2px at chain depth <=20 and 7.7px past depth 40 -- which is the whole
gap between human-clicked lines (0.5-2.7px) and tracked lines (rms 4.0-8.4px).
Direct-to-anchor alone is not viable (28/59 frames, failures coverage-limited
at 7-14% vs the 15% floor). refine_to_anchor makes every directly-registerable
frame a hub and routes the rest through one hop: depth "up to N" -> 1 or 2, 20s
for 30 frames, 15/29 direct. CAVEAT: the 0.00px divergence for direct frames is
tautological; the 2-hop benefit follows from depth but is not independently
measured. Wired into run_field_lines_propose.py, non-destructive.

**REGRESSION TEST ADDED (2026-08-26)**: `run_regression_test.py` +
`tests/fixtures/torovcanyon_3945.json`. 22 assertions pinning the
human-confirmed labelling through every gate, the LOCAL-not-broken
distinction, the metrics validity bound, decoy rejection, six-line model
shape, and confusable-pair surfacing. Runs in seconds, no video needed.
Section 7 reintroduces each historical bug and asserts the matching check
flips -- all three caught. Motivated by the fact that three gates rejecting
ground truth, a ranking term rewarding wrong answers, and a check that
returned False for everything were ALL caught by the person, never by code.
Run it before touching field_lines.py or field_model.py.

**METRICS UNBLOCKED (2026-08-26)**: `calibration.is_measurable` adds the
validity bound that `is_in_field` structurally lacks -- 14% of what is_in_field
accepts on canvas_002 lies past its 54.7m horizon, where positions (and any
speed from a pair of them) are noise. run_metrics_test.py now filters with it
and prints the bound + drop count. Re-run t=3945s+30s: 72 usable speed samples
(vs 8 originally), team A 1.16 m/s mean, team B 1.07 m/s, explicitly covering
only the near 55% of the field. try_auto_calibrate now SKIPS a canvas whose own
calibration fails calibration_covers_evidence (canvas_004, 27.9m) while still
serving LOCAL ones (canvas_002/003, 54.7m). results/metrics_finding.md carries
a correction note. REMAINING: canvas_004 needs a manual redo.

**GATE STACK WAS REJECTING GROUND TRUTH (2026-08-26)**: the full search
returned 0 of ~81,000 candidates. Two gates I built -- projected_field_is_sane
and extent_is_plausible -- rejected the person's verified-correct labelling,
both by judging field coordinates PAST THE HORIZON (same error as
field_in_front_of_camera in Result 3.14). Fixed by clipping both to
field_validity_region, plus robust percentile bounds instead of min/max in
_bbox_in_field. All 9 gates now pass the correct labelling; canvas_004 is still
correctly rejected. RULE: any check reading a projected field coordinate must
first ask whether it is inside field_validity_region.

**STILL BLOCKING**: the true far sideline is not in the tracked candidate set,
so the search picks tracked[3] (the NEXT FIELD OVER, y~50m per the person's
watcher measurement). NOT caused by extrapolation merging -- tested both ways,
30-32px either way. The endzone paint stub is not surviving detection as its
own line: short, adjacent to a stronger parallel line, partly occluded by the
gear pile and standing players (Result 3.12).

**BREAKTHROUGH (2026-08-25, measured by the person): watchers mark the
unpainted sideline, and it is the FIRST signal that discriminates.**
`frisbee_analysis/watchers.py`. YOLO foot points across a segment, fused in
reference coords, filtered by persistence (>=20 distinct frames within 12px)
and RANSAC'd into rows. On torovcanyon t=3945-4005s: 29 standing positions
behind 1,435 detections, straight to 4.20px over 1,210px, sitting +4.0m
outside a 37m field. Reproduces the person's numbers closely (their 1,060 pts
/4.5px/1,330px, box 47px vs 46.6px, y 41.1m vs 41.0m).
`watchers_outside_field` is a labelling GATE -- watchers stand outside the
boundary -- and it SEPARATES the pair that defeated support, length,
persistence, flank-matching, VP structure and reprojection error: True for the
person's correct labelling, False for the wrong far sideline. Wired into
fit_field_model as an optional watcher_rows gate.
CAVEATS: it is a DIRECTION cue not a POSITION one (standoff ~4m and drifting
along the field; standoff_from_paint is the route to fixing that using endzone
paint where people and paint coexist). And a straight row of people is not by
itself a sideline -- the fence/path row is STRAIGHTER (1.0px); separate on box
height (36 vs 47px), not straightness.

**RETRACTION (2026-08-25)**: the claim that the tracker's far-sideline estimate
beat the person's clicks (37% vs 11% paint under it) was WRONG -- that detection
follows the turf edge BEHIND the row of standing players, diverging from the
true sideline by 73px across the frame. Paint support cannot distinguish a field
marking from a high-contrast scene boundary parallel behind it, and on this
footage the far sideline is exactly where such a boundary exists. Fitting the
person's four clicked lines directly gives 2.74px max reprojection (vs 29.4px
for every fit that used the detection) with residuals 1.8/0.7/0.6/0.0px.
Standing weakness: every ranking in field_model.py leans on support/length/
persistence, which measure evidence STRENGTH, not correctness of identity.

**CORRECT LABELLING CONFIRMED (2026-08-25)**: canvas_002's original labelling
is right -- (1,7) blue near sideline, (2,8) far sideline (white, endzone only),
(1,2) end/back line, (3,4) near goal line. Fitted on the tracked lines it gives
1.5 / 6.7 / 2.4 / 1.8px with both far-end lines correctly `estimated unseen`.
Fixed a real bug found doing this: `_bbox_in_field` projected only polygon
corners, which through a LOCAL calibration blow up past the horizon (pan extent
came out as y=-163.9..6.6m on a 37m field), poisoning
extent_supports_observation so build_field_model demoted its OWN fitted lines
to `estimated`. Now samples the region and keeps only points in front of the
camera. AUTO-RANKING STILL DOES NOT PICK THE RIGHT LABELLING -- use --assign.

**RESOLVED (2026-08-25)**: the person asked why the sideline couldn't just be
extended from the white paint inside the endzone -- which is exactly what a
line correspondence does. Cones were never needed. Two over-constraints in the
labelling search were hiding every valid model: (1) requiring the CROSS lines
to share a vanishing point, which the long line + short endzone stub never do
(fixed: only the sideline pair must be concurrent, geometry gates handle the
rest); (2) ranking sideline families by member count, which buried the
{blue, far sideline} cluster at rank 24 of 61 despite it carrying 431k support
vs the top cluster's 208k (fixed: rank by evidence). Top model is now blue as a
sideline at 1.5px residual, far sideline from endzone paint at 6.7px, near end
line 2.0px, near goal line 1.8px, both far-end lines correctly `estimated
unseen`. Also measured: the tracker's version of the far sideline is BETTER
than the original 3 clicks (37% vs 11% paint under it).

Remaining ambiguity is genuine, not a defect: which END of the field the camera
is at (symmetric geometry), and which sideline is called "left" (convention).

**What still needs a human, and why it is structural, not tuning**:
deciding WHICH detected line is which. Four line correspondences determine a
homography exactly, so every candidate labelling reproduces its own lines
perfectly and the reprojection error says nothing about correctness
(measured: a deliberately scrambled labelling scored 7.11px and passed every
check). And a broadcast framing often shows only ONE sideline. So
`run_field_lines_propose.py` detects and tracks automatically, then asks the
person to name the lines it found -- much less error-prone than clicking
points along lines they have to locate by eye.

**⚠️ ACTION NEEDED: re-do the `torovcanyon.mp4` calibrations.** Building the
above turned up a real bug: the calibration behind bank `canvas_002` (and
`canvas_003`, and `outputs/calibration.json`) folds the field into a
self-intersecting bowtie -- vertex 7 at (-1017,-157) while vertex 8, on the
same back line, lands at (8050,1076). Reprojection error (2.74px) and
`vertices_within_bounds` (True) BOTH pass it, because neither can see the
shape of the projected field. NOTE: an earlier version of this entry blamed
a sideline "clicked on bare turf" -- that was WRONG (the person corrected
it: those clicks are on the field's BLUE near sideline, which the
brightness-only detector of the time could not see). All four clicked lines
are on real paint; which label is inconsistent is not yet identified, and
re-fitting the same four correspondences from the tracker's own longer
lines still fails the check. Three of four bank canvases are affected, so
the `is_in_field` filtering and speed numbers in
`results/metrics_finding.md` are unreliable for that video until it is
redone. New check `projected_field_is_sane` is now wired into
`run_calibrate.py`, `calibration_bank.add_entry` and `run_calibrate_auto.py`
so a repeat gets caught at click time.

**Measured scope**: works on `torovcanyon.mp4` (turf with painted markings);
correctly finds nothing on `ojuc.mp4` (bare grass, cones only -- there is no
paint to find, so that footage still needs `run_calibrate.py`); fails on
`videoplayback.mp4` at 640x360 for a measured reason, and that is a UFA-size
field rather than the WFDF geometry this targets anyway.

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
- Seed annotation candidates found on Roboflow Universe (all CC-BY-4.0),
  API key now set up (`.env`, gitignored):
  - [Ultimate Player](https://universe.roboflow.com/ultimetrics/ultimate-player) —
    1,626 images, player class. **CORRECTED**: previously said "pretrained
    model available" -- wrong, MEASURED via direct API query: `versions: 0`,
    zero trained models exist, dataset only.
  - [Tracking](https://universe.roboflow.com/frisbee-tracker/tracking-w8biu) —
    423 images, frisbee/observer/player classes, 26 trained versions.
  - [Frisbee Tracking](https://universe.roboflow.com/tracking-f1jov/frisbee-tracking) —
    521 images, disc/observer/player classes, 3 trained versions.
  - (Disc-focused ones exist too — [Ultimate Frisbee Disc Detector](https://universe.roboflow.com/eelke-van-foeken-sfimp/ultimate-frisbee-disc-detector),
    509 images — not needed for #3/#4, keep in mind if disc detection is revisited.)

  **✅ DONE, MEASURED: fine-tuned locally on "Tracking", no clear win over
  generic COCO on the person's real footage** (see `results/finetune_finding.md`,
  `run_download_roboflow_dataset.py` + `run_finetune_detector.py` +
  `run_finetune_compare_test.py`). None of the three Universe projects offer
  downloadable pretrained weights via the free API -- only hosted cloud
  inference (footage-upload concern) or a dataset to train yourself.
  Downloaded the dataset, fine-tuned `yolo11n` locally (25 epochs, CPU-only,
  ~3 hours, nothing of the person's footage ever uploaded). Strong metrics
  on Roboflow's own val split (player mAP50=0.957) did NOT transfer to a
  visible improvement on the person's actual footage in a direct side-by-
  side (comparable detection coverage, no fewer false positives, no better
  recall on two spot-checked frames) -- real domain shift. Keeping generic
  `yolo11n.pt` for player detection; the fine-tuned model's only unique value
  is `frisbee`/`observer` classes (weak but nonzero) that COCO categorically
  lacks, relevant only if disc detection comes back into scope.
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

**✅ First end-to-end pass DONE, plumbing confirmed, numbers not yet
trustworthy** (see `results/metrics_finding.md`, `run_metrics_test.py`).
Wires everything built so far together for real: registration
(`mosaic.py`) → calibration (`calibration.py`) → YOLO detection + ByteTrack
→ team classification (saturation split) → `is_in_field` filter → team-level
on-field counts + instantaneous speed (m/s), using consecutive-*sampled*-
frame track matches only (sidesteps the ~90% fragmentation from
`results/tracking_finding.md` rather than solving it — no full-lifespan
identity needed for instantaneous speed). Visual check: `is_in_field`
correctly dropped two people standing on the running track outside the
sideline. Numbers themselves (team A/B mean speed ~1.3–1.5 m/s) are **not
trustworthy yet** — only 8 speed samples from one 12s window, and the
calibration in use predates `line_details` so its own reprojection error
is unverified.

Next: re-run `run_calibrate.py` (current version, saves `line_details`) and
check reprojection error before trusting anything downstream; then re-run
`run_metrics_test.py` over a longer/wider window (ideally spanning most of a
point) for a real sample size. Full `Track`-schema emission (positions over
a whole point, not just per-frame counts/instant speed) is still open —
`run_metrics_test.py` is a metrics-shaped proof of concept, not yet that
object.

---

## Hard constraints to remember
- Pi Zero 2 W CANNOT run this pipeline (no GPU; SigLIP/YOLO need one). Process on
  a laptop/desktop with a GPU. Keep any "run live on the Pi" idea out of scope.
- Existing footage only — can't refilm, can't add a second camera.
- Scope: the person has other summer projects. Track A + a reduced Track B is a
  realistic summer; the full thing is not.
