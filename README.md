# Ultimate Frisbee Video Analysis — Project Handoff

Goal: an AI system that analyses ultimate frisbee game footage to produce, in
priority order:

1. **Turnovers & point outcomes** (highest priority)
2. **Passes & completion rate**
3. **Play time per player**
4. **Speeds / distance covered** (lowest priority)

This package is **Track A** — the coordinate-space possession logic, developed
and tested against real ground-truth data (UFATrack) with no video required.
It exists to answer the make-or-break question *before* building the CV pipeline:
**can the disc-holder (and therefore passes/turnovers) be recovered without
reliable disc detection?**

## Current status (honest)

**NO-GO on possession-from-motion-alone, with the current feature set.**
Measured on real UFATrack data:

- The "pivot rule" signal is **moderate**: the holder is the slowest offense
  player 40% of the time, in the slowest 3 67% of the time. Speed is a strong
  *filter*, weak *pinpoint*.
- The turnover-blindness bug is **fixed**: defense players used to get a
  permanent `-inf` emission score, making turnovers structurally impossible
  to predict regardless of tuning. Removing that veto took predicted
  turnovers from 0/5 to 3/5 (in-sample, untuned).
- But **cross-validated tuning does not clear the go/no-go bar**: 5-fold CV
  (stratified so each fold holds out one of the 5 known turnovers) gives
  **transition F1 ≈ 0.29–0.35, turnover recall 0–1 of 5 held out** — well
  under the ~0.5 GO threshold, and even under the ~0.4 "insufficient" line.
- A 400-config random search over the emission weights **did not beat the
  original hand-set weights on held-out data — it did worse** (overfitting a
  16-track selection set). See `results/tuning_finding.md` for the full
  fold-by-fold numbers and the baseline-vs-tuned comparison.

**Verdict: motion + marking features are not enough for possession from
UFATrack coordinate data alone.** Per `NEXT_STEPS.md` A4's fallback, the next
real option is disc detection as a *primary* signal (not sidestepped by this
bet), or rescoping to metrics #3/#4 (play time, speed/distance), which don't
need possession. See `NEXT_STEPS.md` and `results/tuning_finding.md`.

**Decision (2026-08-17): rescoped to Track B, metrics #3/#4.** Real footage
is now available (`videos/`, gitignored — not redistributed).

- **B1 (mosaic registration): CONFIRMED.** Wide-framed background (treeline,
  tents) registers reliably via ORB+RANSAC (74–84 inliers at realistic short
  time gaps); tight close-ups don't (0 inliers, as predicted) and need pose
  interpolation rather than forced registration. Built the real pipeline
  (`frisbee_analysis/mosaic.py`), not just the test — which caught and fixed
  a real bug (a stale registration anchor across a broadcast cut) and
  produced a visually-verified stitched mosaic. See `results/mosaic_finding.md`.
- **B3 (player detection), zero-training feasibility: STRONG.** Off-the-shelf
  COCO-pretrained YOLO, run at native resolution (NOT the library's own
  640px default, which completely missed the field of play), gets real
  detection coverage across the whole field including far-side players. See
  `results/detection_finding.md`.
- **Scene-aware field line detection + cross-pan tracking: WORKS on painted
  fields** (`results/field_lines_finding.md`, `frisbee_analysis/field_lines.py`).
  Restricting detection to a colour+texture playing-surface mask and treating
  paint as a multi-scale *ridge* rather than a Canny edge takes one frame from
  888 naive Hough segments to 6 real lines; pooling across a 60-frame pan
  recovers 3 of the 4 lines a human clicked, to within 1.4-6.3px, with no
  clicking. Deciding *which* line is which still needs a human — that limit is
  structural (4 lines fit a homography exactly, so the error can't rank
  labellings), not a tuning gap. `run_field_lines_propose.py` is the resulting
  tool. Correctly finds nothing on bare-grass coned fields.
- **⚠️ Known-bad calibrations**: the same work found that 3 of 4 bank
  calibrations project the field as a self-intersecting bowtie — one that
  2.74px reprojection error and `vertices_within_bounds` both passed, because
  neither can see the *shape* of the projected field. Which of the four
  labelled lines is inconsistent is not yet identified. `torovcanyon.mp4`
  needs recalibrating, and the speed numbers in `results/metrics_finding.md`
  are unreliable until it is. New check `projected_field_is_sane` now gates
  this at click time.
- **Output is six connected lines**, not a list of fragments
  (`frisbee_analysis/field_model.py`): two sidelines, two end lines, two goal
  lines, each tagged `observed` or `estimated` (computed from the homography +
  WFDF dimensions). Necessary because half that field's ultimate boundary is
  not painted at all.
- **Paint is not always brighter than grass.** That field's near sideline is
  BLUE: same brightness as the turf (V 56–68 vs 52–71), different hue (84–101
  vs 36–42). Detection now runs a second ridge channel on Lab chroma distance
  from the local turf colour, and reports each line's paint colour — which is
  how a person tells lines apart on a multi-sport field.

## What's here

```
frisbee_analysis/         the package
  schema.py               Track dataclass — the data contract everything speaks
  ufatrack_loader.py      loads real UFATrack CSVs -> Track (verified schema)
  features.py             kinematic + marking features
  possession.py           HMM/Viterbi possession detector + naive baseline
  evaluate.py             transition metrics + event derivation
  ultimate_config.py      WFDF field geometry (Track B)
  mosaic.py               Track B: frame-to-mosaic registration (optional cv2 import)
  calibration.py          Track B: image->field homography, lines-only fitting + checks
  calibration_bank.py     Track B: reuse an already-calibrated camera framing
  field_lines.py          Track B: scene-aware line/cone detection + cross-pan tracking
  video_overlays.py       Track B: per-video static broadcast-overlay masks
run_pivot_test.py         Track A: the pivot-rule viability test (runs on real data)
run_viability.py          Track A: hand-set-config harness (frame acc, transition F1, turnovers)
run_tune.py               Track A: cross-validated search + honest go/no-go verdict
run_mosaic_test.py        Track B: mosaic-registration pipeline test (needs real footage)
run_mosaic_visualize.py   Track B: builds an actual stitched mosaic image
run_player_detection_test.py  Track B: zero-training YOLO feasibility test
run_calibrate.py          Track B: manual line-click calibration (GUI, needs a display)
run_field_lines_test.py   Track B: measures line detection/tracking against ground truth
run_field_lines_propose.py  Track B: auto-detect lines, human names them, fit calibration
run_regression_test.py    Track B: pins the known-good answers -- RUN BEFORE EDITING field_lines/field_model
tests/fixtures/           the human-confirmed labelling, so the test needs no video
ufatrack_data/            the real UFATrack dataset (20 possessions, CC-BY-4.0)
videos/                   real footage (gitignored — not committed/redistributed)
outputs/                  generated images e.g. mosaics, detections (gitignored)
NEXT_STEPS.md             prioritised task list for continuing
CONTEXT.md                full design rationale + decisions made + dead ends
results/pivot_finding.md      the measured pivot-rule result, written up
results/tuning_finding.md     the measured tuning + go/no-go result, written up
results/mosaic_finding.md     the measured B1 mosaic-registration feasibility result
results/detection_finding.md  the measured B3 zero-training detection feasibility result
results/calibration_finding.md   the measured calibration findings (incl. what didn't work)
results/field_lines_finding.md   the measured scene-aware line detection/tracking result
```

## Run it

Track A (no GPU, no video needed):
```bash
pip install numpy scipy
python run_pivot_test.py     # the pivot-rule signal on real data
python run_viability.py      # hand-set configs: frame acc, transition F1, turnovers
python run_tune.py           # cross-validated search -- the actual go/no-go number
```

Track B (needs `pip install -r requirements.txt`'s full list, and real footage in `videos/`):
```bash
python run_mosaic_test.py videos/your_clip.mp4 <start_sec> <duration_sec> <sample_fps>

# scene-aware field lines: detect + track across a pan, then name what was found
python3 run_field_lines_test.py videos/your_clip.mp4 <start_sec> <duration_sec>
python3 run_field_lines_propose.py videos/your_clip.mp4 <start_sec> <duration_sec>
python3 run_field_lines_propose.py videos/your_clip.mp4 <start_sec> <duration_sec> 1 \
    --assign 0=3,4 5=1,2 6=2,8 2=7,8      # 3,4 = goal line; see the printed table
python run_mosaic_visualize.py videos/your_clip.mp4 outputs/mosaic.jpg <start_sec> <duration_sec> <sample_fps>
python run_player_detection_test.py videos/your_clip.mp4 <t_sec> outputs/detections.jpg
```

## The one rule that matters most

**Evaluate against `holder_id` (ground truth), never against disc position.**
In UFATrack the disc position is *reconstructed from* the holder annotation by
interpolation — fitting to it is circular. This is baked into the loader (the
disc row is dropped) but keep it in mind for any new code.

## Data attribution

UFATrack: open-starlab/UFATrack, CC-BY-4.0. Oakland Spiders vs Salt Lake Shred,
2025-06-27, 2nd quarter. Cite it if this becomes public / portfolio work.
