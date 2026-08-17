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

## What's here

```
frisbee_analysis/         the package
  schema.py               Track dataclass — the data contract everything speaks
  ufatrack_loader.py      loads real UFATrack CSVs -> Track (verified schema)
  features.py             kinematic + marking features
  possession.py           HMM/Viterbi possession detector + naive baseline
  evaluate.py             transition metrics + event derivation
  ultimate_config.py      WFDF field geometry (for the future CV pipeline)
run_pivot_test.py         the pivot-rule viability test (runs on real data)
run_viability.py          hand-set-config harness (frame acc, transition F1, turnovers)
run_tune.py               cross-validated search + honest go/no-go verdict
ufatrack_data/            the real UFATrack dataset (20 possessions, CC-BY-4.0)
NEXT_STEPS.md             prioritised task list for continuing
CONTEXT.md                full design rationale + decisions made + dead ends
results/pivot_finding.md    the measured pivot-rule result, written up
results/tuning_finding.md   the measured tuning + go/no-go result, written up
```

## Run it

```bash
pip install numpy scipy
python run_pivot_test.py     # the pivot-rule signal on real data
python run_viability.py      # hand-set configs: frame acc, transition F1, turnovers
python run_tune.py           # cross-validated search -- the actual go/no-go number
```

Both default to `ufatrack_data/`. No GPU, no video needed.

## The one rule that matters most

**Evaluate against `holder_id` (ground truth), never against disc position.**
In UFATrack the disc position is *reconstructed from* the holder annotation by
interpolation — fitting to it is circular. This is baked into the loader (the
disc row is dropped) but keep it in mind for any new code.

## Data attribution

UFATrack: open-starlab/UFATrack, CC-BY-4.0. Oakland Spiders vs Salt Lake Shred,
2025-06-27, 2nd quarter. Cite it if this becomes public / portfolio work.
