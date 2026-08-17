# Design Context & Decision Log

Why the project is shaped the way it is, decisions made, and dead ends already
ruled out — so you don't re-litigate them.

## The core bet

Detecting the disc directly is the hardest CV problem here — a disc is small,
motion-blurred, sky-backed, and occluded at every catch. Multiple research
efforts (Stanford CS231n projects; the UFATrack/UltimateTrack datasets) either
failed at automatic disc detection or sidestepped it by annotating the *holder*
and interpolating disc position.

So the bet: **infer possession from player motion (the pivot rule — the thrower
keeps a foot planted) instead of from the disc.** If that works, passes and
turnovers (metrics #1, #2) don't depend on solving disc detection.

**Status of the bet: NO-GO, decided.** The pivot signal is real but only
moderate (holder is slowest 40% of the time). After fixing a structural bug
that made turnovers unpredictable regardless of tuning, and cross-validating
a 400-config weight search against a no-tuning baseline, transition F1 lands
at 0.29–0.35 held-out — well under the ~0.5 GO bar, even under the ~0.4
"insufficient" line. Motion + marking features don't carry enough signal for
possession on this data. See `results/tuning_finding.md` and NEXT_STEPS A1–A4
for the full measurement and the two remaining paths (disc detection as a
primary signal, or rescope to metrics #3/#4 which don't need possession).

## Architecture: the Track schema is the seam

Everything speaks one data contract (`schema.Track`): coordinate data in, per-
frame holder out. UFATrack, the (removed) synthetic generator, and the future
YOLO pipeline all produce the SAME Track. This is why possession logic can be
built and validated today with zero footage, and will plug into the real
pipeline unchanged. Preserve this seam.

## Decisions made

- **Metrics split by calibration-dependence.** #1/#2 (possession, passes,
  turnovers) can largely be done in image space / relative motion — robust to
  bad calibration. #3/#4 (play time as position, speed, distance) NEED good
  metric homography. Lucky, because the footage situation hurts #3/#4 most and
  #1/#2 least.
- **Offline, not streaming.** Possession uses Viterbi over whole possessions
  (looks at the future to fix the present). This is why streaming SDKs are wrong
  here (see dead ends).
- **Evaluate on transition F1, not frame accuracy.** Frame accuracy is dominated
  by long holds and hides transition errors; transitions are what passes/
  turnovers are built from.
- **Fractional coordinates + FieldConfig.** UFATrack is a UFA field (109.73 x
  48.77 m); the person films WFDF (100 x 37 m). Working in [0,1] avoids learning
  field-specific spatial priors.

## Dead ends already ruled out (don't revisit without a new reason)

- **NVIDIA DeepStream / GStreamer.** Streaming/causal by construction; conflicts
  with the offline global-decode design. Solves a throughput problem the project
  doesn't have (2 video files, unlimited time). Also needs an NVIDIA GPU the
  person may not have. Rejected.
- **"Tag players once, track all game."** Trackers (ByteTrack etc.) don't
  preserve identity through occlusion — and ultimate stacks are worst-case
  occlusion. Identity must be re-anchored (at each pull) + resolved per-tracklet,
  not tagged once.
- **Broadcast/pro footage as the validation set.** Fine for detector PRETRAINING,
  wrong for validation — different camera, uniform kits, resolution mismatch.
  Validate on the person's own footage. (UFATrack pro data is fine for Track A
  because Track A is coordinate-space, not pixels.)
- **roboflow/sports `BallTracker`.** It's a centroid smoother that assumes the
  ball was already detected — useless when disc detection is the thing that
  fails. Their README even lists ball tracking as an unsolved challenge. Use
  their homography + team classifier; not their ball code.
- **Full automation (no manual input).** The person initially wanted zero manual
  input. Research teams with GPU budgets chose manual holder annotation over
  automatic disc detection. A little manual input (pull-formation tagging) makes
  identity tractable and gives ground truth for free. (This is a recommendation,
  not yet implemented.)

## Footage constraints (from the person)

- Existing footage only — no refilming, no second camera.
- Fixed high mount (good: pure rotation, clean homography) that pans/tilts/zooms.
- **Often only part of the field visible per frame** — the binding constraint.
  Drives the mosaic-registration approach (NEXT_STEPS B1).
- Some fields lined, some coned. Coned is the harder/general case; build for it.
- Bird's-eye-ish (high mount), which is good — reduces the foreshortening that
  would wreck speed/distance from a low sideline angle.

## Hardware

- Person owns a Raspberry Pi Zero 2 W. It CANNOT run YOLO/SigLIP (no GPU).
  The pipeline is a laptop/desktop-with-GPU workflow. Keep Pi out of scope.

## What was removed from this bundle

An earlier synthetic possession generator (fake data with the pivot rule baked
in) was used to build the pipeline before real data was available. Now that
UFATrack is loaded and working, the synthetic generator is dropped to avoid
confusion — real ground truth supersedes it. If you want it back for unit tests,
it generated Track objects with explicit airborne (-1) frames; note UFATrack
instead labels a holder almost every frame.

## Prompting-history note

This project was scoped across a long conversation. Key reversals worth knowing:
the pivot-rule signal was initially assumed STRONG; real data showed it MODERATE
(40%). The marking feature was hypothesised to add real signal; first pass shows
~0.02 F1. A third: tuning the emission weights via a wide random search was
assumed to help (or at worst do nothing); cross-validated, it actively hurt —
it overfit a 16-track selection set and generalized worse than the original
hand-set weights (see `results/tuning_finding.md`). All three corrections came
from measuring, not assuming — keep that habit, especially "more tuning" when
the ground-truth sample is this small.
