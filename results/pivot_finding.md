# Pivot-Rule Finding (measured on real UFATrack)

Question: is the disc-holder the slowest / most-stationary offense player?

## Method
For every frame with a labelled offense holder (5,655 frames across 20
possessions, ~9.5 min), rank the 7 offense players by smoothed speed and record
where the true holder falls (rank 0 = slowest).

## Result

| holder's speed rank | frames |
|---|---|
| 0 (slowest) | 39.8% |
| 1 | 15.5% |
| 2 | 11.7% |
| 3 | 8.5% |
| 4 | 7.4% |
| 5 | 8.2% |
| 6 | 8.9% |

- Holder is the slowest offense player: **39.8%**
- Holder in slowest 2: **55.3%**
- Holder in slowest 3: **67.0%**
- Holder speed / offense-median speed: **0.64**

Random baseline (7 players) would be ~14.3% at rank 0.

## Interpretation
The pivot signal is **real but moderate**. Speed is a strong FILTER (rule out 4
of 7 players via top-3) but a weak PINPOINT (40% exact). This is why the detector
uses multiple features + temporal (HMM) decoding rather than speed alone — and
why the go/no-go still hinges on getting transition F1 up with tuning / disc
detection.

## Detector result (first pass, untuned)
- transition F1 ≈ 0.36 (HMM) vs 0.27 (naive) — HMM helps
- predicted 0 of 5 turnovers — must fix (switch_penalty + cross-team handling)
- accel + marking features added only ~0.02 F1 so far
