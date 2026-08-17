# Tuning Finding — turnover fix + cross-validated search (measured on real UFATrack)

Closes `NEXT_STEPS.md` A1 (fix turnover blindness), A2 (tune + cross-validate),
A4 (go/no-go verdict).

## A1 — the turnover fix

`_emission_logprob` (`frisbee_analysis/possession.py`) gave every defense
player `-inf` emission at *every* frame, unconditionally. This was an
absolute veto: no value of `switch_penalty` could ever let Viterbi select a
defense player, so a turnover (holder switching teams) was structurally
impossible to predict, regardless of tuning. Fixed by:
- Making `dist_to_defender` symmetric (renamed `dist_to_marker` in
  `features.py`): nearest-opponent distance for every player, offense or
  defense, not just offense.
- Removing the veto in `_emission_logprob`; defense players now get a real
  score, reduced by a new `defense_bias` (default 3.0) log-cost prior, since
  defense holding the disc is rare and should need real evidence to select.

**Immediate effect** (untuned defaults, `run_viability.py`, in-sample on all
20 tracks): turnovers predicted went from **0/5 → 3/5**. F1 unchanged
(~0.34-0.35). Confirms the structural blindness is gone — see A2 below for
whether tuning helps beyond that.

## A2 — cross-validated search

**Method.** Only 20 possessions exist, and only 5 contain a ground-truth
turnover (tracks 4, 8, 9, 11, 20 — one each, always the possession's final
segment). Used stratified 5-fold CV: each fold's held-out set = 1 turnover
track + 3 non-turnover tracks. Nested per fold: random-searched 400+3
(3 hand-set seed configs + 400 random draws) `HMMConfig` candidates over
`w_rel_speed` (0.5–6.0), `w_accel` (0–3.0), `w_dist_def` (0–3.0),
`switch_penalty` (0.5–10.0), `defense_bias` (0–8.0) on the *other* 4 folds
(16 tracks), picked the best by mean transition F1, then scored that
untouched on the held-out fold. Compared against a **no-tuning baseline**:
the fixed hand-set default (`w_rel_speed=3, w_accel=1, w_dist_def=1.5,
switch_penalty=4, defense_bias=3`, the "speed+accel+mark" config already in
`run_viability.py`) scored on the exact same held-out folds, with zero
fitting — this isolates whether the search adds real signal or just
overfits.

## Result

| fold | held out (turnover track) | tuned held-out F1 | tuned turnover | baseline held-out F1 | baseline turnover |
|---|---|---|---|---|---|
| 1 | 4  | 0.253 | MISS | (part of aggregate below) | |
| 2 | 8  | 0.357 | MISS | | |
| 3 | 9  | 0.279 | MISS | | |
| 4 | 11 | 0.410 | MISS | | |
| 5 | 20 | 0.154 | MISS | | |

| | mean transition F1 | range | turnover recall |
|---|---|---|---|
| **Search-tuned (5-fold CV)** | 0.291 | 0.154–0.410 | **0/5** |
| **No-tuning baseline (same folds)** | **0.346** | 0.222–0.454 | **1/5** |
| *(for reference)* untuned, in-sample on all 20 | 0.346 | — | 3/5 |

**The 400-config random search did not beat the fixed hand-set baseline on
held-out data — it did worse, on both F1 and turnover recall.** It overfits
the 16-track selection set: transition F1 on 16 possessions with few
transition events is a high-variance objective over a 5-parameter continuous
space, and a wide blind search finds configs that score well on that noise
rather than on real signal. (The "final config fit on all 20 tracks" that
`run_tune.py` also prints, F1=0.404 in-sample, is subject to the same
overfitting risk and should not be trusted as a generalization estimate
either — it's shown only for completeness.)

## Verdict (A4)

Per `NEXT_STEPS.md`'s stated go/no-go bar (F1 > ~0.5 → GO, < ~0.4 → NO-GO):
**both the tuned and untuned held-out estimates (0.29–0.35) are well under
0.4.** The turnover-blindness *bug* is fixed (turnovers are no longer
structurally unpredictable — 3/5 in-sample, 1/5 held-out with zero
overfitting risk), but overall transition quality is not close to usable, and
tuning within this feature set does not close the gap — it can actively hurt,
given how little data there is.

**NO-GO on possession-from-motion-alone with the current feature set.**
Per `NEXT_STEPS.md` A4's own fallback: next steps are either (a) disc
detection as a *primary* signal rather than an assist (a harder CV problem,
not sidestepped by this bet), or (b) rescope to metrics #3/#4 (play time,
speed/distance), which don't need possession and were already flagged as
"still a real project" on their own.

## Lesson worth keeping

Prior corrected assumptions in this project (see `CONTEXT.md`): pivot signal
assumed strong, measured moderate (40%); marking feature assumed helpful,
measured +0.02 F1. Add a third: **tuning was assumed to help (or at worst do
nothing) — measured, it made held-out performance and turnover recall worse**
via overfitting on a 16-track selection set. With this little ground-truth
data, a narrow/regularized search (or no search at all) beats a wide blind
one. Keep measuring before trusting "more tuning" as a fix.
