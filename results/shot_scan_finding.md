# Shot-Scan Finding: how many distinct camera framings actually exist

Prompted by: "do a scan for how many shots" -- checking whether the
"broadcast reuses a small set of recurring framings" hypothesis (behind the
calibration bank) actually holds, and building the "skip non-gameplay shots"
filter discussed alongside it.

## Method

`run_shot_scan.py`: samples a window every 5s, detects camera cuts (same
incremental logic as `mosaic.py`'s `register_sequence`), then clusters cuts
into distinct recurring framings (match-or-new-cluster against every
cluster found so far, same mechanism as `calibration_bank.py`).

## Result 1: the first ~15-20 minutes of this video is NOT the game

First test ran on the first 10 minutes and found 46 clusters -- checked by
eye and every single one sampled was a produced highlight/intro package
(title cards, dramatic slow-motion catches from a different camera, player
introduction graphics), not the live broadcast. The actual live game camera
(the familiar wide elevated view used throughout this project's other
findings) doesn't start until roughly the 20-minute mark. **Any future
batch processing of this video needs to skip the intro, not just start at
t=0.**

## Result 2: the broadcast cuts between MULTIPLE distinct cameras, not one panning camera

Re-tested on 15 real minutes of gameplay (20:00-35:00). Checked several
cluster representative frames by eye: found genuine ground-level sideline
shots, close-up thrower-tracking shots, and referee/player close-ups mixed
constantly with the elevated wide view. This is a real production style
(cut to different physical cameras), not just one fixed camera's pan/zoom --
corrects the assumption that "fixed mount" meant "one shot to calibrate."

## Result 3: the grass-color filter usefully cuts volume, imperfectly

Built `frisbee_analysis/shot_filter.py` (`looks_like_field_view`): plain
HSV grass-green fraction, threshold 0.35. On the 15-minute test: **102 raw
cuts -> 43 field-view candidates** -- more than half discarded as
presumed-uncalibratable ground-level/close-up/crowd shots. Spot-checked
several filtered-in frames: correctly kept a real (if zoomed-differently)
elevated view. One known false positive during threshold-picking (not from
this run): a near-ground sideline shot with heavy background tree foliage
scored 0.50 (foliage is also green) -- a texture-based refinement (edge
density within the green mask, hypothesis: turf smoother than foliage) did
NOT cleanly fix this, foliage scored LOWER edge density than turf in that
case, opposite of the hypothesis. Left as an imperfect candidate filter (see
the module docstring) rather than chasing a perfect classifier -- a false
positive here just costs one wasted downstream calibration attempt.

## Result 4: even restricted to real field views, ~30 distinct framings in 15 minutes

This is the important correction to the original plan. Among the 43
field-view candidates, strict pairwise matching (same mechanism the
calibration bank uses) found **30 distinct clusters** -- not the "handful of
recurring framings" hypothesis expected. The elevated camera's own zoom/pan
RANGE is wide enough that different zoom states of the SAME physical camera
often don't share enough visual overlap to register against each other,
even though a human would recognize them as "the same broadcast angle."

## Verdict: the calibration burden is real, but the master mosaic (not just the bank) is the right next investment

The current calibration bank (matching a new shot against separate saved
reference frames one at a time) will keep asking for manual calibration more
often than originally hoped, because strict pairwise matching under-counts
how much these framings actually have in common -- two zoom states of the
same camera might not match EACH OTHER directly, but could both plausibly
match a large accumulated mosaic that has coverage from both. This is
exactly the generalization the person proposed (mosaic-based matching
instead of frame-pair matching) -- these numbers are the concrete case for
building it, not just a nice-to-have.

## Recommendation

1. Any batch/automated processing of this video must skip roughly the first
   20 minutes (produced intro, not the game).
2. The grass-color filter (`shot_filter.py`) is worth keeping wired into any
   batch scan -- real, measured volume reduction, cheap, and false positives
   are low-cost (caught downstream by calibration itself failing).
3. Don't expect the current bank to reduce manual calibration to "a
   handful" of sessions for a full game -- ~30 distinct framings per 15
   real minutes suggests a much larger number across a full game using
   strict pairwise matching alone.
4. The master-mosaic upgrade (collage-style, not blended -- see
   `calibration_finding.md`'s Addendum 4 washout note) is the next
   worthwhile investment specifically BECAUSE of this finding, not despite
   it -- it should recover matches that strict single-frame-pair matching
   misses.
