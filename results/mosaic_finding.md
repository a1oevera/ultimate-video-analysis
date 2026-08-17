# Mosaic-Registration Finding (measured on real footage)

Closes `NEXT_STEPS.md` B1 — the feasibility question: can each frame be
registered to a whole-field mosaic via background features, given the camera
often shows only part of the field?

## Method

Source: a ~90 min broadcast recording (`videos/ojuc.mp4`, ROGERS tv,
youth/club tournament, fixed elevated camera that pans/tilts/zooms — not
committed to the repo, see `.gitignore`). Sampled 10 frames evenly across the
game plus 2 frames adjacent (2s, 5s) to one of them, to compare registration
quality across different time gaps. Masked the static broadcast overlay
(scoreboard box top-left, watermark top-right) before feature detection — an
unmasked overlay would inject spurious "always-aligned" keypoints that don't
reflect real camera motion. Ran ORB (3000 features) + brute-force Hamming
matcher + Lowe's ratio test (0.75) + RANSAC homography (5.0px threshold)
between frame pairs, counting RANSAC inliers as the registration-quality
signal.

## Result

| pair | framing | time gap | good matches | RANSAC inliers |
|---|---|---|---|---|
| frame A vs frame A+2s | wide vs wide | 2s | 128 | **84** |
| frame A vs frame A+5s | wide vs wide | 5s | 103 | **74** |
| frame A vs frame B | wide vs wide | ~54s | 33 | 21 |
| frame A vs frame C | wide vs wide | ~36min | 62 | 37 |
| frame B vs frame J | wide vs wide | ~63min | 28 | 16 |
| frame C vs frame D | **wide vs TIGHT close-up** | — | 6 | **0** |

## Interpretation

**Wide-framed shots register well, especially at the short time gaps a real
incremental mosaic pipeline would actually use** (74–84 inliers at 2–5s,
comfortably enough for a stable homography). Registration quality degrades
gracefully as the time gap grows (down to 16 inliers at ~63 min) but never
collapses for wide-vs-wide pairs — the treeline, tents, and other background
structure around this field give ORB plenty to work with. This matches
`CONTEXT.md`'s prediction: "treeline/buildings good, open park bad" — this
footage has a treeline, and it shows.

**Tight close-up frames cannot be registered from background alone (0
inliers).** When the camera zooms in on a play, there's no shared background
left to match against — this is the expected hard case, now confirmed rather
than assumed.

## Verdict: B1 feasibility CONFIRMED, with a required design constraint

Mosaic registration is viable on this footage — proceed to building it. But
the pipeline MUST include **per-frame quality gating**: attempt registration
only when enough inlier matches are found (e.g. require ≥15–20 inliers, based
on the measured gap between working wide pairs and the failing tight pair);
frames that fail should have their camera pose interpolated from the nearest
successfully-registered neighbors rather than forced through a bad
homography. Given the camera is a fixed mount (pose changes smoothly, no
cuts), interpolation across a tight-zoom gap is a reasonable bridge, not a
hack.

Also confirmed necessary: mask static broadcast overlay graphics (scoreboard,
watermark, or any station bug) before feature detection on any footage that
has them — they're free "matches" that say nothing about real camera motion.

## Note on the source video

The test used the person's own uploaded broadcast footage. It is NOT
committed to this repo (`videos/` is gitignored — the file is ~2.2GB, and
broadcast video of real athletes belongs to whoever holds those rights, not
in a public repo). Only aggregate measurements are recorded here.
