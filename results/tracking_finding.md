# Multi-Frame Tracking Finding (measured on real footage)

Follows up `results/detection_finding.md`'s B3 next step: track detections
across frames instead of scoring per-frame boxes in isolation. Ran YOLO11n
(imgsz=1920, per the earlier finding) + `supervision`'s ByteTrack over two
15-second real windows (120 frames each, ~8fps sampling) and measured track
persistence, not just whether tracking runs at all.

## Method

`run_tracking_test.py`: per frame, detect (`classes=[0]`, conf 0.25,
imgsz=1920), feed into `ByteTrack.update_with_detections`, and track each
assigned `tracker_id` across the window — how many frames it's actually
detected in, how many unique IDs get created, and what fraction are
"stable" (span ≥80% of the window) vs. fragmented (a handful of frames, or
even just one, before the tracker loses and re-creates them as a new ID).

Ran it on two different real scenes for comparison: a **tight, cluttered
shot** (huddle/sideline area, t=320-335s) and a **wide, open-field shot that
pans/zooms during the window following the play** (t=270-285s).

## Result

| | tight/cluttered window | wide/panning window |
|---|---|---|
| mean detections/frame | 42.2 | 12.2 |
| unique track IDs created | 152 | 94 |
| mean frames each track actually detected | 33.3 | 15.6 |
| single-frame tracks (never re-matched) | 15/152 (10%) | 7/94 (7%) |
| **stable tracks (≥80% of window)** | **18/152 (12%)** | **4/94 (4%)** |

**Both scenes show heavy fragmentation, for different reasons.** In the
tight/cluttered window, visually inspecting the last frame
(`outputs/tracking_sample.jpg`, local only) shows track IDs already in the
140s–150s range for a handful of visibly near-stationary people — the same
few physical people cycling through many IDs, consistent with dense
occlusion/crossing bodies breaking IOU-based association. In the wide/panning
window, the far lower stable-track fraction (4%) lines up with
`results/detection_finding.md`'s finding that distant players sit right at
the edge of detectability — a player who drops below threshold for even one
frame breaks the track. The wide window's last frame happens to be a
motion-blurred mid-air disc catch with no confident detections at all — fast
action hurting recall right when a play is actually happening is itself a
relevant data point, not just a rendering fluke.

## Verdict

**Naive per-frame detection + off-the-shelf tracking does NOT give usable
persistent identity.** ~90% of created track IDs in both windows are
fragments, not full-window tracks. This is a measured confirmation — not
just the prior expectation — of what `CONTEXT.md` already ruled out as a
dead end: "tag players once, track all game" doesn't survive occlusion, and
ultimate stacks/huddles are close to worst-case for it.

This is exactly why `NEXT_STEPS.md` B4 already specifies re-anchoring
identity at each pull (a clean, wide, spaced-out reference frame) plus
per-tracklet appearance voting, rather than trusting continuous tracking
across a whole point. Nothing here changes that plan — it measures the
problem the plan was already designed around, on real footage instead of
assumption.

## Note on the source frames

The annotated sample images are local only (`outputs/`, gitignored) — real
people's faces and jersey numbers, not committed. Only aggregate
measurements are recorded here.
