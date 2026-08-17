# Player Detection Finding (measured on real footage)

Closes the first half of `NEXT_STEPS.md` B3 — the cheap feasibility question:
before investing in the seed-dataset download + fine-tuning pipeline, how far
does an off-the-shelf, zero-custom-training detector get? Same "measure
before building" approach as B1.

## Method

Ran YOLO11n (COCO-pretrained, `ultralytics`, class 0 = "person", no ultimate
frisbee-specific training at all) on a single representative wide gameplay
frame from the real footage (`videos/ojuc.mp4`, not committed). Compared
default inference settings against full-resolution inference, and checked the
result by eye, not just by detection count (same "build the visual" lesson
as `results/mosaic_finding.md`).

## Result

| config | detections (conf≥0.25) | far-field-sized (<0.15% frame area) |
|---|---|---|
| default (imgsz=640, YOLO's own default) | 20 | **0** |
| imgsz=1920 (native resolution) | 69 | **41** |
| imgsz=1920, conf≥0.05 (looser) | 130 | 90 |

**Default settings completely failed at the actual task.** All 20 detections
at imgsz=640 were near-camera sideline/bench people — the entire field of
play, containing dozens of visibly-detectable players, had ZERO detections.
This isn't a capability gap in the model; it's a resolution artifact: YOLO
downscales the full 1920×1080 frame to its inference size, and at 640px an
already-small far-side player shrinks below what the network can resolve.
This is exactly the risk `NEXT_STEPS.md` flagged: "measure far-side-player
recall specifically."

**Running inference at native resolution (imgsz=1920) fixes it.** 41 of 69
detections were far-field-sized, and eyeballing the annotated frame
(`outputs/detection_sample.jpg`, local only — not committed, real people)
confirms real coverage across the visible field, including players near both
edges, not just false positives inflating the count.

## Verdict

**Zero-custom-training YOLO is a viable starting point for B3 — with one
required setting, not the library default.** `run_player_detection_test.py`
now defaults to `imgsz=1920` for this reason and says so in its docstring, so
this doesn't silently regress back to the failing case.

This meaningfully de-risks B3: a usable player-position signal may not
require the full seed-dataset-download + fine-tuning investment before
getting *something* working end-to-end. Fine-tuning on real ultimate footage
(the original B3 plan) still matters for recall/precision at production
quality and for anything beyond generic "person" (jersey numbers, team
color), but "start from scratch" is no longer the only option before a first
working pipeline exists.

## Note on the source frame

The test frame and annotated output come from the person's own uploaded
broadcast footage. Neither is committed to this repo — see
`results/mosaic_finding.md`'s note on the same point. Only aggregate
measurements are recorded here.
