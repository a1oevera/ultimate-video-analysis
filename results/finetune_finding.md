# Roboflow Fine-Tune Finding (measured, corrects a wrong assumption in NEXT_STEPS.md)

Prompted by: "should i run these on roboflow instead of here" -- checking
whether a Roboflow Universe pretrained ultimate-frisbee detector could
replace the generic COCO-pretrained `yolo11n.pt` this project uses zero-shot.

## Method

Queried all three Roboflow Universe candidates listed in `NEXT_STEPS.md` B3
directly via the API (not just the Universe listing page) for their actual
trained-version count:

| project | images | trained versions |
|---|---|---|
| Ultimate Player (ultimetrics) | 1,626 | **0** |
| Tracking (frisbee-tracker) | 423 | 26 |
| Frisbee Tracking (tracking-f1jov) | 521 | 3 |

**Correction to NEXT_STEPS.md**: it said "pretrained model available" for
Ultimate Player. Wrong -- `versions: 0` in the API response means zero
trained models exist for that project at all, only the raw annotated
dataset.

Checked what Roboflow's free API actually offers even for the two projects
that DO have trained versions (confirmed via `roboflow.core.version.Version`
source): only (a) hosted cloud inference (sends frames to Roboflow's
servers -- same redistribution concern that keeps `videos/` out of this
repo, now sharper since it'd be the person's OWN footage going out) or (b)
downloading the labeled DATASET for training your own model. No path to
download ready-made weights and run them locally without training first.

**Chose the fully-local option**: downloaded the "Tracking" dataset (423
images, classes `frisbee`/`observer`/`player`, CC BY 4.0) via
`run_download_roboflow_dataset.py`, fine-tuned `yolo11n.pt` on it locally
via `run_finetune_detector.py` -- 25 epochs, imgsz=640, CPU-only (measured
~8.8 min/epoch from the first epoch, ~3.0 hours total). Nothing of the
person's own footage was ever uploaded anywhere; only the public dataset was
downloaded.

## Result

Training-set metrics (Roboflow's own val split, NOT the person's footage):

| class | precision | recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| player | 0.965 | 0.883 | 0.957 | 0.581 |
| observer | 0.782 | 0.667 | 0.574 | 0.353 |
| frisbee | 0.752 | 0.119 | 0.156 | 0.065 |

Strong player numbers -- but this is a DIFFERENT dataset (different camera,
compression, jersey colors) from the person's own broadcast video, so this
alone says nothing about real-world performance. Frisbee/observer are weak,
consistent with this project's established read that disc detection is
genuinely hard (see `CONTEXT.md`'s core bet section).

**Direct side-by-side on the person's real footage** (`run_finetune_compare_test.py`,
same frames used throughout `results/metrics_finding.md`): generic COCO
yolo11n vs the fine-tuned model, same frame, same conf/imgsz settings.

- Frame t=1170s: generic 10 detections (mean conf 0.78), fine-tuned 10
  detections (mean conf 0.61) -- same count, fine-tuned LESS confident.
- Frame t=1165s: generic 15 detections (mean conf 0.51), fine-tuned 13
  detections (mean conf 0.56) -- roughly the same coverage by eye
  (`outputs/finetune_compare2_*.jpg`, local only), no obvious missed players
  or new false positives either direction.

## Verdict: no clear win for player detection, real domain shift

Despite strong metrics on its own training data, the fine-tuned model does
NOT show a clear improvement over the generic COCO detector on the person's
actual footage -- comparable detection coverage, and if anything slightly
lower confidence on one of two spot-checked frames. This is a real domain-
shift result, not a wasted effort: it directly answers the question that
prompted this ("should i run these on roboflow instead of here") with a
measurement instead of an assumption in either direction.

**Recommendation**: keep `run_metrics_test.py` on generic `yolo11n.pt` for
player detection -- switching would add complexity (class remapping,
retraining if footage changes) without a demonstrated benefit. The
fine-tuned model DOES add one thing generic COCO categorically cannot: a
`frisbee` and `observer` class, which don't exist in COCO at all. Weak
(mAP50=0.156 for frisbee) but nonzero -- worth keeping `runs/detect/
frisbee_finetune/weights/best.pt` around if disc detection ever comes back
into scope (currently parked per `NEXT_STEPS.md` A4), not for player
counting today.
