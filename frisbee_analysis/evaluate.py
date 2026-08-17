"""
Evaluation + event derivation.

Report TWO accuracies -- they answer different questions:
  frame_accuracy   : fraction of frames with correct holder. Dominated by long
                     holds; misleading on its own.
  transition metrics: precision/recall on possession CHANGES within a tolerance.
                     THIS governs pass/turnover quality. Watch this one.

derive_events turns a holder sequence into the ranked metrics (passes,
turnovers, completion rate). Trivial GIVEN a good holder sequence -- which is
why possession is make-or-break and everything else is bookkeeping.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .schema import Track


def frame_accuracy(pred, gt):
    mask = gt >= 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(pred[mask] == gt[mask]))


def possession_segments(holder):
    segs, cur, start = [], None, 0
    for t, h in enumerate(holder):
        if h == -1:
            continue
        if cur is None:
            cur, start = h, t
        elif h != cur:
            segs.append((start, t - 1, cur))
            cur, start = h, t
    if cur is not None:
        segs.append((start, len(holder) - 1, cur))
    return segs


def transition_frames(holder):
    return [s[0] for s in possession_segments(holder)[1:]]


def transition_metrics(pred, gt, tol=5):
    pt, gtt = transition_frames(pred), transition_frames(gt)
    matched, errs, tp = set(), [], 0
    for p in pt:
        best, bestd = None, tol + 1
        for i, g in enumerate(gtt):
            if i in matched:
                continue
            d = abs(p - g)
            if d <= tol and d < bestd:
                best, bestd = i, d
        if best is not None:
            matched.add(best); errs.append(bestd); tp += 1
    fp, fn = len(pt) - tp, len(gtt) - tp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1,
            "mean_frame_error": float(np.mean(errs)) if errs else None,
            "n_gt": len(gtt), "n_pred": len(pt)}


def derive_events(track: Track, holder):
    segs = possession_segments(holder)
    team = track.team
    passes = turnovers = 0
    for (_, _, h1), (_, _, h2) in zip(segs[:-1], segs[1:]):
        if team[h1] == team[h2]:
            passes += 1
        else:
            turnovers += 1
    total = passes + turnovers
    return {"passes": passes, "turnovers": turnovers,
            "completion_rate": passes / total if total else None,
            "n_touches": len(segs)}


@dataclass
class SequenceResult:
    frame_acc: float
    transitions: dict
    events_pred: dict
    events_gt: dict


def evaluate_sequence(track: Track, pred) -> SequenceResult:
    return SequenceResult(
        frame_acc=frame_accuracy(pred, track.holder_id),
        transitions=transition_metrics(pred, track.holder_id),
        events_pred=derive_events(track, pred),
        events_gt=derive_events(track, track.holder_id),
    )
