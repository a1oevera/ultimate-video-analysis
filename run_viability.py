"""
Possession-detector viability harness on real UFATrack.

Runs the multi-feature HMM possession detector + naive baseline and reports:
  - frame accuracy (misleading alone)
  - TRANSITION precision/recall/F1 (the metric that governs pass/turnover
    quality) -- this is the go/no-go number
  - derived events (passes, turnovers, completion rate) vs ground truth

THIS IS THE GO/NO-GO TEST. The emission weights in HMMConfig are NOT yet tuned.
First run shows the starting point; the real next task is to sweep w_accel,
w_dist_def, and switch_penalty to maximise transition F1 (see NEXT_STEPS.md).

Run:  python run_viability.py [path-to-ufatrack_data]
"""
import sys
import numpy as np
from frisbee_analysis import (load_ufatrack, viterbi_decode,
                              naive_argmax_baseline, evaluate_sequence, HMMConfig)

path = sys.argv[1] if len(sys.argv) > 1 else "ufatrack_data"
tracks = load_ufatrack(path)

# Try a couple of configs so the effect of the marking feature is visible.
configs = {
    "speed only":        HMMConfig(w_rel_speed=3.0, w_accel=0.0, w_dist_def=0.0),
    "speed+accel":       HMMConfig(w_rel_speed=3.0, w_accel=1.0, w_dist_def=0.0),
    "speed+accel+mark":  HMMConfig(w_rel_speed=3.0, w_accel=1.0, w_dist_def=1.5),
}

for name, cfg in configs.items():
    fa, f1, prec, rec = [], [], [], []
    tp_turn = gt_turn = pred_turn = 0
    for t in tracks:
        pred = viterbi_decode(t, cfg)
        res = evaluate_sequence(t, pred)
        fa.append(res.frame_acc)
        f1.append(res.transitions["f1"])
        prec.append(res.transitions["precision"])
        rec.append(res.transitions["recall"])
        gt_turn += res.events_gt["turnovers"]
        pred_turn += res.events_pred["turnovers"]
    print(f"\n=== {name} ===")
    print(f"  frame accuracy:        {np.nanmean(fa):.3f}")
    print(f"  transition precision:  {np.mean(prec):.3f}")
    print(f"  transition recall:     {np.mean(rec):.3f}")
    print(f"  transition F1:         {np.mean(f1):.3f}")
    print(f"  turnovers: gt={gt_turn}  predicted={pred_turn}")

# baseline for comparison
base_fa = []
base_f1 = []
cfg = configs["speed+accel+mark"]
for t in tracks:
    pred = naive_argmax_baseline(t, cfg)
    res = evaluate_sequence(t, pred)
    base_fa.append(res.frame_acc)
    base_f1.append(res.transitions["f1"])
print(f"\n=== naive (no HMM, speed+accel+mark) ===")
print(f"  frame accuracy: {np.nanmean(base_fa):.3f}   transition F1: {np.mean(base_f1):.3f}")

print("""
--------------------------------------------------------------------
GO / NO-GO (read transition F1, not frame accuracy):
  F1 > ~0.5  -> possession is recoverable; build the CV pipeline.
  F1 < ~0.4  -> motion+marking insufficient; possession needs disc detection
                as a primary signal, or reconsider scope.
The HMM should beat the naive baseline on F1. If it doesn't, drop it.
These configs are hand-set, not tuned. Run run_tune.py for the
cross-validated search + the actual go/no-go verdict (results/tuning_finding.md).
--------------------------------------------------------------------""")
