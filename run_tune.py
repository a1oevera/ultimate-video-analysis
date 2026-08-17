"""
Tuning harness: cross-validated search over HMMConfig to maximise transition
F1, now that A1 (frisbee_analysis/possession.py, features.py) removed the
permanent -inf veto on defense holders.

Only 20 possessions exist, and only 5 contain a ground-truth turnover (tracks
4, 8, 9, 11, 20 -- one turnover each, always the final segment in the file).
So this uses STRATIFIED 5-fold CV: each fold's held-out set gets exactly one
turnover track + 3 non-turnover tracks. That's what makes "turnover recall =
k/5" a clean per-fold hit/miss instead of a percentage on a near-empty sample.

Nested per fold: search on the OTHER 4 folds (16 tracks) to pick a config,
then score that config -- untouched -- on the held-out fold. This is what
keeps the reported F1 an honest generalisation estimate rather than a number
fit and scored on the same 20 tracks.

The "final shipped config" at the end is fit on all 20 tracks. Its F1 is NOT
the generalisation estimate -- read the cross-validated summary for that.

Run:  python run_tune.py [path-to-ufatrack_data] [n_configs]
"""
import sys
import random
import numpy as np
from frisbee_analysis import (load_ufatrack, viterbi_decode, evaluate_sequence,
                              HMMConfig, derive_events, possession_segments,
                              compute_features)

path = sys.argv[1] if len(sys.argv) > 1 else "ufatrack_data"
N_CONFIGS = int(sys.argv[2]) if len(sys.argv) > 2 else 400
TOL = 5  # same transition-match tolerance as evaluate.transition_metrics

RANGES = {
    "w_rel_speed": (0.5, 6.0),
    "w_accel": (0.0, 3.0),
    "w_dist_def": (0.0, 3.0),
    "switch_penalty": (0.5, 10.0),
    "defense_bias": (0.0, 8.0),
}

tracks = load_ufatrack(path)
# Features don't depend on HMMConfig weights -- precompute once per track and
# reuse across all N_CONFIGS candidates. Without this, hundreds of identical
# feature recomputations dominate runtime (each involves per-frame O(N^2)
# distance-to-marker work).
feats_by_track = [compute_features(t) for t in tracks]
turnover_idx = [i for i, t in enumerate(tracks)
                if derive_events(t, t.holder_id)["turnovers"] > 0]
non_turnover_idx = [i for i in range(len(tracks)) if i not in turnover_idx]
assert len(turnover_idx) == 5, f"expected 5 turnover tracks, got {len(turnover_idx)}"

# stratified folds: one turnover track + 3 non-turnover tracks each
random.Random(0).shuffle(non_turnover_idx)
folds = [[t] for t in turnover_idx]
for i, idx in enumerate(non_turnover_idx):
    folds[i % 5].append(idx)


def sample_config(rng):
    return HMMConfig(**{k: rng.uniform(*v) for k, v in RANGES.items()})


# Seed the search with the hand-set defaults from possession.py/run_viability.py
# so random search can never end up worse than the manual starting point.
SEED_CONFIGS = [
    HMMConfig(w_rel_speed=3.0, w_accel=0.0, w_dist_def=0.0),
    HMMConfig(w_rel_speed=3.0, w_accel=1.0, w_dist_def=0.0),
    HMMConfig(w_rel_speed=3.0, w_accel=1.0, w_dist_def=1.5),
]


def mean_f1(cfg, idxs):
    f1s = [evaluate_sequence(tracks[i], viterbi_decode(tracks[i], cfg, feats_by_track[i]))
           .transitions["f1"] for i in idxs]
    return float(np.mean(f1s))


def search_best(idxs, seed):
    rng = random.Random(seed)
    best_cfg, best_f1 = None, -1.0
    candidates = SEED_CONFIGS + [sample_config(rng) for _ in range(N_CONFIGS)]
    for cfg in candidates:
        f1 = mean_f1(cfg, idxs)
        if f1 > best_f1:
            best_f1, best_cfg = f1, cfg
    return best_cfg, best_f1


def turnover_hit(track, pred, tol=TOL):
    """Did any predicted transition land within `tol` frames of the true
    cross-team (turnover) transition?"""
    gt_segs = possession_segments(track.holder_id)
    pred_segs = possession_segments(pred)
    team = track.team
    gt_turn_frames = [s2[0] for s1, s2 in zip(gt_segs[:-1], gt_segs[1:])
                       if team[s1[2]] != team[s2[2]]]
    pred_trans_frames = [s[0] for s in pred_segs[1:]]
    return any(abs(g - p) <= tol for g in gt_turn_frames for p in pred_trans_frames)


def fmt_cfg(cfg):
    return (f"w_rel_speed={cfg.w_rel_speed:.2f} w_accel={cfg.w_accel:.2f} "
            f"w_dist_def={cfg.w_dist_def:.2f} switch_penalty={cfg.switch_penalty:.2f} "
            f"defense_bias={cfg.defense_bias:.2f}")


fold_f1, fold_prec, fold_rec, fold_hits = [], [], [], []

for fi, held_out in enumerate(folds):
    selection = [i for i in range(len(tracks)) if i not in held_out]
    best_cfg, sel_f1 = search_best(selection, seed=fi)

    f1s, precs, recs = [], [], []
    turnover_i = next(i for i in held_out if i in turnover_idx)
    hit = None
    for i in held_out:
        t = tracks[i]
        pred = viterbi_decode(t, best_cfg, feats_by_track[i])
        res = evaluate_sequence(t, pred)
        f1s.append(res.transitions["f1"])
        precs.append(res.transitions["precision"])
        recs.append(res.transitions["recall"])
        if i == turnover_i:
            hit = turnover_hit(t, pred)

    fold_f1.append(float(np.mean(f1s)))
    fold_prec.append(float(np.mean(precs)))
    fold_rec.append(float(np.mean(recs)))
    fold_hits.append(hit)

    held_labels = [i + 1 for i in held_out]
    print(f"\n=== Fold {fi + 1}/5 (held out: tracks {held_labels}) ===")
    print(f"  best cfg (from 16-track selection, sel F1={sel_f1:.3f}): {fmt_cfg(best_cfg)}")
    print(f"  held-out: precision={fold_prec[-1]:.3f} recall={fold_rec[-1]:.3f} "
          f"F1={fold_f1[-1]:.3f}")
    print(f"  turnover (track {turnover_i + 1}): {'HIT' if hit else 'MISS'}")

n_hits = sum(1 for h in fold_hits if h)
print(f"\n=== Cross-validated summary (5 folds, honest estimate) ===")
print(f"  mean transition F1: {np.mean(fold_f1):.3f}  "
      f"(range: {min(fold_f1):.3f}-{max(fold_f1):.3f}, std: {np.std(fold_f1):.3f})")
print(f"  mean precision:     {np.mean(fold_prec):.3f}")
print(f"  mean recall:        {np.mean(fold_rec):.3f}")
print(f"  turnover recall:    {n_hits}/5")
print("  NOTE: N=20 possessions, 5 folds -- this is a wide band, not a point")
print("  estimate. Treat the range above as the honest uncertainty, not the mean alone.")

# No-tuning baseline: the SAME fixed hand-set config (no search, no fitting at
# all) scored on the exact same held-out folds. This isolates whether the
# random search is adding real signal or just overfitting the 16-track
# selection set (transition F1 on 16 possessions is a noisy objective over a
# 5-parameter continuous space -- a real overfitting risk with this little data).
baseline_cfg = SEED_CONFIGS[-1]
base_f1, base_hits = [], []
for held_out in folds:
    f1s = []
    turnover_i = next(i for i in held_out if i in turnover_idx)
    hit = None
    for i in held_out:
        t = tracks[i]
        pred = viterbi_decode(t, baseline_cfg, feats_by_track[i])
        f1s.append(evaluate_sequence(t, pred).transitions["f1"])
        if i == turnover_i:
            hit = turnover_hit(t, pred)
    base_f1.append(float(np.mean(f1s)))
    base_hits.append(hit)

print(f"\n=== No-tuning baseline (fixed cfg, same held-out folds, zero fitting) ===")
print(f"  cfg: {fmt_cfg(baseline_cfg)}")
print(f"  mean transition F1: {np.mean(base_f1):.3f}  "
      f"(range: {min(base_f1):.3f}-{max(base_f1):.3f})")
print(f"  turnover recall:    {sum(1 for h in base_hits if h)}/5")
verdict = "search beats the fixed baseline" if np.mean(fold_f1) > np.mean(base_f1) \
    else "search did NOT beat the fixed baseline -- likely overfitting the 16-track selection set"
print(f"  -> {verdict}")

final_cfg, final_sel_f1 = search_best(list(range(len(tracks))), seed=100)
print(f"\n=== Final shipped config (fit on all 20 tracks; NOT the generalisation estimate) ===")
print(f"  {fmt_cfg(final_cfg)}")
print(f"  (in-sample F1 on all 20: {final_sel_f1:.3f} -- do not report this as the headline number)")
