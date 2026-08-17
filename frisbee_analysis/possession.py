"""
Possession detector: infer the holder sequence from kinematics.

Two stages:
  1. Per-frame emission: how holder-like is each offense player? Combines
     rel_speed (slow=holder), accel (low=holder), dist_to_defender (close=holder,
     since the holder is marked). Weights are tunable; the whole point of the
     multi-feature approach is that the MEASURED speed signal alone is only ~40%.
  2. Viterbi decode over an HMM: possession is sticky (few switches per point),
     so temporal smoothing turns a noisy 40%-per-frame signal into much better
     TRANSITION detection. Transitions are what passes/turnovers are built from.

OFFLINE by design (Viterbi needs the whole sequence) -- fine for recorded games,
and the reason a streaming SDK (DeepStream) is the wrong tool.

Defense players are NOT vetoed: the state space covers every player, offense
and defense alike, so a turnover (holder switching teams) is representable.
Defense holding is rare, so `defense_bias` is a soft per-frame log-cost prior
against it -- it must be overcome by real emission signal plus the sticky
`switch_penalty`, not just noise. This is deliberately a soft prior, not a
hard rule: all 5 ground-truth turnovers in UFATrack are terminal (possession
never reverts to offense afterward), but hard-coding that would overfit 5
examples.

STATUS: emission weights are tuned via run_tune.py (cross-validated grid/
random search maximising transition F1). See results/tuning_finding.md.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .schema import Track
from .features import compute_features, FeatureConfig


@dataclass
class HMMConfig:
    w_rel_speed: float = 3.0      # weight on (low) relative speed
    w_accel: float = 0.0          # weight on (low) acceleration; try >0
    w_dist_def: float = 0.0       # weight on (close) marker; try >0
    switch_penalty: float = 4.0   # log-cost of changing holder between frames
    max_holder_speed_frac: float = 2.5  # veto: too fast to hold
    defense_bias: float = 3.0     # log-cost prior against a defense holder
    feature_cfg: FeatureConfig = None


def _emission_logprob(track: Track, cfg: HMMConfig, feats: dict = None) -> np.ndarray:
    """(T, N) log-emission over ALL players (offense and defense). No explicit
    airborne state: UFATrack labels a holder almost every frame, so we model
    holder-only. If you later need airborne, add a state (see git history /
    synthetic version).

    `feats` lets a caller precompute compute_features() once and reuse it
    across many HMMConfig weight combinations on the same track (features
    don't depend on the weights) -- see run_tune.py, which sweeps hundreds of
    configs and would otherwise recompute identical features every time."""
    feats = feats if feats is not None else compute_features(track, cfg.feature_cfg or FeatureConfig())
    rel = feats["rel_speed"]
    acc = feats["accel"]
    dd = feats["dist_to_marker"]
    T, N = rel.shape
    emis = np.full((T, N), -np.inf)
    offense = track.team == 0

    # normalise accel and dist to comparable scale (z-ish, across everyone)
    def _norm(a):
        m = np.nanmean(a)
        s = np.nanstd(a) + 1e-9
        return (a - m) / s

    accz = _norm(acc)
    ddz = _norm(dd)

    for j in range(N):
        score = -cfg.w_rel_speed * rel[:, j]
        score += -cfg.w_accel * accz[:, j]        # low accel -> higher
        score += -cfg.w_dist_def * ddz[:, j]      # close marker -> higher
        if not offense[j]:
            score = score - cfg.defense_bias      # rare; needs real evidence
        score[rel[:, j] > cfg.max_holder_speed_frac] = -np.inf
        emis[:, j] = score
    return emis


def viterbi_decode(track: Track, cfg: HMMConfig = HMMConfig(), feats: dict = None) -> np.ndarray:
    emis = _emission_logprob(track, cfg, feats)
    T, N = emis.shape
    dp = np.full((T, N), -np.inf)
    back = np.zeros((T, N), dtype=int)
    dp[0] = emis[0]
    for t in range(1, T):
        # staying costs 0, switching costs switch_penalty
        stay = dp[t - 1]
        best_other = dp[t - 1] - cfg.switch_penalty
        # for each target j: max(stay_j, max_k!=j (dp_k - penalty))
        global_best = dp[t - 1].max() - cfg.switch_penalty
        argbest = int(dp[t - 1].argmax())
        for j in range(N):
            if stay[j] >= (global_best if argbest != j else
                           (np.partition(dp[t-1], -2)[-2] - cfg.switch_penalty
                            if N > 1 else -np.inf)):
                back[t, j] = j
                dp[t, j] = stay[j] + emis[t, j]
            else:
                # switch from the best other state
                if argbest != j:
                    back[t, j] = argbest
                    dp[t, j] = dp[t-1, argbest] - cfg.switch_penalty + emis[t, j]
                else:
                    k = int(np.argsort(dp[t-1])[-2])
                    back[t, j] = k
                    dp[t, j] = dp[t-1, k] - cfg.switch_penalty + emis[t, j]

    path = np.zeros(T, dtype=int)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return path


def naive_argmax_baseline(track: Track, cfg: HMMConfig = HMMConfig()) -> np.ndarray:
    """Per-frame best emission, no temporal smoothing. Measures what the HMM buys."""
    emis = _emission_logprob(track, cfg)
    return np.argmax(emis, axis=1)
